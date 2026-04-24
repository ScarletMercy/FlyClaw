from __future__ import annotations

import asyncio
import fnmatch
import logging
import time
from typing import Optional

from langchain_core.tools import tool

from src.tools.exceptions import ToolExecutionError

logger = logging.getLogger("myclaw.exec")

_cached_config = None


def _get_config():
    global _cached_config
    if _cached_config is None:
        try:
            from src.config import load_config

            _cached_config = load_config()
        except Exception as e:
            logger.warning("Failed to load exec config: %s", e)
    return _cached_config


def reset_config_cache():
    global _cached_config, _exec_semaphore
    _cached_config = None
    _exec_semaphore = None


class ApprovalNeededError(Exception):
    def __init__(self, command: str, denylisted: bool):
        self.command = command
        self.denylisted = denylisted
        super().__init__(f"Approval needed for: {command[:100]}")


_DEFAULT_DENY_PATTERNS = [
    "rm -rf /",
    "rm -rf /*",
    "rm -rf .*",
    "mkfs*",
    "dd if=*of=/dev/*",
    "> /dev/sd*",
    "shutdown*",
    "reboot*",
    "init 0",
    "init 6",
    "systemctl *poweroff*",
    "systemctl *reboot*",
    "systemctl *halt*",
    ":(){ :|:& };:",
    "chmod -R 777 /",
    "chown -R * /",
    "curl*|*sh",
    "wget*|*sh",
    "python -c*import os*",
    "python3 -c*import os*",
    "nc -l*",
    "ncat*",
    "/etc/passwd",
    "/etc/shadow",
    "nohup*",
    "crontab*",
]

# Patterns that indicate shell features used to bypass denylists
_SHELL_BYPASS_PATTERNS = [
    ("sh -c", "sh -c"),
    ("bash -c", "bash -c"),
    ("zsh -c", "zsh -c"),
    ("/bin/sh", "/bin/sh"),
    ("/bin/bash", "/bin/bash"),
    ("| sh", "| sh"),
    ("| bash", "| bash"),
    ("eval ", "eval "),
]

_exec_semaphore: Optional[asyncio.Semaphore] = None


async def _exec_streaming(
    proc: asyncio.subprocess.Process,
    command: str,
    timeout: int,
    no_output_timeout: int,
    max_output: int,
    start: float,
) -> str:
    """Execute a command with streaming output and no-output timeout detection.

    When the process produces no output for no_output_timeout seconds,
    it is killed and partial output is returned.
    """
    output_event = asyncio.Event()
    out_buf: list[bytes] = []
    err_buf: list[bytes] = []
    readers_remaining = 0
    killed = False

    async def _read_stream(stream: asyncio.StreamReader, buffer: list[bytes]):
        nonlocal readers_remaining
        try:
            async for chunk in stream:
                if chunk:
                    buffer.append(chunk)
                    output_event.set()
        except Exception:
            pass
        finally:
            readers_remaining -= 1
            if readers_remaining <= 0:
                output_event.set()  # Wake up main loop so it can detect process exit

    # Spawn stream readers
    tasks = []
    if proc.stdout:
        readers_remaining += 1
        tasks.append(asyncio.create_task(_read_stream(proc.stdout, out_buf)))
    if proc.stderr:
        readers_remaining += 1
        tasks.append(asyncio.create_task(_read_stream(proc.stderr, err_buf)))

    try:
        deadline = start + timeout
        while True:
            # Check overall timeout
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                proc.kill()
                await proc.wait()
                duration = time.monotonic() - start
                logger.warning("[exec-audit] TIMEOUT dur=%.1fs cmd=%.200s", duration, command)
                raise ToolExecutionError(f"Command timed out after {timeout}s")

            # Wait for output (bounded by both timeouts)
            output_event.clear()
            wait_timeout = min(remaining, no_output_timeout)
            try:
                await asyncio.wait_for(output_event.wait(), timeout=wait_timeout)
            except asyncio.TimeoutError:
                if time.monotonic() >= deadline:
                    proc.kill()
                    await proc.wait()
                    duration = time.monotonic() - start
                    logger.warning("[exec-audit] TIMEOUT dur=%.1fs cmd=%.200s", duration, command)
                    raise ToolExecutionError(f"Command timed out after {timeout}s")

                # No-output timeout: kill the process
                killed = True
                proc.kill()
                await proc.wait()
                break

            # Check if process exited
            if proc.returncode is not None:
                # Drain remaining readers briefly
                await asyncio.gather(*tasks, return_exceptions=True)
                break
    except ToolExecutionError:
        raise
    except Exception as e:
        proc.kill()
        await proc.wait()
        duration = time.monotonic() - start
        logger.error("[exec-audit] ERROR dur=%.1fs cmd=%.200s: %s", duration, command, e)
        raise ToolExecutionError(f"{type(e).__name__}: {e}")
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    # Build output from buffers
    exit_code = proc.returncode or 0
    output_parts = []
    if out_buf:
        output_parts.append(b"".join(out_buf).decode("utf-8", errors="replace"))
    if err_buf:
        output_parts.append(f"[stderr]\n{b''.join(err_buf).decode('utf-8', errors='replace')}")
    output = "\n".join(output_parts) or "(no output)"

    if killed:
        output += f"\n[killed: no output for {no_output_timeout}s]"

    if len(output.encode("utf-8", errors="replace")) > max_output:
        encoded = output.encode("utf-8", errors="replace")[:max_output]
        output = encoded.decode("utf-8", errors="replace") + f"\n... [truncated at {max_output} bytes] ...\n"

    if exit_code != 0 and not killed:
        output += f"\n[exit code: {exit_code}]"

    duration = time.monotonic() - start
    logger.info(
        "[exec-audit] exit=%d dur=%.1fs killed=%s cmd=%.200s",
        exit_code,
        duration,
        killed,
        command,
    )
    return output


def _get_semaphore(max_concurrent: int) -> asyncio.Semaphore:
    global _exec_semaphore
    if _exec_semaphore is None:
        _exec_semaphore = asyncio.Semaphore(max_concurrent)
    return _exec_semaphore


def _is_denylisted(command: str, deny_patterns: list[str]) -> tuple[bool, str]:
    cmd_lower = command.strip().lower()
    for pattern in deny_patterns:
        pattern_lower = pattern.lower()
        if fnmatch.fnmatch(cmd_lower, pattern_lower):
            return True, pattern
        if pattern_lower in cmd_lower:
            return True, pattern
    return False, ""


def _has_shell_bypass(command: str) -> tuple[bool, str]:
    """Detect shell features commonly used to bypass denylists.

    Catches patterns like: sh -c 'rm -rf /', echo x | sh, /bin/bash -c ...
    """
    cmd_lower = command.strip().lower()
    for trigger, label in _SHELL_BYPASS_PATTERNS:
        if trigger in cmd_lower:
            return True, label
    # Detect command substitution $(...) and backticks
    if "$(" in command or "`" in command:
        return True, "command substitution"
    return False, ""


@tool
async def exec_command(
    command: str,
    timeout: Optional[int] = 30,
    workdir: Optional[str] = None,
) -> str:
    """Execute a shell command and return its output.

    Args:
        command: The shell command to execute.
        timeout: Timeout in seconds. Default 30.
        workdir: Working directory. Defaults to current directory.
    """
    cfg = _get_config()

    deny_patterns = (cfg.tools.exec.deny_patterns if cfg else None) or _DEFAULT_DENY_PATTERNS
    max_output = (cfg.tools.exec.max_output_bytes if cfg else None) or 102400
    max_concurrent = (cfg.tools.exec.max_concurrent if cfg else None) or 3
    approval_mode = (cfg.tools.exec.approval_mode if cfg else None) or "off"
    no_output_timeout_val = (cfg.tools.exec.no_output_timeout_seconds if cfg else None) or 0
    sandbox_enabled = cfg.tools.exec.sandbox_enabled if cfg else True
    sandbox_allowed_dirs = cfg.tools.exec.sandbox_allowed_dirs if cfg else ["."]
    sandbox_env_whitelist = cfg.tools.exec.sandbox_env_whitelist if cfg else ["PATH", "HOME", "USERPROFILE", "SYSTEMROOT", "LANG", "PYTHONPATH"]

    # Sandbox: working directory restriction
    if sandbox_enabled:
        from pathlib import Path
        workspace = Path(cfg.agents.workspace).resolve()
        if workdir:
            wd = Path(workdir).resolve()
        else:
            wd = Path(".").resolve()

        allowed = [workspace.resolve()] + [Path(d).expanduser().resolve() for d in sandbox_allowed_dirs]
        if not any(str(wd).startswith(str(a)) for a in allowed):
            raise ToolExecutionError(f"Working directory not allowed by sandbox: {wd}")

    blocked, matched = _is_denylisted(command, deny_patterns)

    if blocked:
        logger.warning("[exec-audit] DENIED command matches pattern '%s': %.200s", matched, command)
        raise ToolExecutionError(f"Command blocked by denylist (matched: {matched})")

    bypass, bypass_detail = _has_shell_bypass(command)
    if bypass and approval_mode == "off":
        logger.warning(
            "[exec-audit] BLOCKED shell bypass detected ('%s') with approval_mode=off: %.200s",
            bypass_detail,
            command,
        )
        raise ToolExecutionError(
            f"Command uses shell bypass pattern ('{bypass_detail}') — enable approval_mode to allow"
        )

    if bypass and approval_mode != "off" and approval_mode != "always":
        # Treat shell bypass patterns as requiring approval even in "on_denylist_miss" mode
        from src.tools.approval import get_approval_manager

        mgr = get_approval_manager()
        if not mgr.has_durable_approval("exec_command", command):
            logger.info("[exec-audit] Shell bypass detected ('%s'), requiring approval: %.200s", bypass_detail, command)
            raise ApprovalNeededError(command, False)

    if approval_mode != "off":
        from src.tools.approval import get_approval_manager

        mgr = get_approval_manager()
        needs = mgr.needs_approval("exec_command", command, approval_mode, blocked)
        if needs and not mgr.has_durable_approval("exec_command", command):
            raise ApprovalNeededError(command, blocked)

    sem = _get_semaphore(max_concurrent)
    if sem.locked():
        logger.info("[exec-audit] QUEUED (semaphore full): %.200s", command)

    async with sem:
        start = time.monotonic()
        exit_code = -1
        no_output_timeout = no_output_timeout_val  # captured from config above

        try:
            # Sandbox: sanitize environment
            env = None
            if sandbox_enabled:
                import os
                env = {k: os.environ[k] for k in sandbox_env_whitelist if k in os.environ}

            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workdir,
                env=env,
            )

            if no_output_timeout > 0:
                # Streaming mode: detect no-output timeout
                return await _exec_streaming(proc, command, timeout, no_output_timeout, max_output, start)
            else:
                # Original buffered mode (backward compatible)
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                exit_code = proc.returncode or 0

                output_parts = []
                if stdout:
                    output_parts.append(stdout.decode("utf-8", errors="replace"))
                if stderr:
                    output_parts.append(f"[stderr]\n{stderr.decode("utf-8", errors="replace")}")
                output = "\n".join(output_parts) or "(no output)"

                if len(output.encode("utf-8", errors="replace")) > max_output:
                    encoded = output.encode("utf-8", errors="replace")[:max_output]
                    output = encoded.decode("utf-8", errors="replace") + f"\n... [truncated at {max_output} bytes] ...\n"

                if exit_code != 0:
                    output += f"\n[exit code: {exit_code}]"

                duration = time.monotonic() - start
                logger.info(
                    "[exec-audit] exit=%d dur=%.1fs cmd=%.200s",
                    exit_code,
                    duration,
                    command,
                )
                return output

        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            duration = time.monotonic() - start
            logger.warning("[exec-audit] TIMEOUT dur=%.1fs cmd=%.200s", duration, command)
            raise ToolExecutionError(f"Command timed out after {timeout}s")
        except ToolExecutionError:
            raise
        except Exception as e:
            duration = time.monotonic() - start
            logger.error("[exec-audit] ERROR dur=%.1fs cmd=%.200s: %s", duration, command, e)
            raise ToolExecutionError(f"{type(e).__name__}: {e}")
