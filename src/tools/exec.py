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
    global _cached_config
    _cached_config = None


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

_exec_semaphore: Optional[asyncio.Semaphore] = None


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

    blocked, matched = _is_denylisted(command, deny_patterns)

    if approval_mode != "off":
        from src.tools.approval import get_approval_manager

        mgr = get_approval_manager()
        needs = mgr.needs_approval("exec_command", command, approval_mode, blocked)
        if needs and not mgr.has_durable_approval("exec_command", command):
            raise ApprovalNeededError(command, blocked)

    if blocked:
        logger.warning("[exec-audit] DENIED command matches pattern '%s': %.200s", matched, command)
        raise ToolExecutionError(f"Command blocked by denylist (matched: {matched})")

    sem = _get_semaphore(max_concurrent)
    if sem.locked():
        logger.info("[exec-audit] QUEUED (semaphore full): %.200s", command)

    async with sem:
        start = time.monotonic()
        exit_code = -1
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workdir,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            exit_code = proc.returncode or 0

            output_parts = []
            if stdout:
                output_parts.append(stdout.decode("utf-8", errors="replace"))
            if stderr:
                output_parts.append(f"[stderr]\n{stderr.decode('utf-8', errors='replace')}")
            output = "\n".join(output_parts) or "(no output)"

            if len(output.encode("utf-8", errors="replace")) > max_output:
                output = (
                    output[: max_output // 2] + f"\n... [truncated at {max_output} bytes] ...\n"
                )

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
            duration = time.monotonic() - start
            logger.warning("[exec-audit] TIMEOUT dur=%.1fs cmd=%.200s", duration, command)
            raise ToolExecutionError(f"Command timed out after {timeout}s")
        except Exception as e:
            duration = time.monotonic() - start
            logger.error("[exec-audit] ERROR dur=%.1fs cmd=%.200s: %s", duration, command, e)
            raise ToolExecutionError(f"{type(e).__name__}: {e}")
