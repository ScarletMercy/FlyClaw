from __future__ import annotations

import asyncio
import fnmatch
import json
import logging
import os
import time
from contextvars import ContextVar
from typing import Literal, Optional

from src.tools.exceptions import ToolExecutionError
from src.tools.process import kill_process_tree

logger = logging.getLogger("flyclaw.exec")

_current_thread_id: ContextVar[str] = ContextVar("_current_thread_id", default="")

_current_agent_context: ContextVar[dict] = ContextVar("_current_agent_context", default={})

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


def _resolve_default_workdir() -> str:
    from pathlib import Path

    cfg = _get_config()
    if cfg:
        ws = cfg.agents.workspace
    else:
        from src.config import AgentConfig

        ws = AgentConfig().workspace
    resolved = str(Path(ws).expanduser().resolve())
    Path(resolved).mkdir(parents=True, exist_ok=True)
    return resolved


def reset_config_cache():
    global _cached_config
    _cached_config = None


def _collect_skill_dirs() -> list:
    """复用 App._build_skill_directories() 的集中目录列表，消除重复。"""
    from src._container import get_container

    container = get_container()
    return [p for _, p in container._build_skill_directories()]


def set_sandbox_enabled(enabled: bool) -> None:
    """Toggle sandbox at runtime — updates in-memory config and persists to YAML."""
    cfg = _get_config()
    if cfg is None:
        return
    cfg.tools.exec.sandbox_enabled = enabled
    try:
        from src.config import save_config

        save_config(cfg)
    except Exception as e:
        logger.warning("Failed to persist sandbox config: %s", e)


def is_sandbox_enabled() -> bool:
    cfg = _get_config()
    return cfg.tools.exec.sandbox_enabled if cfg else True


class ApprovalNeededError(Exception):
    def __init__(
        self,
        command: str,
        denylisted: bool,
        timeout: int | None = None,
        auto_deny: bool = False,
        approval_key: str = "",
    ):
        self.command = command
        self.denylisted = denylisted
        self.timeout = timeout
        self.auto_deny = auto_deny
        self.approval_key = approval_key  # Pattern for "always allow" (e.g. "del ", "rm ")
        super().__init__(f"需要审批: {command[:100]}")


_DEFAULT_DENY_PATTERNS = [
    # Recursive delete — always blocked
    "rm -rf",
    "rm -r",
    "rm -R",
    "rmdir /s",
    "rd /s",
    "rd /s/q",
    "rd /s /q",
    "shutil.rmtree",
    # System destruction
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
    "nc -l*",
    "ncat*",
    "/etc/passwd",
    "/etc/shadow",
    "crontab -r",
    # Windows — disk/volume destruction
    "format *",
    "diskpart",
    "bcdedit",
    "bootsect",
    "vssadmin delete*",
    "cipher /w*",
    "fsutil*",
    # Windows — persistence & privilege escalation
    "schtasks*/create*",
    "schtasks*/delete*",
    "net user */add*",
    "net localgroup**/add*",
    "REG ADD *\\Run*",
    "REG ADD *\\RunOnce*",
    "sc create *",
    "sc config *",
    "New-Service *",
    # Windows — defense evasion
    "wevtutil cl*",
    "Set-MpPreference *",
    "taskkill /f *",
    "net stop *",
    "sc stop *",
    # Windows — credential theft
    "reg save hklm*",
    "reg save hklm\\*",
    "ntdsutil *",
    # Windows — LOLBin / download & execute
    "certutil -urlcache*",
    "certutil -f *",
    "bitsadmin /transfer*",
    "mshta *",
    "msiexec *",
    "rundll32 *javascript*",
    "regsvr32 */i:*",
]

# Non-recursive delete — always requires approval (even if approval_mode=off)
_DELETE_APPROVAL_PATTERNS = [
    "rm ",
    "del ",
    "erase ",
    "rmdir ",
    "rd ",
    "os.remove",
    "os.unlink",
    "remove-item",
    "ri ",
]

# Patterns that indicate shell features used to bypass denylists.
# Split into two categories:
# - Unparseable: always blocked regardless of approval_mode (cannot statically analyze)
# - Executor: only blocked if denylist also matches (safe subcommands can pass)

_UNPARSEABLE_SHELL_PATTERNS = [
    # Pipe to shell — with and without space
    ("|sh", "| sh"),
    ("| sh", "| sh"),
    ("|bash", "| bash"),
    ("| bash", "| bash"),
    ("|zsh", "| zsh"),
    ("| zsh", "| zsh"),
    ("|dash", "| dash"),
    ("| dash", "| dash"),
    ("|ksh", "| ksh"),
    ("| ksh", "| ksh"),
    ("|fish", "| fish"),
    ("| fish", "| fish"),
    # Pipe to interpreters that can execute arbitrary code
    ("|python", "| python"),
    ("| python", "| python"),
    ("|perl", "| perl"),
    ("| perl", "| perl"),
    ("|ruby", "| ruby"),
    ("| ruby", "| ruby"),
    ("|node", "| node"),
    ("| node", "| node"),
]

_SHELL_EXECUTOR_PATTERNS = [
    ("sh -c", "sh -c"),
    ("bash -c", "bash -c"),
    ("zsh -c", "zsh -c"),
    ("perl -e", "perl -e"),
    ("ruby -e", "ruby -e"),
    ("node -e", "node -e"),
    ("/bin/sh", "/bin/sh"),
    ("/bin/bash", "/bin/bash"),
    ("eval ", "eval "),
    ("cmd /c ", "cmd /c"),
    ("cmd /r ", "cmd /r"),
    ("cmd /k ", "cmd /k"),
    ("powershell -encodedcommand ", "powershell -encodedcommand"),
    ("powershell -enc ", "powershell -enc"),
    ("powershell -e ", "powershell -e (encoded)"),
    ("pwsh -encodedcommand ", "pwsh -encodedcommand"),
    ("pwsh -enc ", "pwsh -enc"),
    ("pwsh -e ", "pwsh -e (encoded)"),
    ("powershell -command ", "powershell -command"),
    ("pwsh -command ", "pwsh -command"),
]

# Command prefixes whose subcommands should not trigger denylist/delete checks.
# e.g. "git rm", "docker rm" — these operate on their own scope, not the filesystem.
_COMMAND_PREFIX_WHITELIST = frozenset({"git", "docker", "kubectl", "podman", "npm", "yarn", "pnpm"})


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
        except Exception as e:
            logger.debug("Stream read error: %s", e)
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

    async def _safe_kill():
        """Kill process tree, ignoring ProcessLookupError if already exited."""
        await kill_process_tree(proc)

    async def _wait_for_output_or_exit():
        """Wait until we get output OR the process exits."""
        while proc.returncode is None:
            try:
                await asyncio.wait_for(output_event.wait(), timeout=0.5)
                return True  # Got output
            except asyncio.TimeoutError:
                continue
        return False  # Process exited

    try:
        deadline = start + timeout
        while True:
            # Check if process already exited
            if proc.returncode is not None:
                await asyncio.gather(*tasks, return_exceptions=True)
                break

            # Check overall timeout
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                await _safe_kill()
                duration = time.monotonic() - start
                logger.warning("[exec-audit] TIMEOUT dur=%.1fs cmd=%.200s", duration, command)
                raise ToolExecutionError(f"Command timed out after {timeout}s")

            # Wait for output or process exit (bounded by both timeouts)
            wait_timeout = min(remaining, no_output_timeout)

            got_output = await asyncio.wait_for(_wait_for_output_or_exit(), timeout=wait_timeout)

            # Re-check process exit after waiting
            if proc.returncode is not None:
                await asyncio.gather(*tasks, return_exceptions=True)
                break

            if not got_output:
                # No-output timeout
                if time.monotonic() >= deadline:
                    await _safe_kill()
                    duration = time.monotonic() - start
                    logger.warning("[exec-audit] TIMEOUT dur=%.1fs cmd=%.200s", duration, command)
                    raise ToolExecutionError(f"Command timed out after {timeout}s")

                killed = True
                await _safe_kill()
                break

            # Got output, clear event and loop back
            output_event.clear()
    except ToolExecutionError:
        raise
    except Exception as e:
        await _safe_kill()
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


def _get_first_token(command: str) -> str:
    """Extract the first token of a command for prefix whitelisting."""
    stripped = command.strip().lower()
    for ch in stripped:
        if ch in (" ", "\t", "\n", "\r"):
            break
    else:
        return stripped
    return stripped.split(None, 1)[0] if stripped else ""


def _is_denylisted(command: str, deny_patterns: list[str]) -> tuple[bool, str]:
    """Check if a command matches any deny pattern.

    Two-layer matching (both intentional, not redundant):
    1. fnmatch — full-string glob matching for patterns with wildcards (e.g. ``dd if=*of=/dev/*``)
    2. regex substring — ``\\b``-bounded search catches prefixed variants
       that fnmatch misses (e.g. ``sudo rm -rf /`` won't fnmatch ``rm -rf``,
       but ``\\brm\\b`` finds it).
    """
    import re

    cmd_normalized = re.sub(r"\s+", " ", command.strip()).lower()
    for pattern in deny_patterns:
        pattern_lower = pattern.lower()
        if fnmatch.fnmatch(cmd_normalized, pattern_lower):
            return True, pattern
        # fnmatch is full-string — missed if command has a prefix/suffix.
        # regex substring + word boundary catches those cases.
        start_boundary = r"\b" if pattern_lower[0:1].isalnum() or pattern_lower[0:1] == "_" else ""
        end_boundary = r"\b" if pattern_lower[-1:].isalnum() or pattern_lower[-1:] == "_" else ""
        try:
            if re.search(start_boundary + re.escape(pattern_lower) + end_boundary, cmd_normalized):
                return True, pattern
        except re.error:
            if pattern_lower in cmd_normalized:
                return True, pattern
    return False, ""


_SHELL_SPECIAL_VAR_RE = __import__("re").compile(
    r"\$\d"  # $0, $1, ... positional parameters
    r"|\$[!$?#@*%-]",  # $!, $$, $?, $#, $@, $*, $%, $-
)


def _has_unparseable_shell(command: str) -> tuple[bool, str]:
    """Detect unparseable shell constructs that cannot be statically analyzed.

    These are always blocked regardless of approval_mode because the denylist
    cannot reason about runtime-expanded content.

    Detected patterns:
    - Command substitution: $(...) and backticks
    - Variable expansion with braces: ${...}
    - Arithmetic expansion: $((...))
    - Process substitution: <(...) and >(...)
    - Pipe to shell/interpreter: | sh, | bash, | python, etc.
    - Shell special variables: $0-$9, $!, $$, $?, $#, $@, $*
    """
    cmd_lower = command.strip().lower()
    for trigger, label in _UNPARSEABLE_SHELL_PATTERNS:
        if trigger in cmd_lower:
            return True, label
    # Order matters: check more-specific patterns before their general forms.
    # $(( contains $(, so $(( must be checked first.
    if "${" in command:
        return True, "variable expansion"
    if "$((" in command:
        return True, "arithmetic expansion"
    if "$(" in command:
        return True, "command substitution"
    if "`" in command:
        return True, "backtick substitution"
    if "<(" in command or ">(" in command:
        return True, "process substitution"
    m = _SHELL_SPECIAL_VAR_RE.search(command)
    if m:
        return True, f"shell special variable/expansion ({m.group(0)!r})"
    return False, ""


def _has_shell_executor(command: str) -> tuple[bool, str]:
    """Detect shell executor patterns (sh -c, cmd /c, powershell -command, etc.).

    These are NOT inherently dangerous — the denylist already checks the full
    command string for dangerous tokens. They only matter for approval gating.
    """
    cmd_lower = command.strip().lower()
    for trigger, label in _SHELL_EXECUTOR_PATTERNS:
        if trigger in cmd_lower:
            return True, label
    return False, ""


async def exec_command(
    command: str,
    timeout: Optional[int] = 30,
    workdir: Optional[str] = None,
    background: bool = False,
) -> str:
    """Execute a shell command and return its output.

    Args:
        command: The shell command to execute.
        timeout: Timeout in seconds. Default 30.
        workdir: Working directory. Defaults to agents.workspace in config.yaml.
        background: Run in background. Returns session ID immediately.
    """
    if timeout is None:
        timeout = 30
    elif timeout <= 0:
        raise ToolExecutionError(f"timeout 必须为正整数，收到: {timeout}")

    cfg = _get_config()

    deny_patterns = (cfg.tools.exec.deny_patterns if cfg else None) or _DEFAULT_DENY_PATTERNS
    max_output = (cfg.tools.exec.max_output_bytes if cfg else None) or 102400
    approval_mode = (cfg.tools.exec.approval_mode if cfg else None) or "off"
    no_output_timeout_val = (cfg.tools.exec.no_output_timeout_seconds if cfg else None) or 0
    sandbox_enabled = cfg.tools.exec.sandbox_enabled if cfg else True
    sandbox_allowed_dirs = cfg.tools.exec.sandbox_allowed_dirs if cfg else ["."]
    sandbox_env_whitelist = (
        cfg.tools.exec.sandbox_env_whitelist
        if cfg
        else ["PATH", "HOME", "USERPROFILE", "SYSTEMROOT", "SystemRoot", "COMSPEC", "LANG", "PYTHONPATH"]
    )

    if workdir is None:
        workdir = _resolve_default_workdir()

    # Sandbox: working directory restriction
    if sandbox_enabled:
        from pathlib import Path

        workspace = Path(cfg.agents.workspace).expanduser().resolve()
        if workdir:
            wd = Path(workdir).resolve()
        else:
            wd = workspace

        allowed = (
            [workspace.resolve()]
            + [Path(d).expanduser().resolve() for d in sandbox_allowed_dirs]
            + _collect_skill_dirs()
            + [(Path.home() / ".flyclaw" / "temp").resolve()]
        )

        def _is_under(child: Path, parent: Path) -> bool:
            try:
                child.relative_to(parent)
                return True
            except ValueError:
                return False

        if not any(_is_under(wd, a) for a in allowed):
            raise ToolExecutionError(f"当前为沙盒模式，无法访问工作目录之外的路径：{wd}（允许范围：{workspace}）")

        # Use resolved path to prevent TOCTOU symlink race
        workdir = str(wd)

    # Background mode: spawn via ProcessRegistry and return immediately
    if background:
        from src.tools.process import get_process_registry

        blocked_bg, matched_bg = _is_denylisted(command, deny_patterns)
        if blocked_bg:
            raise ToolExecutionError(f"Command blocked by denylist (matched: {matched_bg})")

        # Unparseable shell constructs — always blocked in background mode.
        # These cannot be statically analyzed and are the primary bypass vector.
        unparseable_bg, unparseable_bg_detail = _has_unparseable_shell(command)
        if unparseable_bg:
            logger.warning(
                "[exec-audit] BLOCKED unparseable shell in background mode ('%s'): %.200s",
                unparseable_bg_detail,
                command,
            )
            raise ToolExecutionError(
                f"Command uses unparseable shell construct ('{unparseable_bg_detail}') "
                f"and cannot be run in background mode"
            )

        # Shell executor patterns — blocked in background mode.
        # Approval gating is not feasible for async processes; block outright.
        executor_bg, executor_bg_detail = _has_shell_executor(command)
        if executor_bg:
            logger.warning(
                "[exec-audit] BLOCKED shell executor in background mode ('%s'): %.200s",
                executor_bg_detail,
                command,
            )
            raise ToolExecutionError(
                f"Command uses shell executor ('{executor_bg_detail}') and cannot be run in background mode"
            )

        # Non-recursive delete — blocked in background mode (approval not feasible).
        if _get_first_token(command) not in _COMMAND_PREFIX_WHITELIST:
            delete_bg, delete_bg_pattern = _is_denylisted(command, _DELETE_APPROVAL_PATTERNS)
            if delete_bg:
                logger.warning(
                    "[exec-audit] BLOCKED delete operation in background mode ('%s'): %.200s",
                    delete_bg_pattern,
                    command,
                )
                raise ToolExecutionError(f"Delete operation ('{delete_bg_pattern}') cannot be run in background mode")

        env = None
        if sandbox_enabled:
            env = {k: os.environ[k] for k in sandbox_env_whitelist if k in os.environ}

        registry = get_process_registry()
        session_id = await registry.spawn(command, workdir=workdir, env=env)
        return json.dumps(
            {
                "session_id": session_id,
                "status": "running",
                "command": command[:200],
            }
        )

    blocked, matched = _is_denylisted(command, deny_patterns)

    if blocked:
        logger.warning("[exec-audit] DENIED command matches pattern '%s': %.200s", matched, command)
        raise ToolExecutionError(f"Command blocked by denylist (matched: {matched})")

    # Non-recursive delete — always requires approval (30s timeout, auto-deny)
    # Whitelisted prefixes (git, docker, etc.) skip delete approval — their "rm"
    # subcommands operate within their own scope, not the filesystem directly.
    if _get_first_token(command) not in _COMMAND_PREFIX_WHITELIST:
        delete_match, delete_pattern = _is_denylisted(command, _DELETE_APPROVAL_PATTERNS)
        if delete_match:
            from src.tools.approval import get_approval_manager

            mgr = get_approval_manager()
            if not mgr.has_durable_approval("exec_command", command):
                if not mgr.has_session_approval(_current_thread_id.get(""), "exec_command", command):
                    logger.info(
                        "[exec-audit] Delete operation requires approval ('%s'): %.200s", delete_pattern, command
                    )
                    raise ApprovalNeededError(
                        command,
                        False,
                        timeout=30,
                        auto_deny=True,
                        approval_key=delete_pattern,
                    )

    # Unparseable shell constructs ($(), backticks, | sh, | bash) — always blocked.
    # These cannot be statically analyzed and are the primary vector for bypass attacks.
    unparseable, unparseable_detail = _has_unparseable_shell(command)
    if unparseable:
        logger.warning(
            "[exec-audit] BLOCKED unparseable shell construct ('%s'): %.200s",
            unparseable_detail,
            command,
        )
        raise ToolExecutionError(f"Command uses unparseable shell construct ('{unparseable_detail}')")

    # Shell executor patterns (sh -c, cmd /c, powershell -command, etc.).
    # These are safe to pass through — the denylist already checked the full command
    # for dangerous tokens. Only apply approval gating if configured.
    executor, executor_detail = _has_shell_executor(command)
    if executor and approval_mode not in ("off", ""):
        from src.tools.approval import get_approval_manager

        mgr = get_approval_manager()
        if not mgr.has_durable_approval("exec_command", command):
            if not mgr.has_session_approval(_current_thread_id.get(""), "exec_command", command):
                logger.info(
                    "[exec-audit] Shell executor ('%s'), requiring approval: %.200s",
                    executor_detail,
                    command,
                )
                raise ApprovalNeededError(command, False)

    if approval_mode not in ("off", ""):
        from src.tools.approval import get_approval_manager

        mgr = get_approval_manager()
        needs = mgr.needs_approval("exec_command", command, approval_mode, blocked)
        if needs and not mgr.has_durable_approval("exec_command", command):
            if not mgr.has_session_approval(_current_thread_id.get(""), "exec_command", command):
                raise ApprovalNeededError(command, blocked)

    start = time.monotonic()
    exit_code = -1
    no_output_timeout = no_output_timeout_val  # captured from config above

    try:
        # Sandbox: sanitize environment
        env = None
        if sandbox_enabled:
            env = {k: os.environ[k] for k in sandbox_env_whitelist if k in os.environ}

        proc_kwargs: dict = dict(
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workdir,
            env=env,
        )
        if os.name != "nt":
            proc_kwargs["start_new_session"] = True
        proc = await asyncio.create_subprocess_shell(command, **proc_kwargs)

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
                output_parts.append(f"[stderr]\n{stderr.decode('utf-8', errors='replace')}")
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
        await kill_process_tree(proc)
        duration = time.monotonic() - start
        logger.warning("[exec-audit] TIMEOUT dur=%.1fs cmd=%.200s", duration, command)
        raise ToolExecutionError(f"Command timed out after {timeout}s")
    except ToolExecutionError:
        raise
    except Exception as e:
        duration = time.monotonic() - start
        logger.error("[exec-audit] ERROR dur=%.1fs cmd=%.200s: %s", duration, command, e)
        raise ToolExecutionError(f"{type(e).__name__}: {e}")


async def process_status(
    action: Literal["list", "poll", "wait", "kill", "log"],
    session_id: str = "",
    timeout: int = 300,
    offset: int = 0,
    limit: int = 200,
) -> str:
    """Manage background processes started with exec_command(background=true).

    Args:
        action: One of: list, poll, wait, kill, log.
        session_id: Process session ID (required for poll/wait/kill/log).
        timeout: Timeout in seconds for wait action. Default 300.
        offset: Line offset for log action. Default 0.
        limit: Max lines for log action. Default 200.
    """
    from src.tools.process import get_process_registry

    registry = get_process_registry()
    normalized = action.strip().lower()

    if normalized == "list":
        sessions = registry.list_sessions()
        if not sessions:
            return "No background processes."
        lines = []
        for s in sessions:
            status_icon = "\U0001f504" if s["status"] == "running" else "✅" if s["exit_code"] == 0 else "❌"
            lines.append(f"  {status_icon} [{s['id']}] pid={s['pid']} ({s['elapsed']}s) {s['command']}")
        return "\n".join(lines)

    if normalized == "poll":
        if not session_id:
            return "Error: session_id required for poll."
        result = await registry.poll(session_id)
        return json.dumps(result, ensure_ascii=False)

    if normalized == "wait":
        if not session_id:
            return "Error: session_id required for wait."
        result = await registry.wait(session_id, timeout=timeout)
        return json.dumps(result, ensure_ascii=False)

    if normalized == "kill":
        if not session_id:
            return "Error: session_id required for kill."
        return await registry.kill(session_id)

    if normalized == "log":
        if not session_id:
            return "Error: session_id required for log."
        return await registry.log(session_id, offset=offset, limit=limit)

    return f"Unknown action: '{normalized}'. Use: list, poll, wait, kill, log."


def get_tools() -> list:
    from src.agent.tooldef import ToolDef

    return [
        ToolDef.from_function(exec_command),
        ToolDef.from_function(process_status),
    ]
