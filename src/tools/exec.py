"""exec 工具 — LLM 命令执行流水线。

用法:
  exec_command(command, timeout=300, workdir=None, background=False)
  process_status(action, session_id, ...)   # 后台进程管理

流水线:
  timeout/参数校验 → deny 闸（hardline 底线 + 可配置 denylist，前台/后台共用）
  → workdir 解析（显式参数 > 追踪 cwd > 默认 workspace）+ 沙箱目录校验
  → 前台: 删除类强制审批 → shell executor 审批 → 常规审批
    → provider 执行 → 输出解码/截断落盘 → 退出码语义翻译 → cwd 读回跟踪 → 审计日志
  → 后台: executor 阻断 → 删除阻断 → registry.spawn

威胁模型与安全分层（hardline/denylist/审批/沙箱各管什么、明确不防什么）:
  见 command_guard 模块 docstring。

shell 选择（tools.exec.shell=auto）:
  三平台统一优先 bash（Windows=Git Bash，不可用降级 PowerShell）；system 走
  create_subprocess_shell 系统默认（Windows=cmd，POSIX=/bin/sh）。
"""

from __future__ import annotations

import asyncio
import json
import locale
import logging
import os
import re
import time
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Literal, Optional

from src.tools.command_guard import (
    COMMAND_PREFIX_WHITELIST,
    DEFAULT_DENY_PATTERNS,
    DELETE_APPROVAL_PATTERNS,
    FORCE_KILL_APPROVAL_PATTERNS,
    basename,
    is_denylisted,
    check_hardline,
    has_shell_executor,
)
from src.tools.exceptions import ToolExecutionError
from src.tools.process import kill_process_tree
from src.tools.shell_provider import ShellProvider, msys_to_windows, resolve_shell_provider

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


def reset_config_cache():
    global _cached_config
    _cached_config = None


def _resolve_default_workdir() -> str:
    cfg = _get_config()
    if cfg:
        ws = cfg.agents.workspace
    else:
        from src.config import AgentConfig

        ws = AgentConfig().workspace
    resolved = str(Path(ws).expanduser().resolve())
    Path(resolved).mkdir(parents=True, exist_ok=True)
    return resolved


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


def is_delete_command(command: str) -> bool:
    """命令文本是否匹配删除模式(del/rm/rmdir/erase/remove-item…)。

    供审批提示标注「（删除文件）」复用——"何为删除"在此单一来源
    (command_guard 的 DELETE_APPROVAL_PATTERNS)。display-only：只影响标签
    文案，不影响是否要求审批(那由 exec_command 的 gate 决定)。
    """
    return bool(command) and is_denylisted(command, DELETE_APPROVAL_PATTERNS)[0]


# ---------------------------------------------------------------------------
# workdir / 沙箱目录
#
# workdir 不做字符校验：它只作为 Popen(cwd=...) 的 OS 级参数使用，从不进入
# shell 解析——坏路径只会得到干净的「目录不存在」错误。
# ---------------------------------------------------------------------------


def _sandbox_allowed_dirs(cfg) -> list[Path]:
    """沙箱允许目录列表。cfg 不可为 None——配置不可用的情形由调用方守卫。"""
    from src.instance import temp_dir

    allowed = [Path(cfg.agents.workspace).expanduser().resolve()]
    for d in cfg.tools.exec.sandbox_allowed_dirs:
        allowed.append(Path(d).expanduser().resolve())
    try:
        allowed += [Path(p).resolve() for p in _collect_skill_dirs()]
    except Exception:
        pass
    allowed.append(temp_dir().resolve())
    return allowed


def _sandbox_path_allowed(path: str, cfg) -> bool:
    """路径是否在沙箱允许域内（workdir 闸与 cwd 追踪共用的判定）。

    沙箱关闭不设限。cfg 不可为 None（exec_command 入口保证）。
    """
    if not cfg.tools.exec.sandbox_enabled:
        return True
    wd = Path(path).resolve()
    for parent in _sandbox_allowed_dirs(cfg):
        try:
            wd.relative_to(parent)
            return True
        except ValueError:
            continue
    return False


# ---------------------------------------------------------------------------
# 输出：解码 → 截断落盘 → 退出码语义
# ---------------------------------------------------------------------------


def _robust_decode(data: bytes) -> str:
    """多级解码：BOM → UTF-8 → 系统 locale → GB18030 → latin-1。"""
    if not data:
        return ""
    if len(data) >= 3 and data[:3] == b"\xef\xbb\xbf":
        return data[3:].decode("utf-8", errors="replace")
    if len(data) >= 2 and data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16", errors="replace")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        pass
    for enc in (locale.getpreferredencoding() or "utf-8", "gb18030"):
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("latin-1")


def _persist_overflow(full_output: str) -> str | None:
    """截断前的完整输出落盘（实例临时目录），返回路径；失败静默。保留最近 20 份。"""
    try:
        from src.instance import temp_dir

        cache_dir = temp_dir()
        # 全新环境（如 CI runner）下 temp 目录可能还不存在，写之前先建
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"exec-overflow-{int(time.time())}-{uuid.uuid4().hex[:6]}.txt"
        path.write_text(full_output, encoding="utf-8")
        old = sorted(cache_dir.glob("exec-overflow-*.txt"))[:-20]
        for p in old:
            try:
                p.unlink()
            except OSError:
                pass
        return str(path)
    except Exception as e:
        logger.debug("Failed to persist overflow output: %s", e)
        return None


def _assemble_output(
    stdout_b: bytes,
    stderr_b: bytes,
    exit_code: int,
    killed_reason: str,
    max_output: int,
    command: str,
) -> str:
    parts = []
    if stdout_b:
        parts.append(_robust_decode(stdout_b))
    if stderr_b:
        parts.append(f"[stderr]\n{_robust_decode(stderr_b)}")
    output = "\n".join(parts) or "(no output)"

    if killed_reason:
        output += f"\n[killed: {killed_reason}]"

    encoded = output.encode("utf-8", errors="replace")
    if len(encoded) > max_output:
        path = _persist_overflow(output)
        suffix = f"\n... [truncated at {max_output} bytes"
        if path:
            suffix += f"; full output saved to {path}"
        suffix += "]\n"
        output = encoded[:max_output].decode("utf-8", errors="replace") + suffix

    if exit_code != 0 and not killed_reason:
        note = _interpret_exit_code(command, exit_code)
        output += f"\n[exit code: {exit_code}]"
        if note:
            output += f" — {note}"
    return output


# 常见命令的非错误退出码（管道/链取最后一段判定），标注在输出里，避免模型
# 把正常非零码当失败排查。覆盖 bash / cmd / powershell 三方言常见命令。
# type 不入表：bash 内建是命令名查找（rc=1=命令未找到）、cmd/PS 是文件
# 读取，方言语义冲突且 stderr 已写明原因。
_EXIT_SEMANTICS: list[tuple[set[str], dict[int, str]]] = [
    ({"grep", "egrep", "fgrep", "rg", "ag", "ack", "findstr", "select-string"}, {1: "No matches found"}),
    ({"diff", "colordiff", "compare-object", "fc"}, {1: "Files differ"}),
    ({"test", "["}, {1: "Condition evaluated to false"}),
    ({"which", "where", "command", "get-command"}, {1: "Command not found"}),
    ({"cat", "get-content"}, {1: "File not found or unreadable"}),
    ({"robocopy"}, {1: "Files copied successfully"}),
    ({"find"}, {1: "Some directories inaccessible (partial results may be valid)"}),
    (
        {"curl"},
        {
            6: "Could not resolve host",
            7: "Failed to connect to host",
            22: "HTTP response indicated error (e.g. 404)",
            28: "Operation timed out",
        },
    ),
]


def _interpret_exit_code(command: str, exit_code: int) -> str | None:
    if exit_code == 0:
        return None
    # 退出码由管道/链的最后一段决定
    segments = re.split(r"\|\||&&|[|;\n]", command)
    last = (segments[-1] if segments else command).strip()
    for token in last.split():
        if "=" in token and not token.startswith("-"):
            continue  # VAR=val cmd 的前导赋值
        base = basename(token)
        break
    else:
        return None
    for commands, table in _EXIT_SEMANTICS:
        if base in commands and exit_code in table:
            return table[exit_code]
    return None


# ---------------------------------------------------------------------------
# cwd 跟踪（临时文件采集 pwd -P，跨调用保持工作目录）
# ---------------------------------------------------------------------------


class _CwdTracker:
    """thread_id → shell cwd。有界（256），/reset 可清。"""

    _MAX = 256

    def __init__(self) -> None:
        self._cwd: dict[str, str] = {}

    def get(self, thread_id: str) -> str:
        return self._cwd.get(thread_id, "")

    def set(self, thread_id: str, cwd: str) -> None:
        if len(self._cwd) >= self._MAX and thread_id not in self._cwd:
            first = next(iter(self._cwd))
            self._cwd.pop(first, None)
        self._cwd[thread_id] = cwd

    def clear(self, thread_id: str = "") -> None:
        if thread_id:
            self._cwd.pop(thread_id, None)
        else:
            self._cwd.clear()


_shell_cwd = _CwdTracker()


def reset_shell_cwd(thread_id: str = "") -> None:
    """清空 shell cwd 追踪（/reset 会话重置时调用）。"""
    _shell_cwd.clear(thread_id)


def _finalize_cwd(provider: ShellProvider, cwd_file: str, cfg, thread_id: str) -> None:
    if not cwd_file:
        return
    try:
        raw = provider.read_cwd_file(cwd_file)
        provider.cleanup_cwd_file(cwd_file)
        if not raw:
            return
        new_cwd = msys_to_windows(raw) if (os.name == "nt" and raw.startswith("/")) else raw
        if os.name == "nt" and not re.match(r"[A-Za-z]:[\\/]", new_cwd) and not new_cwd.startswith("\\\\"):
            # MSYS 挂载点（/tmp、/usr）经 pwd -P 原样输出且无盘符——msys_to_windows
            # 不认识它们，而原生 Windows 会把 "/tmp" 解析成「当前盘根」下的另一个
            # 目录（D:\tmp ≠ 挂载源 D:/Temp），跟踪必错。不变量：Windows 上 tracked
            # cwd 必须是盘符/UNC 绝对路径，否则放弃跟踪保持原值。
            return
        if new_cwd and os.path.isdir(new_cwd) and _sandbox_path_allowed(new_cwd, cfg):
            _shell_cwd.set(thread_id, new_cwd)
    except Exception as e:
        logger.debug("Failed to track shell cwd: %s", e)


# ---------------------------------------------------------------------------
# 进程 spawn 与流式执行
# ---------------------------------------------------------------------------


def _build_spawn_env(provider: ShellProvider) -> dict | None:
    """构建子进程 env（前后台共用的 env 策略单点）。

    子进程完整继承父环境；provider 需要覆盖时叠加覆盖（如 PowerShell 清
    PSModulePath 防用户模块注入）。
    """
    overrides = provider.env_overrides()
    if not overrides:
        return None
    base = dict(os.environ)
    base.update(overrides)
    return base


async def _spawn_process(
    provider: ShellProvider,
    script: str,
    raw_command: str,
    workdir: str,
    env: dict | None,
) -> asyncio.subprocess.Process:
    proc_kwargs: dict = dict(
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=workdir,
        env=env,
    )
    if os.name != "nt":
        proc_kwargs["start_new_session"] = True
    if provider.legacy_shell:
        return await asyncio.create_subprocess_shell(raw_command, **proc_kwargs)
    return await asyncio.create_subprocess_exec(provider.shell_path, *provider.spawn_args(script), **proc_kwargs)


async def _exec_streaming(
    proc: asyncio.subprocess.Process,
    command: str,
    timeout: int,
    no_output_timeout: int,
    max_output: int,
    start: float,
    provider: ShellProvider,
) -> str:
    """Execute a command with streaming output and no-output timeout detection.

    When the process produces no output for no_output_timeout seconds,
    it is killed and partial output is returned.
    """
    output_event = asyncio.Event()
    out_buf: list[bytes] = []
    err_buf: list[bytes] = []
    readers_remaining = 0
    killed_reason = ""
    _last_heartbeat = start

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
            # Heartbeat: log every 30s so long-running commands are observable
            _now = time.monotonic()
            if _now - _last_heartbeat >= 30:
                _last_heartbeat = _now
                logger.info(
                    "[exec-heartbeat] elapsed=%.0fs out=%dB err=%dB rc=%s cmd=%.120s",
                    _now - start,
                    sum(len(b) for b in out_buf),
                    sum(len(b) for b in err_buf),
                    proc.returncode,
                    command,
                )

            # Check if process already exited
            if proc.returncode is not None:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=2.0,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "[exec-audit] reader drain timed out after 2s (child may hold pipe), cmd=%.120s",
                        command,
                    )
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
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=2.0,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "[exec-audit] reader drain timed out after 2s (child may hold pipe), cmd=%.120s",
                        command,
                    )
                break

            if not got_output:
                # No-output timeout
                if time.monotonic() >= deadline:
                    await _safe_kill()
                    duration = time.monotonic() - start
                    logger.warning("[exec-audit] TIMEOUT dur=%.1fs cmd=%.200s", duration, command)
                    raise ToolExecutionError(f"Command timed out after {timeout}s")

                killed_reason = f"no output for {no_output_timeout}s"
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
    output = _assemble_output(b"".join(out_buf), b"".join(err_buf), exit_code, killed_reason, max_output, command)

    duration = time.monotonic() - start
    logger.info(
        "[exec-audit] exit=%d dur=%.1fs killed=%s shell=%s cmd=%.200s",
        exit_code,
        duration,
        bool(killed_reason),
        provider.display_name,
        command,
    )
    return output


# ---------------------------------------------------------------------------
# exec_command 主流水线
# ---------------------------------------------------------------------------


async def exec_command(
    command: str,
    timeout: int = 300,
    workdir: Optional[str] = None,
    background: bool = False,
) -> str:
    """Execute a shell command and return its output.

    If a command fails due to timeout, increase the timeout value and retry.

    For file operations prefer the dedicated file tools
    (read_file/write_file/edit_file/list_dir/grep/glob) over shell commands.

    Args:
        command: The shell command to execute.
        timeout: Timeout in seconds. Default 300. Increase this value for long-running commands like npm install or docker build.
        workdir: Working directory. Defaults to agents.workspace in config.yaml.
        background: Run in background. Returns session ID immediately.
    """
    if timeout <= 0:
        raise ToolExecutionError(f"timeout 必须为正整数，收到: {timeout}")
    if timeout > 3600:
        raise ToolExecutionError(f"timeout 上限为 3600 秒，收到: {timeout}")

    cfg = _get_config()
    if cfg is None:
        # 配置加载失败 → 降级到 schema 默认值；在此归一，下方代码不再感知 None
        from src.config import AppConfig

        cfg = AppConfig()

    deny_patterns = cfg.tools.exec.deny_patterns or DEFAULT_DENY_PATTERNS  # 配置默认 []=使用内置默认表
    max_output = cfg.tools.exec.max_output_bytes or 102400  # 0 视为未配置
    approval_mode = cfg.tools.exec.approval_mode
    no_output_timeout_val = cfg.tools.exec.no_output_timeout_seconds  # 0=禁用无输出检测
    sandbox_enabled = cfg.tools.exec.sandbox_enabled
    shell_pref = cfg.tools.exec.shell
    cwd_persistence = cfg.tools.exec.cwd_persistence

    # ── deny 闸（前台/后台共用）：hardline 底线 → 可配置 denylist ──
    hardline = check_hardline(command)
    if hardline:
        logger.warning("[exec-audit] DENIED hardline ('%s'): %.200s", hardline, command)
        raise ToolExecutionError(
            f"Command blocked (hardline): {hardline}. "
            "This class of command is unconditionally denied; "
            "run it yourself in a terminal if truly needed."
        )

    blocked, matched = is_denylisted(command, deny_patterns)
    if blocked:
        logger.warning("[exec-audit] DENIED command matches pattern '%s': %.200s", matched, command)
        raise ToolExecutionError(
            f"Command blocked by denylist (matched: {matched}). "
            "If this is a false positive, adjust tools.exec.deny_patterns in config.yaml."
        )

    provider = resolve_shell_provider(shell_pref)
    thread_id = _current_thread_id.get("")
    use_tracking = cwd_persistence and not provider.legacy_shell

    # Workdir：显式参数 → 追踪 cwd → 默认 workspace。
    explicit_workdir = workdir is not None
    if workdir is None and use_tracking:
        tracked = _shell_cwd.get(thread_id)
        if tracked and os.path.isdir(tracked):
            workdir = tracked
    if workdir is None:
        workdir = _resolve_default_workdir()

    # Sandbox: working directory restriction
    if sandbox_enabled:
        wd = Path(workdir).resolve()
        if not _sandbox_path_allowed(str(wd), cfg):
            workspace = Path(cfg.agents.workspace).expanduser().resolve()
            raise ToolExecutionError(f"当前为沙盒模式，无法访问工作目录之外的路径：{wd}（允许范围：{workspace}）")

        # Use resolved path to prevent TOCTOU symlink race
        workdir = str(wd)

    # Background mode: spawn via ProcessRegistry and return immediately
    if background:
        from src.tools.process import get_process_registry

        # Shell executor patterns — blocked in background mode.
        # Approval gating is not feasible for async processes; block outright.
        executor_bg, executor_bg_detail = has_shell_executor(command)
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
        if _get_first_token(command) not in COMMAND_PREFIX_WHITELIST:
            delete_bg, delete_bg_pattern = is_denylisted(command, DELETE_APPROVAL_PATTERNS)
            if delete_bg:
                logger.warning(
                    "[exec-audit] BLOCKED delete operation in background mode ('%s'): %.200s",
                    delete_bg_pattern,
                    command,
                )
                raise ToolExecutionError(f"Delete operation ('{delete_bg_pattern}') cannot be run in background mode")

        # Force-kill — blocked in background mode (approval not feasible).
        if _get_first_token(command) not in COMMAND_PREFIX_WHITELIST:
            kill_bg, kill_bg_pattern = is_denylisted(command, FORCE_KILL_APPROVAL_PATTERNS)
            if kill_bg:
                logger.warning(
                    "[exec-audit] BLOCKED force-kill in background mode ('%s'): %.200s",
                    kill_bg_pattern,
                    command,
                )
                raise ToolExecutionError(f"Force-kill ('{kill_bg_pattern}') cannot be run in background mode")

        env = _build_spawn_env(provider)

        # 与前台统一 shell 方言：provider 直执行。后台进程无 cwd 采集
        # 需求，build_script 直接透传命令。
        spawn_argv = (
            None
            if provider.legacy_shell
            else [provider.shell_path, *provider.spawn_args(provider.build_script(command, ""))]
        )

        registry = get_process_registry()
        session_id = await registry.spawn(command, workdir=workdir, env=env, argv=spawn_argv)
        return json.dumps(
            {
                "session_id": session_id,
                "status": "running",
                "command": command[:200],
            }
        )

    # ── 审批闸门（顺序：删除审批 → executor → 强杀 → 常规审批）──

    # Non-recursive delete — always requires approval (30s timeout, auto-deny)
    # Whitelisted prefixes (git, docker, etc.) skip delete approval — their "rm"
    # subcommands operate within their own scope, not the filesystem directly.
    if _get_first_token(command) not in COMMAND_PREFIX_WHITELIST:
        delete_match, delete_pattern = is_denylisted(command, DELETE_APPROVAL_PATTERNS)
        if delete_match:
            from src.tools.approval import get_approval_manager

            mgr = get_approval_manager()
            if not mgr.has_session_approval(_current_thread_id.get(""), "exec_command", command):
                logger.info("[exec-audit] Delete operation requires approval ('%s'): %.200s", delete_pattern, command)
                raise ApprovalNeededError(
                    command,
                    False,
                    timeout=30,
                    auto_deny=True,
                    approval_key=delete_pattern,
                )

    # Shell executor patterns (sh -c, cmd /c, powershell -command, etc.).
    # These are safe to pass through — the denylist already checked the full command
    # for dangerous tokens. Only apply approval gating if configured.
    executor, executor_detail = has_shell_executor(command)
    if executor and approval_mode not in ("off", ""):
        from src.tools.approval import get_approval_manager

        mgr = get_approval_manager()
        if not mgr.has_session_approval(_current_thread_id.get(""), "exec_command", command):
            logger.info(
                "[exec-audit] Shell executor ('%s'), requiring approval: %.200s",
                executor_detail,
                command,
            )
            raise ApprovalNeededError(command, False)

    # Force-kill (kill -9 / pkill -KILL / Stop-Process -Force) — always requires
    # approval: recoverable, but can kill the agent's own process tree or stateful
    # services (databases). Whitelisted prefixes (docker kill etc.) skip — their
    # kill operates within their own scope.
    if _get_first_token(command) not in COMMAND_PREFIX_WHITELIST:
        kill_match, kill_pattern = is_denylisted(command, FORCE_KILL_APPROVAL_PATTERNS)
        if kill_match:
            from src.tools.approval import get_approval_manager

            mgr = get_approval_manager()
            if not mgr.has_session_approval(_current_thread_id.get(""), "exec_command", command):
                logger.info("[exec-audit] Force-kill requires approval ('%s'): %.200s", kill_pattern, command)
                raise ApprovalNeededError(
                    command,
                    False,
                    timeout=30,
                    auto_deny=True,
                    approval_key=kill_pattern,
                )

    if approval_mode not in ("off", ""):
        from src.tools.approval import get_approval_manager

        mgr = get_approval_manager()
        needs = mgr.needs_approval("exec_command", command, approval_mode, blocked)
        if needs and not mgr.has_session_approval(_current_thread_id.get(""), "exec_command", command):
            raise ApprovalNeededError(command, blocked)

    start = time.monotonic()
    cwd_file = provider.create_cwd_file() if use_tracking else ""
    script = provider.build_script(command, cwd_file)
    env = _build_spawn_env(provider)
    no_output_timeout = no_output_timeout_val

    try:
        proc = await _spawn_process(provider, script, command, workdir, env)

        if no_output_timeout > 0:
            # Streaming mode: detect no-output timeout
            output = await _exec_streaming(proc, command, timeout, no_output_timeout, max_output, start, provider)
        else:
            # Original buffered mode (backward compatible)
            logger.info("[exec] communicate start: timeout=%ds cmd=%.120s", timeout, command)
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            logger.info("[exec] communicate done: exit=%s", proc.returncode)
            exit_code = proc.returncode or 0
            output = _assemble_output(stdout or b"", stderr or b"", exit_code, "", max_output, command)
            duration = time.monotonic() - start
            logger.info(
                "[exec-audit] exit=%d dur=%.1fs shell=%s cmd=%.200s",
                exit_code,
                duration,
                provider.display_name,
                command,
            )

        # 显式 workdir 是单次覆盖，不读回（不影响会话追踪的 cwd）
        if use_tracking and cwd_file and not explicit_workdir:
            _finalize_cwd(provider, cwd_file, cfg, thread_id)
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
    finally:
        provider.cleanup_cwd_file(cwd_file)


def _get_first_token(command: str) -> str:
    """Extract the first token of a command for prefix whitelisting."""
    stripped = command.strip().lower()
    for ch in stripped:
        if ch in (" ", "\t", "\n", "\r"):
            break
    else:
        return stripped
    return stripped.split(None, 1)[0] if stripped else ""


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


_EXEC_COMMAND_ORIGINAL_DOC: Optional[str] = None


def _shell_guidance(provider: ShellProvider) -> str:
    """追加到 exec_command docstring 的 shell 方言指引，须随实际 provider 变化。"""
    common = (
        "- Do NOT use cat/head/tail to read files — use read_file. "
        "Do NOT use grep/rg/find to search — use the grep tool. Do NOT use ls to list directories — use list_dir. "
        "Reserve exec_command for: builds, installs, git, processes, scripts, network, package managers.\n"
        "- Long-running tasks (servers, watchers, big builds): use background=true — "
        "it returns a session_id immediately; track via process_status (poll/wait/log/kill).\n"
    )
    if provider.legacy_shell:
        dialect = (
            "- Active shell: system default (cmd.exe on Windows, /bin/sh on POSIX).\n"
            "  Use that dialect's syntax (Windows cmd: dir, type, copy, del, findstr).\n"
        )
    elif provider.display_name == "powershell":
        dialect = "- Active shell: PowerShell.\n  Use PowerShell cmdlets and syntax; both / and \\ work in paths.\n"
    elif provider.display_name == "git-bash":
        dialect = (
            f"- Active shell: {provider.display_name} (same dialect on all platforms when available).\n"
            "  Use forward-slash paths (cd C:/Users/foo or /c/Users/foo); "
            "backslashes are bash escape characters.\n"
        )
    else:
        dialect = f"- Active shell: {provider.display_name} (same dialect on all platforms when available).\n"
    return "\nShell dialect notes:\n" + dialect + common


def get_tools() -> list:
    from src.agent.tooldef import ToolDef

    global _EXEC_COMMAND_ORIGINAL_DOC
    if _EXEC_COMMAND_ORIGINAL_DOC is None:
        _EXEC_COMMAND_ORIGINAL_DOC = exec_command.__doc__ or ""

    # Patch the workdir description with the resolved default path so the LLM
    # sees the actual value instead of a cryptic config variable name, and
    # append shell-dialect guidance. Always patch from the original docstring
    # so this is idempotent across hot-reloads.
    _resolved = _resolve_default_workdir()
    _cfg = _get_config()
    _provider = resolve_shell_provider(_cfg.tools.exec.shell if _cfg else "auto")
    exec_command.__doc__ = _EXEC_COMMAND_ORIGINAL_DOC.replace(
        "Defaults to agents.workspace in config.yaml.",
        f"Defaults to {_resolved}.",
    ) + _shell_guidance(_provider)

    return [
        ToolDef.from_function(exec_command, timeout=3600),
        ToolDef.from_function(process_status),
    ]
