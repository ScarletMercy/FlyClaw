r"""Shell provider 抽象 — 显式选择跨平台执行 shell，三平台统一优先 bash：

- POSIX: which bash → /usr/bin/bash → /bin/bash → /bin/sh（dash 兜底，保底 Termux/Alpine）
- Windows: Git Bash（git 同级 → which bash（排除 SystemRoot 下的 WSL 启动器）
  → Program Files\Git），找不到时降级 PowerShell；两者都没有时退回系统 shell
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import uuid
from abc import ABC, abstractmethod

_MSYS_PATH_RE = re.compile(r"^/([a-zA-Z])(/.*)?$")


def msys_to_windows(path: str) -> str:
    """把 Git Bash 风格路径（/c/Users/...）转回 Windows 原生路径（C:\\Users\\...）。"""
    m = _MSYS_PATH_RE.match(path)
    if not m:
        return path
    drive = m.group(1).upper()
    rest = m.group(2) or "\\"
    return f"{drive}:{rest.replace('/', chr(92))}"


def _quote_posix(text: str) -> str:
    """POSIX 单引号转义。"""
    return text.replace("'", "'\\''")


def _is_wsl_launcher(path: str) -> bool:
    """bash.exe 是否位于 SystemRoot 下（System32/bash.exe 是 WSL 启动器）。"""
    systemroot = os.environ.get("SystemRoot") or r"C:\Windows"
    try:
        real = os.path.realpath(path)
    except OSError:
        return False
    return os.path.normcase(real).startswith(os.path.normcase(systemroot + os.sep))


class ShellProvider(ABC):
    """一种 shell 方言的执行封装。

    build_script 把用户命令包装成「执行 + 采集 cwd」的脚本；
    spawn_args 给出传给解释器的参数列表；runner 用
    create_subprocess_exec(shell_path, *spawn_args(script)) 执行。
    """

    legacy_shell = False  # True = 不经解释器包装，直接 create_subprocess_shell

    @property
    @abstractmethod
    def shell_path(self) -> str: ...

    @property
    @abstractmethod
    def display_name(self) -> str: ...

    @property
    def available(self) -> bool:
        return bool(self.shell_path)

    @abstractmethod
    def build_script(self, command: str, cwd_file: str) -> str:
        """包装命令。cwd_file 非空时执行后写入 shell 视角的真实 cwd。"""

    @abstractmethod
    def spawn_args(self, script: str) -> list[str]: ...

    def env_overrides(self) -> dict[str, str]:
        return {}

    # ── cwd 采集临时文件 ──

    def create_cwd_file(self) -> str:
        path = os.path.join(tempfile.gettempdir(), f"flyclaw-cwd-{uuid.uuid4().hex[:8]}")
        # 统一正斜杠：路径会内插进 bash 单引号脚本，反斜杠会变成字面量导致重定向失败
        return path.replace("\\", "/")

    def read_cwd_file(self, cwd_file: str) -> str | None:
        try:
            with open(cwd_file, encoding="utf-8-sig") as f:
                return f.read().strip()
        except OSError:
            return None

    def cleanup_cwd_file(self, cwd_file: str) -> None:
        if not cwd_file:
            return
        try:
            os.unlink(cwd_file)
        except OSError:
            pass


class BashProvider(ShellProvider):
    """bash（POSIX 原生 / Windows Git Bash）。"""

    def __init__(self) -> None:
        self._shell = self._detect_shell()

    @property
    def shell_path(self) -> str:
        return self._shell

    @property
    def display_name(self) -> str:
        return "bash" if os.name != "nt" else "git-bash"

    @staticmethod
    def _detect_shell() -> str:
        if os.name != "nt":
            return (
                shutil.which("bash")
                or ("/usr/bin/bash" if os.path.isfile("/usr/bin/bash") else None)
                or ("/bin/bash" if os.path.isfile("/bin/bash") else None)
                or "/bin/sh"
            )
        # Windows: Git Bash。git 所在目录旁的 bash.exe 最可靠（git 与 bash 同发行）。
        git_path = shutil.which("git")
        if git_path:
            git_bin = os.path.dirname(git_path)
            for candidate in (
                os.path.join(git_bin, "bash.exe"),
                os.path.normpath(os.path.join(git_bin, "..", "bin", "bash.exe")),
                os.path.join(git_bin, "..", "usr", "bin", "bash.exe"),
            ):
                if os.path.isfile(candidate):
                    return candidate
        found = shutil.which("bash")
        if found and os.path.isfile(found) and not _is_wsl_launcher(found):
            return found
        for base in (
            os.environ.get("ProgramFiles", r"C:\Program Files"),
            os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs"),
        ):
            if not base:
                continue
            candidate = os.path.join(base, "Git", "bin", "bash.exe")
            if os.path.isfile(candidate):
                return candidate
        return ""

    def build_script(self, command: str, cwd_file: str) -> str:
        # eval + 单引号包裹：命令含换行/结尾为 && 等 shell 操作符时仍整体求值；
        # cmd 失败时 && 短路，退出码保留 cmd 的，cwd 维持原值。
        if not cwd_file:
            return command
        return f"eval '{_quote_posix(command)}' && pwd -P >| '{_quote_posix(cwd_file)}'"

    def spawn_args(self, script: str) -> list[str]:
        return ["-c", script]


class PowerShellProvider(ShellProvider):
    """Windows PowerShell 5+ 兜底（仅当 Git Bash 不可用）。"""

    _PS = "powershell"

    @property
    def shell_path(self) -> str:
        return self._PS

    @property
    def display_name(self) -> str:
        return "powershell"

    @property
    def available(self) -> bool:
        return os.name == "nt" and shutil.which(self._PS) is not None

    _EXIT_PROBE = "\n; $_ec = if ($null -ne $LASTEXITCODE) { $LASTEXITCODE } elseif ($?) { 0 } else { 1 }"

    def build_script(self, command: str, cwd_file: str) -> str:
        if not cwd_file:
            return f"{command}{self._EXIT_PROBE}\n; exit $_ec"
        escaped = cwd_file.replace("'", "''")
        return (
            f"{command}{self._EXIT_PROBE}\n"
            f"; (Get-Location).Path | Out-File -FilePath '{escaped}' "
            f"-Encoding utf8 -NoNewline\n"
            f"; exit $_ec"
        )

    def spawn_args(self, script: str) -> list[str]:
        return ["-NoProfile", "-NonInteractive", "-Command", script]

    def env_overrides(self) -> dict[str, str]:
        # 阻断用户 PSModulePath 注入的模块自动加载
        return {"PSMODULEPATH": ""}


class SystemShellProvider(ShellProvider):
    """不经解释器包装，create_subprocess_shell 直执行（Windows=cmd，
    POSIX=/bin/sh）。配置 tools.exec.shell=system 时启用。"""

    legacy_shell = True

    @property
    def shell_path(self) -> str:
        return "system"

    @property
    def display_name(self) -> str:
        return "system"

    def build_script(self, command: str, cwd_file: str) -> str:
        return command  # 无 cwd 采集

    def spawn_args(self, script: str) -> list[str]:
        return [script]  # 未使用（runner 走 legacy 分支）


_provider_cache: dict[str, ShellProvider] = {}


def resolve_shell_provider(preference: str = "auto") -> ShellProvider:
    """按配置选择 provider。结果缓存（探测涉及磁盘查找，进程内只做一次）。"""
    key = preference if preference in ("auto", "system") else "auto"
    cached = _provider_cache.get(key)
    if cached is not None:
        return cached

    if key == "system":
        provider: ShellProvider = SystemShellProvider()
    elif os.name != "nt":
        provider = BashProvider()
    else:
        bash = BashProvider()
        if bash.available:
            provider = bash
        else:
            ps = PowerShellProvider()
            provider = ps if ps.available else SystemShellProvider()

    _provider_cache[key] = provider
    return provider
