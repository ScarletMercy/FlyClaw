"""ShellProvider 单元测试：探测、脚本包装、路径换算。"""

import os
import platform
from unittest.mock import patch

import pytest

from src.tools.shell_provider import (
    BashProvider,
    PowerShellProvider,
    ShellProvider,
    SystemShellProvider,
    msys_to_windows,
    resolve_shell_provider,
)


class TestMsysToWindows:
    def test_drive_path(self):
        assert msys_to_windows("/c/Users/foo") == "C:\\Users\\foo"

    def test_drive_root(self):
        assert msys_to_windows("/d") == "D:\\"

    def test_non_msys_unchanged(self):
        assert msys_to_windows("/usr/bin") == "/usr/bin"
        assert msys_to_windows("C:\\Users\\foo") == "C:\\Users\\foo"
        assert msys_to_windows("relative/path") == "relative/path"


class TestBashProvider:
    def test_spawn_args(self):
        p = BashProvider()
        args = p.spawn_args("echo hi")
        assert args == ["-c", "echo hi"]

    def test_build_script_passthrough_without_cwd_file(self):
        p = BashProvider()
        assert p.build_script("echo hi", "") == "echo hi"

    def test_build_script_captures_cwd(self):
        p = BashProvider()
        script = p.build_script("echo hi", "/tmp/cwdfile")
        assert "eval " in script
        assert "pwd -P" in script
        assert "/tmp/cwdfile" in script
        # cwd 采集必须在命令成功后才执行
        assert "&&" in script

    def test_build_script_escapes_single_quotes(self):
        p = BashProvider()
        script = p.build_script("echo 'a b'", "/tmp/f")
        # 命令里的单引号必须被转义，不能提前终止 eval 的引用
        assert "'\\''" in script
        # 转义后的脚本应能被 bash 正确解析（这里仅验证引号配对结构）
        assert script.count("'") % 2 == 0

    def test_detection_returns_path_or_empty(self):
        p = BashProvider()
        # 探测要么找到 bash（含 Git Bash），要么返回空串（可用性由上层降级处理）
        assert p.shell_path == "" or p.available

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows-only WSL launcher check")
    def test_windows_detection_skips_wsl_bash(self):
        system_bash = r"C:\Windows\System32\bash.exe"
        with (
            patch(
                "src.tools.shell_provider.shutil.which",
                side_effect=lambda name: None if name == "git" else system_bash,
            ),
            patch.dict(
                os.environ,
                {"ProgramFiles": "Z:\\__none__", "ProgramFiles(x86)": "Z:\\__none__", "LOCALAPPDATA": "Z:\\__none__"},
            ),
        ):
            assert BashProvider._detect_shell() == ""


class TestPowerShellProvider:
    def test_build_script_captures_cwd_and_exit_code(self):
        p = PowerShellProvider()
        script = p.build_script("Get-ChildItem", "C:\\temp\\f")
        assert "$LASTEXITCODE" in script
        assert "Get-Location" in script
        assert "Out-File" in script
        assert "exit $_ec" in script
        assert "C:\\temp\\f" in script

    def test_build_script_without_cwd_file_still_reports_exit_code(self):
        p = PowerShellProvider()
        script = p.build_script("dir", "")
        assert script.startswith("dir")
        assert "$LASTEXITCODE" in script
        assert "exit $_ec" in script

    def test_spawn_args_no_profile(self):
        p = PowerShellProvider()
        args = p.spawn_args("dir")
        assert args[0] == "-NoProfile"
        assert args[1] == "-NonInteractive"
        assert args[2] == "-Command"

    def test_env_overrides_clear_psmodulepath(self):
        assert PowerShellProvider().env_overrides() == {"PSMODULEPATH": ""}

    def test_unavailable_off_windows(self):
        if platform.system() != "Windows":
            assert PowerShellProvider().available is False


class TestSystemShellProvider:
    def test_legacy_flag(self):
        p = SystemShellProvider()
        assert p.legacy_shell is True

    def test_build_script_passthrough(self):
        p = SystemShellProvider()
        assert p.build_script("echo hi", "/tmp/f") == "echo hi"

    def test_spawn_args_unused(self):
        p = SystemShellProvider()
        assert p.spawn_args("echo hi") == ["echo hi"]


class TestResolveShellProvider:
    def test_system_preference(self):
        p = resolve_shell_provider("system")
        assert isinstance(p, SystemShellProvider)

    def test_caching(self):
        assert resolve_shell_provider("system") is resolve_shell_provider("system")

    def test_unknown_preference_falls_back_to_auto(self):
        p = resolve_shell_provider("nonsense")
        assert isinstance(p, ShellProvider)

    def test_auto_returns_available_provider(self):
        p = resolve_shell_provider("auto")
        assert isinstance(p, ShellProvider)
        # auto 一定给出一个可执行策略：bash 可用、或 PowerShell 可用、或 system 兜底
        assert p.available or p.legacy_shell

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows-only fallback chain")
    def test_windows_chain_prefers_bash(self):
        # 本机（Git Bash 环境）auto 应解析为 BashProvider
        p = resolve_shell_provider("auto")
        assert isinstance(p, BashProvider)
