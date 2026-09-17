"""exec 流水线单元/集成测试：解码链、退出码语义、workdir 执行、
截断落盘、cwd 跟踪、guard 集成。"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.tools.exceptions import ToolExecutionError
from src.tools.exec import (
    _assemble_output,
    _interpret_exit_code,
    _robust_decode,
    _shell_cwd,
    _shell_guidance,
    exec_command,
)
from src.tools.shell_provider import BashProvider, PowerShellProvider, SystemShellProvider


def _mock_config(**overrides):
    """真实 AppConfig 构造的最小 exec 测试配置（沙箱关闭、缓冲模式）。

    用真实 pydantic 模型而非 MagicMock：exec 直读 tools.exec 字段，
    MagicMock 未显式设置的属性会以 truthy 假值泄漏进业务逻辑。
    """
    from src.config import AppConfig

    cfg = AppConfig()
    cfg.agents.workspace = "."
    cfg.tools.exec.sandbox_enabled = False
    cfg.tools.exec.no_output_timeout_seconds = 0
    for key, value in overrides.items():
        setattr(cfg.tools.exec, key, value)
    return cfg


# ---------------------------------------------------------------------------
# 解码链
# ---------------------------------------------------------------------------


class TestRobustDecode:
    def test_ascii(self):
        assert _robust_decode(b"hello") == "hello"

    def test_utf8_chinese(self):
        assert _robust_decode("中文输出".encode("utf-8")) == "中文输出"

    def test_utf8_bom(self):
        assert _robust_decode(b"\xef\xbb\xbfhello") == "hello"

    def test_utf16_bom(self):
        assert _robust_decode("hi".encode("utf-16")) == "hi"

    def test_gbk_chinese(self):
        # GBK 编码的「中文」：非 UTF-8 字节序列，必须经 locale/gb18030 兜底还原
        assert _robust_decode(b"\xd6\xd0\xce\xc4") == "中文"

    def test_latin1_fallback(self):
        assert _robust_decode(b"caf\xe9") == "café"

    def test_empty(self):
        assert _robust_decode(b"") == ""


# ---------------------------------------------------------------------------
# 退出码语义
# ---------------------------------------------------------------------------


class TestInterpretExitCode:
    def test_zero_is_none(self):
        assert _interpret_exit_code("grep x", 0) is None

    def test_grep_no_match(self):
        assert _interpret_exit_code("grep -q pattern file", 1) == "No matches found"

    def test_pipeline_takes_last_segment(self):
        assert _interpret_exit_code("cat f | grep -q pattern", 1) == "No matches found"
        assert _interpret_exit_code("build && grep -q pattern x", 1) == "No matches found"

    def test_env_assignment_skipped(self):
        assert _interpret_exit_code("LC_ALL=C grep -q p f", 1) == "No matches found"

    def test_path_prefix_stripped(self):
        assert _interpret_exit_code("/usr/bin/grep -q p f", 1) == "No matches found"

    def test_exe_suffix_stripped(self):
        assert _interpret_exit_code("findstr /c:hi x.txt", 1) == "No matches found"

    def test_type_not_annotated(self):
        # type 不入语义表：方言语义冲突（bash=命令名查找，cmd/PS=文件读取）
        assert _interpret_exit_code("type nonexistent_cmd", 1) is None

    def test_robocopy_success_code(self):
        note = _interpret_exit_code("robocopy a b", 1)
        assert note and "copied" in note

    def test_git_exit_one_not_annotated(self):
        assert _interpret_exit_code("git push origin main", 1) is None

    def test_true_exit_code_is_error(self):
        assert _interpret_exit_code("ls -la", 2) is None

    def test_unknown_command(self):
        assert _interpret_exit_code("somescript", 3) is None


# ---------------------------------------------------------------------------
# 截断落盘
# ---------------------------------------------------------------------------


class TestAssembleOutput:
    def test_passthrough(self):
        out = _assemble_output(b"hello", b"", 0, "", 102400, "echo")
        assert out == "hello"

    def test_stderr_section(self):
        out = _assemble_output(b"out", b"err", 0, "", 102400, "x")
        assert "[stderr]" in out and "err" in out

    def test_no_output_placeholder(self):
        out = _assemble_output(b"", b"", 0, "", 102400, "x")
        assert out == "(no output)"

    def test_exit_code_line(self):
        out = _assemble_output(b"", b"", 2, "", 102400, "ls -la")
        assert "[exit code: 2]" in out

    def test_exit_code_semantic_note(self):
        out = _assemble_output(b"", b"", 1, "", 102400, "grep -q p f")
        assert "[exit code: 1] — No matches found" in out

    def test_real_failure_not_labeled_not_an_error(self):
        out = _assemble_output(b"", b"", 1, "", 102400, "cat missing.txt")
        assert "File not found or unreadable" in out
        assert "(not an error)" not in out
        out = _assemble_output(b"", b"", 22, "", 102400, "curl -s https://x/y")
        assert "(not an error)" not in out

    def test_killed_reason(self):
        out = _assemble_output(b"partial", b"", -1, "no output for 60s", 102400, "x")
        assert "[killed: no output for 60s]" in out
        # 被杀的进程没有有意义的退出码，不追加 exit 行
        assert "[exit code:" not in out

    def test_truncation_persists_full_output(self, tmp_path):
        # 隔离到尚不存在的子目录，精确复现全新环境（~/.flyclaw/temp 不存在）
        with patch("src.instance.temp_dir", return_value=tmp_path / "temp"):
            long_text = "x" * 5000 + "END-MARKER"
            out = _assemble_output(long_text.encode(), b"", 0, "", 1024, "x")
        assert "[truncated at 1024 bytes" in out
        assert "full output saved to" in out
        # 完整内容已落盘（隔离目录内，由 pytest 自动清理）
        path = out.split("full output saved to ")[1].strip().split("]")[0]
        assert os.path.isfile(path)
        with open(path, encoding="utf-8") as f:
            assert "END-MARKER" in f.read()


# ---------------------------------------------------------------------------
# 集成：guard / 引导 / workdir 校验 / cwd 跟踪
# ---------------------------------------------------------------------------


class TestExecPipelineIntegration:
    @pytest.mark.asyncio
    async def test_hardline_blocks_sudo_shutdown(self):
        with patch("src.tools.exec._get_config", return_value=_mock_config()):
            with pytest.raises(ToolExecutionError, match="hardline"):
                await exec_command("sudo shutdown -h now")

    @pytest.mark.asyncio
    async def test_denylist_still_blocks_rm_rf(self):
        # 子路径递归删除不归 hardline 管，由 denylist 兜底（误报有配置出口）
        with patch("src.tools.exec._get_config", return_value=_mock_config()):
            with pytest.raises(ToolExecutionError, match="blocked by denylist"):
                await exec_command("rm -rf /tmp/x")

    @pytest.mark.asyncio
    async def test_background_hardline_blocks_sudo_shutdown(self):
        with patch("src.tools.exec._get_config", return_value=_mock_config()):
            with pytest.raises(ToolExecutionError, match="hardline"):
                await exec_command("sudo shutdown -h now", background=True)

    @pytest.mark.asyncio
    async def test_background_still_blocks_denylist(self):
        # ncat 不在 hardline 清单，由 denylist（ncat*）兜底拦截
        with patch("src.tools.exec._get_config", return_value=_mock_config()):
            with pytest.raises(ToolExecutionError, match="blocked by denylist"):
                await exec_command("ncat example.com 80", background=True)

    @pytest.mark.asyncio
    async def test_same_grade_dangers_denied(self):
        # sudo 前缀 / 整段引号包裹的同级危险命令经归一化剥壳后必须 deny
        with patch("src.tools.exec._get_config", return_value=_mock_config()):
            for cmd in [
                "sudo certutil -urlcache -f http://x y",
                "'schtasks /create /tn x'",
                "del /s *.tmp",
                "Remove-Item -Recurse C:\\tmp\\x",
            ]:
                with pytest.raises(ToolExecutionError, match="denylist"):
                    await exec_command(cmd)

    @pytest.mark.asyncio
    async def test_force_kill_requires_approval(self):
        from unittest.mock import MagicMock

        from src.tools.exec import ApprovalNeededError

        mgr = MagicMock()
        mgr.has_session_approval.return_value = False
        with (
            patch("src.tools.exec._get_config", return_value=_mock_config()),
            patch("src.tools.approval.get_approval_manager", return_value=mgr),
        ):
            with pytest.raises(ApprovalNeededError):
                await exec_command("kill -9 123")

    @pytest.mark.asyncio
    async def test_force_kill_blocked_in_background(self):
        with patch("src.tools.exec._get_config", return_value=_mock_config()):
            with pytest.raises(ToolExecutionError, match="background"):
                await exec_command("kill -9 123", background=True)

    @pytest.mark.asyncio
    async def test_workdir_is_os_parameter_not_shell_parsed(self):
        """元字符路径只是普通 OS 参数（Popen cwd），得到的是「目录不存在」
        错误，不会被 shell 解析执行。"""
        with patch("src.tools.exec._get_config", return_value=_mock_config()):
            with pytest.raises(ToolExecutionError) as ei:
                await exec_command("echo hi", workdir="/tmp/a;b", timeout=30)
            # 是 Popen 的目录不存在错误
            assert "disallowed character" not in str(ei.value)

    @pytest.mark.asyncio
    async def test_cjk_workdir_executes(self):
        """中文路径作为显式 workdir 必须可用。"""
        _shell_cwd.clear()
        with tempfile.TemporaryDirectory(prefix="工作目录-", dir=os.getcwd()) as tmp:
            target = tmp.replace("\\", "/")
            with patch("src.tools.exec._get_config", return_value=_mock_config()):
                result = await exec_command("pwd", timeout=30, workdir=target)
                assert os.path.basename(tmp) in result
                # 显式 workdir 是单次覆盖，不污染追踪
                assert _shell_cwd.get("") == ""
        _shell_cwd.clear()

    def test_explicit_empty_allowed_dirs_keeps_process_cwd_out(self):
        from src.tools.exec import _sandbox_allowed_dirs

        with tempfile.TemporaryDirectory(prefix="ws-", dir=os.getcwd()) as ws:
            cfg = _mock_config()
            cfg.agents.workspace = ws
            cfg.tools.exec.sandbox_allowed_dirs = []
            allowed = _sandbox_allowed_dirs(cfg)
            assert Path(os.getcwd()).resolve() not in allowed

    @pytest.mark.asyncio
    async def test_basic_execution_still_works(self):
        with patch("src.tools.exec._get_config", return_value=_mock_config()):
            result = await exec_command("echo pipeline-ok")
        assert "pipeline-ok" in result

    @pytest.mark.asyncio
    async def test_exit_code_semantic_note_in_result(self):
        with patch("src.tools.exec._get_config", return_value=_mock_config()):
            result = await exec_command("grep -q definitely-not-present /dev/null", timeout=30)
        assert "[exit code: 1] — No matches found" in result

    @pytest.mark.asyncio
    async def test_degraded_config_still_executes(self):
        """配置加载失败（cfg=None）时降级 schema 默认值照常执行。

        HOME/USERPROFILE 重定向到临时目录，默认 workspace（~/.flyclaw/workspace）
        的 mkdir 副作用随之落在临时目录内。
        """
        _shell_cwd.clear()
        with tempfile.TemporaryDirectory(prefix="flyclaw-degraded-home-") as home:
            with (
                patch("src.tools.exec._get_config", return_value=None),
                patch.dict(os.environ, {"HOME": home, "USERPROFILE": home}),
            ):
                result = await exec_command("echo degraded-ok", timeout=30)
            assert "degraded-ok" in result
        _shell_cwd.clear()

    @pytest.mark.asyncio
    async def test_background_env_fully_inherited(self):
        """契约：子进程完整继承父环境（前台/后台一致）。

        哨兵变量对后台子进程可见即为继承生效。
        """
        import json

        from src.tools.exec import process_status

        cfg = _mock_config()
        cfg.tools.exec.sandbox_enabled = True
        with (
            patch("src.tools.exec._get_config", return_value=cfg),
            patch.dict(os.environ, {"FLYCLAW_PROBE": "inherited-marker"}),
        ):
            out = await exec_command("env | grep -c FLYCLAW_PROBE", timeout=30, background=True)
            sid = json.loads(out)["session_id"]
            res = json.loads(await process_status("wait", session_id=sid, timeout=15))
        assert res["exit_code"] == 0
        assert "1" in res["output"]


class TestShellGuidance:
    """docstring 指引必须随实际 provider 方言变化，兜底 shell 时不得再讲 Git Bash。"""

    class _GitBash(BashProvider):
        # _shell_guidance 只按 display_name 分方言；桩定两个平台变体，测试不随宿主平台漂移
        @property
        def display_name(self) -> str:
            return "git-bash"

    class _PosixBash(BashProvider):
        @property
        def display_name(self) -> str:
            return "bash"

    def test_git_bash_guidance_mentions_forward_slash(self):
        g = _shell_guidance(self._GitBash())
        assert "forward-slash" in g

    def test_posix_bash_guidance_no_windows_advice(self):
        g = _shell_guidance(self._PosixBash())
        assert "Git Bash" not in g
        assert "Windows" not in g

    def test_system_guidance_no_git_bash_advice(self):
        g = _shell_guidance(SystemShellProvider())
        assert "Git Bash" not in g
        assert "cmd" in g

    def test_powershell_guidance(self):
        g = _shell_guidance(PowerShellProvider())
        assert "PowerShell" in g
        assert "Git Bash" not in g


class TestCwdTracking:
    @pytest.mark.asyncio
    async def test_cwd_tracked_across_calls(self):
        _shell_cwd.clear()
        # 建在当前工作区下，避开 MSYS 挂载点（/tmp 转不出 Windows 路径，
        # _finalize_cwd 会拒绝跟踪）
        with tempfile.TemporaryDirectory(prefix="flyclaw-cwd-test-", dir=os.getcwd()) as tmp:
            target = tmp.replace("\\", "/")
            cfg = _mock_config()
            with patch("src.tools.exec._get_config", return_value=cfg):
                await exec_command(f'cd "{target}"', timeout=30)
                tracked = _shell_cwd.get("")
                assert tracked and os.path.basename(tracked) == os.path.basename(tmp), f"tracked={tracked!r}"

                # 第二次调用不带 workdir，应落在追踪的 cwd 里
                result = await exec_command("pwd", timeout=30)
                assert os.path.basename(tmp) in result
        _shell_cwd.clear()

    @pytest.mark.asyncio
    async def test_tracked_cwd_with_cjk_dir_not_rejected(self):
        """追踪 cwd 来自 pwd -P（真实路径，可含 CJK），必须可原样用于下次执行。"""
        _shell_cwd.clear()
        with tempfile.TemporaryDirectory(prefix="flyclaw-cwd-中文-", dir=os.getcwd()) as tmp:
            target = tmp.replace("\\", "/")
            cfg = _mock_config()
            with patch("src.tools.exec._get_config", return_value=cfg):
                await exec_command(f'cd "{target}"', timeout=30)
                tracked = _shell_cwd.get("")
                assert tracked, "CJK 目录的追踪 cwd 不应被丢弃"
                # 下一次调用用追踪 cwd 执行
                result = await exec_command("pwd", timeout=30)
                assert os.path.basename(tmp) in result
        _shell_cwd.clear()

    @pytest.mark.asyncio
    async def test_explicit_workdir_does_not_pollute_tracking(self):
        _shell_cwd.clear()
        with tempfile.TemporaryDirectory(prefix="flyclaw-cwd-test2-", dir=os.getcwd()) as tmp:
            target = tmp.replace("\\", "/")
            with patch("src.tools.exec._get_config", return_value=_mock_config()):
                before = _shell_cwd.get("")
                # workdir 显式指向 tmp：命令在 tmp 执行，但追踪 cwd 不应被污染
                result = await exec_command("pwd", timeout=30, workdir=target)
                after = _shell_cwd.get("")
                assert before == after or after == ""
                assert os.path.basename(tmp) in result
        _shell_cwd.clear()

    @pytest.mark.asyncio
    async def test_cwd_readback_failure_keeps_output(self):
        cfg = _mock_config()
        with (
            patch("src.tools.exec._get_config", return_value=cfg),
            patch("src.tools.exec._sandbox_path_allowed", side_effect=RuntimeError("boom")),
        ):
            result = await exec_command("echo finalize-ok", timeout=30)
        assert "finalize-ok" in result

    @pytest.mark.skipif(os.name != "nt", reason="MSYS 挂载点仅存在于 Windows Git Bash")
    @pytest.mark.asyncio
    async def test_msys_mount_root_not_tracked(self):
        """cd /tmp（MSYS 挂载点）无盘符，不得被原生 Windows 解析成
        「当前盘根」下的另一目录（D:\\tmp ≠ 挂载源 D:/Temp）而错误跟踪。"""
        _shell_cwd.clear()
        with patch("src.tools.exec._get_config", return_value=_mock_config()):
            await exec_command("cd /tmp", timeout=30)
            assert _shell_cwd.get("") == "", "挂载点路径不应被跟踪"
        _shell_cwd.clear()
