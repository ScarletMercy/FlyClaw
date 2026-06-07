"""Tests for exec timeout validation boundaries."""

from unittest.mock import patch, MagicMock

import pytest

from src.tools.exec import ToolExecutionError, exec_command


def _mock_config():
    """Return a minimal config mock that disables sandbox."""
    cfg = MagicMock()
    cfg.tools.exec.deny_patterns = []
    cfg.tools.exec.max_output_bytes = 102400
    cfg.tools.exec.approval_mode = "off"
    cfg.tools.exec.no_output_timeout_seconds = 0
    cfg.tools.exec.sandbox_enabled = False
    cfg.tools.exec.sandbox_allowed_dirs = []
    cfg.tools.exec.sandbox_env_whitelist = ["PATH"]
    cfg.agents.workspace = "."
    return cfg


class TestExecTimeoutLimit:
    @pytest.mark.asyncio
    async def test_timeout_exceeds_3600_raises(self):
        """timeout > 3600 should raise ToolExecutionError about upper limit."""
        with pytest.raises(ToolExecutionError, match="上限为 3600"):
            await exec_command("echo hi", timeout=3601)

    @pytest.mark.asyncio
    async def test_timeout_exactly_3600_passes(self):
        """timeout == 3600 passes the upper-limit check and executes."""
        with patch("src.tools.exec._get_config", return_value=_mock_config()):
            result = await exec_command("echo hi", timeout=3600)
            assert "hi" in result

    @pytest.mark.asyncio
    async def test_timeout_zero_raises(self):
        """timeout <= 0 should raise ToolExecutionError about positive integer."""
        with pytest.raises(ToolExecutionError, match="正整数"):
            await exec_command("echo hi", timeout=0)

    @pytest.mark.asyncio
    async def test_timeout_negative_raises(self):
        """Negative timeout should raise ToolExecutionError about positive integer."""
        with pytest.raises(ToolExecutionError, match="正整数"):
            await exec_command("echo hi", timeout=-1)

    @pytest.mark.asyncio
    async def test_normal_timeout_executes(self):
        """A reasonable timeout like 100 should work normally."""
        with patch("src.tools.exec._get_config", return_value=_mock_config()):
            result = await exec_command("echo ok", timeout=100)
            assert "ok" in result
