"""Tests for exec timeout validation boundaries."""

from unittest.mock import patch

import pytest

from src.tools.exec import ToolExecutionError, exec_command


def _mock_config():
    """真实 AppConfig 构造的最小 exec 测试配置（沙箱关闭）。

    用真实 pydantic 模型而非 MagicMock：exec 直读 tools.exec 字段，
    MagicMock 未显式设置的属性会以 truthy 假值泄漏进业务逻辑。
    """
    from src.config import AppConfig

    cfg = AppConfig()
    cfg.agents.workspace = "."
    cfg.tools.exec.sandbox_enabled = False
    cfg.tools.exec.no_output_timeout_seconds = 0
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
