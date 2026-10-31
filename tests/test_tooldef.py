"""ToolDef.execute 包装层：裸字符串转 ToolResult，ToolResult 透传。"""

import pytest

from src.agent.tooldef import ToolDef
from src.tools.result import ToolResult


class TestExecuteBridge:
    @pytest.mark.asyncio
    async def test_plain_str_wraps_success(self):
        async def probe() -> str:
            return "ok"

        result = await ToolDef.from_function(probe).execute({})
        assert isinstance(result, ToolResult)
        assert result.content == "ok"
        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_bare_string_is_always_success_no_prefix_sniffing(self):
        """契约：裸字符串一律 success，即使带 [error] 前缀也不嗅探。"""

        async def failing() -> str:
            return "[error] No active channel context"

        result = await ToolDef.from_function(failing).execute({})
        assert result.status == "success"
        assert result.content == "[error] No active channel context"

    @pytest.mark.asyncio
    async def test_toolresult_passes_through_untouched(self):
        async def native() -> ToolResult:
            return ToolResult.error("[error] Invalid URL: x")

        result = await ToolDef.from_function(native).execute({})
        assert result.is_error is True
        assert result.content == "[error] Invalid URL: x"

    @pytest.mark.asyncio
    async def test_none_returns_empty_success(self):
        async def nothing() -> None:
            return None

        result = await ToolDef.from_function(nothing).execute({})
        assert result.content == ""
        assert not result.is_error

    @pytest.mark.asyncio
    async def test_sync_function_supported(self):
        def sync_probe() -> str:
            return "sync ok"

        result = await ToolDef.from_function(sync_probe).execute({})
        assert result.status == "success"
        assert result.content == "sync ok"
