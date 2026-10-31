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
        """契约：裸字符串一律 success，即使内容以 [error] 开头。"""

        async def failing() -> str:
            return "[error] No active channel context"

        result = await ToolDef.from_function(failing).execute({})
        assert result.status == "success"
        assert result.content == "[error] No active channel context"

    @pytest.mark.asyncio
    async def test_toolresult_passes_through_untouched(self):
        async def native() -> ToolResult:
            return ToolResult.error("Invalid URL: x")

        result = await ToolDef.from_function(native).execute({})
        assert result.is_error is True
        assert result.content == "Invalid URL: x"

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


class TestToolLayerFailureVocabulary:
    """防线：工具层失败信令词汇表。

    一、禁止裸 return "[error]..."——裸字符串会被包装层判成 success。
    二、工具结果内容禁止携带 [error]/[blocked] 前缀——失败信令只存在于 status。
    """

    _EXTRA_TARGETS = ("src/agents/delegate.py", "src/agent/loop.py", "src/agent/guardrails.py")

    @staticmethod
    def _targets(root):
        return sorted((root / "src" / "tools").rglob("*.py")) + [
            root / p for p in TestToolLayerFailureVocabulary._EXTRA_TARGETS
        ]

    def test_no_bare_error_string_returns(self):
        import re
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        pattern = re.compile(r"^\s*return\s+f?[\"']\[error\]")

        offenders = [
            f"{path.relative_to(root)}:{i}: {line.strip()}"
            for path in self._targets(root)
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
            if pattern.match(line)
        ]
        assert offenders == []

    def test_no_failure_prefix_literals_in_tool_results(self):
        import re
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        pattern = re.compile(r"[\"']\[(error|blocked)\]")

        offenders = [
            f"{path.relative_to(root)}:{i}: {line.strip()}"
            for path in self._targets(root)
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
            if pattern.search(line)
        ]
        assert offenders == []
