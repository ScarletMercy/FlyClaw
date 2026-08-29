"""Tests for src/tools/ai_tools.py — tool registration."""

from src.tools.ai_tools import get_tools


class TestAiTools:
    def test_get_tools_returns_list(self):
        tools = get_tools()
        assert isinstance(tools, list)
        assert len(tools) == 1

    def test_tool_name(self):
        tools = get_tools()
        assert tools[0].name == "subagent_status"
