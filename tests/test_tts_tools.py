"""Tests for src/tools/tts_tools.py — basic validation logic."""

from src.tools.tts_tools import get_tools


class TestTtsTools:
    def test_get_tools_returns_list(self):
        tools = get_tools()
        assert isinstance(tools, list)
        assert len(tools) == 1

    def test_tool_has_name(self):
        tools = get_tools()
        assert tools[0].name == "text_to_speech"
