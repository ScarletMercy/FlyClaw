"""Tests for judge_memory_with_llm finally-closes model."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestJudgeMemoryWithLlmFinallyClose:
    @pytest.mark.asyncio
    async def test_close_called_on_success(self):
        """model.close() must be called when judge returns a result."""
        mock_model = AsyncMock()
        mock_model.chat.return_value = MagicMock(
            content='{"remember": true, "content": "user likes tea", "category": "preference"}'
        )

        with patch("src.agent.client.ChatClient", return_value=mock_model):
            from src.tools.memory_tools import judge_memory_with_llm

            result = await judge_memory_with_llm("I love tea", "Got it!", "gpt-4", "http://api", "key")

        assert result == ("user likes tea", "preference")
        mock_model.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_called_on_remember_false(self):
        """model.close() must be called even when judge decides not to remember."""
        mock_model = AsyncMock()
        mock_model.chat.return_value = MagicMock(content='{"remember": false}')

        with patch("src.agent.client.ChatClient", return_value=mock_model):
            from src.tools.memory_tools import judge_memory_with_llm

            result = await judge_memory_with_llm("hello", "hi", "gpt-4", "http://api", "key")

        assert result is None
        mock_model.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_called_on_json_parse_error(self):
        """model.close() must be called even when JSON parsing fails."""
        mock_model = AsyncMock()
        mock_model.chat.return_value = MagicMock(content="not valid json{{{")

        with patch("src.agent.client.ChatClient", return_value=mock_model):
            from src.tools.memory_tools import judge_memory_with_llm

            result = await judge_memory_with_llm("hello", "hi", "gpt-4", "http://api", "key")

        assert result is None
        mock_model.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_called_on_chat_exception(self):
        """model.close() must be called even when chat() raises."""
        mock_model = AsyncMock()
        mock_model.chat.side_effect = RuntimeError("API error")

        with patch("src.agent.client.ChatClient", return_value=mock_model):
            from src.tools.memory_tools import judge_memory_with_llm

            result = await judge_memory_with_llm("hello", "hi", "gpt-4", "http://api", "key")

        assert result is None
        mock_model.close.assert_awaited_once()
