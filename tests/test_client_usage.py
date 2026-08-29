"""Tests for ChatClient usage extraction in ChatResponse."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.client import ChatClient


def _make_openai_resp(
    content: str = "Hello!",
    has_usage: bool = True,
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    total_tokens: int = 150,
    cached_tokens: int | None = None,
    has_details: bool = True,
):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message = MagicMock()
    resp.choices[0].message.content = content
    resp.choices[0].message.tool_calls = []

    if has_usage:
        usage = MagicMock()
        usage.prompt_tokens = prompt_tokens
        usage.completion_tokens = completion_tokens
        usage.total_tokens = total_tokens

        if has_details:
            details = MagicMock()
            details.cached_tokens = cached_tokens
            usage.prompt_tokens_details = details
        else:
            usage.prompt_tokens_details = None

        resp.usage = usage
    else:
        resp.usage = None

    return resp


class TestUsageExtraction:
    @pytest.mark.asyncio
    async def test_usage_extracted_when_present(self):
        resp = _make_openai_resp(
            prompt_tokens=200,
            completion_tokens=80,
            total_tokens=280,
            cached_tokens=120,
        )

        client = ChatClient(base_url="http://fake", api_key="k", model="m")
        client._client = MagicMock()
        client._client.chat.completions.create = AsyncMock(return_value=resp)

        result = await client.chat([{"role": "user", "content": "hi"}])

        assert result.usage is not None
        assert result.usage["prompt_tokens"] == 200
        assert result.usage["completion_tokens"] == 80
        assert result.usage["total_tokens"] == 280
        assert result.usage["cached_tokens"] == 120

    @pytest.mark.asyncio
    async def test_usage_none_when_no_usage(self):
        resp = _make_openai_resp(has_usage=False)

        client = ChatClient(base_url="http://fake", api_key="k", model="m")
        client._client = MagicMock()
        client._client.chat.completions.create = AsyncMock(return_value=resp)

        result = await client.chat([{"role": "user", "content": "hi"}])

        assert result.usage is None

    @pytest.mark.asyncio
    async def test_cached_tokens_zero_when_no_details(self):
        resp = _make_openai_resp(has_details=False, prompt_tokens=100)

        client = ChatClient(base_url="http://fake", api_key="k", model="m")
        client._client = MagicMock()
        client._client.chat.completions.create = AsyncMock(return_value=resp)

        result = await client.chat([{"role": "user", "content": "hi"}])

        assert result.usage is not None
        assert result.usage["cached_tokens"] == 0

    @pytest.mark.asyncio
    async def test_cached_tokens_extracted(self):
        resp = _make_openai_resp(cached_tokens=500)

        client = ChatClient(base_url="http://fake", api_key="k", model="m")
        client._client = MagicMock()
        client._client.chat.completions.create = AsyncMock(return_value=resp)

        result = await client.chat([{"role": "user", "content": "hi"}])

        assert result.usage["cached_tokens"] == 500
