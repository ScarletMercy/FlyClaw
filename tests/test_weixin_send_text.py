"""Tests for weixin send_text chunk failure handling.

Bug: when a middle chunk fails in send_text, already-sent chunks cannot be
recalled and subsequent chunks are never sent, leaving the user with a
truncated message.

Fix: best-effort — continue sending remaining chunks after a failure, and
log a warning instead of aborting the entire loop.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_weixin_channel():
    """Create a WeixinChannel with mocked internals suitable for unit tests."""
    from src.channels.weixin import WeixinChannel

    config = MagicMock()
    config.account_id = "test_account"
    config.base_url = "https://fake-ilink.example.com"
    config.session_file = None
    config.split_multiline_messages = False
    config.send_chunk_delay = 0  # no delay in tests
    config.send_retry_count = 0  # no retries — each chunk gets one attempt

    ch = WeixinChannel.__new__(WeixinChannel)
    # Minimal attribute setup (bypass __init__ which does real I/O)
    ch._config = config
    ch._account_id = config.account_id
    ch._base_url = config.base_url
    ch._send_session = MagicMock()  # truthy — bypasses the None guard
    ch._token = "fake_token"
    ch._split_multiline_messages = config.split_multiline_messages
    ch._send_chunk_delay_seconds = config.send_chunk_delay
    ch._send_chunk_retries = config.send_retry_count
    ch._send_chunk_retry_delay_seconds = 1.0
    ch._token_store = MagicMock()
    ch._token_store.get.return_value = None

    return ch


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSendTextChunkFailureBestEffort:
    """When a middle chunk fails, remaining chunks should still be sent."""

    @pytest.mark.asyncio
    async def test_middle_chunk_failure_sends_remaining_chunks(self):
        """If chunk #2 of 4 fails, chunks #1, #3, #4 should still be delivered."""
        ch = _make_weixin_channel()

        sent_chunks: list[str] = []

        async def fake_send_text_chunk(*, chat_id, chunk, context_token, client_id):
            sent_chunks.append(chunk)
            if chunk == "BBB":
                raise RuntimeError("simulated network error for chunk BBB")

        ch._send_text_chunk = AsyncMock(side_effect=fake_send_text_chunk)

        # Patch _split_text_for_delivery to return 4 predictable chunks
        with patch("src.channels.weixin._split_text_for_delivery", return_value=["AAA", "BBB", "CCC", "DDD"]):
            result = await ch.send_text("chat_123", "long text that gets split")

        # Chunks AAA, CCC, DDD should have been sent despite BBB failing
        assert "AAA" in sent_chunks, "chunk AAA should be sent"
        assert "CCC" in sent_chunks, "chunk CCC should be sent despite BBB failing"
        assert "DDD" in sent_chunks, "chunk DDD should be sent despite BBB failing"

    @pytest.mark.asyncio
    async def test_middle_chunk_failure_returns_last_successful_id(self):
        """send_text should return the client_id of the last *successfully* sent chunk."""
        ch = _make_weixin_channel()

        successful_ids: list[str] = []

        async def fake_send_text_chunk(*, chat_id, chunk, context_token, client_id):
            if chunk == "FAIL":
                raise RuntimeError("boom")
            successful_ids.append(client_id)

        ch._send_text_chunk = AsyncMock(side_effect=fake_send_text_chunk)

        with patch("src.channels.weixin._split_text_for_delivery", return_value=["OK1", "FAIL", "OK2"]):
            result = await ch.send_text("chat_123", "text")

        # Result should be the client_id of OK2 (last successful send)
        assert result is not None, "should return last successful client_id, not None"
        assert len(successful_ids) == 2
        assert result == successful_ids[-1]

    @pytest.mark.asyncio
    async def test_all_chunks_fail_returns_none(self):
        """If every chunk fails, send_text should return None."""
        ch = _make_weixin_channel()

        async def always_fail(*, chat_id, chunk, context_token, client_id):
            raise RuntimeError("always fails")

        ch._send_text_chunk = AsyncMock(side_effect=always_fail)

        with patch("src.channels.weixin._split_text_for_delivery", return_value=["A", "B", "C"]):
            result = await ch.send_text("chat_123", "text")

        assert result is None

    @pytest.mark.asyncio
    async def test_no_failure_returns_last_client_id(self):
        """When all chunks succeed, behavior is unchanged — return last client_id."""
        ch = _make_weixin_channel()

        sent_ids: list[str] = []

        async def fake_send_text_chunk(*, chat_id, chunk, context_token, client_id):
            sent_ids.append(client_id)

        ch._send_text_chunk = AsyncMock(side_effect=fake_send_text_chunk)

        with patch("src.channels.weixin._split_text_for_delivery", return_value=["A", "B"]):
            result = await ch.send_text("chat_123", "text")

        assert result is not None
        assert result == sent_ids[-1]
        assert len(sent_ids) == 2
