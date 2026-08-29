"""Tests for QQ typing indicator circuit breaker: closed → open → half-open."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.channels.qq import QQChannel


def _make_qq():
    """Create a lightweight QQChannel without real config/auth."""
    ch = QQChannel.__new__(QQChannel)
    ch._typing_disabled = False
    ch._typing_fail_count = 0
    ch._typing_disabled_since = 0.0
    ch._typing_cooldown = 300.0
    ch._typing_probing = False
    ch._http_client = MagicMock()  # truthy
    ch._token_manager = MagicMock()  # truthy
    ch._seq_counter = 0
    return ch


# ---------------------------------------------------------------------------
# Closed state
# ---------------------------------------------------------------------------


class TestTypingClosedState:
    @pytest.mark.asyncio
    async def test_sends_normally_on_success(self):
        ch = _make_qq()
        with (
            patch("src.channels.qq._parse_chat_id", return_value=("c2c", "uid")),
            patch.object(ch, "_api_post", new_callable=AsyncMock, return_value={"code": 0}),
        ):
            result = await ch._send_typing("c2c:uid")
        assert result is True
        assert ch._typing_fail_count == 0
        assert ch._typing_disabled is False

    @pytest.mark.asyncio
    async def test_three_failures_opens_circuit(self):
        ch = _make_qq()
        with (
            patch("src.channels.qq._parse_chat_id", return_value=("c2c", "uid")),
            patch.object(ch, "_api_post", new_callable=AsyncMock, return_value=None),
            patch("src.channels.qq._time") as mock_time,
        ):
            now = 1000.0
            mock_time.monotonic.side_effect = [now, now + 1, now + 2]
            for i in range(3):
                await ch._send_typing("c2c:uid")
        assert ch._typing_disabled is True
        assert ch._typing_disabled_since > 0
        assert ch._typing_fail_count == 3

    @pytest.mark.asyncio
    async def test_success_resets_fail_count(self):
        ch = _make_qq()
        ch._typing_fail_count = 2
        with (
            patch("src.channels.qq._parse_chat_id", return_value=("c2c", "uid")),
            patch.object(ch, "_api_post", new_callable=AsyncMock, return_value={"code": 0}),
        ):
            await ch._send_typing("c2c:uid")
        assert ch._typing_fail_count == 0


# ---------------------------------------------------------------------------
# Open state (blocked during cooldown)
# ---------------------------------------------------------------------------


class TestTypingOpenState:
    @pytest.mark.asyncio
    async def test_blocks_during_cooldown(self):
        ch = _make_qq()
        ch._typing_disabled = True
        ch._typing_disabled_since = 1000.0
        with (
            patch("src.channels.qq._parse_chat_id", return_value=("c2c", "uid")),
            patch("src.channels.qq._time") as mock_time,
        ):
            mock_time.monotonic.return_value = 1100.0  # 100s elapsed < 300s cooldown
            result = await ch._send_typing("c2c:uid")
        assert result is False

    @pytest.mark.asyncio
    async def test_probing_blocks_other_tasks(self):
        ch = _make_qq()
        ch._typing_disabled = True
        ch._typing_probing = True  # another task is already probing
        result = await ch._send_typing("c2c:uid")
        assert result is False


# ---------------------------------------------------------------------------
# Half-open state (probe after cooldown)
# ---------------------------------------------------------------------------


class TestTypingHalfOpenState:
    @pytest.mark.asyncio
    async def test_probe_success_reopens_circuit(self):
        ch = _make_qq()
        ch._typing_disabled = True
        ch._typing_disabled_since = 1000.0
        with (
            patch("src.channels.qq._parse_chat_id", return_value=("c2c", "uid")),
            patch.object(ch, "_api_post", new_callable=AsyncMock, return_value={"code": 0}),
            patch("src.channels.qq._time") as mock_time,
        ):
            mock_time.monotonic.return_value = 1400.0  # 400s > 300s cooldown
            result = await ch._send_typing("c2c:uid")
        assert result is True
        assert ch._typing_disabled is False
        assert ch._typing_fail_count == 0
        assert ch._typing_probing is False

    @pytest.mark.asyncio
    async def test_probe_failure_extends_cooldown(self):
        ch = _make_qq()
        ch._typing_disabled = True
        ch._typing_disabled_since = 1000.0
        with (
            patch("src.channels.qq._parse_chat_id", return_value=("c2c", "uid")),
            patch.object(ch, "_api_post", new_callable=AsyncMock, return_value=None),
            patch("src.channels.qq._time") as mock_time,
        ):
            mock_time.monotonic.return_value = 1400.0  # cooldown elapsed
            result = await ch._send_typing("c2c:uid")
        assert result is False
        assert ch._typing_disabled is True
        assert ch._typing_disabled_since == 1400.0  # reset cooldown start
        assert ch._typing_probing is False


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestTypingEdgeCases:
    @pytest.mark.asyncio
    async def test_non_c2c_returns_false(self):
        ch = _make_qq()
        with patch("src.channels.qq._parse_chat_id", return_value=("group", "gid")):
            result = await ch._send_typing("group:gid")
        assert result is False

    @pytest.mark.asyncio
    async def test_exception_clears_probing_flag(self):
        ch = _make_qq()
        ch._typing_disabled = True
        ch._typing_disabled_since = 1000.0
        with (
            patch("src.channels.qq._parse_chat_id", return_value=("c2c", "uid")),
            patch.object(ch, "_api_post", new_callable=AsyncMock, side_effect=RuntimeError("boom")),
            patch("src.channels.qq._time") as mock_time,
        ):
            mock_time.monotonic.return_value = 1400.0  # cooldown elapsed → probe
            result = await ch._send_typing("c2c:uid")
        assert result is False
        assert ch._typing_probing is False  # finally block must clear it

    @pytest.mark.asyncio
    async def test_no_http_client_returns_false(self):
        ch = _make_qq()
        ch._http_client = None
        with patch("src.channels.qq._parse_chat_id", return_value=("c2c", "uid")):
            result = await ch._send_typing("c2c:uid")
        assert result is False
