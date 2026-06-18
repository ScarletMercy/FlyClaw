"""Tests for QQ WebSocket reconnect/session-recovery logic.

Covers the three fixes for the 2026-06-17 incident:
- Bug #0: close-code driven session/token reset (the 4009 RESUME death-loop)
- Bug #1: _cleanup must not cancel in-flight agent turns on reconnect
"""

import asyncio
import json
import logging
from unittest.mock import AsyncMock, patch

import pytest


class _FakeTokenMgr:
    """Minimal stand-in for _QQTokenMgr — get_token + clear_cache."""

    def __init__(self):
        self.cleared = False

    async def get_token(self):
        return "fake-token"

    def clear_cache(self):
        self.cleared = True


class _FakeResp:
    """Fake httpx response for the /gateway fetch inside connect()."""

    def json(self):
        return {"url": "ws://fake"}


class _FakeWS:
    """Fake websockets connection: yields HELLO once, then idles.

    connect() reads HELLO via recv(); _recv_loop then async-iterates and
    blocks (no dispatches). Heartbeat interval is huge so the heartbeat task
    never fires during the test.
    """

    def __init__(self):
        self.sent: list[str] = []
        self.closed = False
        self._hello_yielded = False

    async def recv(self):
        if not self._hello_yielded:
            self._hello_yielded = True
            return json.dumps({"op": 10, "d": {"heartbeat_interval": 600000}})
        await asyncio.sleep(3600)

    async def send(self, data):
        self.sent.append(data)

    async def close(self):
        self.closed = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(3600)


def _make_client():
    from src.channels.qq import _QQWebSocketClient

    return _QQWebSocketClient(
        "app_id",
        _FakeTokenMgr(),
        lambda *a, **kw: None,
        logging.getLogger("test.qq"),
    )


# ── Bug #0: _apply_close_code ────────────────────────────────────────────────


class TestApplyCloseCode:
    def test_4009_clears_session(self):
        c = _make_client()
        c._session_id = "e39d9a41"
        c._last_seq = 99
        c._apply_close_code(4009)
        assert c._session_id is None
        assert c._last_seq is None

    def test_4007_clears_session(self):
        c = _make_client()
        c._session_id = "abc"
        c._last_seq = 7
        c._apply_close_code(4007)
        assert c._session_id is None
        assert c._last_seq is None

    def test_4004_clears_session_and_token(self):
        c = _make_client()
        c._session_id = "abc"
        c._last_seq = 3
        c._apply_close_code(4004)
        assert c._session_id is None
        assert c._last_seq is None
        assert c._token_mgr.cleared is True

    def test_4009_does_not_clear_token(self):
        # 4009 is session-side only; the token is still valid.
        c = _make_client()
        c._session_id = "abc"
        c._apply_close_code(4009)
        assert c._token_mgr.cleared is False

    def test_graceful_close_keeps_session(self):
        # 1000/1001/network drops must preserve session → next connect RESUMEs.
        c = _make_client()
        c._session_id = "abc"
        c._last_seq = 5
        c._apply_close_code(1000)
        assert c._session_id == "abc"
        assert c._last_seq == 5
        assert c._token_mgr.cleared is False

    def test_none_code_keeps_session(self):
        c = _make_client()
        c._session_id = "abc"
        c._apply_close_code(None)
        assert c._session_id == "abc"


# ── Bug #1: _cleanup inflight handling ───────────────────────────────────────


class TestCleanupInflight:
    async def test_default_does_not_cancel_inflight(self):
        # Reconnect path: _cleanup() must leave in-flight agent turns running.
        c = _make_client()
        c._running = True
        c._ws = None  # nothing to close

        cancelled = asyncio.Event()

        async def long_running_agent_turn():
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        task = c._spawn_task(long_running_agent_turn())
        await asyncio.sleep(0)  # let the task start
        assert task in c._inflight

        await c._cleanup()  # default cancel_inflight=False

        await asyncio.sleep(0)
        assert not cancelled.is_set(), "inflight task should NOT be cancelled on reconnect"
        assert task in c._inflight, "task should remain tracked"
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def test_cancel_inflight_true_cancels_on_shutdown(self):
        # stop() path: _cleanup(cancel_inflight=True) tears everything down.
        c = _make_client()
        c._running = True
        c._ws = None

        cancelled = asyncio.Event()

        async def agent_turn():
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        task = c._spawn_task(agent_turn())
        await asyncio.sleep(0)
        assert task in c._inflight

        await c._cleanup(cancel_inflight=True)

        assert cancelled.is_set(), "inflight task SHOULD be cancelled on stop()"
        assert c._inflight == set()

    async def test_cleanup_closes_ws(self):
        c = _make_client()

        class _FakeWS:
            def __init__(self):
                self.closed = False

            async def close(self):
                self.closed = True

        ws = _FakeWS()
        c._ws = ws
        await c._cleanup()
        assert ws.closed is True
        assert c._ws is None


# ── Bug #1 + #0 + #3: connect() handshake decision (runtime path) ────────────


class TestConnectHandshake:
    """Drive the real connect() with a patched websockets.connect + gateway
    fetch, and assert the RESUME-vs-IDENTIFY decision and the Bug #3 raise.
    This is the closest runtime check short of a live QQ gateway."""

    async def _connect(self, client, fake):
        with (
            patch("websockets.connect", new=AsyncMock(return_value=fake)),
            patch("src.channels.qq.api_request_with_retry", new=AsyncMock(return_value=_FakeResp())),
        ):
            await client.connect()

    async def test_resume_with_session_then_identify_after_4009(self):
        # Reproduces the logged incident's recovery contract:
        # dead session present → RESUME; 4009 clears it → reconnect IDENTIFYs.
        from src.channels.qq import OP_IDENTIFY, OP_RESUME

        client = _make_client()
        client._running = True

        # 1) Dead session → connect() must send RESUME
        client._session_id = "dead-session"
        client._last_seq = 42
        fake1 = _FakeWS()
        await self._connect(client, fake1)
        ops1 = [json.loads(s).get("op") for s in fake1.sent]
        assert OP_RESUME in ops1, f"expected RESUME with session set, got {ops1}"
        await client.stop()

        # 2) 4009 clears session
        client._apply_close_code(4009)
        assert client._session_id is None
        assert client._last_seq is None

        # 3) Reconnect → must IDENTIFY (no session to resume)
        client._running = True
        fake2 = _FakeWS()
        await self._connect(client, fake2)
        ops2 = [json.loads(s).get("op") for s in fake2.sent]
        assert OP_IDENTIFY in ops2, f"expected IDENTIFY after 4009 cleared session, got {ops2}"
        await client.stop()

    async def test_connect_raises_on_non_hello_frame(self):
        # Bug #3: a bad first frame must raise (not silently return) so the
        # outer _schedule_reconnect retry loop owns backoff.
        client = _make_client()
        client._running = True

        class _BadWS:
            async def recv(self):
                return json.dumps({"op": 1})  # not HELLO

            async def send(self, data):
                pass

            async def close(self):
                pass

        with (
            patch("websockets.connect", new=AsyncMock(return_value=_BadWS())),
            patch("src.channels.qq.api_request_with_retry", new=AsyncMock(return_value=_FakeResp())),
        ):
            with pytest.raises(RuntimeError):
                await client.connect()

    async def test_4009_close_in_recv_loop_triggers_identify_on_reconnect(self):
        # End-to-end-ish: drive the actual recv_loop against a fake ws that
        # raises a real ConnectionClosed(4009), then assert the NEXT connect
        # IDENTIFYs. Verifies _apply_close_code fires on the real close path.
        from src.channels.qq import OP_IDENTIFY, OP_HELLO
        from websockets.exceptions import ConnectionClosedError
        from websockets.frames import Close

        client = _make_client()
        client._running = True
        client._session_id = "dead-session"
        client._last_seq = 7

        close_exc = ConnectionClosedError(Close(4009, "Session timed out"), None)

        class _ClosingWS:
            """HELLO via recv(), then the async-for raises 4009 immediately."""

            def __init__(self):
                self.sent = []
                self.closed = False
                self._hello_yielded = False

            async def recv(self):
                if not self._hello_yielded:
                    self._hello_yielded = True
                    return json.dumps({"op": OP_HELLO, "d": {"heartbeat_interval": 600000}})
                raise close_exc

            async def send(self, data):
                self.sent.append(data)

            async def close(self):
                self.closed = True

            def __aiter__(self):
                return self

            async def __anext__(self):
                raise close_exc

        fake1 = _ClosingWS()
        await self._connect(client, fake1)
        # connect() returned; recv_loop (background) now hits 4009 → clears session.
        # Give the background recv_task a moment to process the close.
        for _ in range(50):
            await asyncio.sleep(0.01)
            if client._session_id is None:
                break
        assert client._session_id is None, "4009 in recv_loop should have cleared session"
        await client.stop()

        # Next connect must IDENTIFY.
        client._running = True
        fake2 = _FakeWS()
        await self._connect(client, fake2)
        ops = [json.loads(s).get("op") for s in fake2.sent]
        assert OP_IDENTIFY in ops, f"expected IDENTIFY after recv_loop 4009, got {ops}"
        await client.stop()


# ── Reconnect persistence: never give up ─────────────────────────────────────


class TestReconnectNeverGivesUp:
    async def test_reconnect_retries_past_old_cap(self):
        # Regression guard: reconnect must NOT cap at any attempt count. A
        # long outage must self-heal, not silently go offline forever. We make
        # connect() fail persistently and assert the loop kept going well past
        # the old _MAX_RECONNECT_ATTEMPTS=100 ceiling.
        client = _make_client()
        client._running = True

        calls = {"n": 0}

        async def _always_fails():
            calls["n"] += 1
            if calls["n"] > 110:
                client._running = False  # break the otherwise-infinite loop
            raise RuntimeError("persistent failure")

        async def _fast_sleep(_delay):
            # yield once to the loop without recursing into the patched asyncio.sleep
            ev = asyncio.Event()
            ev.set()
            await ev.wait()

        client.connect = _always_fails  # await self.connect() calls it bare (no self)
        with patch("asyncio.sleep", _fast_sleep):
            await client._schedule_reconnect()

        assert calls["n"] > 100, f"reconnect stopped at {calls['n']} — must never give up"


# ── Bug A: stop() must cancel a parked boot-time connect/reconnect task ──────


class TestStopCancelsStartTask:
    async def test_stop_cancels_parked_ws_start_task(self):
        # Regression guard: if the boot-time _start_ws task is parked inside
        # _schedule_reconnect's asyncio.sleep (persistent boot failure) when
        # stop() runs, stop() must cancel it promptly — not leave it lingering
        # up to 60s after shutdown.
        from src.channels.qq import QQChannel
        from src.config import AppConfig

        config = AppConfig()
        config.channels.qq.enabled = False  # avoid start()'s real network work
        ch = QQChannel(config.channels.qq)

        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def parked():
            started.set()
            try:
                await asyncio.sleep(600)  # simulates the backoff sleep
            except asyncio.CancelledError:
                cancelled.set()
                raise

        ch._ws_start_task = asyncio.create_task(parked())
        await started.wait()

        # stop() must cancel the parked task well before 600s.
        await asyncio.wait_for(ch.stop(), timeout=5.0)
        assert cancelled.is_set(), "stop() failed to cancel the parked _ws_start_task"
