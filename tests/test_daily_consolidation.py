"""Tests for run_daily_consolidation and helpers."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.state import AgentState


def _make_container(
    state_store=None,
    agent_loop=None,
    session_registry=None,
    min_messages=10,
    qq=None,
    weixin=None,
):
    container = MagicMock()
    container.state_store = state_store
    container.agent_loop = agent_loop
    container.session_registry = session_registry
    container.qq = qq
    container.weixin = weixin
    container.config.consolidation.min_messages = min_messages
    return container


def _make_state(messages, chat_id="chat_123", channel="qq", created_at=None):
    return AgentState(
        messages=messages,
        chat_id=chat_id,
        channel=channel,
        created_at=created_at if created_at is not None else time.time(),
    )


def _make_messages(n_user, n_assistant=None):
    if n_assistant is None:
        n_assistant = n_user
    msgs = []
    for i in range(n_user):
        msgs.append({"role": "user", "content": f"user msg {i}"})
    for i in range(n_assistant):
        msgs.append({"role": "assistant", "content": f"assistant msg {i}"})
    return msgs


_PATCH_CONSOLIDATE = "src.services.daily_consolidation._consolidate_session"
_PATCH_NOTIFY = "src.services.daily_consolidation._send_notification"
_PATCH_NEW_SESSION = "src.services.daily_consolidation._create_new_session"
_PATCH_REVIEW = "src.skills.review.spawn_background_review"


# ─── Guard: early return ─────────────────────────────────────────────────────


class TestGuardConditions:
    @pytest.mark.asyncio
    async def test_no_state_store_returns_empty(self):
        container = _make_container(state_store=None, agent_loop=MagicMock())
        from src.services.daily_consolidation import run_daily_consolidation

        result = await run_daily_consolidation(container)
        assert result["sessions_processed"] == 0
        assert result["sessions_skipped"] == 0

    @pytest.mark.asyncio
    async def test_no_agent_loop_returns_empty(self):
        store = MagicMock()
        container = _make_container(state_store=store, agent_loop=None)
        from src.services.daily_consolidation import run_daily_consolidation

        result = await run_daily_consolidation(container)
        assert result["sessions_processed"] == 0

    @pytest.mark.asyncio
    async def test_empty_threads_returns_empty(self):
        store = MagicMock()
        store.list_threads = AsyncMock(return_value=[])
        container = _make_container(state_store=store, agent_loop=MagicMock())
        from src.services.daily_consolidation import run_daily_consolidation

        result = await run_daily_consolidation(container)
        assert result["sessions_processed"] == 0
        assert result["sessions_skipped"] == 0


# ─── Session filtering ───────────────────────────────────────────────────────


class TestSessionFiltering:
    @pytest.mark.asyncio
    async def test_skip_below_min_messages(self):
        store = MagicMock()
        state = _make_state(_make_messages(2, 2))
        store.list_threads = AsyncMock(return_value=["t1"])
        store.load = AsyncMock(return_value=state)

        container = _make_container(state_store=store, agent_loop=MagicMock(), min_messages=10)
        from src.services.daily_consolidation import run_daily_consolidation

        result = await run_daily_consolidation(container)
        assert result["sessions_skipped"] == 1
        assert result["sessions_processed"] == 0

    @pytest.mark.asyncio
    async def test_skip_below_3_user_messages(self):
        store = MagicMock()
        msgs = _make_messages(2, 20)
        state = _make_state(msgs)
        store.list_threads = AsyncMock(return_value=["t1"])
        store.load = AsyncMock(return_value=state)

        container = _make_container(state_store=store, agent_loop=MagicMock(), min_messages=10)
        from src.services.daily_consolidation import run_daily_consolidation

        result = await run_daily_consolidation(container)
        assert result["sessions_skipped"] == 1

    @pytest.mark.asyncio
    async def test_skip_does_not_create_new_session(self):
        """Skipped sessions must NOT rotate — prevents exponential session growth."""
        store = MagicMock()
        state = _make_state(_make_messages(2, 2))
        store.list_threads = AsyncMock(return_value=["t1"])
        store.load = AsyncMock(return_value=state)

        container = _make_container(state_store=store, agent_loop=MagicMock(), min_messages=10)

        with patch(_PATCH_NEW_SESSION, new_callable=AsyncMock) as mock_new:
            from src.services.daily_consolidation import run_daily_consolidation

            result = await run_daily_consolidation(container)

        assert result["sessions_skipped"] == 1
        mock_new.assert_not_called()

    @pytest.mark.asyncio
    async def test_skip_none_state(self):
        store = MagicMock()
        store.list_threads = AsyncMock(return_value=["t1"])
        store.load = AsyncMock(return_value=None)

        container = _make_container(state_store=store, agent_loop=MagicMock())
        from src.services.daily_consolidation import run_daily_consolidation

        result = await run_daily_consolidation(container)
        # None state → no session exists, not counted as "skipped"
        assert result["sessions_skipped"] == 0


# ─── Normal flow ─────────────────────────────────────────────────────────────


class TestNormalFlow:
    @pytest.mark.asyncio
    async def test_process_session_calls_consolidate(self):
        store = MagicMock()
        msgs = _make_messages(5, 10)
        state = _make_state(msgs)
        store.list_threads = AsyncMock(return_value=["qq:group:abc"])
        store.load = AsyncMock(return_value=state)

        agent_loop = MagicMock()
        agent_loop.invalidate_memory_cache = MagicMock()

        container = _make_container(state_store=store, agent_loop=agent_loop, session_registry=None)

        with (
            patch(_PATCH_CONSOLIDATE, return_value="saved 2 memories"),
            patch(_PATCH_NOTIFY, new_callable=AsyncMock),
            patch(_PATCH_NEW_SESSION, new_callable=AsyncMock),
        ):
            from src.services.daily_consolidation import run_daily_consolidation

            result = await run_daily_consolidation(container)

        assert result["sessions_processed"] == 1

    @pytest.mark.asyncio
    async def test_creates_new_session_via_registry(self):
        store = MagicMock()
        msgs = _make_messages(5, 10)
        state = _make_state(msgs)
        store.list_threads = AsyncMock(return_value=["qq:group:abc"])
        store.load = AsyncMock(return_value=state)

        registry = MagicMock()
        registry.new_session = AsyncMock(return_value="new_session_id")

        agent_loop = MagicMock()
        agent_loop.invalidate_memory_cache = MagicMock()

        container = _make_container(state_store=store, agent_loop=agent_loop, session_registry=registry)

        with (
            patch(_PATCH_CONSOLIDATE, return_value=""),
            patch(_PATCH_NOTIFY, new_callable=AsyncMock),
        ):
            from src.services.daily_consolidation import run_daily_consolidation

            await run_daily_consolidation(container)

        registry.new_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalidate_memory_cache_called(self):
        store = MagicMock()
        msgs = _make_messages(5, 10)
        state = _make_state(msgs)
        store.list_threads = AsyncMock(return_value=["qq:group:abc"])
        store.load = AsyncMock(return_value=state)

        agent_loop = MagicMock()
        agent_loop.invalidate_memory_cache = MagicMock()

        container = _make_container(state_store=store, agent_loop=agent_loop, session_registry=None)

        with (
            patch(_PATCH_CONSOLIDATE, return_value=""),
            patch(_PATCH_NOTIFY, new_callable=AsyncMock),
            patch(_PATCH_NEW_SESSION, new_callable=AsyncMock),
        ):
            from src.services.daily_consolidation import run_daily_consolidation

            await run_daily_consolidation(container)

        agent_loop.invalidate_memory_cache.assert_called_once()


# ─── Notifications ───────────────────────────────────────────────────────────


class TestNotifications:
    @pytest.mark.asyncio
    async def test_notification_dreaming_and_wake(self):
        store = MagicMock()
        msgs = _make_messages(5, 10)
        state = _make_state(msgs, chat_id="chat_abc", channel="qq")
        store.list_threads = AsyncMock(return_value=["qq:group:abc"])
        store.load = AsyncMock(return_value=state)

        agent_loop = MagicMock()
        agent_loop.invalidate_memory_cache = MagicMock()

        container = _make_container(state_store=store, agent_loop=agent_loop)

        notifications = []

        async def _capture_notify(c, channel, chat_id, text):
            notifications.append(text)

        with (
            patch(_PATCH_CONSOLIDATE, return_value="saved memory"),
            patch(_PATCH_NOTIFY, side_effect=_capture_notify),
            patch(_PATCH_NEW_SESSION, new_callable=AsyncMock),
        ):
            from src.services.daily_consolidation import run_daily_consolidation

            await run_daily_consolidation(container)

        assert len(notifications) == 2
        assert "dreaming" in notifications[0]
        assert "wake up" in notifications[1]
        assert "saved memory" in notifications[1]

    @pytest.mark.asyncio
    async def test_notification_skipped_when_no_chat_id(self):
        store = MagicMock()
        msgs = _make_messages(5, 10)
        state = _make_state(msgs, chat_id="", channel="qq")
        store.list_threads = AsyncMock(return_value=["qq:group:abc"])
        store.load = AsyncMock(return_value=state)

        agent_loop = MagicMock()
        agent_loop.invalidate_memory_cache = MagicMock()

        qq_channel = MagicMock()
        qq_channel.send_text = AsyncMock()

        container = _make_container(state_store=store, agent_loop=agent_loop, qq=qq_channel)

        with (
            patch(_PATCH_CONSOLIDATE, return_value=""),
            patch(_PATCH_NEW_SESSION, new_callable=AsyncMock),
        ):
            from src.services.daily_consolidation import run_daily_consolidation

            await run_daily_consolidation(container)

        qq_channel.send_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_consolidation_failure_sends_error_notification(self):
        store = MagicMock()
        msgs = _make_messages(5, 10)
        state = _make_state(msgs, chat_id="chat_abc", channel="qq")
        store.list_threads = AsyncMock(return_value=["qq:group:abc"])
        store.load = AsyncMock(return_value=state)

        agent_loop = MagicMock()
        container = _make_container(state_store=store, agent_loop=agent_loop)

        notifications = []

        async def _capture_notify(c, channel, chat_id, text):
            notifications.append(text)

        async def _fail_consolidate(agent_loop, config, messages):
            raise RuntimeError("LLM timeout")

        with (
            patch(_PATCH_CONSOLIDATE, side_effect=_fail_consolidate),
            patch(_PATCH_NOTIFY, side_effect=_capture_notify),
        ):
            from src.services.daily_consolidation import run_daily_consolidation

            result = await run_daily_consolidation(container)

        assert result["sessions_processed"] == 0
        assert len(result["errors"]) == 1
        assert len(notifications) == 2
        assert "dreaming" in notifications[0]
        assert "consolidation failed" in notifications[1]


# ─── Summary counting ────────────────────────────────────────────────────────


class TestSummaryCounting:
    @pytest.mark.asyncio
    async def test_memories_saved_counted_from_summary(self):
        store = MagicMock()
        msgs = _make_messages(5, 10)
        state = _make_state(msgs)
        store.list_threads = AsyncMock(return_value=["qq:group:abc"])
        store.load = AsyncMock(return_value=state)

        agent_loop = MagicMock()
        agent_loop.invalidate_memory_cache = MagicMock()

        container = _make_container(state_store=store, agent_loop=agent_loop)

        with (
            patch(_PATCH_CONSOLIDATE, return_value="memory saved: user likes tea"),
            patch(_PATCH_NOTIFY, new_callable=AsyncMock),
            patch(_PATCH_NEW_SESSION, new_callable=AsyncMock),
        ):
            from src.services.daily_consolidation import run_daily_consolidation

            result = await run_daily_consolidation(container)

        assert result["memories_saved"] == 1
        assert result["skills_updated"] == 0

    @pytest.mark.asyncio
    async def test_skills_updated_counted_from_summary(self):
        store = MagicMock()
        msgs = _make_messages(5, 10)
        state = _make_state(msgs)
        store.list_threads = AsyncMock(return_value=["qq:group:abc"])
        store.load = AsyncMock(return_value=state)

        agent_loop = MagicMock()
        agent_loop.invalidate_memory_cache = MagicMock()

        container = _make_container(state_store=store, agent_loop=agent_loop)

        with (
            patch(_PATCH_CONSOLIDATE, return_value="skill created: auto_format"),
            patch(_PATCH_NOTIFY, new_callable=AsyncMock),
            patch(_PATCH_NEW_SESSION, new_callable=AsyncMock),
        ):
            from src.services.daily_consolidation import run_daily_consolidation

            result = await run_daily_consolidation(container)

        assert result["skills_updated"] == 1
        assert result["memories_saved"] == 0


# ─── Age filtering ────────────────────────────────────────────────────────────


class TestAgeFiltering:
    @pytest.mark.asyncio
    async def test_skip_session_older_than_24h(self):
        store = MagicMock()
        state = _make_state(_make_messages(5, 10), created_at=time.time() - 48 * 3600)
        store.list_threads = AsyncMock(return_value=["old_session"])
        store.load = AsyncMock(return_value=state)

        container = _make_container(state_store=store, agent_loop=MagicMock())
        from src.services.daily_consolidation import run_daily_consolidation

        result = await run_daily_consolidation(container)
        assert result["sessions_skipped"] == 1
        assert result["sessions_processed"] == 0

    @pytest.mark.asyncio
    async def test_process_session_within_24h(self):
        store = MagicMock()
        state = _make_state(_make_messages(5, 10), created_at=time.time() - 3600)
        store.list_threads = AsyncMock(return_value=["qq:group:abc"])
        store.load = AsyncMock(return_value=state)

        agent_loop = MagicMock()
        agent_loop.invalidate_memory_cache = MagicMock()

        container = _make_container(state_store=store, agent_loop=agent_loop, session_registry=None)

        with (
            patch(_PATCH_CONSOLIDATE, return_value="saved 1 memory"),
            patch(_PATCH_NOTIFY, new_callable=AsyncMock),
            patch(_PATCH_NEW_SESSION, new_callable=AsyncMock),
        ):
            from src.services.daily_consolidation import run_daily_consolidation

            result = await run_daily_consolidation(container)

        assert result["sessions_processed"] == 1


# ─── _consolidate_session internals (patch spawn, not the function itself) ────


class TestConsolidateSession:
    """Test _consolidate_session by patching spawn_background_review,
    letting the real _consolidate_session logic execute."""

    @pytest.mark.asyncio
    async def test_calls_spawn_with_correct_args(self):
        agent_loop = MagicMock()
        agent_loop._client = MagicMock()
        agent_loop._tools = ["tool_a", "tool_b"]
        config = MagicMock()
        messages = [{"role": "user", "content": "hello"}]

        async def _fake_spawn(*args, **kwargs):
            return "saved memory"

        with patch(_PATCH_REVIEW, side_effect=_fake_spawn) as mock_review:
            from src.services.daily_consolidation import _consolidate_session

            result = await _consolidate_session(agent_loop=agent_loop, config=config, messages=messages)

        assert result == "saved memory"
        mock_review.assert_called_once()
        _, kwargs_called = mock_review.call_args
        assert kwargs_called.get("review_skills") is True
        assert kwargs_called.get("review_memory") is True
        assert kwargs_called.get("messages_snapshot") == messages

    @pytest.mark.asyncio
    async def test_passes_client_and_tools_from_agent_loop(self):
        mock_client = MagicMock()
        mock_tools = ["t1"]
        agent_loop = MagicMock()
        agent_loop._client = mock_client
        agent_loop._tools = mock_tools
        config = MagicMock()
        messages = [{"role": "user", "content": "hi"}]

        async def _fake_spawn(*args, **kwargs):
            return "ok"

        with patch(_PATCH_REVIEW, side_effect=_fake_spawn) as mock_review:
            from src.services.daily_consolidation import _consolidate_session

            await _consolidate_session(agent_loop=agent_loop, config=config, messages=messages)

        kwargs_called = mock_review.call_args[1]
        assert kwargs_called["client"] is mock_client
        assert kwargs_called["tools"] is mock_tools

    @pytest.mark.asyncio
    async def test_returns_summary_string(self):
        agent_loop = MagicMock()
        agent_loop._client = MagicMock()
        agent_loop._tools = []
        config = MagicMock()
        messages = [{"role": "user", "content": "test"}]

        async def _fake_spawn(*args, **kwargs):
            return "2 memories saved, 1 skill updated"

        with patch(_PATCH_REVIEW, side_effect=_fake_spawn):
            from src.services.daily_consolidation import _consolidate_session

            result = await _consolidate_session(agent_loop=agent_loop, config=config, messages=messages)

        assert result == "2 memories saved, 1 skill updated"

    @pytest.mark.asyncio
    async def test_propagates_spawn_exception(self):
        agent_loop = MagicMock()
        agent_loop._client = MagicMock()
        agent_loop._tools = []
        config = MagicMock()
        messages = []

        async def _failing_spawn(*args, **kwargs):
            raise RuntimeError("spawn failed")

        with patch(_PATCH_REVIEW, side_effect=_failing_spawn):
            from src.services.daily_consolidation import _consolidate_session

            with pytest.raises(RuntimeError, match="spawn failed"):
                await _consolidate_session(agent_loop=agent_loop, config=config, messages=messages)


# ─── Diary key uniqueness (regression: nightly overwrite) ──────────────────────


class TestDiaryKeyUniqueness:
    """Regression: nightly runs must not overwrite prior diary entries.

    day_counter resets every run; before the creation date was encoded in the
    key, every ~1-day-old session reused "1天的日记1" and the ON CONFLICT(key)
    DO UPDATE in save_memory clobbered the previous night's summary.
    """

    @pytest.mark.asyncio
    async def test_two_nights_do_not_overwrite(self, tmp_path):
        from collections import defaultdict

        from src.services.daily_consolidation import _save_session_summary
        from src.tools.memory_tools import MemoryStore

        store = MemoryStore(db_path=str(tmp_path / "mem.db"))
        await store.initialize()

        messages = [
            {"role": "user", "content": "帮我写一个排序算法"},
            {"role": "assistant", "content": "好的，这是快速排序"},
            {"role": "user", "content": "部署到生产环境"},
            {"role": "assistant", "content": "已部署，监控正常"},
        ]

        fake_resp = MagicMock()
        fake_resp.content = "用户讨论了排序和部署"
        config = MagicMock()

        now = time.time()
        # Two sessions both days_ago == 1 (within 0–48h), but 24h apart so they
        # land on different calendar dates — mimicking rotation creating a fresh
        # session each night.
        created_at_n1 = now - 36 * 3600  # 36h ago → int(1.5) == 1
        created_at_n2 = now - 12 * 3600  # 12h ago → max(1, int(0.5)) == 1

        with (
            patch("src.tools.memory_tools.get_memory_store", return_value=store),
            patch("src.agent.client.ChatClient") as mock_chat_cls,
        ):
            mock_client = MagicMock()
            mock_client.chat = AsyncMock(return_value=fake_resp)
            mock_client.close = AsyncMock()
            mock_chat_cls.return_value = mock_client

            # Each nightly run starts with a fresh day_counter.
            await _save_session_summary(config, created_at_n1, messages, defaultdict(int))
            await _save_session_summary(config, created_at_n2, messages, defaultdict(int))

        memories = await store.list_all(limit=100)
        episodic = [m for m in memories if m.get("category") == "episodic"]
        keys = [m["key"] for m in episodic]

        assert len(episodic) == 2, f"Night 2 overwrote night 1 (expected 2 entries, got {len(episodic)}): {keys}"
        assert len(set(keys)) == 2, f"Diary keys collided: {keys}"

        await store.close()
