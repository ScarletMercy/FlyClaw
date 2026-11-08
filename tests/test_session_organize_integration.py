"""会话整理全链路集成测试：真 SQLite + 真状态文件 + 真编排，仅 mock LLM 边界。

单元测试各自验证 SQL/门槛/推进语义；本文件把
run_session_organize -> run_daily_consolidation -> 老化 -> list_pending_threads SQL
-> 内容门槛 -> mark_organized/死信 -> 状态文件落盘
串成闭环，跨多轮运行验证扫描语义：老会话补漏、零重复处理、失败重试与死信、
低门槛死线程老化与复活、游标推进/不推进。
"""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.state import AgentState, StateStore

_PATCH_CONSOLIDATE = "src.services.daily_consolidation._consolidate_session"
_PATCH_NOTIFY = "src.services.daily_consolidation._send_notification"
_PATCH_SAVE_SUMMARY = "src.services.daily_consolidation._save_session_summary"
_PATCH_REACHABLE = "src.services.daily_consolidation._llm_reachable"
_PATCH_DATA_DIR = "src.instance.data_dir"


def _msgs(tag: str, n_user=8, n_assistant=4) -> list[dict]:
    """过门槛的消息：>=10 条总数且 >=3 user。tag 前缀用于区分会话/触发失败。"""
    return [{"role": "user", "content": f"{tag} u{i}"} for i in range(n_user)] + [
        {"role": "assistant", "content": f"{tag} a{i}"} for i in range(n_assistant)
    ]


def _make_container(store, min_messages=10, registry=None):
    container = MagicMock()
    container.state_store = store
    agent_loop = MagicMock()
    agent_loop.invalidate_memory_cache = MagicMock()
    container.agent_loop = agent_loop
    container.session_registry = registry
    container.qq = None
    container.weixin = None
    container.config.consolidation.min_messages = min_messages
    container.config.consolidation.stale_idle_days = 30
    container.config.consolidation.max_attempts = 3
    return container


async def _insert_session(store, thread_id, messages, created_at, chat_id="c1", updated_at=None):
    """原生 INSERT 指定 created_at/updated_at（save() 恒写 now，造不出老会话/死线程）。"""
    state = AgentState(messages=messages, chat_id=chat_id, channel="qq")
    conn = await store._get_conn()
    await conn.execute(
        "INSERT INTO sessions (thread_id, messages, metadata, frozen_system_prompt, created_at, updated_at) "
        "VALUES (?, ?, ?, '', ?, ?)",
        (
            thread_id,
            json.dumps(state.messages, ensure_ascii=False),
            json.dumps(state.meta_dict(), ensure_ascii=False),
            created_at,
            updated_at if updated_at is not None else created_at,
        ),
    )
    await conn.commit()


async def _run_organize(container, tmp_path, consolidate=None, notify_spy=None, reachable=False):
    """跑真 run_session_organize（含锁与状态推进），data_dir 指向 tmp_path。"""
    from src.services.consolidation_state import run_session_organize

    async def _notify(_container, _channel, chat_id, text):
        if notify_spy is not None:
            notify_spy.append((chat_id, text))

    with (
        patch(_PATCH_CONSOLIDATE, consolidate or AsyncMock(return_value="summary")),
        patch(_PATCH_NOTIFY, _notify),
        patch(_PATCH_SAVE_SUMMARY, AsyncMock()),
        patch(_PATCH_REACHABLE, AsyncMock(return_value=reachable)),
        patch(_PATCH_DATA_DIR, return_value=tmp_path),
    ):
        return await run_session_organize(container)


async def _load_state(tmp_path):
    from src.services.consolidation_state import load_state

    with patch(_PATCH_DATA_DIR, return_value=tmp_path):
        return await load_state()


class TestSessionOrganizeFullLoop:
    @pytest.mark.asyncio
    async def test_first_run_catches_up_old_sessions(self, tmp_path):
        """首跑全量补漏：40 天前的老会话照样被整理打标，dreaming 按 created_at 正序，
        为每个 legacy thread 开新会话，游标从 None 推进落盘。"""
        store = StateStore(str(tmp_path / "s.db"))
        old_ts = time.time() - 40 * 86400
        await _insert_session(store, "old", _msgs("old"), old_ts, chat_id="chat_old")
        await _insert_session(store, "new", _msgs("new"), time.time(), chat_id="chat_new")

        reg = MagicMock()
        reg.new_session = AsyncMock(return_value="s_new")
        container = _make_container(store, registry=reg)
        notify_spy: list[tuple[str, str]] = []

        result = await _run_organize(container, tmp_path, notify_spy=notify_spy)

        assert result["sessions_processed"] == 2
        assert result["errors"] == []
        assert await store.list_pending_threads() == []
        dreaming_order = [chat for chat, text in notify_spy if "dreaming" in text]
        assert dreaming_order == ["chat_old", "chat_new"]
        assert reg.new_session.await_count == 2
        st = await _load_state(tmp_path)
        assert st.last_session_organize_at is not None
        await store.close()

    @pytest.mark.asyncio
    async def test_second_run_empty_scan(self, tmp_path):
        """全部整理后第二轮空扫：0 处理 0 跳过 0 错误，游标再次推进。"""
        store = StateStore(str(tmp_path / "s.db"))
        await _insert_session(store, "t1", _msgs("a"), time.time())
        container = _make_container(store)

        r1 = await _run_organize(container, tmp_path)
        assert r1["sessions_processed"] == 1
        last1 = (await _load_state(tmp_path)).last_session_organize_at

        r2 = await _run_organize(container, tmp_path)
        assert r2["sessions_processed"] == 0
        assert r2["sessions_skipped"] == 0
        assert r2["errors"] == []
        last2 = (await _load_state(tmp_path)).last_session_organize_at
        assert last2 >= last1
        await store.close()

    @pytest.mark.asyncio
    async def test_below_threshold_idle_session_aged_then_revived(self, tmp_path):
        """低门槛且 30 天无活动的死线程被老化进终态；用户回归攒够门槛后复活并被整理。"""
        store = StateStore(str(tmp_path / "s.db"))
        tiny = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        await _insert_session(store, "tiny", tiny, time.time() - 40 * 86400)
        container = _make_container(store)

        r1 = await _run_organize(container, tmp_path)
        assert r1["sessions_aged"] == 1
        assert r1["sessions_skipped"] == 0  # 老化出局，不再进扫描
        assert (await store.load("tiny")).status == 2

        r2 = await _run_organize(container, tmp_path)
        assert r2["sessions_aged"] == 0 and r2["sessions_skipped"] == 0  # 终态不重扫

        state = await store.load("tiny")
        state.messages.extend(_msgs("more", n_user=5, n_assistant=5))
        await store.save("tiny", state)

        r3 = await _run_organize(container, tmp_path)
        assert r3["sessions_revived"] == 1
        assert r3["sessions_processed"] == 1
        assert (await store.load("tiny")).status == 1
        await store.close()

    @pytest.mark.asyncio
    async def test_below_threshold_active_session_not_aged(self, tmp_path):
        """低门槛但近期活跃的会话不老化——仍有机会攒够门槛。"""
        store = StateStore(str(tmp_path / "s.db"))
        tiny = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        await _insert_session(store, "tiny", tiny, time.time() - 40 * 86400, updated_at=time.time())
        container = _make_container(store)

        r1 = await _run_organize(container, tmp_path)
        assert r1["sessions_aged"] == 0
        assert r1["sessions_skipped"] == 1  # 门槛不够被跳过，但留在待处理
        assert await store.list_pending_threads() == ["tiny"]
        await store.close()

    @pytest.mark.asyncio
    async def test_chronic_failure_dead_letters_after_max_attempts(self, tmp_path):
        """连续失败达到 max_attempts 进死信：不再重试、不再发 dreaming。
        计数只在有成功证人的轮次生效；零成功的一轮（全局故障）不烧预算。"""
        store = StateStore(str(tmp_path / "s.db"))
        await _insert_session(store, "bad", _msgs("x"), time.time(), chat_id="chat_bad")
        container = _make_container(store)

        async def _fail_bad(agent_loop, config, messages):
            if messages[0]["content"].startswith("x"):
                raise RuntimeError("poison")
            return "summary"

        for night in (1, 2):
            await _insert_session(store, f"ok{night}", _msgs(f"ok{night}"), time.time(), chat_id="chat_ok")
            r = await _run_organize(container, tmp_path, consolidate=_fail_bad)
            assert r["sessions_dead_lettered"] == 0
            assert (await store.load("bad")).attempts == night

        async def _fail_all(agent_loop, config, messages):
            raise RuntimeError("outage")

        await _insert_session(store, "ok3", _msgs("ok3"), time.time(), chat_id="chat_ok")
        r = await _run_organize(container, tmp_path, consolidate=_fail_all)
        assert r["sessions_dead_lettered"] == 0
        assert (await store.load("bad")).attempts == 2
        assert (await store.load("ok3")).attempts == 0

        r4 = await _run_organize(container, tmp_path, consolidate=_fail_bad)
        assert r4["sessions_dead_lettered"] == 1
        assert (await store.load("bad")).status == 3

        notify_spy: list[tuple[str, str]] = []
        ok_mock = AsyncMock(return_value="summary")
        r5 = await _run_organize(container, tmp_path, consolidate=ok_mock, notify_spy=notify_spy)
        assert r5["sessions_processed"] == 0
        assert r5["sessions_skipped"] == 0
        assert notify_spy == []
        assert ok_mock.await_count == 0
        await store.close()

    @pytest.mark.asyncio
    async def test_lone_poison_dead_letters_via_probe(self, tmp_path):
        """孤立毒会话：零成功但探测成功 → 照常计数三振；探测失败的晚不烧预算。"""
        store = StateStore(str(tmp_path / "s.db"))
        await _insert_session(store, "bad", _msgs("x"), time.time(), chat_id="chat_bad")
        container = _make_container(store)

        async def _fail(agent_loop, config, messages):
            raise RuntimeError("poison")

        r = await _run_organize(container, tmp_path, consolidate=_fail, reachable=False)
        assert r["sessions_dead_lettered"] == 0
        assert (await store.load("bad")).attempts == 0

        for night in (1, 2, 3):
            r = await _run_organize(container, tmp_path, consolidate=_fail, reachable=True)
            assert (await store.load("bad")).attempts == night
        assert r["sessions_dead_lettered"] == 1
        assert (await store.load("bad")).status == 3

        notify_spy: list[tuple[str, str]] = []
        ok_mock = AsyncMock(return_value="summary")
        r = await _run_organize(container, tmp_path, consolidate=ok_mock, notify_spy=notify_spy, reachable=True)
        assert r["sessions_processed"] == 0
        assert notify_spy == []
        assert ok_mock.await_count == 0
        await store.close()

    @pytest.mark.asyncio
    async def test_dead_session_revives_on_new_activity(self, tmp_path):
        """死信会话有新活动 → save 重开为待处理，下轮照常整理。"""
        store = StateStore(str(tmp_path / "s.db"))
        await _insert_session(store, "bad", _msgs("x"), time.time())
        container = _make_container(store)

        for _ in range(3):
            await store.record_failed_attempt("bad", 3)
        assert (await store.load("bad")).status == 3
        assert await store.list_pending_threads() == []

        state = await store.load("bad")
        state.messages.extend(_msgs("more", n_user=2, n_assistant=2))
        await store.save("bad", state)
        assert await store.list_pending_threads() == ["bad"]

        r = await _run_organize(container, tmp_path)
        assert r["sessions_processed"] == 1
        assert (await store.load("bad")).status == 1
        await store.close()

    @pytest.mark.asyncio
    async def test_failure_no_advance_zero_reprocess(self, tmp_path):
        """部分失败：游标不推进、失败会话不打标下轮重试、成功会话零重复处理。"""
        from src.services.consolidation_state import ConsolidationState, save_state

        with patch(_PATCH_DATA_DIR, return_value=tmp_path):
            await save_state(ConsolidationState(last_session_organize_at=1000.0))

        store = StateStore(str(tmp_path / "s.db"))
        await _insert_session(store, "ok", _msgs("OK"), time.time() - 7200)
        await _insert_session(store, "bad", _msgs("FAIL"), time.time() - 3600)

        async def _flaky(agent_loop, config, messages):
            if messages[0]["content"].startswith("FAIL"):
                raise RuntimeError("boom")
            return "summary"

        container = _make_container(store)
        r1 = await _run_organize(container, tmp_path, consolidate=_flaky)
        assert r1["sessions_processed"] == 1
        assert len(r1["errors"]) == 1 and "bad" in r1["errors"][0]
        assert await store.list_pending_threads() == ["bad"]
        assert (await _load_state(tmp_path)).last_session_organize_at == 1000.0

        ok_mock = AsyncMock(return_value="summary")
        r2 = await _run_organize(container, tmp_path, consolidate=ok_mock)
        assert r2["sessions_processed"] == 1
        assert ok_mock.await_count == 1
        assert await store.list_pending_threads() == []
        assert (await _load_state(tmp_path)).last_session_organize_at > 1000.0
        await store.close()

    @pytest.mark.asyncio
    async def test_mark_organized_failure_records_error(self, tmp_path):
        """mark_organized 落库失败：计入 errors、游标不推进、下轮重扫该会话。"""
        store = StateStore(str(tmp_path / "s.db"))
        await _insert_session(store, "t1", _msgs("a"), time.time())
        container = _make_container(store)

        with patch.object(store, "mark_organized", new=AsyncMock(side_effect=RuntimeError("db locked"))):
            r1 = await _run_organize(container, tmp_path)
        assert r1["sessions_processed"] == 1
        assert any("mark_organized" in e for e in r1["errors"])
        assert (await _load_state(tmp_path)).last_session_organize_at is None

        r2 = await _run_organize(container, tmp_path)
        assert r2["sessions_processed"] == 1  # 重扫重试，落标成功
        assert r2["errors"] == []
        assert await store.list_pending_threads() == []
        await store.close()

    @pytest.mark.asyncio
    async def test_empty_db_first_run_advances(self, tmp_path):
        """空库首跑：零处理零跳过零错误，游标照常推进（无错即推进）。"""
        store = StateStore(str(tmp_path / "s.db"))
        container = _make_container(store)

        result = await _run_organize(container, tmp_path)

        assert result["sessions_processed"] == 0
        assert result["sessions_skipped"] == 0
        assert result["errors"] == []
        assert (await _load_state(tmp_path)).last_session_organize_at is not None
        await store.close()
