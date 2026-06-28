"""SessionRegistry 恢复的私聊/群隔离（含 /new 多会话 key 带 scope）。"""

from __future__ import annotations

import pytest

from src.agent.state import AgentState, MemoryStateStore
from src.session.tracker import SessionRegistry


def _msg_state() -> AgentState:
    return AgentState(messages=[{"role": "user", "content": "hi"}])


async def _seed_store() -> MemoryStateStore:
    store = MemoryStateStore()
    for tid in (
        # 默认线程
        "qq:user:43792F51FB004A795033172075EE6ED4",
        "qq:user:26B061CB3F2D00FD2C41A4BD306616B2",
        "qq:group:G1",
        "qq:group:G2",
        # 新格式多会话（/new 产出，scope 在 seg2）
        "qq:dm:s1",
        "qq:group:G1:s1",
        # 旧格式 legacy 多会话（向后兼容）
        "qq:s1:43792F51FB004A795033172075EE6ED4",
        "weixin:dm",
    ):
        await store.save(tid, _msg_state())
    return store


@pytest.mark.asyncio
async def test_new_session_thread_id_format():
    """new_session 产出的 thread_id 必须是 {user_key}:{sid}（sid 落末段，根治 sN 歧义）。"""
    reg = SessionRegistry()
    await reg.new_session("qq:dm")
    await reg.new_session("qq:group:G1")
    dm_sessions = reg.list_sessions("qq:dm")
    grp_sessions = reg.list_sessions("qq:group:G1")
    assert dm_sessions[0]["thread_id"] == "qq:dm:s1"
    assert grp_sessions[0]["thread_id"] == "qq:group:G1:s1"


@pytest.mark.asyncio
async def test_dm_recovery_excludes_group_default_and_multi():
    """DM /old 恢复不得含群默认线程(qq:group:G1)也不得含群多会话(qq:group:G1:s1)。"""
    store = await _seed_store()
    reg = SessionRegistry()
    recovered = await reg.find_all_channel_threads("qq:dm", store)
    assert set(recovered) == {
        "qq:user:43792F51FB004A795033172075EE6ED4",
        "qq:user:26B061CB3F2D00FD2C41A4BD306616B2",
        "qq:dm:s1",
        "qq:s1:43792F51FB004A795033172075EE6ED4",
    }


@pytest.mark.asyncio
async def test_group_recovery_excludes_dm_threads():
    """群 /old 恢复不得含任何私聊线程（默认/多会话/legacy）。"""
    store = await _seed_store()
    reg = SessionRegistry()
    recovered = await reg.find_all_channel_threads("qq:group:G1", store)
    assert set(recovered) == {"qq:group:G2", "qq:group:G1:s1"}


@pytest.mark.asyncio
async def test_dm_orphaned_short_circuits_then_fallback_recovers():
    store = await _seed_store()
    reg = SessionRegistry()
    assert await reg.find_orphaned_threads("qq:dm", store) == []
    assert await reg.find_all_channel_threads("qq:dm", store) != []


@pytest.mark.asyncio
async def test_orphan_scan_matches_hash_and_skips_registered():
    """3 段 key 走真正的扫描分支:按末段 hash 匹配,排除已注册/异渠道/异 hash/非 3 段。"""
    store = MemoryStateStore()
    for tid in (
        "qq:s1:ABC123",  # 命中:同渠道 + 末段 hash
        "qq:s2:ABC123",  # 命中
        "qq:s1:OTHER",  # 末段 hash 不匹配
        "weixin:s1:ABC123",  # 异渠道
        "qq:user:ABC123:s1",  # 4 段,不符合 3 段形状
    ):
        await store.save(tid, _msg_state())

    reg = SessionRegistry()
    await reg.recover_sessions("qq:user:ABC123", ["qq:s1:ABC123"])  # 注册一个 → 应被排除

    recovered = await reg.find_orphaned_threads("qq:user:ABC123", store)
    assert set(recovered) == {"qq:s2:ABC123"}


@pytest.mark.asyncio
async def test_orphan_scan_recovers_legacy_group_threads():
    """群 key(3 段)走扫描:用末段群 id 找回 legacy 群多会话,且不串群/不串私聊。"""
    store = MemoryStateStore()
    for tid in (
        "qq:s1:G1",  # 本群 legacy 多会话 → 命中
        "qq:s1:G2",  # 另一个群 → 不串
        "qq:s1:ABC123",  # 私聊 → 不串
    ):
        await store.save(tid, _msg_state())

    reg = SessionRegistry()
    recovered = await reg.find_orphaned_threads("qq:group:G1", store)
    assert set(recovered) == {"qq:s1:G1"}
