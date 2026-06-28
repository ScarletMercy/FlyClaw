"""群记忆活链路集成测试。

把这条链路串起来: set_memory_session(group) → agent loop 内部 task 调度
→ memory() 工具读 get_memory_scope() → 落到 GroupMemoryStore[正确 group_id]。

用脚本化假 LLM 触发 memory 工具调用,真实 tmp 双 store 落库后断言 DB 状态。
专测 ContextVar 接线与跨群隔离——单测各自 mock 上下游,抓不到这条链路。
关键回归信号: 群 scope 下 save 后 DM store 必须为空(ContextVar 若未穿透工具
会静默退回 DM scope,即 #5 latent 风险)。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.client import ChatResponse
from src.agent.loop import AgentLoop
from src.agent.state import AgentState, MemoryStateStore
from src.tools.memory_tools import (
    GroupMemoryStore,
    MemoryStore,
    get_tools,
    set_memory_session,
)

_PATCH_GET_STORE = "src.tools.memory_tools.get_memory_store"
_PATCH_SET_SESSION = "src.tools.memory_tools.set_memory_session"


# ─── helpers ──────────────────────────────────────────────────────────────────


def _make_tc(name: str, args: dict | None = None, tc_id: str | None = None):
    @dataclass
    class TC:
        id: str
        function: Any

    @dataclass
    class Fn:
        name: str
        arguments: str

    return TC(id=tc_id or f"call_{name}", function=Fn(name=name, arguments=json.dumps(args or {})))


def _make_config():
    config = MagicMock()
    config.agents = MagicMock()
    config.agents.max_tool_rounds = 50
    config.agents.tool_output_cache_chars = 8000
    config.agents.lock_timeout = 5.0
    config.agents.system_prompt = "You are helpful."
    config.agents.workspace = "."
    config.agents.bootstrap_files = []
    config.agents.timezone = "UTC"
    config.auth = None
    config.tools = MagicMock()
    config.tools.policy = MagicMock()
    config.tools.policy.allow = ["*"]
    config.tools.policy.deny = []
    config.tools.policy.owner_only = []
    config.tools.guardrails = MagicMock()
    config.tools.guardrails.enabled = False
    config.compression = MagicMock()
    config.compression.enabled = False
    return config


def _router(dm: MemoryStore, grp: GroupMemoryStore):
    """patch get_memory_store 的 side_effect: 按 chat_type 路由到 tmp 双 store。"""

    def _f(db_path=None, chat_type="p2p"):
        return grp if chat_type == "group" else dm

    return _f


async def _make_stores(tmp_path):
    dm = MemoryStore(str(tmp_path / "dm.db"))
    await dm.initialize()
    grp = GroupMemoryStore(str(tmp_path / "grp.db"))
    await grp.initialize()
    return dm, grp


def _loop_with_memory(client) -> AgentLoop:
    return AgentLoop(client=client, tools=get_tools(), state_store=MemoryStateStore(), config=_make_config())


def _state(text: str, chat_type: str, chat_id: str) -> AgentState:
    return AgentState(
        messages=[{"role": "user", "content": text}],
        chat_id=chat_id,
        chat_type=chat_type,
        sender_id="u1",
        channel="qq",
    )


@pytest.fixture(autouse=True)
def _reset_scope():
    """每个用例前后复位 memory scope,避免 ContextVar 跨用例泄漏。"""
    set_memory_session("p2p", "")
    yield
    set_memory_session("p2p", "")


# ─── A. 群 scope → memory(save) → GroupMemoryStore ────────────────────────────


@pytest.mark.asyncio
async def test_group_scope_routes_save_to_group_store(tmp_path):
    dm, grp = await _make_stores(tmp_path)
    try:
        client = AsyncMock()
        client.chat.side_effect = [
            ChatResponse(
                content="",
                tool_calls=[
                    _make_tc(
                        "memory",
                        {"action": "save", "content": "Alice 喜欢茶", "key": "alice_tea", "category": "preference"},
                    )
                ],
            ),
            ChatResponse(content="已记住", tool_calls=[]),
        ]
        loop = _loop_with_memory(client)
        set_memory_session("group", "G1")

        with patch(_PATCH_GET_STORE, side_effect=_router(dm, grp)):
            await loop.run(_state("记住我喜欢茶", "group", "G1"), "qq:group:G1")

        assert len(await grp.list_all(group_id="G1")) == 1
        # ContextVar 穿透失败的回归信号: 群 scope 的 save 不得落入 DM
        assert len(await dm.list_all()) == 0
    finally:
        await dm.close()
        await grp.close()


# ─── B. 跨群隔离 + ContextVar 穿透 agent 内部 create_task ──────────────────────


@pytest.mark.asyncio
async def test_cross_group_isolation_and_contextvar_propagation(tmp_path):
    dm, grp = await _make_stores(tmp_path)
    try:
        # Phase 1: G1 存一条
        c1 = AsyncMock()
        c1.chat.side_effect = [
            ChatResponse(
                content="",
                tool_calls=[
                    _make_tc(
                        "memory",
                        {"action": "save", "content": "Alice 喜欢茶", "key": "alice_tea", "category": "preference"},
                    )
                ],
            ),
            ChatResponse(content="ok", tool_calls=[]),
        ]
        loop1 = _loop_with_memory(c1)
        set_memory_session("group", "G1")
        with patch(_PATCH_GET_STORE, side_effect=_router(dm, grp)):
            await loop1.run(_state("记住", "group", "G1"), "qq:group:G1")

        # Phase 2: G2 试图 recall G1 的 key(经工具)
        c2 = AsyncMock()
        c2.chat.side_effect = [
            ChatResponse(content="", tool_calls=[_make_tc("memory", {"action": "get", "key": "alice_tea"})]),
            ChatResponse(content="查不到", tool_calls=[]),
        ]
        loop2 = _loop_with_memory(c2)
        set_memory_session("group", "G2")
        s2 = _state("Alice 喜欢什么", "group", "G2")
        with patch(_PATCH_GET_STORE, side_effect=_router(dm, grp)):
            await loop2.run(s2, "qq:group:G2")

        # 隔离: G1 有, G2 空, DM 空
        assert len(await grp.list_all(group_id="G1")) == 1
        assert len(await grp.list_all(group_id="G2")) == 0
        assert len(await dm.list_all()) == 0
        # 工具层隔离: G2 scope 下 get G1 的 key 必须返回 not found
        tool_results = [m.get("content", "") for m in s2.messages if m.get("role") == "tool"]
        assert any("not found" in r for r in tool_results), tool_results
    finally:
        await dm.close()
        await grp.close()


# ─── C. p2p scope 仍走 DM store(回归) ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_p2p_scope_still_uses_dm_store(tmp_path):
    dm, grp = await _make_stores(tmp_path)
    try:
        client = AsyncMock()
        client.chat.side_effect = [
            ChatResponse(
                content="",
                tool_calls=[
                    _make_tc(
                        "memory",
                        {"action": "save", "content": "DM only fact", "key": "dm1", "category": "fact"},
                    )
                ],
            ),
            ChatResponse(content="ok", tool_calls=[]),
        ]
        loop = _loop_with_memory(client)
        set_memory_session("p2p", "")

        with patch(_PATCH_GET_STORE, side_effect=_router(dm, grp)):
            await loop.run(_state("记住", "p2p", "u1"), "qq:user:u1")

        assert len(await dm.list_all()) == 1
        assert len(await grp.list_all(group_id=None)) == 0
    finally:
        await dm.close()
        await grp.close()


# ─── F. daily_consolidation 按群线程设 scope ──────────────────────────────────


def _consolidation_msgs(n_user=8, n_assistant=4):
    return [{"role": "user", "content": f"u{i}"} for i in range(n_user)] + [
        {"role": "assistant", "content": f"a{i}"} for i in range(n_assistant)
    ]


@pytest.mark.asyncio
async def test_daily_consolidation_sets_group_scope_per_thread():
    """整理遍历群线程时必须按 (chat_type, chat_id) 设 memory scope。"""
    from src.services.daily_consolidation import run_daily_consolidation

    store = MemoryStateStore()
    now = time.time()
    await store.save(
        "qq:group:G1",
        AgentState(messages=_consolidation_msgs(), chat_id="G1", chat_type="group", channel="qq", created_at=now),
    )
    await store.save(
        "qq:user:u1",
        AgentState(messages=_consolidation_msgs(), chat_id="u1", chat_type="p2p", channel="qq", created_at=now),
    )

    agent_loop = MagicMock()
    agent_loop.invalidate_memory_cache = MagicMock()
    container = MagicMock()
    container.state_store = store
    container.agent_loop = agent_loop
    container.session_registry = None
    container.qq = None
    container.weixin = None
    container.config.consolidation.min_messages = 10

    with (
        patch("src.services.daily_consolidation._consolidate_session", return_value="summary"),
        patch("src.services.daily_consolidation._send_notification", new_callable=AsyncMock),
        patch("src.services.daily_consolidation._save_session_summary", new_callable=AsyncMock),
        patch(_PATCH_SET_SESSION) as spy,
    ):
        await run_daily_consolidation(container)

    calls = [(c.args[0], c.args[1]) for c in spy.call_args_list]
    assert ("group", "G1") in calls, calls
    assert ("p2p", "") in calls, calls


# ─── G. 审批 resume 删记忆: scope 从 state 推导, 只删目标群 ────────────────────


@pytest.mark.asyncio
async def test_group_memory_delete_approval_targets_correct_group(tmp_path):
    """审批 resume 删记忆时, scope 从保存的 state(chat_type/chat_id) 推导,
    只删目标群, 不碰其他群/DM。覆盖 _resume_inner 的 GroupMemoryStore 分支——
    该分支此前无任何测试执行(scope 来源与工具执行路径不同:从 state 而非 ContextVar)。
    """
    dm, grp = await _make_stores(tmp_path)
    try:
        # 同 key 三处各存一份: G1 / G2 / DM
        await grp.remember("Alice G1", key="alice_tea", category="preference", group_id="G1")
        await grp.remember("Alice G2", key="alice_tea", category="preference", group_id="G2")
        await dm.remember("Alice DM", key="alice_tea", category="preference")

        state_store = MemoryStateStore()
        state = AgentState(
            messages=[
                {"role": "user", "content": "删掉 alice 的记忆"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "tc1",
                            "type": "function",
                            "function": {"name": "memory_delete", "arguments": json.dumps({"keys": ["alice_tea"]})},
                        }
                    ],
                },
            ],
            chat_id="G1",
            chat_type="group",
            sender_id="u1",
            channel="qq",
            pending_approval={
                "request_id": "r1",
                "tool_call_id": "tc1",
                "tool_name": "memory_delete",
                "command_preview": "- [alice_tea]: Alice G1",
                "memory_keys": ["alice_tea"],
            },
        )
        await state_store.save("qq:group:G1", state)

        client = AsyncMock()
        client.chat.return_value = ChatResponse(content="已删除", tool_calls=[])
        loop = AgentLoop(client=client, tools=get_tools(), state_store=state_store, config=_make_config())

        with patch(_PATCH_GET_STORE, side_effect=_router(dm, grp)):
            await loop.resume("qq:group:G1", "allow_once")

        # 只删 G1
        assert len(await grp.list_all(group_id="G1")) == 0
        # G2 不受影响
        g2 = await grp.list_all(group_id="G2")
        assert len(g2) == 1 and g2[0]["content"] == "Alice G2", g2
        # DM 不受影响
        assert len(await dm.list_all()) == 1
    finally:
        await dm.close()
        await grp.close()


# ─── H. 凌晨整理: DM 线程与群线程的存档隔离 ────────────────────────────────────


@pytest.mark.asyncio
async def test_daily_consolidation_isolates_dm_and_group_saves(tmp_path):
    """凌晨整理: DM 线程抽取的日记落 DM store, 群线程落对应群 store, 不串。

    跑真 _save_session_summary(只 mock ChatClient 文本), 验证 per-thread
    set_memory_session 确实让存档 save_memory 落到对的 store。
    """
    from src.services.daily_consolidation import run_daily_consolidation

    dm, grp = await _make_stores(tmp_path)
    sstore = MemoryStateStore()
    now = time.time()
    dm_msgs = [{"role": "user", "content": f"u{i}"} for i in range(5)] + [
        {"role": "assistant", "content": f"a{i}"} for i in range(5)
    ]
    grp_msgs = [{"role": "user", "content": f"gu{i}"} for i in range(5)] + [
        {"role": "assistant", "content": f"ga{i}"} for i in range(5)
    ]
    await sstore.save(
        "qq:user:u1", AgentState(messages=dm_msgs, chat_id="u1", chat_type="p2p", channel="qq", created_at=now)
    )
    await sstore.save(
        "qq:group:G1", AgentState(messages=grp_msgs, chat_id="G1", chat_type="group", channel="qq", created_at=now)
    )

    agent_loop = MagicMock()
    agent_loop.invalidate_memory_cache = MagicMock()
    container = MagicMock()
    container.state_store = sstore
    container.agent_loop = agent_loop
    container.session_registry = None
    container.qq = None
    container.weixin = None
    container.config.consolidation.min_messages = 10

    fake_client = AsyncMock()
    fake_client.chat.return_value = ChatResponse(content="日记摘要")
    fake_client.close = AsyncMock()

    try:
        with (
            patch("src.services.daily_consolidation._consolidate_session", new_callable=AsyncMock, return_value=""),
            patch("src.services.daily_consolidation._send_notification", new_callable=AsyncMock),
            patch(_PATCH_GET_STORE, side_effect=_router(dm, grp)),
            patch("src.agent.client.ChatClient", return_value=fake_client),
        ):
            await run_daily_consolidation(container)

        dm_items = await dm.list_all()
        g1_items = await grp.list_all(group_id="G1")
        # DM 线程日记落 DM, 群线程日记落 G1
        assert len(dm_items) == 1, dm_items
        assert len(g1_items) == 1, g1_items
        assert dm_items[0]["category"] == "episodic"
        assert g1_items[0]["category"] == "episodic"
        # 隔离: 群 store 全量(含所有群)只有 G1 那 1 条, 不含 DM 的
        assert len(await grp.list_all(group_id=None)) == 1
    finally:
        await dm.close()
        await grp.close()
