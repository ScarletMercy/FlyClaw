"""群记忆真实 LLM e2e: 群上下文召回质量 + 跨群不串扰。

需要 config.yaml 里的有效 API 凭据(model 段)。无 key 自动 skip。
与 test_group_memory_integration.py 互补: 那个用假 LLM 测接线,
这个用真 LongCat 测"群 scope 下记忆摘要注入 + 召回"是否真的工作。

Run: pytest tests/test_group_memory_e2e.py -v -s --timeout=200
"""

from __future__ import annotations

import io
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

# Windows GBK 控制台兜底——LLM 回复可能含中文/emoji
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from src.agent.client import ChatClient
from src.agent.loop import AgentLoop
from src.agent.state import AgentState, MemoryStateStore
from src.config import load_config
from src.tools.memory_tools import GroupMemoryStore, get_tools, set_memory_session

# ─── Load model config ───────────────────────────────────────────────────────

_config = load_config()
_API_KEY = _config.model.api_key or ""
_BASE_URL = _config.model.base_url or "https://api.openai.com/v1"
_MODEL = _config.model.name or "gpt-4o"

requires_api = pytest.mark.skipif(not _API_KEY, reason="No API key in config")


def _make_config():
    config = MagicMock()
    config.agents.max_tool_rounds = 10
    config.agents.tool_output_cache_chars = 8000
    config.agents.lock_timeout = 5.0
    config.agents.system_prompt = "你是一个助手。根据已知记忆如实回答;不知道就说不知道。"
    config.agents.workspace = "."
    config.agents.bootstrap_files = []
    config.agents.timezone = "Asia/Shanghai"
    config.auth = None
    config.tools.policy.allow = ["*"]
    config.tools.policy.deny = []
    config.tools.policy.owner_only = []
    config.tools.guardrails.enabled = False
    config.compression.enabled = False
    config.memory_store.enabled = True
    config.model.base_url = _BASE_URL
    config.model.api_key = _API_KEY
    config.model.name = _MODEL
    return config


def _router(group_store: GroupMemoryStore):
    def _f(db_path=None, chat_type="p2p"):
        return group_store

    return _f


async def _run_group_question(client: ChatClient, group_store: GroupMemoryStore, gid: str, question: str) -> str:
    """在 group scope 下跑一轮,返回助手回复。"""
    state_store = MemoryStateStore()
    loop = AgentLoop(client=client, tools=get_tools(), state_store=state_store, config=_make_config())
    state = AgentState(
        messages=[{"role": "user", "content": question}],
        chat_id=gid,
        chat_type="group",
        sender_id="u1",
        channel="qq",
    )
    set_memory_session("group", gid)
    with patch("src.tools.memory_tools.get_memory_store", side_effect=_router(group_store)):
        result = await loop.run(state, f"grp_e2e_{int(time.time())}")
    return result.messages[-1].get("content", "")


@requires_api
class TestGroupScopedRecall:
    """群 scope 下,seed 的群事实应被召回(经记忆摘要注入)。"""

    @pytest.mark.asyncio
    async def test_group_scoped_recall_accuracy(self, tmp_path):
        grp = GroupMemoryStore(str(tmp_path / "grp.db"))
        await grp.initialize()
        client = ChatClient(base_url=_BASE_URL, api_key=_API_KEY, model=_MODEL, temperature=0.0)
        try:
            # 不可猜的鲜明 token: 答对 = 必然读过群记忆(LLM 不可能凭空产出这些)
            facts = [
                ("grp_codename", "这个群的机密代号是 PHOENIX-ZX7", "fact"),
                ("grp_passphrase", "管理员设定的入群口令是 NEON-DELTA-42", "fact"),
            ]
            for k, content, cat in facts:
                await grp.remember(content, key=k, category=cat, group_id="G1")

            questions = [
                ("这个群的机密代号是什么?", "phoenix-zx7"),  # 小写比较, 容忍大小写
                ("管理员设定的入群口令是什么?", "neon-delta-42"),
            ]
            results = []
            for q, expect in questions:
                resp = (await _run_group_question(client, grp, "G1", q)).lower()
                results.append((expect, expect in resp))

            correct = sum(1 for _, ok in results if ok)
            # 严阈值: 不可猜 token, 全对才算通过——答对一个就证明群摘要注入工作
            assert correct == len(questions), f"群召回未全中: {correct}/{len(questions)}, results={results}"
        finally:
            await client.close()
            await grp.close()


@requires_api
class TestGroupIsolation:
    """跨群不串扰(双向): G2 能召回自己的, 且召不到 G1 的。"""

    @pytest.mark.asyncio
    async def test_no_cross_group_bleed(self, tmp_path):
        grp = GroupMemoryStore(str(tmp_path / "grp.db"))
        await grp.initialize()
        client = ChatClient(base_url=_BASE_URL, api_key=_API_KEY, model=_MODEL, temperature=0.0)
        try:
            await grp.remember("Alice 的安全码是 ALPHA-7891", key="alice_code", category="contact", group_id="G1")
            await grp.remember("Bob 的安全码是 BETA-3344", key="bob_code", category="contact", group_id="G2")

            # 正向: G2 能召回自己的 Bob —— 证明 G2 scope 真加载了记忆(不是空/失效)
            resp_bob = (await _run_group_question(client, grp, "G2", "Bob 的安全码是多少?")).lower()
            assert "beta-3344" in resp_bob, f"G2 应能召回自己的 Bob: {resp_bob!r}"

            # 反向: G2 召不到 G1 的 Alice —— 证明隔离, 未串入 G1
            resp_alice = (await _run_group_question(client, grp, "G2", "Alice 的安全码是多少?")).upper()
            assert "ALPHA-7891" not in resp_alice, f"跨群串扰: G2 不该知道 G1 的 Alice: {resp_alice!r}"
        finally:
            await client.close()
            await grp.close()
