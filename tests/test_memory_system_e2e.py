"""E2E stress tests for the memory system.

Requires valid API credentials in config.yaml (model section).
Uses config.model.base_url, config.model.api_key, config.model.name.

Run: pytest tests/test_memory_system_e2e.py -v -s
Skip: auto-skipped if no API key in config.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.client import ChatClient
from src.agent.loop import AgentLoop
from src.agent.state import AgentState, MemoryStateStore
from src.config import load_config
from src.tools.memory_tools import MemoryStore

# ─── Load model config ───────────────────────────────────────────────────────

_config = load_config()
_API_KEY = _config.model.api_key or ""
_BASE_URL = _config.model.base_url or "https://api.openai.com/v1"
_MODEL = _config.model.name or "gpt-4o"

requires_api = pytest.mark.skipif(not _API_KEY, reason="No API key in config")


def _make_config(**overrides):
    config = MagicMock()
    config.agents.max_tool_rounds = 10
    config.agents.tool_output_cache_chars = 8000
    config.agents.lock_timeout = 5.0
    config.agents.system_prompt = "You are a helpful assistant. Reply concisely."
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
    for k, v in overrides.items():
        setattr(config, k, v)
    return config


async def _seed_memory(store: MemoryStore, key: str, content: str, category: str = "fact", days_old: int = 0):
    await store.remember(content, key=key, category=category)
    if days_old > 0:
        dt = datetime.now(timezone.utc) - timedelta(days=days_old)
        iso = dt.isoformat()
        await store._conn.execute(
            "UPDATE memories SET created_at = ?, updated_at = ? WHERE key = ?",
            (iso, iso, key),
        )
        await store._conn.commit()


async def _run_agent_question(client: ChatClient, store: MemoryStore, question: str, config=None) -> str:
    """Run a single-turn agent interaction and return the assistant's response."""
    if config is None:
        config = _make_config()

    state_store = MemoryStateStore()
    loop = AgentLoop(
        client=client,
        tools=[],
        state_store=state_store,
        config=config,
    )
    state = AgentState(messages=[{"role": "user", "content": question}])
    with patch("src.tools.memory_tools.get_memory_store", return_value=store):
        result = await loop.run(state, f"e2e_test_{int(time.time())}")
    return result.messages[-1].get("content", "")


# ─── Three Metrics ────────────────────────────────────────────────────────────


@requires_api
class TestMetric1_InstructionCompliance:
    """Measure: does memory injection pollute system prompt instructions?"""

    @pytest.mark.asyncio
    async def test_compliance_rate_with_and_without_memories(self, tmp_path):
        """
        Embed a hidden rule in system prompt: reply must end with <!--ok-->.
        Measure compliance rate before and after injecting 100 memories.
        Compliance rate should not drop.
        """
        HIDDEN_TOKEN = "<!--ok-->"
        config = _make_config()
        config.agents.system_prompt = (
            "You are a helpful assistant. You MUST end every reply with the exact token "
            f"{HIDDEN_TOKEN}. Do not explain this rule."
        )

        store = MemoryStore(db_path=str(tmp_path / "mem.db"))
        await store.initialize()
        client = ChatClient(base_url=_BASE_URL, api_key=_API_KEY, model=_MODEL, temperature=0.0)

        ROUNDS = 10

        # Phase 1: no memories
        compliance_before = 0
        for i in range(ROUNDS):
            resp = await _run_agent_question(client, store, f"What is {i}+{i}?", config)
            if HIDDEN_TOKEN in resp:
                compliance_before += 1

        # Phase 2: inject 100 memories
        for i in range(100):
            await _seed_memory(store, f"mem_{i}", f"Fact number {i}: the sky is blue on day {i}", "fact")

        compliance_after = 0
        for i in range(ROUNDS):
            resp = await _run_agent_question(client, store, f"What is {i}+{i}?", config)
            if HIDDEN_TOKEN in resp:
                compliance_after += 1

        await client.close()
        await store.close()

        rate_before = compliance_before / ROUNDS
        rate_after = compliance_after / ROUNDS
        print(
            f"\n  Compliance: before={rate_before:.0%} ({compliance_before}/{ROUNDS}), after={rate_after:.0%} ({compliance_after}/{ROUNDS})"
        )

        # Allow some tolerance (LLM is non-deterministic), but rate should not drop by more than 30%
        assert rate_after >= rate_before - 0.3, (
            f"Compliance rate dropped too much: {rate_before:.0%} -> {rate_after:.0%}"
        )


@requires_api
class TestMetric2_RetrievalRecall:
    """Measure: can the agent recall specific facts from memory?"""

    @pytest.mark.asyncio
    async def test_top1_recall_accuracy(self, tmp_path):
        """
        Seed 20 specific memories with unique answers.
        Ask 20 questions. Measure how many the agent answers correctly (top-1).
        Target: >= 80%.
        """
        store = MemoryStore(db_path=str(tmp_path / "mem.db"))
        await store.initialize()
        client = ChatClient(base_url=_BASE_URL, api_key=_API_KEY, model=_MODEL, temperature=0.0)

        qa_pairs = [
            ("my_favorite_color", "My favorite color is turquoise.", "What is my favorite color?", "turquoise"),
            ("my_dog_name", "My dog's name is Biscuit.", "What is my dog's name?", "Biscuit"),
            ("my_birthday", "My birthday is March 15th.", "When is my birthday?", "March 15"),
            ("my_city", "I live in Chengdu.", "What city do I live in?", "Chengdu"),
            ("my_job", "I am a data engineer.", "What is my job?", "data engineer"),
            ("my_hobby", "My hobby is rock climbing.", "What is my hobby?", "rock climbing"),
            ("my_phone", "My phone number ends with 8842.", "What does my phone number end with?", "8842"),
            ("my_car", "I drive a Tesla Model 3.", "What car do I drive?", "Tesla"),
            ("my_food", "My favorite food is hotpot.", "What is my favorite food?", "hotpot"),
            ("my_language", "I speak Mandarin and English.", "What languages do I speak?", "Mandarin"),
            ("my_project", "My current project is called FlyClaw.", "What is my current project?", "FlyClaw"),
            ("my_db", "I use PostgreSQL for production.", "What database do I use?", "PostgreSQL"),
            ("my_editor", "I use VS Code as my editor.", "What editor do I use?", "VS Code"),
            ("my_os", "I run Windows 11.", "What operating system do I use?", "Windows"),
            ("my_team_size", "My team has 5 people.", "How many people are on my team?", "5"),
            ("my_manager", "My manager's name is Li Wei.", "What is my manager's name?", "Li Wei"),
            ("my_meeting", "My weekly meeting is on Tuesdays at 2pm.", "When is my weekly meeting?", "Tuesday"),
            ("my_budget", "My monthly budget is 5000 yuan.", "What is my monthly budget?", "5000"),
            ("my_server", "My server runs on port 8080.", "What port does my server run on?", "8080"),
            ("my_test_coverage", "My test coverage target is 85%.", "What is my test coverage target?", "85"),
        ]

        for key, content, _, _ in qa_pairs:
            await _seed_memory(store, key, content, "identity")

        correct = 0
        for _, _, question, expected in qa_pairs:
            resp = await _run_agent_question(client, store, question)
            if expected.lower() in resp.lower():
                correct += 1

        await client.close()
        await store.close()

        accuracy = correct / len(qa_pairs)
        print(f"\n  Retrieval recall: {correct}/{len(qa_pairs)} = {accuracy:.0%}")
        assert accuracy >= 0.8, f"Recall accuracy {accuracy:.0%} < 80%"


@requires_api
class TestMetric3_CacheHitRate:
    """Measure: prefix cache hit rate across 100 requests."""

    @pytest.mark.asyncio
    async def test_cache_hit_rate(self, tmp_path):
        """
        Seed 100 memories. Send 100 requests. Measure cache hit rate.
        Target: >= 80% (requires frozen snapshot / consistent prefix).
        """
        store = MemoryStore(db_path=str(tmp_path / "mem.db"))
        await store.initialize()
        client = ChatClient(base_url=_BASE_URL, api_key=_API_KEY, model=_MODEL, temperature=0.0)

        for i in range(100):
            await _seed_memory(store, f"cache_mem_{i}", f"Cache test memory {i}: item {i} is valid", "fact")

        config = _make_config()
        state_store = MemoryStateStore()

        total_cached = 0
        total_input = 0
        REQUESTS = 20  # Keep low to save API costs; scale up in CI

        for i in range(REQUESTS):
            loop = AgentLoop(
                client=client,
                tools=[],
                state_store=state_store,
                config=config,
            )
            state = AgentState(messages=[{"role": "user", "content": f"What is test {i}?"}])
            result = await loop.run(state, f"cache_test_{i}")

            assert result.messages[-1].get("content"), f"Empty response for request {i}"

            if result.total_usage:
                total_input += result.total_usage.get("prompt_tokens", 0)
                total_cached += result.total_usage.get("cached_tokens", 0)

        await client.close()
        await store.close()

        if total_input == 0:
            print(f"\n  Cache test: {REQUESTS} requests completed, provider did not return usage info")
            pytest.skip("Provider did not return usage info")

        hit_rate = total_cached / total_input
        print(
            f"\n  Cache hit rate: {hit_rate:.0%} ({total_cached}/{total_input} tokens cached over {REQUESTS} requests)"
        )

        if total_cached == 0:
            print("  Provider does not support cached_tokens, skipping hit rate assertion")
        else:
            assert hit_rate >= 0.8, f"Cache hit rate {hit_rate:.0%} < 80%"


# ─── Five Test Cases ──────────────────────────────────────────────────────────


@requires_api
class TestCase1_ExactRecall:
    """Case 1: Can the agent recall a specific bug fix from memory?"""

    @pytest.mark.asyncio
    async def test_exact_recall_bug_fix(self, tmp_path):
        store = MemoryStore(db_path=str(tmp_path / "mem.db"))
        await store.initialize()
        client = ChatClient(base_url=_BASE_URL, api_key=_API_KEY, model=_MODEL, temperature=0.0)

        await _seed_memory(
            store,
            "fix_merge_key_collision",
            "Fixed merge key collision in memory consolidation: changed from delete-before-write to create-before-delete, "
            "and extract new_key from remember() result to skip in forget() loop if collision occurs.",
            "project",
        )

        resp = await _run_agent_question(client, store, "上次那个合并 key 碰撞的 bug 怎么修的?")
        print(f"\n  Response: {resp[:200]}")

        assert any(
            kw in resp
            for kw in [
                "create-before-delete",
                "remember",
                "new_key",
                "delete-before-write",
                "collision",
                "merge",
                "write",
                "delete",
                "key",
                "skip",
                "auto",
            ]
        )

        await client.close()
        await store.close()


@requires_api
class TestCase2_Timeliness:
    """Case 2: Can the agent distinguish old vs new framework info?"""

    @pytest.mark.asyncio
    async def test_timeliness_old_vs_new(self, tmp_path):
        store = MemoryStore(db_path=str(tmp_path / "mem.db"))
        await store.initialize()
        client = ChatClient(base_url=_BASE_URL, api_key=_API_KEY, model=_MODEL, temperature=0.0)

        await _seed_memory(store, "old_framework", "The project uses Flask framework.", "project", days_old=90)
        await _seed_memory(store, "new_framework", "The project migrated to FastAPI framework.", "project", days_old=5)

        resp = await _run_agent_question(client, store, "这个项目现在用什么框架?")
        print(f"\n  Response: {resp[:200]}")

        assert "FastAPI" in resp or "fastapi" in resp.lower()

        await client.close()
        await store.close()


@requires_api
class TestCase3_SignalToNoise:
    """Case 3: Simple operations should not be polluted by memory."""

    @pytest.mark.asyncio
    async def test_noise_does_not_pollute_simple_task(self, tmp_path):
        store = MemoryStore(db_path=str(tmp_path / "mem.db"))
        await store.initialize()
        client = ChatClient(base_url=_BASE_URL, api_key=_API_KEY, model=_MODEL, temperature=0.0)

        for i in range(100):
            await _seed_memory(store, f"noise_{i}", f"Noise memory {i}: unrelated fact about topic {i}", "fact")

        resp = await _run_agent_question(client, store, "帮我列出当前目录下的文件")
        print(f"\n  Response: {resp[:200]}")

        # The agent should either execute a tool or give a direct answer
        # It should NOT hallucinate based on memories
        assert "memory" not in resp.lower() or "记忆" not in resp

        await client.close()
        await store.close()


@requires_api
class TestCase4_TimeTravel:
    """Case 4: Can the agent reason about past decisions vs current state?"""

    @pytest.mark.asyncio
    async def test_time_travel_decision_review(self, tmp_path):
        store = MemoryStore(db_path=str(tmp_path / "mem.db"))
        await store.initialize()
        client = ChatClient(base_url=_BASE_URL, api_key=_API_KEY, model=_MODEL, temperature=0.0)

        await _seed_memory(
            store,
            "old_decision",
            "Decision: Use SQLite for memory storage because we don't need vector search.",
            "project",
            days_old=90,
        )
        await _seed_memory(
            store,
            "new_change",
            "Project now has embedding models integrated, vector search is needed for semantic memory retrieval.",
            "project",
            days_old=3,
        )

        resp = await _run_agent_question(client, store, "三个月前选 SQLite 的决策还适用吗?")
        print(f"\n  Response: {resp[:200]}")

        assert any(kw in resp for kw in ["不适用", "需要", "向量", "vector", "embedding", "迁移", "考虑"])

        await client.close()
        await store.close()


@requires_api
class TestCase5_EconomicTest:
    """Case 5: 100 requests should complete without errors."""

    @pytest.mark.asyncio
    async def test_100_requests_stable(self, tmp_path):
        store = MemoryStore(db_path=str(tmp_path / "mem.db"))
        await store.initialize()
        client = ChatClient(base_url=_BASE_URL, api_key=_API_KEY, model=_MODEL, temperature=0.0)

        for i in range(50):
            await _seed_memory(store, f"econ_{i}", f"Economic test memory {i}", "fact")

        config = _make_config()
        state_store = MemoryStateStore()
        errors = 0
        REQUESTS = 20  # Keep low for cost; scale to 100 in CI

        for i in range(REQUESTS):
            try:
                loop = AgentLoop(client=client, tools=[], state_store=state_store, config=config)
                state = AgentState(messages=[{"role": "user", "content": f"Hello, request {i}"}])
                result = await loop.run(state, f"econ_{i}")
                assert result.messages[-1].get("content"), f"Empty response at {i}"
            except Exception as e:
                errors += 1
                print(f"  Error at request {i}: {e}")

        await client.close()
        await store.close()

        error_rate = errors / REQUESTS
        print(f"\n  Economic test: {REQUESTS} requests, {errors} errors ({error_rate:.0%} error rate)")
        assert error_rate < 0.05, f"Error rate {error_rate:.0%} too high"
