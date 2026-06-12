"""E2E tests for the 3am daily consolidation memory extraction pipeline.

Uses real LLM calls to verify that spawn_background_review correctly extracts
user facts from conversation history. Each test follows a two-phase pattern:

  Phase 1 (Extraction): Run spawn_background_review on a conversation → save memories
  Phase 2 (Cross-validation): Inject extracted memories into a fresh agent and verify
    it can answer correctly — same model, different angle.

Requires valid API credentials in config.yaml (model section).
Run: pytest tests/test_daily_consolidation_e2e.py -v -s
Skip: auto-skipped if no API key in config.
"""

from __future__ import annotations

import io
import json
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

# Fix Windows GBK console — LLM responses may contain emoji/Unicode
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import pytest

from src.agent.client import ChatClient
from src.agent.loop import AgentLoop
from src.agent.state import AgentState, MemoryStateStore
from src.config import load_config
from src.skills.review import spawn_background_review
from src.tools.memory_tools import MemoryStore

# ─── Load model config ───────────────────────────────────────────────────────

_config = load_config()
_API_KEY = _config.model.api_key or ""
_BASE_URL = _config.model.base_url or "https://api.openai.com/v1"
_MODEL = _config.model.name or "gpt-4o"

requires_api = pytest.mark.skipif(not _API_KEY, reason="No API key in config")


# ─── Helpers ─────────────────────────────────────────────────────────────────


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


def _build_conversation(turns: list[tuple[str, str]]) -> list[dict]:
    """Convert [(role, content), ...] to message dicts."""
    return [{"role": role, "content": content} for role, content in turns]


def _get_review_tools() -> list:
    """Build the tool list for the review agent (memory + skill_view + skill_manage)."""
    from src.tools.memory_tools import get_tools as memory_tools
    from src.skills.manager import get_tools as skill_tools

    return memory_tools() + skill_tools()


async def _run_review(
    store: MemoryStore,
    client: ChatClient,
    messages: list[dict],
    config=None,
) -> str:
    """Run spawn_background_review and return the summary string.

    Patches get_memory_store to use the test store and get_container
    to prevent skill tool failures.
    """
    if config is None:
        config = _make_config()

    tools = _get_review_tools()
    container_mock = MagicMock()
    container_mock.skills_cache = []

    with (
        patch("src.tools.memory_tools.get_memory_store", return_value=store),
        patch("src._container.get_container", return_value=container_mock),
    ):
        # Same params as production _consolidate_session:
        # review_skills=True + review_memory=True = COMBINED_REVIEW_PROMPT
        summary = await spawn_background_review(
            client=client,
            tools=tools,
            config=config,
            messages_snapshot=messages,
            review_skills=True,
            review_memory=True,
            max_rounds=10,
        )
    return summary


async def _run_retrieval_question(
    client: ChatClient,
    store: MemoryStore,
    question: str,
    config=None,
) -> str:
    """Cross-validation: run a single-turn agent with extracted memories injected."""
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
        result = await loop.run(state, f"consolidation_verify_{int(time.time())}")
    return result.messages[-1].get("content", "")


def _memory_contents(store_memories: list[dict]) -> list[str]:
    """Extract content strings from memory list."""
    return [m.get("content", "") for m in store_memories]


def _contains_any(text: str, terms: list[str]) -> bool:
    """Check if text contains any of the terms (case-insensitive)."""
    lower = text.lower()
    return any(t.lower() in lower for t in terms)


def _all_memory_texts_contain(memories: list[dict], term: str) -> bool:
    """Check if any memory content contains the term."""
    return any(term in m.get("content", "") for m in memories)


# ─── Test 1: Identity extraction + cross-validation ──────────────────────────


@requires_api
class TestAccuracy_IdentityRecall:
    """Verify identity facts are extracted and can be recalled by a fresh agent."""

    @pytest.mark.asyncio
    async def test_identity_extraction_and_recall(self, tmp_path):
        store = MemoryStore(db_path=str(tmp_path / "mem.db"))
        await store.initialize()
        client = ChatClient(base_url=_BASE_URL, api_key=_API_KEY, model=_MODEL, temperature=0.0)

        messages = _build_conversation(
            [
                ("user", "你好，我叫韩飞摩，是一个全栈开发工程师"),
                ("assistant", "你好韩飞摩！很高兴认识你，有什么我可以帮你的吗？"),
                ("user", "我的公司是星辰科技，在杭州"),
                ("assistant", "星辰科技在杭州，了解了。你需要什么帮助？"),
                ("user", "我现在的职位是技术总监"),
                ("assistant", "技术总监，很厉害！有什么技术问题需要讨论吗？"),
                ("user", "帮我看看这段代码有没有性能问题"),
                ("assistant", "好的，请把代码发给我看看。"),
                ("user", "def process(data): return [x*2 for x in data]"),
                ("assistant", "这段列表推导式很简洁，性能也不错。如果数据量大可以考虑生成器。"),
                ("user", "好的，还有个问题，Python的装饰器怎么用？"),
                ("assistant", "装饰器是Python的高阶特性，用@语法糖。例如 @decorator 放在函数定义前。"),
            ]
        )

        # Phase 1: extract memories
        summary = await _run_review(store, client, messages)
        print(f"\n  Extraction summary: {summary}")

        memories = await store.list_all(limit=100)
        print(f"  Memories saved: {len(memories)}")
        for m in memories:
            print(f"    [{m.get('category', '?')}] {m.get('content', '')[:80]}")

        # Phase 2: cross-validation — ask a fresh agent
        qa = [
            ("我叫什么名字？", ["韩飞摩"]),
            ("我在哪个城市？", ["杭州"]),
            ("我的职位是什么？", ["技术总监"]),
        ]

        correct = 0
        for question, expected_terms in qa:
            resp = await _run_retrieval_question(client, store, question)
            hit = _contains_any(resp, expected_terms)
            correct += int(hit)
            print(f"    Q: {question} -> {'OK' if hit else 'MISS'} {resp[:60]}")

        await client.close()
        await store.close()

        recall = correct / len(qa)
        print(f"  Identity recall: {correct}/{len(qa)} = {recall:.0%}")
        assert recall >= 0.66, f"Identity recall {recall:.0%} < 66%"


# ─── Test 2: Preference extraction + cross-validation ────────────────────────


@requires_api
class TestAccuracy_PreferenceRecall:
    """Verify user preferences are extracted and can be recalled."""

    @pytest.mark.asyncio
    async def test_preference_extraction_and_recall(self, tmp_path):
        store = MemoryStore(db_path=str(tmp_path / "mem.db"))
        await store.initialize()
        client = ChatClient(base_url=_BASE_URL, api_key=_API_KEY, model=_MODEL, temperature=0.0)

        messages = _build_conversation(
            [
                ("user", "帮我用TypeScript写一个API接口"),
                ("assistant", "好的，我来帮你写一个TypeScript的API接口。你用的是什么框架？"),
                ("user", "以后回复我请用中文，不要用英文回复"),
                ("assistant", "好的，以后我会用中文回复你。"),
                ("user", "还有，代码风格我喜欢用单引号，不用双引号"),
                ("assistant", "明白了，代码中会用单引号风格。"),
                ("user", "以后写代码请默认用 pnpm，不要用 npm"),
                ("assistant", "收到，以后默认用pnpm。让我继续写那个API接口。"),
                ("user", "对了，帮我写个单元测试"),
                ("assistant", "好的，我来写单元测试。"),
                ("user", "测试覆盖率要达到80%以上"),
                ("assistant", "明白，我会确保测试覆盖率达到80%以上。"),
                ("user", "好的，部署的时候用 Docker"),
                ("assistant", "好的，我来准备Docker部署配置。"),
            ]
        )

        summary = await _run_review(store, client, messages)
        print(f"\n  Extraction summary: {summary}")

        memories = await store.list_all(limit=100)
        print(f"  Memories saved: {len(memories)}")
        for m in memories:
            print(f"    [{m.get('category', '?')}] {m.get('content', '')[:80]}")

        qa = [
            ("我的代码风格偏好是什么？", ["单引号"]),
            ("我应该用什么包管理器？", ["pnpm"]),
        ]

        correct = 0
        for question, expected_terms in qa:
            resp = await _run_retrieval_question(client, store, question)
            hit = _contains_any(resp, expected_terms)
            correct += int(hit)
            print(f"    Q: {question} -> {'OK' if hit else 'MISS'} {resp[:60]}")

        await client.close()
        await store.close()

        recall = correct / len(qa)
        print(f"  Preference recall: {correct}/{len(qa)} = {recall:.0%}")
        assert recall >= 0.5, f"Preference recall {recall:.0%} < 50%"


# ─── Test 3: Contact extraction + exact match ────────────────────────────────


@requires_api
class TestAccuracy_ContactRecall:
    """Verify contact info is extracted with exact values (not paraphrased)."""

    @pytest.mark.asyncio
    async def test_contact_extraction_exact(self, tmp_path):
        store = MemoryStore(db_path=str(tmp_path / "mem.db"))
        await store.initialize()
        client = ChatClient(base_url=_BASE_URL, api_key=_API_KEY, model=_MODEL, temperature=0.0)

        messages = _build_conversation(
            [
                ("user", "帮我发个通知邮件"),
                ("assistant", "好的，发给谁？"),
                ("user", "我的邮箱是 hanfeimo@startech.cn，有事可以发邮件"),
                ("assistant", "好的，记下了你的邮箱。"),
                ("user", "手机号是 13812345678，紧急情况可以打这个"),
                ("assistant", "收到，手机号也记下了。"),
                ("user", "微信也可以，微信号是 feimo_han"),
                ("assistant", "好的，三种联系方式都有了。"),
                ("user", "帮我看看项目文档怎么写"),
                ("assistant", "项目文档一般包括README、API文档、架构说明。你需要哪种？"),
                ("user", "写个README吧"),
                ("assistant", "好的，我来写README文档。"),
            ]
        )

        summary = await _run_review(store, client, messages)
        print(f"\n  Extraction summary: {summary}")

        memories = await store.list_all(limit=100)
        print(f"  Memories saved: {len(memories)}")
        for m in memories:
            print(f"    [{m.get('category', '?')}] {m.get('content', '')[:80]}")

        qa = [
            ("我的邮箱是什么？", ["hanfeimo@startech.cn"]),
            ("我的手机号是什么？", ["13812345678"]),
        ]

        correct = 0
        for question, expected_terms in qa:
            resp = await _run_retrieval_question(client, store, question)
            hit = any(t in resp for t in expected_terms)
            correct += int(hit)
            print(f"    Q: {question} -> {'OK' if hit else 'MISS'} {resp[:60]}")

        await client.close()
        await store.close()

        recall = correct / len(qa)
        print(f"  Contact recall: {correct}/{len(qa)} = {recall:.0%}")
        assert recall >= 0.5, f"Contact recall {recall:.0%} < 50%"


# ─── Test 4: Negative — noise rejection ──────────────────────────────────────


@requires_api
class TestNegative_NoiseRejection:
    """One-off tasks and general knowledge should NOT be extracted as memories."""

    @pytest.mark.asyncio
    async def test_no_extraction_from_coding_and_knowledge(self, tmp_path):
        store = MemoryStore(db_path=str(tmp_path / "mem.db"))
        await store.initialize()
        client = ChatClient(base_url=_BASE_URL, api_key=_API_KEY, model=_MODEL, temperature=0.0)

        messages = _build_conversation(
            [
                ("user", "帮我写一个快速排序算法"),
                (
                    "assistant",
                    "好的，这是Python快速排序：\ndef quicksort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[len(arr)//2]\n    left = [x for x in arr if x < pivot]\n    ...",
                ),
                ("user", "量子计算是什么原理？"),
                (
                    "assistant",
                    "量子计算利用量子比特(qubit)的叠加态和纠缠态进行并行计算，在某些问题上比经典计算快很多。",
                ),
                ("user", "运行报错了，IndexError: list index out of range"),
                ("assistant", "这个错误是因为数组越界了。检查一下数组长度和索引范围。"),
                ("user", "什么是Transformer架构？"),
                ("assistant", "Transformer是2017年提出的深度学习架构，核心是自注意力机制，广泛应用于NLP领域。"),
                ("user", "现在帮我写个二分查找"),
                (
                    "assistant",
                    "好的，二分查找：\ndef binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right: ...",
                ),
                ("user", "Python的GIL是什么？"),
                ("assistant", "GIL是全局解释器锁，确保同一时刻只有一个线程执行Python字节码，限制了多线程并行。"),
                ("user", "帮我测试一下这个函数"),
                ("assistant", "好的，写个简单的测试：\nassert binary_search([1,2,3], 2) == 1"),
            ]
        )

        summary = await _run_review(store, client, messages)
        print(f"\n  Extraction summary: {summary}")

        memories = await store.list_all(limit=100)
        print(f"  Memories saved: {len(memories)}")
        for m in memories:
            print(f"    [{m.get('category', '?')}] {m.get('content', '')[:80]}")

        # Cross-validation: ask about previous coding tasks — agent should NOT know
        if memories:
            contents = _memory_contents(memories)
            all_contents = " ".join(contents)
            forbidden = ["快速排序", "量子", "Transformer", "GIL", "二分查找"]
            noise_count = sum(1 for f in forbidden if f in all_contents)
            print(f"  Noise in memories: {noise_count}/{len(forbidden)} forbidden terms found")
            assert noise_count <= 1, f"Too much noise: {noise_count} forbidden terms in extracted memories"

        await client.close()
        await store.close()

        assert len(memories) <= 2, f"Expected ≤2 memories, got {len(memories)}"


# ─── Test 5: Adversarial — no hallucination ──────────────────────────────────


@requires_api
class TestAdversarial_NoHallucination:
    """Extracted memories must not contain fabricated facts."""

    @pytest.mark.asyncio
    async def test_no_hallucinated_facts(self, tmp_path):
        store = MemoryStore(db_path=str(tmp_path / "mem.db"))
        await store.initialize()
        client = ChatClient(base_url=_BASE_URL, api_key=_API_KEY, model=_MODEL, temperature=0.0)

        messages = _build_conversation(
            [
                ("user", "我养了一只猫叫小橘"),
                ("assistant", "小橘听起来很可爱！是什么品种的猫？"),
                ("user", "是橘猫，3岁了"),
                ("assistant", "橘猫很温顺，3岁正是活泼的时候。"),
                ("user", "我的GitHub用户名是 feimo-dev"),
                ("assistant", "记下了，feimo-dev。"),
                ("user", "服务器端口是 8080"),
                ("assistant", "了解，端口8080。"),
                ("user", "帮我看看这个配置文件"),
                ("assistant", "好的，请发配置文件内容。"),
                ("user", "server { listen 8080; server_name localhost; }"),
                ("assistant", "这是一个Nginx配置，监听8080端口。看起来没问题。"),
                ("user", "帮我加个SSL证书配置"),
                ("assistant", "好的，加上SSL配置：需要指定证书路径。"),
            ]
        )

        summary = await _run_review(store, client, messages)
        print(f"\n  Extraction summary: {summary}")

        memories = await store.list_all(limit=100)
        print(f"  Memories saved: {len(memories)}")
        for m in memories:
            print(f"    [{m.get('category', '?')}] {m.get('content', '')[:80]}")

        # Phase 2: check for correct recall
        contents = _memory_contents(memories)
        all_contents = " ".join(contents)

        # Must contain at least one real fact
        real_terms = ["小橘", "feimo-dev", "8080"]
        real_found = sum(1 for t in real_terms if t in all_contents)
        print(f"  Real facts found: {real_found}/{len(real_terms)}")
        assert real_found >= 1, f"Expected ≥1 real fact, found {real_found}"

        # Must NOT contain fabricated facts
        fake_terms = ["小花", "dog", "GitHub用户名是abc", "汪汪", "养了一只狗"]
        fake_found = sum(1 for t in fake_terms if t.lower() in all_contents.lower())
        print(f"  Fabricated facts found: {fake_found}")
        assert fake_found == 0, f"Found {fake_found} fabricated terms in memories"

        # Phase 3: cross-validation -- recall real fact, reject fake
        resp_real = await _run_retrieval_question(client, store, "我的宠物叫什么名字？")
        print(f"    Q: pet name -> {resp_real[:60]}")
        assert _contains_any(resp_real, ["小橘"]), f"Expected '小橘' in response"

        resp_fake = await _run_retrieval_question(client, store, "我的狗叫什么名字？")
        print(f"    Q: dog name -> {resp_fake[:60]}")
        # Should say "don't know" or "no dog" — not hallucinate a dog name
        assert "小橘" not in resp_fake or "猫" in resp_fake or "没有" in resp_fake or "不" in resp_fake, (
            "Agent hallucinated a dog when user only has a cat"
        )

        await client.close()
        await store.close()


# ─── Test 6: Comprehensive Recall + Precision metric ─────────────────────────


@requires_api
class TestMetric_RecallPrecisionScore:
    """Quantitative metric: measure recall and precision of extraction."""

    @pytest.mark.asyncio
    async def test_recall_precision_with_cross_validation(self, tmp_path):
        store = MemoryStore(db_path=str(tmp_path / "mem.db"))
        await store.initialize()
        client = ChatClient(base_url=_BASE_URL, api_key=_API_KEY, model=_MODEL, temperature=0.0)

        # 8 extractable facts + noise padding
        messages = _build_conversation(
            [
                # Fact 1: name
                ("user", "你好，我叫韩飞摩"),
                ("assistant", "你好韩飞摩！"),
                # Fact 2: company
                ("user", "我的公司是星辰科技"),
                ("assistant", "星辰科技，了解了。"),
                # Fact 3: city
                ("user", "公司在杭州"),
                ("assistant", "杭州是个好地方。"),
                # Fact 4: role
                ("user", "我是技术总监"),
                ("assistant", "技术总监，很厉害。"),
                # Fact 5: email
                ("user", "邮箱是 hanfeimo@startech.cn"),
                ("assistant", "记下了。"),
                # Fact 6: language preference
                ("user", "以后请用中文回复"),
                ("assistant", "好的，以后用中文回复。"),
                # Fact 7: package manager preference
                ("user", "包管理器用 pnpm"),
                ("assistant", "明白，用pnpm。"),
                # Fact 8: project framework
                ("user", "我的项目用 FastAPI"),
                ("assistant", "FastAPI是个很好的框架。"),
                # Noise padding
                ("user", "帮我写个冒泡排序"),
                (
                    "assistant",
                    "好的，冒泡排序：\ndef bubble_sort(arr):\n    for i in range(len(arr)):\n        for j in range(len(arr)-i-1):\n            if arr[j] > arr[j+1]: arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr",
                ),
                ("user", "什么是RESTful API？"),
                (
                    "assistant",
                    "RESTful API是一种遵循REST架构风格的API设计，使用HTTP方法（GET/POST/PUT/DELETE）操作资源。",
                ),
            ]
        )

        # Phase 1: extraction
        summary = await _run_review(store, client, messages)
        print(f"\n  Extraction summary: {summary}")

        memories = await store.list_all(limit=100)
        print(f"  Memories saved: {len(memories)}")
        for m in memories:
            print(f"    [{m.get('category', '?')}] {m.get('content', '')[:80]}")

        contents = _memory_contents(memories)
        all_contents = " ".join(contents)

        # 8 ground truth facts with verification terms
        ground_truth = [
            ("name", "韩飞摩"),
            ("company", "星辰科技"),
            ("city", "杭州"),
            ("role", "技术总监"),
            ("email", "hanfeimo@startech.cn"),
            ("lang_pref", "中文"),
            ("pkg_pref", "pnpm"),
            ("framework", "FastAPI"),
        ]

        # Calculate Recall
        facts_found = 0
        for label, term in ground_truth:
            found = term in all_contents
            facts_found += int(found)
            print(f"    Fact '{label}' ({term}): {'OK' if found else 'MISS'}")

        recall = facts_found / len(ground_truth)
        print(f"  Recall: {facts_found}/{len(ground_truth)} = {recall:.0%}")

        # Calculate Precision (noise = memories that contain coding/general knowledge terms)
        noise_terms = ["冒泡", "排序", "RESTful", "REST", "bubble"]
        noise_count = sum(1 for c in contents for t in noise_terms if t in c)
        precision = (len(memories) - noise_count) / len(memories) if memories else 1.0
        print(f"  Precision: {len(memories) - noise_count}/{len(memories)} = {precision:.0%}")

        # Phase 2: cross-validation — ask about each fact
        qa_pairs = [
            ("我叫什么名字？", "韩飞摩"),
            ("我的公司叫什么？", "星辰科技"),
            ("我在哪个城市？", "杭州"),
            ("我的职位是什么？", "技术总监"),
            ("我的邮箱是什么？", "hanfeimo@startech.cn"),
            ("我的项目用什么框架？", "FastAPI"),
        ]

        cv_correct = 0
        for question, expected in qa_pairs:
            resp = await _run_retrieval_question(client, store, question)
            hit = expected in resp
            cv_correct += int(hit)
            print(f"    CV Q: {question} -> {'OK' if hit else 'MISS'} {resp[:60]}")

        cv_rate = cv_correct / len(qa_pairs)
        print(f"  Cross-validation rate: {cv_correct}/{len(qa_pairs)} = {cv_rate:.0%}")

        await client.close()
        await store.close()

        # Assertions
        assert recall >= 0.5, f"Recall {recall:.0%} < 50%"
        assert precision >= 0.7, f"Precision {precision:.0%} < 70%"
        # Cross-validation should be reasonably high, proving memories are useful
        assert cv_rate >= 0.5, f"Cross-validation rate {cv_rate:.0%} < 50%"


# ─── Test 7: Long conversation with real session data ────────────────────────

_REAL_DB = "C:/Users/86198/.myclaw/data/session_index.db"
# Sessions to test: diverse channels, sizes, and content
_REAL_SESSIONS = [
    # (label, thread_id, min_chars)
    ("qq:s1/98k", "qq:s1:1097C11341995ACED68981D1786676B7", 50_000),
    ("qq:user/129k", "qq:user:1097C11341995ACED68981D1786676B7", 50_000),
    ("qq:s2/48k", "qq:s2:1097C11341995ACED68981D1786676B7", 10_000),
    ("feishu:user/63k", "feishu:user:ou_bebd12fb70fcc8fc0ee6583f9d45fd1b", 10_000),
]


def _load_real_session(db_path: str, thread_id: str) -> list[dict]:
    """Load messages from a real session database."""
    import sqlite3

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT role, content FROM messages WHERE thread_id=? ORDER BY id",
        (thread_id,),
    )
    messages = []
    for role, content in cur.fetchall():
        if not content:
            continue
        mapped = {"human": "user", "ai": "assistant"}.get(role, role)
        messages.append({"role": mapped, "content": content})
    conn.close()
    return messages


@requires_api
class TestLongConversation_RealSession:
    """Verify extraction on real production sessions of varying sizes.

    Tests multiple real sessions from session_index.db, each with different
    content patterns (music creation, general chat, tool testing, etc.).
    """

    @pytest.mark.asyncio
    async def test_extraction_from_real_sessions(self, tmp_path):
        import os

        if not os.path.exists(_REAL_DB):
            pytest.skip(f"Real session DB not found: {_REAL_DB}")

        store = MemoryStore(db_path=str(tmp_path / "mem.db"))
        await store.initialize()
        client = ChatClient(base_url=_BASE_URL, api_key=_API_KEY, model=_MODEL, temperature=0.0)

        results = []

        for label, thread_id, min_chars in _REAL_SESSIONS:
            messages = _load_real_session(_REAL_DB, thread_id)
            total_chars = sum(len(m["content"]) for m in messages)
            total_msgs = len(messages)

            if total_chars < min_chars:
                print(f"\n  SKIP {label}: {total_chars / 1000:.0f}k < {min_chars / 1000:.0f}k min")
                continue

            # Show user message summary
            user_msgs = [m["content"] for m in messages if m["role"] == "user"]
            print(f"\n{'=' * 60}")
            print(f"  {label}: {total_msgs} msgs, {total_chars / 1000:.0f}k chars, {len(user_msgs)} user msgs")

            # Run extraction with a fresh store per session
            session_store = MemoryStore(db_path=str(tmp_path / f"{label.replace('/', '_')}.db"))
            await session_store.initialize()

            summary = await _run_review(session_store, client, messages)

            memories = await session_store.list_all(limit=100)
            print(f"  Summary: {summary or '(none)'}")
            print(f"  Memories: {len(memories)}")
            for m in memories:
                print(f"    [{m.get('category', '?')}] {m.get('content', '')[:100]}")

            await session_store.close()

            results.append(
                {
                    "label": label,
                    "chars": total_chars,
                    "msgs": total_msgs,
                    "user_msgs": len(user_msgs),
                    "memories": len(memories),
                    "memory_contents": [m.get("content", "") for m in memories],
                }
            )

        await client.close()
        await store.close()

        # Summary report
        print(f"\n{'=' * 60}")
        print(f"  SUMMARY: {len(results)} sessions tested")
        for r in results:
            print(
                f"    {r['label']}: {r['msgs']} msgs / {r['chars'] / 1000:.0f}k chars -> {r['memories']} memories extracted"
            )

        # Sanity: all sessions should complete without error
        assert len(results) >= 1, "No sessions were tested"
        for r in results:
            for content in r["memory_contents"]:
                assert len(content) > 2, f"Empty memory in {r['label']}"


# ─── Test 8: Diary summary ───────────────────────────────────────────────────


@requires_api
class TestDiarySummary:
    """Verify _save_session_summary generates episodic diary entries."""

    @pytest.mark.asyncio
    async def test_diary_generates_episodic(self, tmp_path):
        from collections import defaultdict

        from src.services.daily_consolidation import _save_session_summary

        store = MemoryStore(db_path=str(tmp_path / "mem.db"))
        await store.initialize()
        client = ChatClient(base_url=_BASE_URL, api_key=_API_KEY, model=_MODEL, temperature=0.0)

        messages = _build_conversation(
            [
                ("user", "帮我写一个排序算法"),
                ("assistant", "好的，这是快速排序：def quicksort(arr): ..."),
                ("user", "测试一下性能"),
                ("assistant", "性能测试结果：100万条数据耗时 0.3 秒"),
                ("user", "帮我优化一下"),
                ("assistant", "优化后使用原地排序，内存减少50%。"),
                ("user", "部署到生产环境"),
                ("assistant", "已部署，监控正常。"),
                ("user", "好的，今天先这样"),
                ("assistant", "好的，随时找我！"),
            ]
        )

        created_at = time.time() - 3600  # 1 hour ago
        config = _make_config()
        day_counter: dict[int, int] = defaultdict(int)

        with patch("src.tools.memory_tools.get_memory_store", return_value=store):
            await _save_session_summary(config, created_at, messages, day_counter)

        memories = await store.list_all(limit=100)
        episodic = [m for m in memories if m.get("category") == "episodic"]
        print(f"\n  Episodic memories: {len(episodic)}")
        for m in episodic:
            print(f"    key={m['key']}: {m['content'][:100]}")

        assert len(episodic) == 1, f"Expected 1 episodic, got {len(episodic)}"
        entry = episodic[0]
        assert "日记1" in entry["key"], f"Key should contain '日记1', got: {entry['key']}"
        assert len(entry["content"]) <= 200, f"Summary too long: {len(entry['content'])} chars"
        assert len(entry["content"]) > 5, "Summary too short"

        await client.close()
        await store.close()

    @pytest.mark.asyncio
    async def test_diary_counter_increments(self, tmp_path):
        """Multiple sessions on the same day get sequential diary keys."""
        from collections import defaultdict

        from src.services.daily_consolidation import _save_session_summary

        store = MemoryStore(db_path=str(tmp_path / "mem.db"))
        await store.initialize()
        client = ChatClient(base_url=_BASE_URL, api_key=_API_KEY, model=_MODEL, temperature=0.0)
        config = _make_config()
        created_at = time.time() - 3600

        day_counter: dict[int, int] = defaultdict(int)

        msgs1 = _build_conversation(
            [
                ("user", "帮我写个排序"),
                ("assistant", "好的"),
                ("user", "优化一下"),
                ("assistant", "优化完成"),
                ("user", "部署"),
                ("assistant", "已部署"),
            ]
        )
        msgs2 = _build_conversation(
            [
                ("user", "帮我写个搜索"),
                ("assistant", "好的"),
                ("user", "测试"),
                ("assistant", "通过"),
                ("user", "上线"),
                ("assistant", "已上线"),
            ]
        )

        with patch("src.tools.memory_tools.get_memory_store", return_value=store):
            await _save_session_summary(config, created_at, msgs1, day_counter)
            await _save_session_summary(config, created_at, msgs2, day_counter)

        memories = await store.list_all(limit=100)
        episodic = [m for m in memories if m.get("category") == "episodic"]
        keys = [m["key"] for m in episodic]
        print(f"\n  Diary keys: {keys}")

        assert len(episodic) == 2, f"Expected 2 episodic, got {len(episodic)}"
        assert any("日记1" in k for k in keys), f"Missing '日记1' in {keys}"
        assert any("日记2" in k for k in keys), f"Missing '日记2' in {keys}"

        await client.close()
        await store.close()
