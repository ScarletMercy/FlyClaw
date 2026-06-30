"""Integration tests for memory consolidation pipeline.

Uses real MemoryStore (tmp_path SQLite) with mocked ChatClient/LLM.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.memory_tools import MemoryStore


@pytest.fixture
async def store(tmp_path):
    s = MemoryStore(db_path=str(tmp_path / "test_memories.db"))
    await s.initialize()
    yield s
    await s.close()


def _make_container():
    container = MagicMock()
    container.config.memory_store.enabled = True
    container.config.model.base_url = "http://fake"
    container.config.model.api_key = "fake-key"
    container.config.model.name = "fake-model"
    container.config.agents.workspace = "."
    return container


def _mock_chat_client():
    """AsyncMock that returns an AsyncMock instance when called (constructor)."""
    mock_instance = AsyncMock()
    return MagicMock(return_value=mock_instance)


async def _seed(store: MemoryStore, count: int, category: str = "fact"):
    for i in range(count):
        await store.remember(f"Memory content {i}", key=f"mem_{i}", category=category)


async def _seed_backdated(store: MemoryStore, key: str, content: str, category: str, days_old: int):
    await store.remember(content, key=key, category=category)
    dt = datetime.now(timezone.utc) - timedelta(days=days_old)
    iso = dt.isoformat()
    await store._conn.execute(
        "UPDATE memories SET created_at = ?, updated_at = ?, updated_ts = ? WHERE key = ?",
        (iso, iso, dt.timestamp(), key),
    )
    await store._conn.commit()


_PATCH_STORE = "src.tools.memory_tools.get_memory_store"
_PATCH_LLM = "src.services.memory_consolidation._ask_llm"
_PATCH_CLIENT = "src.agent.client.ChatClient"


# ─── Skip / guard tests ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_empty_store_skips(store):
    container = _make_container()
    with patch(_PATCH_STORE, return_value=store):
        from src.services.memory_consolidation import run_memory_consolidation

        result = await run_memory_consolidation(container)
    assert result["total_memories"] == 0
    assert result["categories_processed"] == 0


@pytest.mark.asyncio
async def test_single_memory_skips(store):
    await store.remember("Only one", key="solo", category="fact")
    container = _make_container()
    with patch(_PATCH_STORE, return_value=store):
        from src.services.memory_consolidation import run_memory_consolidation

        result = await run_memory_consolidation(container)
    assert result["total_memories"] == 1
    assert result["categories_processed"] == 0


@pytest.mark.asyncio
async def test_disabled_config_skips(store):
    await _seed(store, 10)
    container = _make_container()
    container.config.memory_store.enabled = False
    with patch(_PATCH_STORE, return_value=store):
        from src.services.memory_consolidation import run_memory_consolidation

        result = await run_memory_consolidation(container)
    assert result["total_memories"] == 0
    assert result["categories_processed"] == 0


# ─── Merge tests ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_merge_creates_before_deletes(store):
    await _seed(store, 6, category="fact")
    plan = {
        "merge": [{"from_keys": ["mem_0", "mem_1"], "to_content": "Merged memory content", "to_category": "fact"}],
        "delete": [],
        "keep": ["mem_2", "mem_3", "mem_4", "mem_5"],
    }
    container = _make_container()
    with (
        patch(_PATCH_STORE, return_value=store),
        patch(_PATCH_LLM, side_effect=lambda c, p: plan),
        patch(_PATCH_CLIENT, _mock_chat_client()),
    ):
        from src.services.memory_consolidation import run_memory_consolidation

        result = await run_memory_consolidation(container)

    assert result["merged"] == 1
    keys = [m["key"] for m in await store.list_all(limit=100)]
    assert "mem_0" not in keys
    assert "mem_1" not in keys
    contents = [m["content"] for m in await store.list_all(limit=100)]
    assert "Merged memory content" in contents


@pytest.mark.asyncio
async def test_merge_key_collision_safe(store):
    """Auto-key collides with from_key -> new entry is NOT deleted."""
    content_a = "A" * 40 + " part A"
    content_b = "A" * 40 + " part B"
    await store.remember(content_a, key="collision_key", category="fact")
    await store.remember(content_b, key="other_key", category="fact")
    await _seed(store, 4, category="fact")

    merged_content = "A" * 40 + " merged AB"
    plan = {
        "merge": [{"from_keys": ["collision_key", "other_key"], "to_content": merged_content, "to_category": "fact"}],
        "delete": [],
        "keep": ["mem_0", "mem_1", "mem_2", "mem_3"],
    }
    container = _make_container()
    with (
        patch(_PATCH_STORE, return_value=store),
        patch(_PATCH_LLM, side_effect=lambda c, p: plan),
        patch(_PATCH_CLIENT, _mock_chat_client()),
    ):
        from src.services.memory_consolidation import run_memory_consolidation

        result = await run_memory_consolidation(container)

    assert result["merged"] == 1
    contents = [m["content"] for m in await store.list_all(limit=100)]
    assert merged_content in contents


@pytest.mark.asyncio
async def test_delete_old_facts(store):
    await _seed_backdated(store, "old_bug", "old bug fix record", "fact", 200)
    await _seed(store, 5, category="fact")

    plan = {"merge": [], "delete": ["old_bug"], "keep": [f"mem_{i}" for i in range(5)]}
    container = _make_container()
    with (
        patch(_PATCH_STORE, return_value=store),
        patch(_PATCH_LLM, side_effect=lambda c, p: plan),
        patch(_PATCH_CLIENT, _mock_chat_client()),
    ):
        from src.services.memory_consolidation import run_memory_consolidation

        result = await run_memory_consolidation(container)

    assert result["deleted"] == 1
    keys = [m["key"] for m in await store.list_all(limit=100)]
    assert "old_bug" not in keys


@pytest.mark.asyncio
async def test_keep_identity_even_old(store):
    await _seed_backdated(store, "user_name", "username: zhangsan", "identity", 400)
    await _seed(store, 5, category="identity")

    plan = {"merge": [], "delete": [], "keep": ["user_name"] + [f"mem_{i}" for i in range(5)]}
    container = _make_container()
    with (
        patch(_PATCH_STORE, return_value=store),
        patch(_PATCH_LLM, side_effect=lambda c, p: plan),
        patch(_PATCH_CLIENT, _mock_chat_client()),
    ):
        from src.services.memory_consolidation import run_memory_consolidation

        result = await run_memory_consolidation(container)

    assert result["deleted"] == 0
    keys = [m["key"] for m in await store.list_all(limit=100)]
    assert "user_name" in keys


@pytest.mark.asyncio
async def test_llm_failure_graceful(store):
    await _seed(store, 6, category="fact")

    async def _fail(client, prompt):
        raise RuntimeError("LLM timeout")

    container = _make_container()
    with (
        patch(_PATCH_STORE, return_value=store),
        patch(_PATCH_LLM, side_effect=_fail),
        patch(_PATCH_CLIENT, _mock_chat_client()),
    ):
        from src.services.memory_consolidation import run_memory_consolidation

        result = await run_memory_consolidation(container)

    assert len(result["errors"]) == 1
    assert "fact" in result["errors"][0]
    assert result["total_memories"] == 6


@pytest.mark.asyncio
async def test_full_pipeline_mixed(store):
    for i in range(6):
        await store.remember(f"fact {i}", key=f"fact_{i}", category="fact")
    for i in range(4):
        await store.remember(f"identity {i}", key=f"id_{i}", category="identity")
    for i in range(3):
        await store.remember(f"preference {i}", key=f"pref_{i}", category="preference")
    for i in range(3):
        await store.remember(f"contact {i}", key=f"contact_{i}", category="contact")
    for i in range(4):
        await store.remember(f"project {i}", key=f"proj_{i}", category="project")

    plans = {
        "fact": {
            "merge": [{"from_keys": ["fact_0", "fact_1"], "to_content": "merged fact", "to_category": "fact"}],
            "delete": ["fact_2"],
            "keep": ["fact_3", "fact_4", "fact_5"],
        },
        "identity": {"merge": [], "delete": [], "keep": [f"id_{i}" for i in range(4)]},
        "preference": {"merge": [], "delete": [], "keep": [f"pref_{i}" for i in range(3)]},
        "contact": {"merge": [], "delete": [], "keep": [f"contact_{i}" for i in range(3)]},
        "project": {
            "merge": [{"from_keys": ["proj_0", "proj_1"], "to_content": "merged project", "to_category": "project"}],
            "delete": [],
            "keep": ["proj_2", "proj_3"],
        },
    }

    # Track which categories were asked about
    asked_categories: list[str] = []

    async def _plan_by_category(client, prompt):
        # Match by checking which category's memories appear in the prompt
        for cat, p in plans.items():
            # Check if this category's first memory key appears in the items
            first_key = p.get("keep", [None])[0] or (
                p.get("merge", [{}])[0].get("from_keys", [None])[0] if p.get("merge") else None
            )
            if first_key and f"key: {first_key}" in prompt:
                asked_categories.append(cat)
                return p
        return {"merge": [], "delete": [], "keep": []}

    container = _make_container()
    with (
        patch(_PATCH_STORE, return_value=store),
        patch(_PATCH_LLM, side_effect=_plan_by_category),
        patch(_PATCH_CLIENT, _mock_chat_client()),
    ):
        from src.services.memory_consolidation import run_memory_consolidation

        result = await run_memory_consolidation(container)

    assert result["total_memories"] == 20
    assert result["merged"] == 2
    assert result["deleted"] == 1
    assert result["categories_processed"] == 5

    keys = {m["key"] for m in await store.list_all(limit=100)}
    assert "fact_0" not in keys
    assert "fact_1" not in keys
    assert "fact_2" not in keys
    assert "id_0" in keys
    assert "pref_0" in keys


# ---------------------------------------------------------------------------
# Partial failure & edge-case tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_returns_none_skips_category(store):
    """When _ask_llm returns None (non-JSON response), the category should be skipped."""
    await _seed(store, 6, category="fact")
    container = _make_container()
    with (
        patch(_PATCH_STORE, return_value=store),
        patch(_PATCH_LLM, return_value=None),
        patch(_PATCH_CLIENT, _mock_chat_client()),
    ):
        from src.services.memory_consolidation import run_memory_consolidation

        result = await run_memory_consolidation(container)
    assert len(result["errors"]) == 0
    assert result["merged"] == 0
    assert result["deleted"] == 0


@pytest.mark.asyncio
async def test_malformed_plan_skips_invalid_ops(store):
    """A plan with invalid data (empty from_keys, non-string delete) should not crash."""
    await _seed(store, 6, category="fact")
    plan = {
        "merge": [{"from_keys": [], "to_content": ""}],
        "delete": [123, "", None],
        "keep": "not_a_list",
    }

    async def _return_plan(client, prompt):
        return plan

    container = _make_container()
    with (
        patch(_PATCH_STORE, return_value=store),
        patch(_PATCH_LLM, side_effect=_return_plan),
        patch(_PATCH_CLIENT, _mock_chat_client()),
    ):
        from src.services.memory_consolidation import run_memory_consolidation

        result = await run_memory_consolidation(container)
    # No crash; original data should still be intact
    keys = {m["key"] for m in await store.list_all(limit=100)}
    assert all(f"mem_{i}" in keys for i in range(6))


@pytest.mark.asyncio
async def test_merge_nonexistent_from_keys_safe(store):
    """Merge with non-existent from_keys should create new entry; forget is a no-op."""
    await _seed(store, 5, category="fact")
    plan = {
        "merge": [
            {
                "from_keys": ["nonexistent_1", "nonexistent_2"],
                "to_content": "merged content",
                "to_category": "fact",
            }
        ],
        "delete": [],
        "keep": [f"mem_{i}" for i in range(5)],
    }

    async def _return_plan(client, prompt):
        return plan

    container = _make_container()
    with (
        patch(_PATCH_STORE, return_value=store),
        patch(_PATCH_LLM, side_effect=_return_plan),
        patch(_PATCH_CLIENT, _mock_chat_client()),
    ):
        from src.services.memory_consolidation import run_memory_consolidation

        result = await run_memory_consolidation(container)
    # Merge should complete (new entry created, forget on missing keys is safe)
    assert result["merged"] == 1


@pytest.mark.asyncio
async def test_delete_nonexistent_key_safe(store):
    """Deleting a key that doesn't exist should not raise."""
    await _seed(store, 5, category="fact")
    plan = {
        "merge": [],
        "delete": ["ghost_key", "another_ghost"],
        "keep": [f"mem_{i}" for i in range(5)],
    }

    async def _return_plan(client, prompt):
        return plan

    container = _make_container()
    with (
        patch(_PATCH_STORE, return_value=store),
        patch(_PATCH_LLM, side_effect=_return_plan),
        patch(_PATCH_CLIENT, _mock_chat_client()),
    ):
        from src.services.memory_consolidation import run_memory_consolidation

        result = await run_memory_consolidation(container)
    assert result["deleted"] == 2
    keys = {m["key"] for m in await store.list_all(limit=100)}
    assert all(f"mem_{i}" for i in range(5))


@pytest.mark.asyncio
async def test_partial_category_failure(store):
    """If one category's LLM call fails, other categories should still be processed."""
    for i in range(6):
        await store.remember(f"fact {i}", key=f"fact_{i}", category="fact")
    for i in range(4):
        await store.remember(f"identity {i}", key=f"id_{i}", category="identity")

    async def _fail_fact_then_succeed(client, prompt):
        if "fact_0" in prompt:
            raise RuntimeError("LLM timeout for fact")
        return {"merge": [], "delete": [], "keep": [f"id_{i}" for i in range(4)]}

    container = _make_container()
    with (
        patch(_PATCH_STORE, return_value=store),
        patch(_PATCH_LLM, side_effect=_fail_fact_then_succeed),
        patch(_PATCH_CLIENT, _mock_chat_client()),
    ):
        from src.services.memory_consolidation import run_memory_consolidation

        result = await run_memory_consolidation(container)

    assert len(result["errors"]) == 1
    assert "fact" in result["errors"][0]
    # Fact memories should be untouched (failed category)
    keys = {m["key"] for m in await store.list_all(limit=100)}
    assert "fact_0" in keys
    # Identity was processed (keep)
    assert "id_0" in keys
