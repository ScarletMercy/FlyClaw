"""Tests for _consolidate_store group isolation — group memories consolidated per group_id."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.memory_tools import GroupMemoryStore


@pytest.mark.asyncio
async def test_consolidate_store_group_isolation(tmp_path):
    """_consolidate_store 对群记忆按 group_id 隔离整理，不串群。"""
    store = GroupMemoryStore(str(tmp_path / "g.db"))
    await store.initialize()
    try:
        for i in range(6):
            await store.remember(f"groupA fact {i}", key=f"a_{i}", category="fact", group_id="groupA")
            await store.remember(f"groupB fact {i}", key=f"b_{i}", category="fact", group_id="groupB")

        plan_a = {
            "merge": [{"from_keys": ["a_0", "a_1"], "to_content": "groupA merged fact", "to_category": "fact"}],
            "delete": ["a_2"],
            "keep": ["a_3", "a_4", "a_5"],
        }

        now = datetime.now(timezone.utc)
        result: dict = {
            "categories_processed": 0,
            "total_memories": 0,
            "merged": 0,
            "deleted": 0,
            "kept": 0,
            "errors": [],
        }

        mock_client = AsyncMock()
        gmemories_a = await store.list_all(limit=2000, group_id="groupA")
        assert len(gmemories_a) == 6

        from src.services.memory_consolidation import _consolidate_store

        with patch("src.services.memory_consolidation._ask_llm", side_effect=lambda c, p: plan_a):
            await _consolidate_store(
                store,
                mock_client,
                now,
                now.strftime("%Y-%m-%d"),
                result,
                memories=gmemories_a,
                group_id="groupA",
            )

        assert result["merged"] == 1
        assert result["deleted"] == 1
        assert result["kept"] == 3

        b_items = await store.list_all(group_id="groupB")
        assert len(b_items) == 6

        a_items = await store.list_all(group_id="groupA")
        assert len(a_items) == 4
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_run_consolidation_group_path(tmp_path):
    """run_memory_consolidation 群段真正执行（双 store 路由 + per-group 整理）。"""
    from src.tools.memory_tools import GroupMemoryStore, MemoryStore
    from src.services.memory_consolidation import run_memory_consolidation

    dm_store = MemoryStore(str(tmp_path / "dm.db"))
    await dm_store.initialize()
    group_store = GroupMemoryStore(str(tmp_path / "group.db"))
    await group_store.initialize()

    try:
        for i in range(6):
            await dm_store.remember(f"dm fact {i}", key=f"dm_{i}", category="fact")
            await group_store.remember(f"groupA fact {i}", key=f"a_{i}", category="fact", group_id="groupA")
            await group_store.remember(f"groupB fact {i}", key=f"b_{i}", category="fact", group_id="groupB")

        def llm_side_effect(client, prompt):
            if "groupA" in prompt:
                return {"merge": [], "delete": ["a_0"], "keep": ["a_1", "a_2", "a_3", "a_4", "a_5"]}
            elif "groupB" in prompt:
                return {"merge": [], "delete": ["b_0"], "keep": ["b_1", "b_2", "b_3", "b_4", "b_5"]}
            else:
                return {"merge": [], "delete": ["dm_0"], "keep": ["dm_1", "dm_2", "dm_3", "dm_4", "dm_5"]}

        mock_client = AsyncMock()
        container = MagicMock()
        container.config.memory_store.enabled = True
        container.config.model.base_url = "http://fake"
        container.config.model.api_key = "fake-key"
        container.config.model.name = "fake-model"

        with (
            patch(
                "src.tools.memory_tools.get_memory_store",
                side_effect=lambda db_path=None, chat_type="p2p": group_store if chat_type == "group" else dm_store,
            ),
            patch("src.services.memory_consolidation._ask_llm", side_effect=llm_side_effect),
            patch("src.agent.client.ChatClient", return_value=mock_client),
        ):
            result = await run_memory_consolidation(container)

        assert result["deleted"] == 3
        assert result["categories_processed"] == 3
        assert len(await dm_store.list_all()) == 5
        assert len(await group_store.list_all(group_id="groupA")) == 5
        assert len(await group_store.list_all(group_id="groupB")) == 5
    finally:
        await dm_store.close()
        await group_store.close()
