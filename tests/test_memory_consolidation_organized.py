"""Tests for memory_consolidation organized trigger + whole-category marking."""

from __future__ import annotations

import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.memory_consolidation import _consolidate_store
from src.tools.memory_tools import MemoryStore


def _llm_plan(merge=None, delete=None, keep=None):
    return {"merge": merge or [], "delete": delete or [], "keep": keep or []}


@pytest.mark.asyncio
async def test_category_all_organized_is_skipped(tmp_path):
    """分类内全部 organized=1 → 零 LLM 调用。"""
    store = MemoryStore(str(tmp_path / "m.db"))
    await store.initialize()
    try:
        # 5 条同分类，全标记 organized
        for i in range(5):
            await store.remember(f"内容{i}", key=f"k{i}", category="fact")
        await store.mark_organized([f"k{i}" for i in range(5)])

        memories = await store.list_all(limit=200)
        client = MagicMock()
        result = {
            "categories_processed": 0,
            "merged": 0,
            "deleted": 0,
            "kept": 0,
            "errors": [],
        }
        ask = AsyncMock(return_value=_llm_plan())
        with patch("src.services.memory_consolidation._ask_llm", ask):
            await _consolidate_store(
                store,
                client,
                datetime.datetime.now().astimezone(),
                "2026-08-05",
                result,
                memories=memories,
            )
        ask.assert_not_called()
        assert result["categories_processed"] == 0
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_category_with_unorganized_gets_reviewed_and_marked(tmp_path):
    """分类内有未整理条目 → 审查整类 → 成功后整类标记 organized。"""
    store = MemoryStore(str(tmp_path / "m.db"))
    await store.initialize()
    try:
        for i in range(5):
            await store.remember(f"内容{i}", key=f"k{i}", category="fact")
        # k0,k1 已整理；k2,k3,k4 未整理 → 触发整类审查
        await store.mark_organized(["k0", "k1"])

        memories = await store.list_all(limit=200)
        client = MagicMock()
        result = {"categories_processed": 0, "merged": 0, "deleted": 0, "kept": 0, "errors": []}
        ask = AsyncMock(return_value=_llm_plan(keep=["k0", "k1", "k2", "k3", "k4"]))
        with patch("src.services.memory_consolidation._ask_llm", ask):
            await _consolidate_store(
                store,
                client,
                datetime.datetime.now().astimezone(),
                "2026-08-05",
                result,
                memories=memories,
            )
        # LLM 被调用一次，整类被审查
        assert ask.call_count == 1
        assert result["categories_processed"] == 1
        # 整类标记 organized
        items = {i["key"]: i["organized"] for i in await store.list_all(limit=200)}
        assert all(v == 1 for v in items.values())
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_llm_failure_does_not_mark(tmp_path):
    """LLM 调用抛异常 → 不标记 organized，下次重试。"""
    store = MemoryStore(str(tmp_path / "m.db"))
    await store.initialize()
    try:
        for i in range(5):
            await store.remember(f"内容{i}", key=f"k{i}", category="fact")

        memories = await store.list_all(limit=200)
        client = MagicMock()
        result = {"categories_processed": 0, "merged": 0, "deleted": 0, "kept": 0, "errors": []}
        ask = AsyncMock(side_effect=RuntimeError("LLM down"))
        with patch("src.services.memory_consolidation._ask_llm", ask):
            await _consolidate_store(
                store,
                client,
                datetime.datetime.now().astimezone(),
                "2026-08-05",
                result,
                memories=memories,
            )
        # 未标记
        items = {i["key"]: i["organized"] for i in await store.list_all(limit=200)}
        assert all(v == 0 for v in items.values())
        assert result["errors"]  # 记录了错误
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_llm_returns_none_does_not_mark(tmp_path):
    """LLM 返回非 JSON(plan=None) → 不标记，下次重试。"""
    store = MemoryStore(str(tmp_path / "m.db"))
    await store.initialize()
    try:
        for i in range(5):
            await store.remember(f"内容{i}", key=f"k{i}", category="fact")

        memories = await store.list_all(limit=200)
        client = MagicMock()
        result = {"categories_processed": 0, "merged": 0, "deleted": 0, "kept": 0, "errors": []}
        ask = AsyncMock(return_value=None)
        with patch("src.services.memory_consolidation._ask_llm", ask):
            await _consolidate_store(
                store,
                client,
                datetime.datetime.now().astimezone(),
                "2026-08-05",
                result,
                memories=memories,
            )
        items = {i["key"]: i["organized"] for i in await store.list_all(limit=200)}
        assert all(v == 0 for v in items.values())
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_op_failure_does_not_mark_category():
    """merge/delete 单项失败 → 不标记整类 organized，下次重审失败的合并/删除。"""
    store = MagicMock()  # 用 mock 便于让 forget 抛错
    memories = [
        {"key": f"k{i}", "content": f"内容{i}", "category": "fact", "updated_at": "2026-08-01", "organized": 0}
        for i in range(5)
    ]
    store.remember = AsyncMock(return_value='{"ok": true, "key": "k_new"}')
    store.forget = AsyncMock(side_effect=RuntimeError("db locked"))
    store.mark_organized = AsyncMock(return_value=0)

    result = {"categories_processed": 0, "merged": 0, "deleted": 0, "kept": 0, "errors": []}
    plan = _llm_plan(
        merge=[{"from_keys": ["k0", "k1"], "to_content": "合并", "to_category": "fact"}],
        delete=["k2"],
        keep=["k3", "k4"],
    )
    ask = AsyncMock(return_value=plan)
    with patch("src.services.memory_consolidation._ask_llm", ask):
        await _consolidate_store(
            store,
            MagicMock(),
            datetime.datetime.now().astimezone(),
            "2026-08-05",
            result,
            memories=memories,
        )
    assert result["categories_processed"] == 1
    store.mark_organized.assert_not_awaited()
