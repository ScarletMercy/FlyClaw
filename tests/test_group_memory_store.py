"""Tests for GroupMemoryStore — group_id isolation, FTS composite key, dedup scope."""

import json

import pytest

from src.tools.memory_tools import GroupMemoryStore


async def _make_store(db_path: str) -> GroupMemoryStore:
    s = GroupMemoryStore(db_path)
    await s.initialize()
    return s


class TestGroupMemoryStoreCRUD:
    @pytest.mark.asyncio
    async def test_same_key_different_groups_not_overwritten(self, tmp_path):
        store = await _make_store(str(tmp_path / "g.db"))
        try:
            await store.remember("Alice phone 123", key="alice", group_id="A")
            await store.remember("Alice phone 456", key="alice", group_id="B")

            ra = json.loads(await store.recall("alice", group_id="A"))
            rb = json.loads(await store.recall("alice", group_id="B"))
            assert ra["content"] == "Alice phone 123"
            assert rb["content"] == "Alice phone 456"
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_recall_wrong_group_not_found(self, tmp_path):
        store = await _make_store(str(tmp_path / "g.db"))
        try:
            await store.remember("secret data", key="k1", group_id="A")
            result = json.loads(await store.recall("k1", group_id="B"))
            assert "error" in result
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_upsert_same_group_overwrites(self, tmp_path):
        store = await _make_store(str(tmp_path / "g.db"))
        try:
            await store.remember("old value", key="k", group_id="A")
            await store.remember("new value", key="k", group_id="A")
            result = json.loads(await store.recall("k", group_id="A"))
            assert result["content"] == "new value"
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_forget_isolated_by_group(self, tmp_path):
        store = await _make_store(str(tmp_path / "g.db"))
        try:
            await store.remember("data A", key="shared", group_id="A")
            await store.remember("data B", key="shared", group_id="B")

            await store.forget("shared", group_id="A")

            ra = json.loads(await store.recall("shared", group_id="A"))
            rb = json.loads(await store.recall("shared", group_id="B"))
            assert "error" in ra
            assert rb["content"] == "data B"
        finally:
            await store.close()


class TestGroupMemoryStoreListAll:
    @pytest.mark.asyncio
    async def test_list_all_filtered_by_group(self, tmp_path):
        store = await _make_store(str(tmp_path / "g.db"))
        try:
            await store.remember("A1", key="a1", group_id="A")
            await store.remember("A2", key="a2", group_id="A")
            await store.remember("B1", key="b1", group_id="B")

            a_items = await store.list_all(group_id="A")
            b_items = await store.list_all(group_id="B")
            assert {i["key"] for i in a_items} == {"a1", "a2"}
            assert {i["key"] for i in b_items} == {"b1"}
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_list_all_no_filter_returns_all_groups(self, tmp_path):
        store = await _make_store(str(tmp_path / "g.db"))
        try:
            await store.remember("A1", key="a1", group_id="A")
            await store.remember("B1", key="b1", group_id="B")

            all_items = await store.list_all(group_id=None)
            assert len(all_items) == 2
        finally:
            await store.close()


class TestGroupMemoryStoreDedup:
    @pytest.mark.asyncio
    async def test_dedup_within_same_group_only(self, tmp_path):
        store = await _make_store(str(tmp_path / "g.db"))
        try:
            r1 = json.loads(await store.remember("same content", group_id="A"))
            r2 = json.loads(await store.remember("same content", group_id="A"))
            r3 = json.loads(await store.remember("same content", group_id="B"))

            assert r1.get("dedup") is not True
            assert r2.get("dedup") is True
            assert r3.get("dedup") is not True
        finally:
            await store.close()


class TestGroupMemoryStoreFTS:
    @pytest.mark.asyncio
    async def test_fts_search_isolated_by_group(self, tmp_path):
        store = await _make_store(str(tmp_path / "g.db"))
        try:
            if not store._fts_available:
                pytest.skip("FTS5 not available")

            await store.remember("Python programming tips", key="tips", group_id="A")
            await store.remember("Python cooking recipes", key="recipes", group_id="B")

            a_results = await store.list_all(query="programming", group_id="A")
            b_results = await store.list_all(query="cooking", group_id="B")
            a_cooking = await store.list_all(query="cooking", group_id="A")

            assert len(a_results) == 1
            assert a_results[0]["content"] == "Python programming tips"
            assert len(b_results) == 1
            assert b_results[0]["content"] == "Python cooking recipes"
            assert len(a_cooking) == 0
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_fts_same_key_different_content(self, tmp_path):
        """同 key 不同群不同 content，FTS 复合键确保不串。"""
        store = await _make_store(str(tmp_path / "g.db"))
        try:
            if not store._fts_available:
                pytest.skip("FTS5 not available")

            await store.remember("Python 编程技巧", key="python", group_id="A")
            await store.remember("Python 烹饪食谱", key="python", group_id="B")

            results_a = await store.list_all(query="编程技巧", group_id="A")
            results_b = await store.list_all(query="烹饪食谱", group_id="B")

            assert len(results_a) == 1
            assert results_a[0]["content"] == "Python 编程技巧"
            assert len(results_b) == 1
            assert results_b[0]["content"] == "Python 烹饪食谱"
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_fts_rebuild_uses_composite_key(self, tmp_path):
        """FTS 重建时使用复合键 group_id:key。"""
        db_path = str(tmp_path / "g.db")
        store = GroupMemoryStore(db_path)
        await store.initialize()
        try:
            if not store._fts_available:
                pytest.skip("FTS5 not available")

            await store.remember("Python 编程技巧", key="python", group_id="A")
            await store.remember("Python 烹饪食谱", key="python", group_id="B")

            await store._conn.execute("DROP TABLE IF EXISTS memories_fts")
            store._fts_available = False
            await store._ensure_fts()

            results_a = await store.list_all(query="编程技巧", group_id="A")
            results_b = await store.list_all(query="烹饪食谱", group_id="B")
            assert len(results_a) == 1
            assert results_a[0]["content"] == "Python 编程技巧"
            assert len(results_b) == 1
            assert results_b[0]["content"] == "Python 烹饪食谱"
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_fts_forget_cleans_composite_key(self, tmp_path):
        """forget 后 FTS 中该群的复合键被清除，不影响其他群。"""
        store = await _make_store(str(tmp_path / "g.db"))
        try:
            if not store._fts_available:
                pytest.skip("FTS5 not available")

            await store.remember("Python 编程技巧", key="python", group_id="A")
            await store.remember("Python 烹饪食谱", key="python", group_id="B")

            await store.forget("python", group_id="A")

            a_results = await store.list_all(query="编程技巧", group_id="A")
            b_results = await store.list_all(query="烹饪食谱", group_id="B")
            assert len(a_results) == 0
            assert len(b_results) == 1
        finally:
            await store.close()
