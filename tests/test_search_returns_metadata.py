"""Tests that search results surface metadata column."""

from __future__ import annotations

import pytest

from src.memory.store import MemoryStore


@pytest.fixture
async def store(tmp_path):
    s = MemoryStore(
        db_path=str(tmp_path / "t.db"),
        dimensions=4,
    )
    await s.initialize()
    yield s
    await s.close()


class TestSearchMetadata:
    @pytest.mark.asyncio
    async def test_fts_search_returns_metadata(self, store):
        meta = {"category": "preference", "updated_ts": 1700000000.0, "group_id": ""}
        # Note: content has a space so "深色主题" is its own FTS5 token (unicode61
        # treats consecutive CJK chars as one token).
        await store.add_document("kv:pref1", "我喜欢 深色主题", metadata=meta, chunk=False)
        results = await store._fts_search("深色主题", limit=5)
        assert len(results) == 1
        assert results[0]["metadata"] == meta

    @pytest.mark.asyncio
    async def test_fts_search_metadata_none_when_absent(self, store):
        await store.add_document("doc:no_meta", "plain content", chunk=False)
        results = await store._fts_search("plain content", limit=5)
        assert results[0]["metadata"] is None
