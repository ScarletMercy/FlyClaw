"""Tests for add_document(chunk=False) — skip chunk_markdown, one chunk per call."""

from __future__ import annotations

import pytest

from src.memory.store import MemoryStore


@pytest.fixture
async def store(tmp_path):
    s = MemoryStore(
        db_path=str(tmp_path / "test.db"),
        dimensions=8,
    )
    await s.initialize()
    yield s
    await s.close()


class TestAddDocumentNoChunk:
    @pytest.mark.asyncio
    async def test_chunk_false_inserts_single_chunk(self, store):
        content = "这是一条很长的 KV 记忆，但不会被切块，整条作为一个 chunk 插入。" * 3
        added = await store.add_document("kv:test_key", content, chunk=False)
        assert added == 1
        ids = await store.get_chunk_ids_for_path("kv:test_key")
        assert len(ids) == 1

    @pytest.mark.asyncio
    async def test_chunk_true_default_behaves_like_before(self, store):
        content = "短内容"
        added = await store.add_document("doc:path", content)
        assert added >= 1  # chunk_markdown 至少返回 1
