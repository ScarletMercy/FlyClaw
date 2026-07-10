"""Tests for src/memory/base.py — BaseMemoryStore, FTS5 search, score normalization, result merging."""

from pathlib import Path

import pytest

from src.memory.base import BaseMemoryStore


# ── Concrete test implementation ───────────────────────────


class DummyMemoryStore(BaseMemoryStore):
    """Minimal concrete subclass for testing shared logic."""

    def __init__(self, db_path: str):
        self._vec_init_called = False
        self._vec_close_called = False
        super().__init__(db_path)

    async def _init_vector_backend(self):
        self._vec_init_called = True

    async def _close_vector_backend(self):
        self._vec_close_called = True


@pytest.fixture
async def store(tmp_path):
    db_path = str(tmp_path / "test_memory.db")
    s = DummyMemoryStore(db_path)
    await s.initialize()
    yield s
    await s.close()


# ── Lifecycle ──────────────────────────────────────────────


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_initialize_creates_db(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        s = DummyMemoryStore(db_path)
        await s.initialize()
        assert Path(db_path).exists()
        assert s._vec_init_called is True
        await s.close()

    @pytest.mark.asyncio
    async def test_close_cleans_up(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        s = DummyMemoryStore(db_path)
        await s.initialize()
        await s.close()
        assert s._conn is None
        assert s._vec_close_called is True


# ── add_document / search ──────────────────────────────────


class TestAddDocument:
    @pytest.mark.asyncio
    async def test_add_returns_count(self, store):
        count = await store.add_document("test.md", "# Hello\n\nSome content here")
        assert count >= 1

    @pytest.mark.asyncio
    async def test_add_empty_returns_zero(self, store):
        count = await store.add_document("empty.md", "")
        assert count == 0

    @pytest.mark.asyncio
    async def test_add_with_metadata(self, store):
        count = await store.add_document(
            "meta.md",
            "Some content",
            metadata={"source": "test", "tags": ["a", "b"]},
        )
        assert count >= 1


class TestDeleteDocument:
    @pytest.mark.asyncio
    async def test_delete_removes_chunks(self, store):
        await store.add_document("del.md", "Content to delete")
        deleted = await store.delete_document("del.md")
        assert deleted >= 1

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, store):
        deleted = await store.delete_document("nonexistent.md")
        assert deleted == 0


class TestListDocuments:
    @pytest.mark.asyncio
    async def test_lists_added_docs(self, store):
        await store.add_document("a.md", "Content A")
        await store.add_document("b.md", "Content B")
        docs = await store.list_documents()
        paths = [d["path"] for d in docs]
        assert "a.md" in paths
        assert "b.md" in paths

    @pytest.mark.asyncio
    async def test_empty_list(self, store):
        docs = await store.list_documents()
        assert docs == []


class TestGetChunkIds:
    @pytest.mark.asyncio
    async def test_returns_ids(self, store):
        await store.add_document("ids.md", "First paragraph\n\nSecond paragraph")
        ids = await store.get_chunk_ids_for_path("ids.md")
        assert len(ids) >= 1
        assert all(isinstance(i, int) for i in ids)


# ── Score normalization ────────────────────────────────────


class TestNormalizeFtsScores:
    def test_empty_list(self):
        result = BaseMemoryStore._normalize_fts_scores([])
        assert result == []

    def test_single_result_gets_score_1(self):
        results = [{"fts_score": -5.0, "id": 1}]
        out = BaseMemoryStore._normalize_fts_scores(results)
        assert len(out) == 1
        assert out[0]["score"] == 1.0

    def test_multiple_results_normalized(self):
        results = [
            {"fts_score": -10.0, "id": 1},
            {"fts_score": -2.0, "id": 2},
        ]
        out = BaseMemoryStore._normalize_fts_scores(results, min_score=0.0)
        assert len(out) == 2
        # Higher absolute score → higher normalized
        scores = [r["score"] for r in out]
        assert scores[0] > scores[1] or scores[0] == scores[1]

    def test_min_score_filter(self):
        results = [
            {"fts_score": -10.0, "id": 1},
            {"fts_score": -2.0, "id": 2},
        ]
        # Use high min_score to filter the weak result
        out = BaseMemoryStore._normalize_fts_scores(results, min_score=0.99)
        # Only the best match should survive (if any)
        assert all(r["score"] >= 0.99 for r in out)
