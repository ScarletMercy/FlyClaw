"""Tests for memory store (SQLite + FTS5)."""

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def store(tmp_path):
    """Create and initialize a MemoryStore, closing after test."""
    from src.memory.store import MemoryStore

    s = MemoryStore(db_path=str(tmp_path / "memory.db"))
    asyncio.run(s.initialize())
    yield s
    asyncio.run(s.close())


def _run(coro):
    """Run an async coroutine synchronously for testing."""
    return asyncio.run(coro)


class TestMemoryStore:
    def test_add_and_search(self, store):
        _run(store.add_document("doc1", "Python is a programming language. It is versatile.", {"source": "test"}))
        results = _run(store.search(query_text="Python programming"))
        assert len(results) > 0
        assert "Python" in results[0]["content"]

    def test_add_returns_chunk_count(self, store):
        long_text = "\n\n".join([f"Paragraph {i} about topic {i}." for i in range(20)])
        count = _run(store.add_document("doc1", long_text))
        assert count >= 1

    def test_search_empty_query(self, store):
        _run(store.add_document("doc1", "Some content here"))
        results = _run(store.search(query_text=""))
        assert results == []

    def test_search_no_results(self, store):
        results = _run(store.search(query_text="nonexistent_topic_xyz"))
        assert results == []

    def test_delete_document(self, store):
        _run(store.add_document("doc1", "Unique content about quantum computing."))
        _run(store.delete_document("doc1"))
        results = _run(store.search(query_text="quantum computing"))
        assert results == []

    def test_delete_nonexistent(self, store):
        count = _run(store.delete_document("no_such_doc"))
        assert count == 0

    def test_list_documents(self, store):
        _run(store.add_document("doc1", "Content about Python."))
        _run(store.add_document("doc2", "Content about JavaScript."))
        docs = _run(store.list_documents())
        assert len(docs) == 2
        paths = [d["path"] for d in docs]
        assert "doc1" in paths
        assert "doc2" in paths

    def test_fts_query_formatting(self):
        from src.memory.store import MemoryStore

        assert "OR" in MemoryStore._format_fts_query("hello world")
        assert "test" in MemoryStore._format_fts_query("test")
        # Special chars stripped
        q = MemoryStore._format_fts_query('hello "world" * (test)')
        assert '"' not in q
        assert "*" not in q

    def test_fts_query_only_special_chars_returns_empty(self):
        from src.memory.store import MemoryStore

        assert MemoryStore._format_fts_query('*"') == '""'
        assert MemoryStore._format_fts_query("(") == '""'
        assert MemoryStore._format_fts_query('()""**') == '""'

    def test_fts_query_whitespace_only(self):
        from src.memory.store import MemoryStore

        assert MemoryStore._format_fts_query("   ") == '""'

    def test_overwrite_document(self, store):
        _run(store.add_document("doc1", "Original content about algorithms."))
        _run(store.add_document("doc1", "Updated content about data structures."))
        results = _run(store.search(query_text="data structures"))
        assert len(results) > 0
        # Should find the updated content
        found = any("data structures" in r["content"] for r in results)
        assert found

    def test_normalize_fts_scores_single_result(self):
        from src.memory.store import MemoryStore

        results = [{"fts_score": -2.5, "id": 1}]
        norm = MemoryStore._normalize_fts_scores(results, min_score=0.1)
        assert len(norm) == 1
        assert norm[0]["score"] == 1.0

    def test_vec_to_blob(self):
        from src.memory.store import _vec_to_blob

        vec = [1.0, 2.0, 3.0]
        blob = _vec_to_blob(vec)
        assert len(blob) == 12  # 3 floats * 4 bytes

    def test_get_chunk_ids_for_path(self, store):
        _run(store.add_document("doc1", "Paragraph one about testing.\n\nParagraph two about coverage."))
        ids = _run(store.get_chunk_ids_for_path("doc1"))
        assert len(ids) >= 1
        assert all(isinstance(i, int) for i in ids)


class TestChunker:
    def test_empty_text(self):
        from src.memory.chunker import chunk_markdown

        assert chunk_markdown("") == []
        assert chunk_markdown("   \n\n  ") == []

    def test_single_paragraph(self):
        from src.memory.chunker import chunk_markdown

        chunks = chunk_markdown("Hello world.")
        assert len(chunks) == 1
        assert chunks[0]["text"] == "Hello world."
        assert chunks[0]["index"] == 0

    def test_multiple_paragraphs(self):
        from src.memory.chunker import chunk_markdown

        text = "\n\n".join([f"Paragraph {i} with some content." for i in range(5)])
        chunks = chunk_markdown(text)
        assert len(chunks) >= 1
        # All chunks should have index
        for i, chunk in enumerate(chunks):
            assert chunk["index"] == i

    def test_chunk_overlap(self):
        from src.memory.chunker import chunk_markdown

        # Create text that will produce multiple chunks
        paragraphs = [
            f"This is paragraph number {i} about topic {i}. It contains enough text to be meaningful."
            for i in range(50)
        ]
        text = "\n\n".join(paragraphs)
        chunks = chunk_markdown(text, chunk_tokens=40, overlap_tokens=10)
        assert len(chunks) > 1

    def test_custom_chunk_size(self):
        from src.memory.chunker import chunk_markdown

        text = "\n\n".join([f"Paragraph {i}." for i in range(100)])
        small = chunk_markdown(text, chunk_tokens=20)
        big = chunk_markdown(text, chunk_tokens=200)
        assert len(small) >= len(big)
