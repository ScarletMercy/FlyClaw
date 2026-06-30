"""Base class for memory stores with hybrid search (FTS5 + optional vector backend).

Provides all shared SQLite/FTS5 logic. Subclasses only implement vector-specific hooks.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from src.utils.tz import now_iso
from pathlib import Path
from typing import Optional

import aiosqlite

from src.utils.fts import sanitize_fts5_query

logger = logging.getLogger("flyclaw.memory.base")


class BaseMemoryStore(ABC):
    """Abstract base for memory stores with SQLite chunks + FTS5 + pluggable vector backend."""

    def __init__(self, db_path: str, dimensions: int = 1536, fts_tokenizer: str = "unicode61"):
        self.db_path = db_path
        self.dimensions = dimensions
        self.fts_tokenizer = fts_tokenizer
        self._conn: Optional[aiosqlite.Connection] = None

    # ── Lifecycle ───────────────────────────────────────────

    async def initialize(self) -> None:
        """Create SQLite tables, FTS5 index, then call vector backend init."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA synchronous=NORMAL")

        await self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL,
                chunk_index INTEGER NOT NULL DEFAULT 0,
                content TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);
        """)

        try:
            await self._conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts "
                f"USING fts5(path, content, tokenize='{self.fts_tokenizer}')"
            )
        except Exception as e:
            logger.warning("Failed to create FTS5 table: %s", e)

        await self._conn.commit()
        await self._init_vector_backend()

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None
        await self._close_vector_backend()

    # ── Abstract hooks (vector backend) ─────────────────────

    @abstractmethod
    async def _init_vector_backend(self) -> None:
        """Initialize vector-specific resources (called at end of initialize)."""

    async def _close_vector_backend(self) -> None:
        """Clean up vector resources. Default: no-op."""

    @abstractmethod
    def _has_vector_support(self) -> bool:
        """Return True if vector search is available."""

    @abstractmethod
    async def _vec_search(self, query_embedding: list[float], limit: int) -> list[dict]:
        """Search using vector similarity. Returns list of result dicts."""

    @abstractmethod
    async def _store_embeddings(self, chunk_ids: list[int], embeddings: list[list[float]]) -> None:
        """Store vector embeddings for the given chunk IDs."""

    @abstractmethod
    async def _delete_vectors(self, ids: list[int]) -> None:
        """Delete vectors for the given chunk IDs."""

    # ── Public API (shared) ─────────────────────────────────

    async def add_document(self, path: str, content: str, metadata: Optional[dict] = None) -> int:
        """Chunk content, store in SQLite + FTS5. Returns number of chunks added."""
        from src.memory.chunker import chunk_markdown

        chunks = chunk_markdown(content)
        if not chunks:
            return 0

        now = now_iso()
        meta_json = json.dumps(metadata) if metadata else None
        added = 0

        for chunk in chunks:
            cursor = await self._conn.execute(
                "INSERT INTO chunks (path, chunk_index, content, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
                (path, chunk["index"], chunk["text"], meta_json, now),
            )
            chunk_id = cursor.lastrowid
            added += 1

            try:
                await self._conn.execute(
                    "INSERT INTO chunks_fts(rowid, path, content) VALUES (?, ?, ?)",
                    (chunk_id, path, chunk["text"]),
                )
            except Exception as e:
                logger.warning("FTS5 insert failed for chunk %d: %s", chunk["index"], e)

        await self._conn.commit()
        return added

    async def add_embeddings(self, chunk_ids: list[int], embeddings: list[list[float]]) -> None:
        """Store vector embeddings via the subclass hook."""
        if not chunk_ids:
            return
        await self._store_embeddings(chunk_ids, embeddings)

    async def get_chunk_ids_for_path(self, path: str) -> list[int]:
        cursor = await self._conn.execute("SELECT id FROM chunks WHERE path = ? ORDER BY chunk_index", (path,))
        rows = await cursor.fetchall()
        return [row["id"] for row in rows]

    async def search(
        self,
        query_embedding: Optional[list[float]] = None,
        query_text: str = "",
        max_results: int = 6,
        vector_weight: float = 0.7,
        min_score: float = 0.35,
    ) -> list[dict]:
        """Hybrid search: FTS5 BM25 + optional vector similarity."""
        fts_results = await self._fts_search(query_text, limit=max(20, max_results * 3))

        if not query_embedding or not self._has_vector_support():
            return self._normalize_fts_scores(fts_results[:max_results], min_score)

        vec_results = await self._vec_search(query_embedding, limit=max(24, max_results * 4))

        return self._merge_results(
            fts_results,
            vec_results,
            vector_weight=vector_weight,
            max_results=max_results,
            min_score=min_score,
        )

    async def delete_document(self, path: str) -> int:
        """Delete all chunks for a document path. Returns count deleted."""
        cursor = await self._conn.execute("SELECT id FROM chunks WHERE path = ?", (path,))
        rows = await cursor.fetchall()
        ids = [r["id"] for r in rows]

        if ids:
            placeholders = ",".join("?" * len(ids))
            await self._conn.execute(f"DELETE FROM chunks WHERE id IN ({placeholders})", ids)
            try:
                await self._conn.execute(f"DELETE FROM chunks_fts WHERE rowid IN ({placeholders})", ids)
            except Exception:
                pass
            await self._delete_vectors(ids)
            await self._conn.commit()
        return len(ids)

    async def list_documents(self) -> list[dict]:
        """List all indexed documents with chunk counts."""
        cursor = await self._conn.execute(
            """
            SELECT path, COUNT(*) as chunk_count,
                   MIN(created_at) as first_chunk, MAX(created_at) as last_chunk
            FROM chunks GROUP BY path ORDER BY path
            """
        )
        rows = await cursor.fetchall()
        return [
            {
                "path": row["path"],
                "chunk_count": row["chunk_count"],
                "first_chunk": row["first_chunk"],
                "last_chunk": row["last_chunk"],
            }
            for row in rows
        ]

    # ── FTS5 search (shared) ────────────────────────────────

    async def _fts_search(self, query: str, limit: int = 20) -> list[dict]:
        if not query.strip():
            return []
        results = []
        fts_query = sanitize_fts5_query(query)
        try:
            cursor = await self._conn.execute(
                """
                SELECT c.id, c.path, c.chunk_index, c.content,
                       bm25(chunks_fts) AS score
                FROM chunks_fts f
                JOIN chunks c ON c.id = f.rowid
                WHERE chunks_fts MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (fts_query, limit),
            )
            rows = await cursor.fetchall()
            for row in rows:
                results.append(
                    {
                        "id": row["id"],
                        "path": row["path"],
                        "chunk_index": row["chunk_index"],
                        "content": row["content"],
                        "fts_score": row["score"],
                        "vec_score": 0.0,
                    }
                )
        except Exception as e:
            logger.warning("FTS5 search failed (query=%s): %s", fts_query, e)
        return results

    # ── Score utilities (shared) ────────────────────────────

    @staticmethod
    def _normalize_fts_scores(results: list[dict], min_score: float = 0.35) -> list[dict]:
        """Normalize FTS5 BM25 scores to 0-1 range and filter."""
        if not results:
            return []
        scores = [abs(r["fts_score"]) for r in results]
        s_min, s_max = min(scores), max(scores)

        if s_max == s_min:
            for r in results:
                r["score"] = 1.0
            return results

        s_range = s_max - s_min
        normalized = []
        for r in results:
            norm = (abs(r["fts_score"]) - s_min) / s_range
            if norm >= min_score:
                r["score"] = norm
                normalized.append(r)
        normalized.sort(key=lambda x: x["score"], reverse=True)
        return normalized

    def _merge_results(
        self,
        fts_results: list[dict],
        vec_results: list[dict],
        vector_weight: float = 0.7,
        max_results: int = 6,
        min_score: float = 0.35,
    ) -> list[dict]:
        """Merge FTS5 and vector results with weighted scoring."""
        fts_map: dict[int, dict] = {r["id"]: r for r in fts_results}
        combined: dict[int, dict] = {}

        for r in vec_results:
            cid = r["id"]
            combined[cid] = r.copy()
            if cid in fts_map:
                combined[cid]["fts_score"] = fts_map[cid]["fts_score"]

        for r in fts_results:
            if r["id"] not in combined:
                combined[r["id"]] = r.copy()

        # BM25 返回负值（越负 = 匹配越好），取 abs 后归一化
        fts_scores = [abs(r["fts_score"]) for r in combined.values() if r["fts_score"] != 0]
        vec_scores = [r["vec_score"] for r in combined.values() if r["vec_score"] != 0]

        fts_min = min(fts_scores) if fts_scores else 0
        fts_max = max(fts_scores) if fts_scores else 1
        vec_min = min(vec_scores) if vec_scores else 0
        vec_max = max(vec_scores) if vec_scores else 1
        fts_range = fts_max - fts_min if fts_max != fts_min else 1
        vec_range = vec_max - vec_min if vec_max != vec_min else 1

        for r in combined.values():
            if r["fts_score"]:
                fts_norm = (
                    1.0 if len(fts_scores) <= 1 or fts_max == fts_min else (abs(r["fts_score"]) - fts_min) / fts_range
                )
            else:
                fts_norm = 0

            if r["vec_score"]:
                vec_norm = 1.0 if len(vec_scores) <= 1 or vec_max == vec_min else (r["vec_score"] - vec_min) / vec_range
            else:
                vec_norm = 0

            r["score"] = vector_weight * vec_norm + (1 - vector_weight) * fts_norm

        results = [r for r in combined.values() if r["score"] >= min_score]
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:max_results]
