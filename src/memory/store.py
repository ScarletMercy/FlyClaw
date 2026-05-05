"""Memory store with hybrid search (FTS5 + sqlite-vec).

Falls back to FTS5-only search if sqlite-vec is not installed.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.memory.base import BaseMemoryStore

logger = logging.getLogger("myclaw.memory.store")

_HAS_SQLITE_VEC = False
try:
    import sqlite_vec  # noqa: F401

    _HAS_SQLITE_VEC = True
except ImportError:
    logger.info("sqlite-vec not installed; memory search will use FTS5-only mode")


class MemoryStore(BaseMemoryStore):
    """SQLite-backed memory store with FTS5 (BM25) and optional sqlite-vec (vector search)."""

    # ── Vector backend hooks ────────────────────────────────

    async def _init_vector_backend(self) -> None:
        if not _HAS_SQLITE_VEC:
            logger.info("Memory store initialized (FTS5-only): %s", self.db_path)
            return

        try:
            await self._conn.enable_load_extension(True)
            import sqlite_vec

            try:
                await self._conn.load_extension("vec0")
                logger.info("sqlite-vec extension loaded")
            except Exception:
                try:
                    raw = self._conn._conn
                    sqlite_vec.loadable_vector_extensions(raw)
                    logger.info("sqlite-vec loaded via loadable_vector_extensions")
                except Exception:
                    logger.warning("sqlite-vec extension load failed; FTS5-only mode")
            await self._conn.enable_load_extension(False)
        except Exception as e:
            logger.warning("sqlite-vec init failed: %s", e)

        # Create vec0 table
        try:
            await self._conn.execute("SELECT * FROM chunks_vec LIMIT 0")
        except Exception:
            try:
                await self._conn.execute(
                    f"CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(embedding float[{self.dimensions}])"
                )
            except Exception as e:
                logger.warning("Failed to create vec0 table: %s", e)

        await self._conn.commit()
        logger.info("Memory store initialized: %s (vec=%s)", self.db_path, _HAS_SQLITE_VEC)

    async def _close_vector_backend(self) -> None:
        pass

    def _has_vector_support(self) -> bool:
        return _HAS_SQLITE_VEC

    async def _vec_search(self, query_embedding: list[float], limit: int = 24) -> list[dict]:
        results = []
        try:
            blob = _vec_to_blob(query_embedding)
            cursor = await self._conn.execute(
                """
                SELECT c.id, c.path, c.chunk_index, c.content, v.distance
                FROM chunks_vec v
                JOIN chunks c ON c.id = v.rowid
                WHERE v.embedding MATCH ?
                ORDER BY v.distance
                LIMIT ?
                """,
                (blob, limit),
            )
            rows = await cursor.fetchall()
            for row in rows:
                results.append(
                    {
                        "id": row["id"],
                        "path": row["path"],
                        "chunk_index": row["chunk_index"],
                        "content": row["content"],
                        "fts_score": 0.0,
                        "vec_score": 1.0 - row["distance"],
                    }
                )
        except Exception as e:
            logger.warning("Vector search failed: %s", e)
        return results

    async def _store_embeddings(self, chunk_ids: list[int], embeddings: list[list[float]]) -> None:
        if not _HAS_SQLITE_VEC:
            return
        for chunk_id, embedding in zip(chunk_ids, embeddings):
            try:
                blob = _vec_to_blob(embedding)
                await self._conn.execute(
                    "INSERT INTO chunks_vec (rowid, embedding) VALUES (?, ?)",
                    (chunk_id, blob),
                )
            except Exception as e:
                logger.warning("Failed to insert vector for chunk %d: %s", chunk_id, e)
        await self._conn.commit()

    async def _delete_vectors(self, ids: list[int]) -> None:
        if not _HAS_SQLITE_VEC:
            return
        try:
            placeholders = ",".join("?" * len(ids))
            await self._conn.execute(f"DELETE FROM chunks_vec WHERE rowid IN ({placeholders})", ids)
        except Exception:
            pass


def _vec_to_blob(vec: list[float]) -> bytes:
    """Convert float vector to sqlite-vec blob format."""
    import struct

    return struct.pack(f"<{len(vec)}f", *vec)
