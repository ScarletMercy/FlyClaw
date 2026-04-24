"""Memory store with hybrid search (FTS5 + sqlite-vec).

Uses aiosqlite with WAL mode. Follows the same patterns as src/cron/store.py.
Falls back to FTS5-only search if sqlite-vec is not installed.
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import aiosqlite

logger = logging.getLogger("myclaw.memory.store")

# Try importing sqlite-vec; fallback to FTS5-only if unavailable
_HAS_SQLITE_VEC = False
try:
    import sqlite_vec  # noqa: F401
    _HAS_SQLITE_VEC = True
except ImportError:
    logger.info("sqlite-vec not installed; memory search will use FTS5-only mode")


class MemoryStore:
    """SQLite-backed memory store with FTS5 (BM25) and optional sqlite-vec (vector search)."""

    def __init__(self, db_path: str, dimensions: int = 1536, fts_tokenizer: str = "unicode61"):
        self.db_path = db_path
        self.dimensions = dimensions
        self.fts_tokenizer = fts_tokenizer
        self._conn: Optional[aiosqlite.Connection] = None

    async def initialize(self) -> None:
        """Create tables and indexes."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA synchronous=NORMAL")

        # Load sqlite-vec extension if available
        if _HAS_SQLITE_VEC:
            try:
                await self._conn.enable_load_extension(True)
                # Try loading vec0 extension
                import sqlite_vec
                # sqlite_vec registers via its own mechanism; try direct approach
                try:
                    await self._conn.load_extension("vec0")
                    logger.info("sqlite-vec extension loaded")
                except Exception:
                    # Fallback: try registering via sqlite3 API on raw connection
                    try:
                        raw = self._conn._conn  # aiosqlite stores raw conn as _conn
                        sqlite_vec.loadable_vector_extensions(raw)
                        logger.info("sqlite-vec loaded via loadable_vector_extensions")
                    except Exception:
                        logger.warning("sqlite-vec extension load failed; FTS5-only mode")
                await self._conn.enable_load_extension(False)
            except Exception as e:
                logger.warning("sqlite-vec init failed: %s", e)

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

        # FTS5 table
        try:
            await self._conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
                USING fts5(path, content, tokenize='{self.fts_tokenizer}')
            """)
        except Exception as e:
            logger.warning("Failed to create FTS5 table: %s", e)

        # sqlite-vec table (optional)
        if _HAS_SQLITE_VEC:
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
        logger.info(
            "Memory store initialized: %s (vec=%s)",
            self.db_path,
            _HAS_SQLITE_VEC,
        )

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def add_document(
        self,
        path: str,
        content: str,
        metadata: Optional[dict] = None,
    ) -> int:
        """Chunk content, compute embeddings, store in all tables.

        Returns the number of chunks added.
        """
        from src.memory.chunker import chunk_markdown

        chunks = chunk_markdown(content)
        if not chunks:
            return 0

        now = datetime.now(timezone.utc).isoformat()
        meta_json = json.dumps(metadata) if metadata else None
        added = 0

        for chunk in chunks:
            cursor = await self._conn.execute(
                "INSERT INTO chunks (path, chunk_index, content, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
                (path, chunk["index"], chunk["text"], meta_json, now),
            )
            chunk_id = cursor.lastrowid
            added += 1

            # FTS5 index (use explicit rowid to match chunks.id)
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
        """Add vector embeddings for chunks. Only works if sqlite-vec is available."""
        if not _HAS_SQLITE_VEC or not chunk_ids:
            return

        for chunk_id, embedding in zip(chunk_ids, embeddings):
            try:
                blob = self._vec_to_blob(embedding)
                await self._conn.execute(
                    "INSERT INTO chunks_vec (rowid, embedding) VALUES (?, ?)",
                    (chunk_id, blob),
                )
            except Exception as e:
                logger.warning("Failed to insert vector for chunk %d: %s", chunk_id, e)

        await self._conn.commit()

    async def get_chunk_ids_for_path(self, path: str) -> list[int]:
        """Get all chunk IDs for a document path."""
        cursor = await self._conn.execute(
            "SELECT id FROM chunks WHERE path = ? ORDER BY chunk_index",
            (path,),
        )
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
        """Hybrid search: FTS5 BM25 + sqlite-vec cosine similarity.

        If query_embedding is None or sqlite-vec is not available,
        falls back to FTS5-only search.
        """
        # FTS5 search
        fts_results = await self._fts_search(query_text, limit=max(20, max_results * 3))

        if not query_embedding or not _HAS_SQLITE_VEC:
            # FTS5-only mode
            return self._normalize_fts_scores(fts_results[:max_results], min_score)

        # Vector search
        vec_results = await self._vec_search(query_embedding, limit=max(24, max_results * 4))

        # Merge results
        return self._merge_results(
            fts_results,
            vec_results,
            vector_weight=vector_weight,
            max_results=max_results,
            min_score=min_score,
        )

    async def _fts_search(self, query: str, limit: int = 20) -> list[dict]:
        """Search using FTS5 BM25."""
        if not query.strip():
            return []
        results = []
        # FTS5 MATCH requires proper syntax: OR-separated terms or quoted phrases
        fts_query = self._format_fts_query(query)
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
                results.append({
                    "id": row["id"],
                    "path": row["path"],
                    "chunk_index": row["chunk_index"],
                    "content": row["content"],
                    "fts_score": row["score"],
                    "vec_score": 0.0,
                })
        except Exception as e:
            logger.warning("FTS5 search failed (query=%s): %s", fts_query, e)
        return results

    @staticmethod
    def _format_fts_query(query: str) -> str:
        """Format a natural language query for FTS5 MATCH syntax.

        Splits on spaces, joins with OR. Quotes are preserved as phrase matches.
        """
        import shlex
        try:
            parts = shlex.split(query)
        except ValueError:
            parts = query.split()

        if not parts:
            return query

        fts_terms = []
        for part in parts:
            # Strip special FTS5 characters
            cleaned = part.replace('"', '').replace("*", "").replace("(", "").replace(")", "")
            if cleaned:
                fts_terms.append(cleaned)

        return " OR ".join(fts_terms) if fts_terms else query

    async def _vec_search(self, query_embedding: list[float], limit: int = 24) -> list[dict]:
        """Search using sqlite-vec cosine distance."""
        results = []
        try:
            blob = self._vec_to_blob(query_embedding)
            cursor = await self._conn.execute(
                """
                SELECT c.id, c.path, c.chunk_index, c.content,
                       v.distance
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
                # Convert cosine distance to similarity score (1 - distance)
                vec_score = 1.0 - row["distance"]
                results.append({
                    "id": row["id"],
                    "path": row["path"],
                    "chunk_index": row["chunk_index"],
                    "content": row["content"],
                    "fts_score": 0.0,
                    "vec_score": vec_score,
                })
        except Exception as e:
            logger.warning("Vector search failed: %s", e)
        return results

    def _merge_results(
        self,
        fts_results: list[dict],
        vec_results: list[dict],
        vector_weight: float = 0.7,
        max_results: int = 6,
        min_score: float = 0.35,
    ) -> list[dict]:
        """Merge FTS5 and vector results with weighted scoring."""
        # Build lookup from FTS results
        fts_map: dict[int, dict] = {}
        for r in fts_results:
            fts_map[r["id"]] = r

        # Build combined scores
        combined: dict[int, dict] = {}
        for r in vec_results:
            cid = r["id"]
            combined[cid] = r.copy()
            if cid in fts_map:
                combined[cid]["fts_score"] = fts_map[cid]["fts_score"]

        # Add FTS-only results not in vector results
        for r in fts_results:
            if r["id"] not in combined:
                combined[r["id"]] = r.copy()

        # Normalize scores
        fts_scores = [r["fts_score"] for r in combined.values() if r["fts_score"] != 0]
        vec_scores = [r["vec_score"] for r in combined.values() if r["vec_score"] != 0]

        fts_min = min(fts_scores) if fts_scores else 0
        fts_max = max(fts_scores) if fts_scores else 1
        vec_min = min(vec_scores) if vec_scores else 0
        vec_max = max(vec_scores) if vec_scores else 1

        fts_range = fts_max - fts_min if fts_max != fts_min else 1
        vec_range = vec_max - vec_min if vec_max != vec_min else 1

        for cid, r in combined.items():
            fts_norm = (r["fts_score"] - fts_min) / fts_range if r["fts_score"] else 0
            vec_norm = (r["vec_score"] - vec_min) / vec_range if r["vec_score"] else 0
            r["score"] = vector_weight * vec_norm + (1 - vector_weight) * fts_norm

        # Filter, sort, and limit
        results = [r for r in combined.values() if r["score"] >= min_score]
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:max_results]

    @staticmethod
    def _normalize_fts_scores(results: list[dict], min_score: float = 0.35) -> list[dict]:
        """Normalize FTS5 BM25 scores to 0-1 range and filter.

        When there's only one result (or all scores are equal), assigns score 1.0
        since the match is relevant by definition (BM25 returned it).
        """
        if not results:
            return []
        scores = [abs(r["fts_score"]) for r in results]
        s_min, s_max = min(scores), max(scores)

        # Single result or all equal: all matches are equally relevant
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
            if _HAS_SQLITE_VEC:
                try:
                    await self._conn.execute(f"DELETE FROM chunks_vec WHERE rowid IN ({placeholders})", ids)
                except Exception:
                    pass
            await self._conn.commit()
        return len(ids)

    async def list_documents(self) -> list[dict]:
        """List all indexed documents with chunk counts."""
        cursor = await self._conn.execute(
            """
            SELECT path, COUNT(*) as chunk_count, MIN(created_at) as first_chunk, MAX(created_at) as last_chunk
            FROM chunks
            GROUP BY path
            ORDER BY path
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

    @staticmethod
    def _vec_to_blob(vec: list[float]) -> bytes:
        """Convert float vector to sqlite-vec blob format."""
        import struct
        return struct.pack(f"<{len(vec)}f", *vec)
