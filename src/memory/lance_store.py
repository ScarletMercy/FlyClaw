"""Memory store with hybrid search using LanceDB for vectors + SQLite FTS5 for text.

Requires: pip install lancedb pyarrow
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

from src.memory.base import BaseMemoryStore

logger = logging.getLogger("flyclaw.memory.lance_store")

_HAS_LANCEDB = False
try:
    import lancedb
    import numpy as np
    import pyarrow as pa

    _HAS_LANCEDB = True
except ImportError:
    pass


class LanceMemoryStore(BaseMemoryStore):
    """Memory store backed by LanceDB (vectors) + SQLite (chunks + FTS5)."""

    def __init__(
        self,
        db_path: str,
        dimensions: int = 1536,
        fts_tokenizer: str = "unicode61",
        lancedb_uri: str = "~/.flyclaw/data/memory_lancedb",
    ):
        super().__init__(db_path, dimensions, fts_tokenizer)
        self.lancedb_uri = str(Path(lancedb_uri).expanduser().resolve())
        self._lance_db = None
        self._lance_table = None

    # ── Vector backend hooks ────────────────────────────────

    async def _init_vector_backend(self) -> None:
        if not _HAS_LANCEDB:
            raise ImportError("lancedb and pyarrow are required for backend='lancedb'")

        Path(self.lancedb_uri).parent.mkdir(parents=True, exist_ok=True)
        self._lance_db = lancedb.connect(self.lancedb_uri)

        schema = pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("vector", pa.list_(pa.float32(), self.dimensions)),
            ]
        )

        try:
            self._lance_table = self._lance_db.open_table("memory_vectors")
        except Exception:
            self._lance_table = self._lance_db.create_table("memory_vectors", schema=schema)

        logger.info("LanceDB memory store initialized: sqlite=%s, lance=%s", self.db_path, self.lancedb_uri)

    def _has_vector_support(self) -> bool:
        return self._lance_table is not None

    async def _vec_search(self, query_embedding: list[float], limit: int = 24) -> list[dict]:
        if self._lance_table is None:
            return []
        results = []
        try:
            query_vec = np.array(query_embedding, dtype=np.float32)
            lance_df = self._lance_table.search(query_vec).limit(limit).to_pandas()

            if lance_df.empty:
                return []

            # Batch-fetch chunks from SQLite
            chunk_ids = [int(row["id"]) for _, row in lance_df.iterrows()]
            placeholders = ",".join("?" * len(chunk_ids))
            cursor = await self._conn.execute(
                f"SELECT id, path, chunk_index, content FROM chunks WHERE id IN ({placeholders})",
                chunk_ids,
            )
            chunk_map = {row["id"]: row for row in await cursor.fetchall()}

            for _, row in lance_df.iterrows():
                chunk_id = int(row["id"])
                chunk = chunk_map.get(chunk_id)
                if chunk is None:
                    continue
                results.append(
                    {
                        "id": chunk_id,
                        "path": chunk["path"],
                        "chunk_index": chunk["chunk_index"],
                        "content": chunk["content"],
                        "fts_score": 0.0,
                        "vec_score": 1.0 - float(row.get("_distance", 1.0)),
                    }
                )
        except Exception as e:
            logger.warning("LanceDB vector search failed: %s", e)
        return results

    async def _store_embeddings(self, chunk_ids: list[int], embeddings: list[list[float]]) -> None:
        if self._lance_table is None:
            return
        ids = pa.array(chunk_ids, type=pa.int64())
        vecs = pa.array(
            [np.array(v, dtype=np.float32) for v in embeddings],
            type=pa.list_(pa.float32(), self.dimensions),
        )
        self._lance_table.add(pa.table({"id": ids, "vector": vecs}))

    async def _delete_vectors(self, ids: list[int]) -> None:
        if self._lance_table is None:
            return
        try:
            id_list = ",".join(str(i) for i in ids)
            await asyncio.to_thread(lambda: self._lance_table.delete(f"id IN ({id_list})"))
        except Exception as e:
            logger.warning("Failed to delete vectors from LanceDB: %s", e)
