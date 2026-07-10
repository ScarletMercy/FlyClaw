"""Memory store with hybrid search using LanceDB for vectors + SQLite FTS5 for text.

Requires: pip install lancedb pyarrow
"""

from __future__ import annotations

import asyncio
import json
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
        lancedb_uri: str | None = None,
    ):
        super().__init__(db_path, dimensions, fts_tokenizer)
        if lancedb_uri is None:
            from src.instance import data_dir

            lancedb_uri = str(data_dir() / "memory_lancedb")
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
                # 元数据随向量存储（JSON 字符串，与 SQLite chunks.metadata 同格式），
                # 便于溯源：category/updated_ts 跟向量一起，不依赖 SQLite join
                pa.field("metadata", pa.string()),
                # group_id 独立列：向量检索按群键值过滤（.where），不靠 path 前缀
                pa.field("group_id", pa.string()),
            ]
        )

        try:
            self._lance_table = self._lance_db.open_table("memory_vectors")
            # 旧表 schema 迁移：缺列则补（lancedb add_columns，实测 0.30.2 可用）
            existing_cols = {f.name for f in self._lance_table.schema}
            for col in ("metadata", "group_id"):
                if col not in existing_cols:
                    try:
                        self._lance_table.add_columns({col: "''"})
                        logger.info("LanceDB schema migrated: added %s column", col)
                    except Exception as e:
                        logger.warning("LanceDB schema migration failed (%s column): %s", col, e)
        except Exception:
            self._lance_table = self._lance_db.create_table("memory_vectors", schema=schema)

        logger.info("LanceDB memory store initialized: sqlite=%s, lance=%s", self.db_path, self.lancedb_uri)

    def _has_vector_support(self) -> bool:
        return self._lance_table is not None

    async def _vec_search(
        self, query_embedding: list[float], limit: int = 24, group_id: Optional[str] = None
    ) -> list[dict]:
        if self._lance_table is None:
            return []
        try:
            lance_rows = await asyncio.to_thread(self._sync_lance_search, query_embedding, limit, group_id)

            if not lance_rows:
                return []

            # 只从 SQLite 捞 content + path（metadata 现在随向量存在 LanceDB）
            chunk_ids = [int(r["id"]) for r in lance_rows]
            placeholders = ",".join("?" * len(chunk_ids))
            cursor = await self._conn.execute(
                f"SELECT id, path, chunk_index, content FROM chunks WHERE id IN ({placeholders})",
                chunk_ids,
            )
            chunk_map = {row["id"]: row for row in await cursor.fetchall()}

            results = []
            for r in lance_rows:
                chunk_id = int(r["id"])
                chunk = chunk_map.get(chunk_id)
                if chunk is None:
                    continue
                # metadata 直接从 LanceDB 行取（溯源信息跟向量一起存）
                meta = None
                meta_str = r.get("metadata")
                if meta_str:
                    try:
                        meta = json.loads(meta_str)
                    except (json.JSONDecodeError, TypeError):
                        meta = None
                results.append(
                    {
                        "id": chunk_id,
                        "path": chunk["path"],
                        "chunk_index": chunk["chunk_index"],
                        "content": chunk["content"],
                        "metadata": meta,
                        "fts_score": 0.0,
                        "vec_score": 1.0 - float(r.get("_distance", 2.0)) / 2.0,  # L2²→余弦: cos = 1 - L2²/2
                    }
                )
            return results
        except Exception as e:
            logger.warning("LanceDB vector search failed: %s", e)
            return []

    def _sync_lance_search(self, query_embedding: list[float], limit: int, group_id: Optional[str] = None):
        """同步执行 LanceDB 搜索，由 asyncio.to_thread 调用。

        用 to_list() 而非 to_pandas()——避免 pandas 依赖。
        group_id 非空时按群键值过滤（.where），不靠 path 前缀。
        """
        query_vec = np.array(query_embedding, dtype=np.float32)
        search = self._lance_table.search(query_vec).limit(limit)
        if group_id is not None:
            escaped = group_id.replace("'", "''")
            search = search.where(f"group_id = '{escaped}'")
        return search.to_list()

    async def _store_embeddings(self, chunk_ids: list[int], embeddings: list[list[float]]) -> None:
        if self._lance_table is None:
            return
        # 从 SQLite 捞这些 chunk 的 metadata JSON + group_id，随向量一起写入 LanceDB
        placeholders = ",".join("?" * len(chunk_ids))
        cursor = await self._conn.execute(
            f"SELECT id, metadata, group_id FROM chunks WHERE id IN ({placeholders})",
            chunk_ids,
        )
        rows = {row["id"]: row for row in await cursor.fetchall()}

        ids = []
        vecs = []
        metas = []
        gids = []
        for cid, emb in zip(chunk_ids, embeddings):
            row = rows.get(cid)
            ids.append(cid)
            vecs.append(np.array(emb, dtype=np.float32))
            metas.append(row["metadata"] if row and row["metadata"] else "{}")
            gids.append(row["group_id"] if row else "")

        table = pa.table(
            {
                "id": pa.array(ids, type=pa.int64()),
                "vector": pa.array(vecs, type=pa.list_(pa.float32(), self.dimensions)),
                "metadata": pa.array(metas, type=pa.string()),
                "group_id": pa.array(gids, type=pa.string()),
            }
        )
        await asyncio.to_thread(self._lance_table.add, table)

    async def _delete_vectors(self, ids: list[int]) -> None:
        if self._lance_table is None:
            return
        try:
            id_list = ",".join(str(i) for i in ids)
            await asyncio.to_thread(lambda: self._lance_table.delete(f"id IN ({id_list})"))
        except Exception as e:
            logger.warning("Failed to delete vectors from LanceDB: %s", e)
