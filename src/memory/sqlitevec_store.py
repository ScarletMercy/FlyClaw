"""Memory store with sqlite-vec for vectors + SQLite FTS5 for text.

默认向量后端。vec0 虚表建在与 chunks 同一个 SQLite 库里(不另起目录/文件),
metadata/content/path 经 JOIN chunks 取得--同库免费,不必随向量冗余存储。

Requires: pip install sqlite-vec
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import sqlite_vec

from src.memory.base import BaseMemoryStore

logger = logging.getLogger("flyclaw.memory.sqlitevec_store")

_HAS_SQLITE_VEC = True  # sqlite-vec 是核心依赖;import 失败说明环境异常


class SqliteVecMemoryStore(BaseMemoryStore):
    """Memory store backed by sqlite-vec (vectors) + SQLite (chunks + FTS5)."""

    def __init__(
        self,
        db_path: str,
        dimensions: int = 1536,
        fts_tokenizer: str = "unicode61",
        vec_table: str = "memory_vec",
    ):
        super().__init__(db_path, dimensions, fts_tokenizer)
        self.vec_table = vec_table
        self._vec_ready = False

    # ── Vector backend hooks ────────────────────────────────

    async def _init_vector_backend(self) -> None:
        if not _HAS_SQLITE_VEC:
            raise ImportError("sqlite-vec is required for backend='sqlite_vec'")

        # sqlite-vec 扩展加载进 chunks 所在的同一连接(vec0 虚表与之同库)
        await self._conn.enable_load_extension(True)
        await self._conn.execute("SELECT load_extension(?)", (sqlite_vec.loadable_path(),))

        await self._conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {self.vec_table} "
            f"USING vec0(embedding FLOAT[{self.dimensions}], group_id TEXT)"
        )
        await self._conn.commit()
        self._vec_ready = True

        logger.info(
            "sqlite-vec memory store initialized: sqlite=%s, vec_table=%s, dims=%d",
            self.db_path,
            self.vec_table,
            self.dimensions,
        )

    def _has_vector_support(self) -> bool:
        return self._vec_ready

    async def _vec_search(
        self, query_embedding: list[float], limit: int = 24, group_id: Optional[str] = None
    ) -> list[dict]:
        if not self._vec_ready:
            return []
        try:
            # KNN:embedding MATCH + k。group_id 非空时按 aux 列 pre-filter(实测与 LanceDB .where() 同语义)。
            # distance 为 L2(非 L2²),单位向量下 cos = 1 - L2²/2 -> vec_score = 1 - distance²/2。
            sql = (
                f"SELECT v.rowid AS id, v.distance, c.path, c.chunk_index, c.content, c.metadata "
                f"FROM {self.vec_table} v JOIN chunks c ON c.id = v.rowid "
                f"WHERE v.embedding MATCH ? AND k = ?"
            )
            params: list = [json.dumps(query_embedding), limit]
            if group_id is not None:
                sql += " AND v.group_id = ?"
                params.append(group_id)
            sql += " ORDER BY v.distance"

            cursor = await self._conn.execute(sql, params)
            rows = await cursor.fetchall()

            results = []
            for row in rows:
                meta = None
                if row["metadata"]:
                    try:
                        meta = json.loads(row["metadata"])
                    except (json.JSONDecodeError, TypeError):
                        meta = None
                distance = float(row["distance"])
                results.append(
                    {
                        "id": int(row["id"]),
                        "path": row["path"],
                        "chunk_index": row["chunk_index"],
                        "content": row["content"],
                        "metadata": meta,
                        "fts_score": 0.0,
                        "vec_score": 1.0 - (distance * distance) / 2.0,  # L2 -> 余弦(单位向量)
                    }
                )
            return results
        except Exception as e:
            logger.warning("sqlite-vec vector search failed: %s", e)
            return []

    async def _store_embeddings(self, chunk_ids: list[int], embeddings: list[list[float]]) -> None:
        if not self._vec_ready:
            return
        # 从 chunks 捞 group_id(group_id 是 vec0 aux 列,KNN 过滤要用;metadata 不入库,JOIN 即得)
        placeholders = ",".join("?" * len(chunk_ids))
        cursor = await self._conn.execute(f"SELECT id, group_id FROM chunks WHERE id IN ({placeholders})", chunk_ids)
        rows = {row["id"]: row for row in await cursor.fetchall()}

        for cid, emb in zip(chunk_ids, embeddings):
            row = rows.get(cid)
            gid = row["group_id"] if row else ""
            # vec0 不支持 INSERT OR REPLACE(抛 UNIQUE);先删后插实现 upsert,重新 embed 同 id 不炸
            await self._conn.execute(f"DELETE FROM {self.vec_table} WHERE rowid = ?", (cid,))
            await self._conn.execute(
                f"INSERT INTO {self.vec_table}(rowid, embedding, group_id) VALUES (?, ?, ?)",
                (cid, json.dumps(emb), gid),
            )
        await self._conn.commit()

    async def _delete_vectors(self, ids: list[int]) -> None:
        if not self._vec_ready or not ids:
            return
        try:
            placeholders = ",".join("?" * len(ids))
            await self._conn.execute(f"DELETE FROM {self.vec_table} WHERE rowid IN ({placeholders})", ids)
            await self._conn.commit()
        except Exception as e:
            logger.warning("Failed to delete vectors from sqlite-vec: %s", e)
