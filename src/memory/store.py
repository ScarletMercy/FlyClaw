"""SQLite-backed memory store with FTS5 (BM25) search.

Pure FTS5-only —— 向量搜索/存储/删除能力由 LanceMemoryStore 承担。
BaseMemoryStore 的四个向量钩子（_has_vector_support / _vec_search /
_store_embeddings / _delete_vectors）在此类沿用基类空实现
（FTS5-only store 不支持向量）；真实实现见 LanceMemoryStore。
"""

from __future__ import annotations

import logging

from src.memory.base import BaseMemoryStore

logger = logging.getLogger("flyclaw.memory.store")


class MemoryStore(BaseMemoryStore):
    """SQLite-backed memory store with FTS5 (BM25) search. FTS5-only."""

    async def _init_vector_backend(self) -> None:
        """FTS5-only：无向量后端，仅记日志。"""
        logger.info("Memory store initialized (FTS5-only): %s", self.db_path)
