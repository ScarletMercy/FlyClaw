"""Vector backend factory.

收口向量后端选择:默认 sqlite-vec,lancedb 为可选(opt-in 且需可导入)。
backend=="sqlite"(FTS5-only)不走本工厂,由调用方直接用 MemoryStore。
"""

from __future__ import annotations

import logging
from typing import Optional

from src.memory.base import BaseMemoryStore

logger = logging.getLogger("flyclaw.memory.factory")


def make_vector_store(
    *,
    backend: str,
    db_path: str,
    dimensions: int = 1536,
    fts_tokenizer: str = "unicode61",
    lancedb_uri: Optional[str] = None,
    vec_table: str = "memory_vec",
) -> BaseMemoryStore:
    """按 backend 构造向量后端 store。

    - "lancedb":导入 lancedb 失败时 warn 并回退 sqlite_vec(用户显式选了 lancedb 但没装)。
    - "sqlite_vec" / 其它 / 回退:SqliteVecMemoryStore。

    vec0 与 chunks 同库;lancedb_uri 仅 lancedb 后端用(sqlite-vec 不需要独立目录)。
    """
    if backend == "lancedb":
        try:
            import lancedb  # noqa: F401  直接探测:lance_store.py 内部吞 ImportError,靠它判不出
            from src.memory.lance_store import LanceMemoryStore

            return LanceMemoryStore(
                db_path,
                dimensions=dimensions,
                fts_tokenizer=fts_tokenizer,
                lancedb_uri=lancedb_uri,
            )
        except ImportError:
            logger.warning("backend='lancedb' 但未安装 lancedb(需 pip install -e '.[lancedb]'),回退 sqlite_vec")

    from src.memory.sqlitevec_store import SqliteVecMemoryStore

    return SqliteVecMemoryStore(
        db_path,
        dimensions=dimensions,
        fts_tokenizer=fts_tokenizer,
        vec_table=vec_table,
    )
