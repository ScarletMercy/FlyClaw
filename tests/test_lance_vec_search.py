"""Tests isolating LanceDB vector search path.

这些测试隔离验证向量写入 + 检索，不靠 FTS 兜底——之前 bug（.to_pandas() 需要
pandas 但 venv 没装，_vec_search 静默返回空）就是缺这种隔离测试才藏住。
"""

from __future__ import annotations

import pytest

from src.config import MemoryConfig
from src.memory.lance_store import LanceMemoryStore
from src.memory.search import MemorySearcher


class FakeEmbeddings:
    """固定向量，content 无关——隔离 FTS，只测向量路径。"""

    async def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


@pytest.fixture
async def store(tmp_path):
    s = LanceMemoryStore(
        db_path=str(tmp_path / "v.db"),
        dimensions=4,
        lancedb_uri=str(tmp_path / "v.lance"),
    )
    await s.initialize()
    yield s
    await s.close()


class TestLanceVecSearch:
    @pytest.mark.asyncio
    async def test_vec_search_retrieves_stored_vector(self, store):
        """向量存进去能被 _vec_search 检索到，vec_score 反映距离。"""
        await store.add_document(
            "kv:k1",
            "alpha beta gamma",
            metadata={"category": "fact", "updated_ts": 1.0, "group_id": ""},
            chunk=False,
        )
        ids = await store.get_chunk_ids_for_path("kv:k1")
        await store.add_embeddings([ids[0]], [[0.1, 0.2, 0.3, 0.4]])

        # 相似向量（完全相同）→ 命中，vec_score=1.0（distance=0）
        hits = await store._vec_search([0.1, 0.2, 0.3, 0.4], limit=5)
        assert len(hits) == 1, f"expected 1 hit, got {len(hits)}"
        assert hits[0]["path"] == "kv:k1"
        assert hits[0]["vec_score"] == 1.0, f"expected vec_score=1.0, got {hits[0]['vec_score']}"

    @pytest.mark.asyncio
    async def test_vec_search_dissimilar_vector_low_score(self, store):
        """不相似向量 → vec_score 低（距离远）。"""
        await store.add_document(
            "kv:k1",
            "alpha beta gamma",
            metadata={"category": "fact", "updated_ts": 1.0, "group_id": ""},
            chunk=False,
        )
        ids = await store.get_chunk_ids_for_path("kv:k1")
        await store.add_embeddings([ids[0]], [[0.1, 0.2, 0.3, 0.4]])

        # 不相似向量 —— LanceDB 仍返回最近的（limit 内），但 vec_score 应低
        hits = await store._vec_search([0.9, 0.9, 0.9, 0.9], limit=5)
        assert len(hits) == 1  # LanceDB 总返回 limit 条最近的
        assert hits[0]["vec_score"] < 0.5, f"dissimilar vector should have low vec_score, got {hits[0]['vec_score']}"

    @pytest.mark.asyncio
    async def test_hybrid_uses_vector_when_fts_misses(self, store):
        """FTS 匹配不到时，向量仍能召回——证明向量路径真生效，不是 FTS 兜底。

        这是关键回归测试：之前 .to_pandas() bug 下，FTS miss + vector broken →
        返回空。修复后向量应召回。
        """
        # 内容用 FTS 匹配不到的 token（query 用完全不同的词）
        await store.add_document(
            "kv:k1",
            "alpha beta gamma delta epsilon zeta",
            metadata={"category": "fact", "updated_ts": 1.0, "group_id": ""},
            chunk=False,
        )
        ids = await store.get_chunk_ids_for_path("kv:k1")
        await store.add_embeddings([ids[0]], [[0.1, 0.2, 0.3, 0.4]])

        searcher = MemorySearcher(store, FakeEmbeddings(), MemoryConfig())
        # query_text="kappa" FTS 匹配不到任何 token，但 embed_query 返回的向量
        # 与存储的 [0.1,0.2,0.3,0.4] 完全相同 → 向量路径应召回
        results = await searcher.search("kappa", min_score=0.0)
        assert len(results) >= 1, (
            "vector path should retrieve even when FTS misses — "
            "if empty, vector search is broken (check pandas/.to_list())"
        )
        assert results[0]["path"] == "kv:k1"

    @pytest.mark.asyncio
    async def test_no_vector_stored_returns_empty_vec_search(self, store):
        """没存向量时 _vec_search 返回空（不崩）。"""
        await store.add_document(
            "kv:k1",
            "alpha beta",
            metadata={"category": "fact", "updated_ts": 1.0, "group_id": ""},
            chunk=False,
        )
        # 不调 add_embeddings
        hits = await store._vec_search([0.1, 0.2, 0.3, 0.4], limit=5)
        # 没向量 → LanceDB 表空 → 返回空
        assert hits == []

    @pytest.mark.asyncio
    async def test_metadata_stored_in_lancedb_for_traceability(self, store):
        """元数据随向量存进 LanceDB（不只 SQLite），_vec_search 直接返回——溯源不依赖 SQLite join。

        直接查 LanceDB 表断言 metadata 列存在且有值，证明元数据真的跟向量在一起。
        """
        import json as _json

        meta = {"category": "preference", "updated_ts": 1700000000.0, "group_id": "g1"}
        await store.add_document("kv:k1", "alpha beta", metadata=meta, chunk=False)
        ids = await store.get_chunk_ids_for_path("kv:k1")
        await store.add_embeddings([ids[0]], [[0.1, 0.2, 0.3, 0.4]])

        # 1. _vec_search 返回的 metadata 从 LanceDB 行直接取
        hits = await store._vec_search([0.1, 0.2, 0.3, 0.4], limit=5)
        assert len(hits) == 1
        assert hits[0]["metadata"] == meta, f"metadata should round-trip via LanceDB, got {hits[0]['metadata']}"
        assert hits[0]["metadata"]["category"] == "preference"
        assert hits[0]["metadata"]["updated_ts"] == 1700000000.0
        assert hits[0]["metadata"]["group_id"] == "g1"

        # 2. 直接查 LanceDB 表，确认 metadata 是表的列且有值（不靠 SQLite）
        import numpy as np

        lance_rows = store._lance_table.search(np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)).limit(5).to_list()
        assert len(lance_rows) == 1
        assert "metadata" in lance_rows[0], "metadata should be a column in LanceDB table"
        assert _json.loads(lance_rows[0]["metadata"]) == meta, (
            "LanceDB metadata column should hold the original metadata JSON"
        )

    @pytest.mark.asyncio
    async def test_add_embeddings_to_legacy_schema_table(self, tmp_path):
        """升级场景：旧 LanceDB 表只有 {id, vector} 列（无 metadata），新代码 add_embeddings
        含 metadata 列时不应静默失败。

        回归 bug #2：_init_vector_backend open_table 不迁移 schema，旧表存在时
        _store_embeddings 写含 metadata 列的 pa.table → add() schema mismatch →
        生产中被 index_document 的 except 吞掉（search.py:89），向量静默写不进。
        """
        import lancedb
        import pyarrow as pa

        from src.memory.lance_store import LanceMemoryStore

        lance_uri = str(tmp_path / "legacy.lance")
        db = lancedb.connect(lance_uri)
        # 旧 schema：只有 id + vector，没有 metadata 列（升级前的表结构）
        legacy_schema = pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("vector", pa.list_(pa.float32(), 4)),
            ]
        )
        db.create_table("memory_vectors", schema=legacy_schema)

        store = LanceMemoryStore(
            db_path=str(tmp_path / "sqlite.db"),
            dimensions=4,
            lancedb_uri=lance_uri,
        )
        await store.initialize()  # open_table 复用旧表，不迁移 schema

        await store.add_document("kv:k1", "alpha beta", metadata={"category": "fact"}, chunk=False)
        ids = await store.get_chunk_ids_for_path("kv:k1")

        # add_embeddings 写含 metadata 列的 pa.table 到旧 schema 表
        try:
            await store.add_embeddings([ids[0]], [[0.1, 0.2, 0.3, 0.4]])
            embed_ok = True
            embed_err = ""
        except Exception as e:
            embed_ok = False
            embed_err = f"{type(e).__name__}: {e}"

        assert embed_ok, f"add_embeddings 到旧 schema 表应成功（应迁移 schema 或兼容），got: {embed_err}"
        hits = await store._vec_search([0.1, 0.2, 0.3, 0.4], limit=5)
        assert len(hits) == 1, "向量应写入并检索到"
        assert hits[0]["metadata"] == {"category": "fact"}, f"metadata 应随向量写入 LanceDB，got {hits[0]['metadata']}"
        await store.close()
