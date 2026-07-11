"""Tests isolating sqlite-vec vector search path (默认向量后端)。

镜像 test_lance_vec_search.py,验证 sqlite-vec 写入 + 检索 + group_id pre-filter
+ source 标签 + 阈值过滤,行为与 LanceDB 路径一致。metadata 经 JOIN chunks 取得
(与 chunks 同库,不入 vec0)。
"""

from __future__ import annotations

import pytest

from src.config import MemoryConfig
from src.memory.search import MemorySearcher
from src.memory.sqlitevec_store import SqliteVecMemoryStore


class FakeEmbeddings:
    """固定向量,content 无关--隔离 FTS,只测向量路径。"""

    async def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    async def close(self):
        pass


class ConfigurableEmbeddings:
    """查询向量可配置--测 semantic_min_score 过滤需控制 vec_score 高低。"""

    def __init__(self, query_vec: list[float]):
        self._vec = query_vec

    async def embed_query(self, text: str) -> list[float]:
        return self._vec

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._vec for _ in texts]

    async def close(self):
        pass


@pytest.fixture
async def store(tmp_path):
    s = SqliteVecMemoryStore(db_path=str(tmp_path / "v.db"), dimensions=4)
    await s.initialize()
    yield s
    await s.close()


class TestSqliteVecSearch:
    @pytest.mark.asyncio
    async def test_vec_search_retrieves_stored_vector(self, store):
        """向量存进去能被 _vec_search 检索到,vec_score 反映距离。"""
        await store.add_document(
            "kv:k1",
            "alpha beta gamma",
            metadata={"category": "fact", "updated_ts": 1.0, "group_id": ""},
            chunk=False,
        )
        ids = await store.get_chunk_ids_for_path("kv:k1")
        await store.add_embeddings([ids[0]], [[0.1, 0.2, 0.3, 0.4]])

        # 相似向量(完全相同)-> distance=0 -> vec_score=1.0
        hits = await store._vec_search([0.1, 0.2, 0.3, 0.4], limit=5)
        assert len(hits) == 1, f"expected 1 hit, got {len(hits)}"
        assert hits[0]["path"] == "kv:k1"
        assert hits[0]["vec_score"] == 1.0, f"expected vec_score=1.0, got {hits[0]['vec_score']}"

    @pytest.mark.asyncio
    async def test_vec_search_dissimilar_vector_low_score(self, store):
        """不相似向量 -> vec_score 低(距离远)。"""
        await store.add_document(
            "kv:k1",
            "alpha beta gamma",
            metadata={"category": "fact", "updated_ts": 1.0, "group_id": ""},
            chunk=False,
        )
        ids = await store.get_chunk_ids_for_path("kv:k1")
        await store.add_embeddings([ids[0]], [[0.1, 0.2, 0.3, 0.4]])

        # L2²=1.74 -> vec_score = 1 - L2²/2 = 0.13。钉死绝对值,防公式回归:
        # 若误把 sqlite-vec 的 L2 当 L2²(改成 1 - distance/2)会得 0.34;
        # 若 sqlite-vec 改返回 L2² 而代码仍 1 - distance²/2 会得 -0.51。
        # 两者都偏离 0.13 -> 被抓。(sqlite-vec 实测返回 L2,非 L2²。)
        hits = await store._vec_search([0.9, 0.9, 0.9, 0.9], limit=5)
        assert len(hits) == 1
        assert hits[0]["vec_score"] == pytest.approx(0.13, abs=0.01), (
            f"vec_score 应钉在 1-L2²/2=0.13, got {hits[0]['vec_score']}"
        )

    @pytest.mark.asyncio
    async def test_hybrid_uses_vector_when_fts_misses(self, store):
        """FTS 匹配不到时,向量仍能召回--证明向量路径真生效。"""
        await store.add_document(
            "kv:k1",
            "alpha beta gamma delta epsilon zeta",
            metadata={"category": "fact", "updated_ts": 1.0, "group_id": ""},
            chunk=False,
        )
        ids = await store.get_chunk_ids_for_path("kv:k1")
        await store.add_embeddings([ids[0]], [[0.1, 0.2, 0.3, 0.4]])

        searcher = MemorySearcher(store, FakeEmbeddings(), MemoryConfig())
        # query_text="kappa" FTS miss,但向量与存储相同 -> 向量召回
        results = await searcher.search("kappa")
        assert len(results) >= 1, "vector path should retrieve even when FTS misses"
        assert results[0]["path"] == "kv:k1"

    @pytest.mark.asyncio
    async def test_no_vector_stored_returns_empty_vec_search(self, store):
        """没存向量时 _vec_search 返回空(不崩)。"""
        await store.add_document(
            "kv:k1",
            "alpha beta",
            metadata={"category": "fact", "updated_ts": 1.0, "group_id": ""},
            chunk=False,
        )
        hits = await store._vec_search([0.1, 0.2, 0.3, 0.4], limit=5)
        assert hits == []

    @pytest.mark.asyncio
    async def test_metadata_roundtrips_via_chunks_join(self, store):
        """metadata 不入 vec0,经 JOIN chunks 取得--round-trip 仍正确。

        与 LanceDB 测试对称:那边证 metadata 随向量存 LanceDB;这边证 metadata
        从 chunks 取(vec0 只存 vector+group_id)。两者都保证 _vec_search 返回 metadata。
        """
        meta = {"category": "preference", "updated_ts": 1700000000.0, "group_id": "g1"}
        await store.add_document("kv:k1", "alpha beta", metadata=meta, chunk=False)
        ids = await store.get_chunk_ids_for_path("kv:k1")
        await store.add_embeddings([ids[0]], [[0.1, 0.2, 0.3, 0.4]])

        hits = await store._vec_search([0.1, 0.2, 0.3, 0.4], limit=5)
        assert len(hits) == 1
        assert hits[0]["metadata"] == meta, f"metadata should round-trip via chunks JOIN, got {hits[0]['metadata']}"

    @pytest.mark.asyncio
    async def test_group_id_prefilter(self, store):
        """group_id 过滤是 pre-filter:KNN 先按群过滤再取 k,A 组更近也不串进 B 组查询。"""
        await store.add_document("kv:a1", "alpha", metadata={"group_id": "gA"}, chunk=False)
        await store.add_document("kv:b1", "bravo", metadata={"group_id": "gB"}, chunk=False)
        id_a = (await store.get_chunk_ids_for_path("kv:a1"))[0]
        id_b = (await store.get_chunk_ids_for_path("kv:b1"))[0]
        await store.add_embeddings([id_a], [[0.1, 0.2, 0.3, 0.4]])  # gA,与 query 同(最近)
        await store.add_embeddings([id_b], [[0.9, 0.9, 0.9, 0.9]])  # gB,远

        # 查 gB:尽管 gA 的向量更近,pre-filter 应只返回 gB
        hits_b = await store._vec_search([0.1, 0.2, 0.3, 0.4], limit=5, group_id="gB")
        assert len(hits_b) == 1, f"pre-filter should return only gB, got {len(hits_b)}"
        assert hits_b[0]["path"] == "kv:b1"

        # 查 gA:只返回 gA
        hits_a = await store._vec_search([0.1, 0.2, 0.3, 0.4], limit=5, group_id="gA")
        assert len(hits_a) == 1
        assert hits_a[0]["path"] == "kv:a1"

    @pytest.mark.asyncio
    async def test_search_results_carry_source_label(self, store):
        """search() 返回带 source 标签,语义组在前,同文档两路命中不去重。"""
        await store.add_document(
            "kv:k1",
            "redis 部署笔记",
            metadata={"category": "fact", "updated_ts": 1.0, "group_id": ""},
            chunk=False,
        )
        ids = await store.get_chunk_ids_for_path("kv:k1")
        await store.add_embeddings([ids[0]], [[0.1, 0.2, 0.3, 0.4]])

        searcher = MemorySearcher(store, FakeEmbeddings(), MemoryConfig())
        results = await searcher.search("redis")
        sources = [r["source"] for r in results]
        assert "semantic" in sources and "exact" in sources, f"两路都应命中, got {sources}"
        assert sources.index("semantic") < sources.index("exact"), "语义组应排在 exact 前"
        assert len(results) == 2, f"同文档两路命中应各保留一条, got {len(results)}"

    @pytest.mark.asyncio
    async def test_search_exact_label_carries_fts_data_when_vector_filtered(self, store):
        """向量被 semantic_min_score 过滤 + FTS 命中 -> 只剩 exact 标签。"""
        await store.add_document(
            "kv:k1",
            "redis 部署笔记",
            metadata={"category": "fact", "updated_ts": 1.0, "group_id": ""},
            chunk=False,
        )
        ids = await store.get_chunk_ids_for_path("kv:k1")
        await store.add_embeddings([ids[0]], [[0.1, 0.2, 0.3, 0.4]])

        # vec_score≈0.13 < 0.30 默认阈值被滤;query="redis" FTS 命中
        searcher = MemorySearcher(store, ConfigurableEmbeddings([0.9, 0.9, 0.9, 0.9]), MemoryConfig())
        results = await searcher.search("redis")
        sources = [r["source"] for r in results]
        assert sources == ["exact"], f"向量被滤应只剩 exact, got {sources}"
        assert results[0]["path"] == "kv:k1"

    @pytest.mark.asyncio
    async def test_semantic_min_score_filters_low_similarity(self, store):
        """semantic_min_score 真过滤:低阈值召回 0.13,高阈值过滤。"""
        await store.add_document(
            "kv:k1",
            "alpha beta gamma",
            metadata={"category": "fact", "updated_ts": 1.0, "group_id": ""},
            chunk=False,
        )
        ids = await store.get_chunk_ids_for_path("kv:k1")
        await store.add_embeddings([ids[0]], [[0.1, 0.2, 0.3, 0.4]])

        emb_low = ConfigurableEmbeddings([0.9, 0.9, 0.9, 0.9])  # vec_score≈0.13

        r_low = await MemorySearcher(store, emb_low, MemoryConfig(semantic_min_score=0.0)).search("kappa")
        assert any(r["source"] == "semantic" for r in r_low), f"阈值0.0应召回0.13分, got {r_low}"

        r_high = await MemorySearcher(store, emb_low, MemoryConfig(semantic_min_score=0.30)).search("kappa")
        assert all(r["source"] != "semantic" for r in r_high), f"阈值0.30应过滤0.13分, got {r_high}"

    @pytest.mark.asyncio
    async def test_semantic_min_score_boundary_inclusive(self, store):
        """>= 边界:vec_score 恰等于阈值时召回(>= 而非 >)。"""
        await store.add_document(
            "kv:k1",
            "alpha beta gamma",
            metadata={"category": "fact", "updated_ts": 1.0, "group_id": ""},
            chunk=False,
        )
        ids = await store.get_chunk_ids_for_path("kv:k1")
        await store.add_embeddings([ids[0]], [[0.1, 0.2, 0.3, 0.4]])

        emb = ConfigurableEmbeddings([0.9, 0.9, 0.9, 0.9])
        actual = (await store._vec_search([0.9, 0.9, 0.9, 0.9], limit=5))[0]["vec_score"]

        r_eq = await MemorySearcher(store, emb, MemoryConfig(semantic_min_score=actual)).search("kappa")
        assert any(r["source"] == "semantic" for r in r_eq), f"分恰等于阈值应召回(>=), got {r_eq}"

        r_above = await MemorySearcher(store, emb, MemoryConfig(semantic_min_score=actual + 0.001)).search("kappa")
        assert all(r["source"] != "semantic" for r in r_above), f"阈值略高于分应过滤, got {r_above}"

    @pytest.mark.asyncio
    async def test_reembed_same_chunk_id_upsert(self, store):
        """重新 embed 同一 chunk_id 不炸 UNIQUE(vec0 不支持 OR REPLACE),且向量被更新。

        回归:INSERT OR REPLACE 在 vec0 上抛 UNIQUE,重新索引/重 embed 同 id 会崩。
        改 delete-then-insert 后:不报错、不重复、旧向量被替换。
        """
        await store.add_document("kv:k1", "alpha", metadata={"group_id": ""}, chunk=False)
        cid = (await store.get_chunk_ids_for_path("kv:k1"))[0]
        await store.add_embeddings([cid], [[0.1, 0.2, 0.3, 0.4]])
        # 重新 embed 同 id,不同向量--不炸
        await store.add_embeddings([cid], [[0.9, 0.8, 0.7, 0.6]])

        # 新向量在,且不重复
        hits_new = await store._vec_search([0.9, 0.8, 0.7, 0.6], limit=5)
        assert len(hits_new) == 1, f"upsert should not duplicate, got {len(hits_new)}"
        assert hits_new[0]["vec_score"] == 1.0
        # 旧向量已替换:查旧向量不应得 vec_score=1.0(否则旧向量还残留)
        hits_old = await store._vec_search([0.1, 0.2, 0.3, 0.4], limit=5)
        assert hits_old[0]["vec_score"] < 1.0, "old vector should be replaced, not coexist"
