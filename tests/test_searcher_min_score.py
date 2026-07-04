"""Tests for MemorySearcher.search min_score override and metadata passthrough."""

from __future__ import annotations

import pytest

from src.config import MemoryConfig
from src.memory.store import MemoryStore
from src.memory.search import MemorySearcher


class FakeEmbeddings:
    async def embed_query(self, text):
        return [0.1, 0.2, 0.3, 0.4]

    async def embed_texts(self, texts):
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


@pytest.fixture
async def searcher(tmp_path):
    store = MemoryStore(
        db_path=str(tmp_path / "t.db"),
        dimensions=4,
    )
    await store.initialize()
    await store.add_document(
        "kv:k1", "我喜欢 深色主题", metadata={"category": "preference", "updated_ts": 1.0, "group_id": ""}, chunk=False
    )
    cfg = MemoryConfig()
    s = MemorySearcher(store, FakeEmbeddings(), cfg)
    yield s
    await s.close()


@pytest.fixture
async def searcher_multi(tmp_path):
    """3 条不同相关度的文档（BM25 分数递减），用于验证 min_score 真过滤。

    - k1: vim 出现 3 次，短文档 → 高 BM25
    - k2: vim 出现 1 次，短文档 → 中 BM25
    - k3: vim 出现 1 次，长文档（稀释）→ 低 BM25
    """
    store = MemoryStore(
        db_path=str(tmp_path / "multi.db"),
        dimensions=4,
    )
    await store.initialize()
    await store.add_document(
        "kv:k1",
        "vim vim vim editor",
        metadata={"category": "fact", "updated_ts": 1.0, "group_id": ""},
        chunk=False,
    )
    await store.add_document(
        "kv:k2",
        "vim editor",
        metadata={"category": "fact", "updated_ts": 2.0, "group_id": ""},
        chunk=False,
    )
    await store.add_document(
        "kv:k3",
        "vim editor and many other words to dilute bm25 score here making doc long",
        metadata={"category": "fact", "updated_ts": 3.0, "group_id": ""},
        chunk=False,
    )
    cfg = MemoryConfig()
    s = MemorySearcher(store, FakeEmbeddings(), cfg)
    yield s
    await s.close()


class TestSearchMinScore:
    @pytest.mark.asyncio
    async def test_min_score_param_accepted(self, searcher):
        """min_score 参数被接受（不 TypeError）。基本冒烟。"""
        results = await searcher.search("深色主题", min_score=0.0)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_min_score_filters_low_relevance(self, searcher_multi):
        """min_score 真过滤：高阈值返回少于全量，且最高分那条始终在。

        这是核心断言——不是只测参数被接受，而是测阈值实际滤掉低相关度结果。
        """
        # min_score=0.0 → 全部 3 条（归一化后最低分=0.0，>= 0.0 通过）
        results_all = await searcher_multi.search("vim", min_score=0.0)
        assert len(results_all) == 3, f"min_score=0.0 should return all 3, got {len(results_all)}"

        # min_score=0.9 → 只剩最高相关度（k1，vim 3x）
        results_high = await searcher_multi.search("vim", min_score=0.9)
        assert len(results_high) < 3, (
            f"min_score=0.9 should filter some out, got {len(results_high)} — BM25 可能未区分（检查文档长度/词频）"
        )
        assert len(results_high) >= 1, "top match should always pass"
        # 最高分一定是 k1（vim 出现 3 次，BM25 最高）
        assert results_high[0]["path"] == "kv:k1", f"top result should be kv:k1, got {results_high[0]['path']}"

    @pytest.mark.asyncio
    async def test_min_score_threshold_monotonic(self, searcher_multi):
        """阈值单调：min_score 越高，返回越少。"""
        n_low = len(await searcher_multi.search("vim", min_score=0.0))
        n_mid = len(await searcher_multi.search("vim", min_score=0.5))
        n_high = len(await searcher_multi.search("vim", min_score=0.95))
        assert n_low >= n_mid >= n_high, f"not monotonic: {n_low} >= {n_mid} >= {n_high}"
        assert n_high >= 1, "highest threshold should still return top match"

    @pytest.mark.asyncio
    async def test_metadata_in_result(self, searcher):
        results = await searcher.search("深色主题", min_score=0.0)
        assert results[0]["metadata"]["category"] == "preference"
