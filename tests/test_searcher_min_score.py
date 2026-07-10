"""Tests for MemorySearcher threshold filtering and metadata passthrough.

阈值过滤由 MemoryConfig.fts_min_score 控制（FTS5-only store 走 exact 路）。
"""

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

    async def close(self):
        pass


@pytest.fixture
async def searcher(tmp_path):
    store = MemoryStore(db_path=str(tmp_path / "t.db"), dimensions=4)
    await store.initialize()
    await store.add_document(
        "kv:k1", "我喜欢 深色主题", metadata={"category": "preference", "updated_ts": 1.0, "group_id": ""}, chunk=False
    )
    s = MemorySearcher(store, FakeEmbeddings(), MemoryConfig())
    yield s
    await s.close()


@pytest.fixture
async def multi_store(tmp_path):
    """3 条不同相关度的文档(BM25 分数递减),验证 fts_min_score 真过滤。

    MemoryStore 是 FTS5-only(_has_vector_support=False),走 exact 路,
    阈值由 config.fts_min_score 控制。
    - k1: vim 出现 3 次,短文档 -> 高 BM25
    - k2: vim 出现 1 次,短文档 -> 中 BM25
    - k3: vim 出现 1 次,长文档(稀释)-> 低 BM25
    """
    store = MemoryStore(db_path=str(tmp_path / "multi.db"), dimensions=4)
    await store.initialize()
    await store.add_document(
        "kv:k1", "vim vim vim editor", metadata={"category": "fact", "updated_ts": 1.0, "group_id": ""}, chunk=False
    )
    await store.add_document(
        "kv:k2", "vim editor", metadata={"category": "fact", "updated_ts": 2.0, "group_id": ""}, chunk=False
    )
    await store.add_document(
        "kv:k3",
        "vim editor and many other words to dilute bm25 score here making doc long",
        metadata={"category": "fact", "updated_ts": 3.0, "group_id": ""},
        chunk=False,
    )
    yield store
    await store.close()


def _searcher(store, **cfg_overrides):
    cfg = MemoryConfig(**cfg_overrides)
    return MemorySearcher(store, FakeEmbeddings(), cfg)


class TestSearchMinScore:
    @pytest.mark.asyncio
    async def test_fts_min_score_filters_low_relevance(self, multi_store):
        """fts_min_score 真过滤:低阈值返回全量,高阈值只剩最高相关度。"""
        # fts_min_score=0.0 -> 全部 3 条(归一化后最低分=0.0,>= 0.0 通过)
        results_all = await _searcher(multi_store, fts_min_score=0.0).search("vim")
        assert len(results_all) == 3, f"fts_min_score=0.0 should return all 3, got {len(results_all)}"

        # fts_min_score=0.9 -> 只剩最高相关度(k1,vim 3x)
        results_high = await _searcher(multi_store, fts_min_score=0.9).search("vim")
        assert len(results_high) < 3, f"fts_min_score=0.9 should filter some out, got {len(results_high)}"
        assert len(results_high) >= 1, "top match should always pass"
        assert results_high[0]["path"] == "kv:k1", f"top result should be kv:k1, got {results_high[0]['path']}"

    @pytest.mark.asyncio
    async def test_fts_min_score_threshold_monotonic(self, multi_store):
        """阈值单调:fts_min_score 越高,返回越少。"""
        n_low = len(await _searcher(multi_store, fts_min_score=0.0).search("vim"))
        n_mid = len(await _searcher(multi_store, fts_min_score=0.5).search("vim"))
        n_high = len(await _searcher(multi_store, fts_min_score=0.95).search("vim"))
        assert n_low >= n_mid >= n_high, f"not monotonic: {n_low} >= {n_mid} >= {n_high}"
        assert n_high >= 1, "highest threshold should still return top match"

    @pytest.mark.asyncio
    async def test_metadata_in_result(self, searcher):
        results = await searcher.search("深色主题")
        assert results[0]["metadata"]["category"] == "preference"
