"""Tests for EmbeddingProvider.embed_texts via mocked httpx."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from src.config import ModelConfig
from src.memory.embeddings import EmbeddingProvider


@pytest.fixture
def mock_httpx(monkeypatch):
    """monkeypatch httpx.AsyncClient，返回 (queue, FakeResponse)。

    queue 里放 FakeResponse 或 Exception（post 时抛）。
    """
    queue: list = []

    class FakeAsyncClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None, headers=None):
            if not queue:
                raise RuntimeError("no response queued")
            item = queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

    class FakeResponse:
        def __init__(self, status_code=200, json_data=None, text=""):
            self.status_code = status_code
            self._json = json_data or {}
            self.text = text

        def json(self):
            return self._json

        def raise_for_status(self):
            if self.status_code >= 400:
                raise httpx.HTTPStatusError(f"HTTP {self.status_code}", request=None, response=self)

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    return queue, FakeResponse


def _make_provider() -> EmbeddingProvider:
    """MemoryConfig 没有 embedding_model/embedding_dimensions/base_url 字段，
    EmbeddingProvider.__init__ 用 getattr 取，所以用 SimpleNamespace 直接构造。
    """
    cfg = SimpleNamespace(
        api_key="sk-test",
        base_url="https://api.example.com",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=3,
    )
    model = ModelConfig(api_key="sk-test", base_url="https://api.example.com")
    return EmbeddingProvider(cfg, model)


class TestEmbedTexts:
    @pytest.mark.asyncio
    async def test_empty_input(self, mock_httpx):
        ep = _make_provider()
        assert await ep.embed_texts([]) == []

    @pytest.mark.asyncio
    async def test_success_sorted_by_index(self, mock_httpx):
        queue, FakeResp = mock_httpx
        # API 返回 index 1 在前，embed_texts 必须按 index 排序
        queue.append(
            FakeResp(
                200,
                json_data={
                    "data": [
                        {"index": 1, "embedding": [0.2, 0.2, 0.2]},
                        {"index": 0, "embedding": [0.1, 0.1, 0.1]},
                    ]
                },
            )
        )
        ep = _make_provider()
        result = await ep.embed_texts(["a", "b"])
        assert result == [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]

    @pytest.mark.asyncio
    async def test_http_error_raises(self, mock_httpx):
        queue, FakeResp = mock_httpx
        queue.append(FakeResp(401, text="unauthorized"))
        ep = _make_provider()
        with pytest.raises(httpx.HTTPStatusError):
            await ep.embed_texts(["a"])

    @pytest.mark.asyncio
    async def test_network_error_raises(self, mock_httpx):
        queue, _ = mock_httpx
        queue.append(ConnectionError("network down"))
        ep = _make_provider()
        with pytest.raises(ConnectionError):
            await ep.embed_texts(["a"])

    @pytest.mark.asyncio
    async def test_query_via_embed_query(self, mock_httpx):
        """embed_query 单条入口也走同一路径。"""
        queue, FakeResp = mock_httpx
        queue.append(FakeResp(200, json_data={"data": [{"index": 0, "embedding": [0.5, 0.5, 0.5]}]}))
        ep = _make_provider()
        result = await ep.embed_query("hello")
        assert result == [0.5, 0.5, 0.5]

    @pytest.mark.asyncio
    async def test_dim_mismatch_raises(self, mock_httpx):
        """返回向量维度与配置不符时抛 RuntimeError（provider 忽略 dimensions 参数的显式失败）。"""
        queue, FakeResp = mock_httpx
        # _make_provider 用 embedding_dimensions=3；返回 4 维 → 校验抛
        queue.append(
            FakeResp(
                200,
                json_data={"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3, 0.4]}]},
            )
        )
        ep = _make_provider()
        with pytest.raises(RuntimeError):
            await ep.embed_texts(["a"])
