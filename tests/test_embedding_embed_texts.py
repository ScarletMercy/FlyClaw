"""Tests for EmbeddingProvider.embed_texts via mocked openai SDK client."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from openai import AuthenticationError

from src.config import ModelConfig
from src.memory import embeddings as emb_module
from src.memory.embeddings import EmbeddingProvider


@pytest.fixture
def mock_sdk(monkeypatch):
    """monkeypatch src.memory.embeddings.AsyncOpenAI，返回 (queue, FakeCreateResponse)。

    queue 里放 FakeCreateResponse 或 Exception（create 时抛）。
    EmbeddingProvider 在模块级 `from openai import AsyncOpenAI`，故需 patch
    emb_module.AsyncOpenAI 才能生效。
    """
    queue: list = []

    class FakeEmbedding:
        def __init__(self, index, embedding):
            self.index = index
            self.embedding = embedding

    class FakeCreateResponse:
        def __init__(self, data):
            self.data = data

    class FakeEmbeddings:
        def __init__(self):
            self.calls: list[dict] = []  # 记录每次 create 的 kwargs

        async def create(self, **kwargs):
            self.calls.append(kwargs)
            if not queue:
                raise RuntimeError("no response queued")
            item = queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

    class FakeAsyncOpenAI:
        def __init__(self, *a, **k):
            self.embeddings = FakeEmbeddings()

        async def close(self):
            pass

    monkeypatch.setattr(emb_module, "AsyncOpenAI", FakeAsyncOpenAI)
    return queue, FakeCreateResponse, FakeEmbedding


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
    async def test_empty_input(self, mock_sdk):
        ep = _make_provider()
        assert await ep.embed_texts([]) == []

    @pytest.mark.asyncio
    async def test_success_sorted_by_index(self, mock_sdk):
        queue, FakeResp, FakeEmb = mock_sdk
        # API 返回 index 1 在前，embed_texts 必须按 index 排序
        queue.append(
            FakeResp(
                [
                    FakeEmb(1, [0.2, 0.2, 0.2]),
                    FakeEmb(0, [0.1, 0.1, 0.1]),
                ]
            )
        )
        ep = _make_provider()
        result = await ep.embed_texts(["a", "b"])
        assert result == [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]

    @pytest.mark.asyncio
    async def test_http_error_raises(self, mock_sdk):
        queue, _, _ = mock_sdk
        req = httpx.Request("POST", "https://api.example.com/embeddings")
        resp = httpx.Response(401, request=req, text='{"error":{"message":"unauthorized"}}')
        queue.append(AuthenticationError("unauthorized", response=resp, body=None))
        ep = _make_provider()
        with pytest.raises(AuthenticationError):
            await ep.embed_texts(["a"])

    @pytest.mark.asyncio
    async def test_network_error_raises(self, mock_sdk):
        queue, _, _ = mock_sdk
        queue.append(ConnectionError("network down"))
        ep = _make_provider()
        with pytest.raises(ConnectionError):
            await ep.embed_texts(["a"])

    @pytest.mark.asyncio
    async def test_query_via_embed_query(self, mock_sdk):
        """embed_query 单条入口也走同一路径。"""
        queue, FakeResp, FakeEmb = mock_sdk
        queue.append(FakeResp([FakeEmb(0, [0.5, 0.5, 0.5])]))
        ep = _make_provider()
        result = await ep.embed_query("hello")
        assert result == [0.5, 0.5, 0.5]

    @pytest.mark.asyncio
    async def test_dim_mismatch_raises(self, mock_sdk):
        """返回向量维度与配置不符时抛 RuntimeError（provider 忽略 dimensions 参数的显式失败）。"""
        queue, FakeResp, FakeEmb = mock_sdk
        # _make_provider 用 embedding_dimensions=3；返回 4 维 → 校验抛
        queue.append(FakeResp([FakeEmb(0, [0.1, 0.2, 0.3, 0.4])]))
        ep = _make_provider()
        with pytest.raises(RuntimeError):
            await ep.embed_texts(["a"])
