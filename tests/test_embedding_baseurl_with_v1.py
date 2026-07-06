"""回归测试：base_url 已含 /v1 时不得再追加 /v1（历史 bug：…/v1/v1/embeddings → 404）。

OpenAI 兼容 provider 的 base_url 惯例已带版本段（DeepSeek `…/v1`、Groq `…/openai/v1`、
智谱 `…/paas/v4`、DashScope `…/compatible-mode/v1`）。旧代码手拼 `{base_url}/v1/embeddings`
会得到 `…/v1/v1/embeddings` → 404。改走 openai SDK 后，base_url 应透传给 SDK，不自己追加 /v1。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.config import MemoryStoreConfig, ModelConfig
from src.memory import embeddings as emb_module
from src.memory.embeddings import EmbeddingProvider


class TestBaseUrlWithV1:
    def test_from_vector_config_does_not_double_v1(self):
        """带 /v1 的 base_url 必须原样透传给 SDK，不追加 /v1。"""
        ms = MemoryStoreConfig(
            vector_enabled=True,
            vector_model="text-embedding-3-small",
            vector_base_url="https://api.deepseek.com/v1",
            vector_api_key="sk-x",
            vector_dimensions=1536,
        )
        ep = EmbeddingProvider.from_vector_config(ms, ModelConfig())
        # 内部存储的 base_url 就是原值，没有再叠 /v1
        assert ep._base_url == "https://api.deepseek.com/v1"
        assert "v1/v1" not in ep._base_url

    def test_init_path_does_not_double_v1(self):
        """__init__ 路径同样不得追加 /v1。"""
        cfg = SimpleNamespace(
            api_key="sk-x",
            base_url="https://api.deepseek.com/v1",
            embedding_model="text-embedding-3-small",
            embedding_dimensions=1536,
        )
        ep = EmbeddingProvider(cfg, ModelConfig(api_key="sk-x", base_url="https://api.deepseek.com/v1"))
        assert ep._base_url == "https://api.deepseek.com/v1"
        assert "v1/v1" not in ep._base_url

    def test_sdk_client_constructed_with_original_base_url(self, monkeypatch):
        """AsyncOpenAI 收到的 base_url 必须是原值，不被追加 /v1。"""
        captured: dict = {}

        class FakeEmbeddings:
            async def create(self, **kwargs):
                return None

        class FakeAsyncOpenAI:
            def __init__(self, *a, **k):
                captured["base_url"] = k.get("base_url")
                self.embeddings = FakeEmbeddings()

            async def close(self):
                pass

        monkeypatch.setattr(emb_module, "AsyncOpenAI", FakeAsyncOpenAI)

        ms = MemoryStoreConfig(
            vector_enabled=True,
            vector_model="text-embedding-3-small",
            vector_base_url="https://api.deepseek.com/v1",
            vector_api_key="sk-x",
            vector_dimensions=1536,
        )
        ep = EmbeddingProvider.from_vector_config(ms, ModelConfig())
        ep._get_client()  # 触发 SDK 客户端构造
        assert captured["base_url"] == "https://api.deepseek.com/v1"
        assert captured["base_url"] != "https://api.deepseek.com/v1/v1"

    @pytest.mark.asyncio
    async def test_embed_with_v1_base_url_calls_create(self, monkeypatch):
        """端到端：带 /v1 的 base_url 下 embed_texts 能命中 embeddings.create（不再 404）。"""
        called: dict = {}

        class FakeEmbedding:
            def __init__(self, index, embedding):
                self.index = index
                self.embedding = embedding

        class FakeResp:
            def __init__(self, data):
                self.data = data

        class FakeEmbeddings:
            async def create(self, **kwargs):
                called["kwargs"] = kwargs
                return FakeResp([FakeEmbedding(0, [0.1] * 1536)])

        class FakeAsyncOpenAI:
            def __init__(self, *a, **k):
                self.embeddings = FakeEmbeddings()

            async def close(self):
                pass

        monkeypatch.setattr(emb_module, "AsyncOpenAI", FakeAsyncOpenAI)

        ms = MemoryStoreConfig(
            vector_enabled=True,
            vector_model="text-embedding-3-small",
            vector_base_url="https://api.deepseek.com/v1",
            vector_api_key="sk-x",
            vector_dimensions=1536,
        )
        ep = EmbeddingProvider.from_vector_config(ms, ModelConfig())
        result = await ep.embed_texts(["hi"])
        assert result == [[0.1] * 1536]
        # create 只收到 model/input（按 OpenAI 规范，dimensions 可选不发），且未把 /v1 拼进 URL
        assert called["kwargs"]["model"] == "text-embedding-3-small"
        assert called["kwargs"]["input"] == ["hi"]
        assert "dimensions" not in called["kwargs"]
