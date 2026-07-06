"""Tests for EmbeddingProvider.from_vector_config."""

from __future__ import annotations

from src.config import MemoryStoreConfig, ModelConfig
from src.memory.embeddings import EmbeddingProvider


class TestFromVectorConfig:
    def test_reads_vector_fields(self) -> None:
        ms = MemoryStoreConfig(
            vector_enabled=True,
            vector_model="bge-m3",
            vector_base_url="https://api.example.com",
            vector_api_key="sk-xxx",
            vector_dimensions=1024,
        )
        model = ModelConfig()
        ep = EmbeddingProvider.from_vector_config(ms, model)
        assert ep._model == "bge-m3"
        assert ep._dimensions == 1024
        # base_url 透传，不自己追加 /v1
        assert ep._base_url == "https://api.example.com"
        assert ep._api_key == "sk-xxx"
        # SDK 客户端拿到的 base_url 也是原值（SDK 内部会规范化加尾斜杠，但不再叠加 /v1）
        client = ep._get_client()
        assert str(client.base_url).rstrip("/") == "https://api.example.com"
        assert client.api_key == "sk-xxx"

    def test_no_fallback_to_model_config(self) -> None:
        """vector_* 留空时不该 fallback 到 model_config。"""
        ms = MemoryStoreConfig(vector_enabled=True, vector_base_url="", vector_api_key="")
        model = ModelConfig(api_key="should-not-leak")
        ep = EmbeddingProvider.from_vector_config(ms, model)
        assert ep._api_key == ""
        assert ep._base_url == ""
        assert "should-not-leak" not in ep._api_key
        assert "should-not-leak" not in ep._base_url
