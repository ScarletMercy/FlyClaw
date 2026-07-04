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
        assert ep._url == "https://api.example.com/v1/embeddings"
        assert "sk-xxx" in ep._headers["Authorization"]

    def test_no_fallback_to_model_config(self) -> None:
        """vector_* 留空时不该 fallback 到 model_config。"""
        ms = MemoryStoreConfig(vector_enabled=True, vector_base_url="", vector_api_key="")
        model = ModelConfig(api_key="should-not-leak")
        ep = EmbeddingProvider.from_vector_config(ms, model)
        assert ep._url == "/v1/embeddings"
        assert "should-not-leak" not in ep._headers["Authorization"]
