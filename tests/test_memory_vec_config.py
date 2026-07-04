"""Tests for MemoryStoreConfig vector archival fields."""

from __future__ import annotations

from src.config import MemoryStoreConfig


class TestMemoryStoreConfigVector:
    def test_vector_defaults_disabled(self) -> None:
        ms = MemoryStoreConfig()
        assert ms.vector_enabled is False
        assert ms.vector_model == "text-embedding-3-small"
        assert ms.vector_dimensions == 1536
        assert ms.vector_keep_recent_n == 20
        assert ms.vector_keep_recent_days == 7

    def test_vector_db_path_default(self) -> None:
        ms = MemoryStoreConfig()
        assert ms.vector_db_path == "~/.flyclaw/data/memory_kv_vec.db"

    def test_vector_enabled_with_required_fields(self) -> None:
        ms = MemoryStoreConfig(
            enabled=True,
            vector_enabled=True,
            vector_model="bge-m3",
            vector_base_url="https://api.example.com",
            vector_api_key="sk-xxx",
        )
        assert ms.vector_enabled is True
        assert ms.vector_model == "bge-m3"
