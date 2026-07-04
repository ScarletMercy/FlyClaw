"""Tests for setup wizard vector sub-step."""

from __future__ import annotations

import builtins
from unittest.mock import patch

import pytest

from src.setup import _step_memory_store


class TestWizardVector:
    def test_disabled_skips_vector(self, monkeypatch):
        config = {"memory_store": {"enabled": True}}
        # 启用记忆存储=yes(已enabled), 启用向量=no
        inputs = iter(["yes", "no"])
        monkeypatch.setattr(builtins, "input", lambda *a, **k: next(inputs))
        _step_memory_store(config)
        assert config["memory_store"]["vector_enabled"] is False

    def test_enabled_with_validated_inputs(self, monkeypatch):
        config = {"memory_store": {"enabled": True}}
        inputs = iter(
            [
                "yes",  # 启用记忆存储
                "yes",  # 启用向量
                "text-embedding-3-small",  # model
                "https://api.example.com",  # base_url
                "sk-xxx",  # api_key
                "",  # dimensions (用默认)
            ]
        )
        monkeypatch.setattr(builtins, "input", lambda *a, **k: next(inputs))
        monkeypatch.setattr(
            "src.setup._verify_embedding_api_key",
            lambda b, k, m: (True, "1536"),
        )
        _step_memory_store(config)
        ms = config["memory_store"]
        assert ms["vector_enabled"] is True
        assert ms["vector_model"] == "text-embedding-3-small"
        assert ms["vector_dimensions"] == 1536

    def test_validation_fail_then_abandon(self, monkeypatch):
        config = {"memory_store": {"enabled": True}}
        inputs = iter(
            [
                "yes",  # 启用记忆存储
                "yes",  # 启用向量
                "text-embedding-3-small",
                "https://api.example.com",
                "sk-bad",
                "2",  # 验证失败 → 选放弃
            ]
        )
        monkeypatch.setattr(builtins, "input", lambda *a, **k: next(inputs))
        monkeypatch.setattr(
            "src.setup._verify_embedding_api_key",
            lambda b, k, m: (False, "HTTP 401"),
        )
        _step_memory_store(config)
        assert config["memory_store"]["vector_enabled"] is False
