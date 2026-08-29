"""Tests for setup wizard vector sub-step."""

from __future__ import annotations

import builtins


from src.setup import _step_memory_store


class TestWizardVector:
    def test_disabled_skips_vector(self, monkeypatch):
        config = {"memory_store": {"enabled": True}}
        # save审批模式=model, 启用记忆存储=yes(已enabled), 启用向量=no
        inputs = iter(["model", "yes", "no"])
        monkeypatch.setattr(builtins, "input", lambda *a, **k: next(inputs))
        _step_memory_store(config)
        assert config["memory_store"]["vector_enabled"] is False

    def test_enabled_with_validated_inputs(self, monkeypatch):
        config = {"memory_store": {"enabled": True}}
        inputs = iter(
            [
                "model",  # save审批模式
                "yes",  # 启用记忆存储
                "yes",  # 启用向量
                "text-embedding-3-small",  # model
                "https://api.example.com",  # base_url
                "sk-xxx",  # api_key
                # 维度不再问用户，由验证探测得出
            ]
        )
        monkeypatch.setattr(builtins, "input", lambda *a, **k: next(inputs))
        # 探测返回的原生维度——用明显假的非默认值（999≠1536），
        # 证明向导存的是探测结果，而非拍脑袋默认或用户输入。真实维度模型相关，不由测试 dictate。
        probed = "999"
        monkeypatch.setattr(
            "src.setup._verify_embedding_api_key",
            lambda b, k, m: (True, probed),
        )
        _step_memory_store(config)
        ms = config["memory_store"]
        assert ms["vector_enabled"] is True
        assert ms["vector_model"] == "text-embedding-3-small"
        assert ms["vector_dimensions"] == int(probed)  # 存的是探测值，不再问用户

    def test_validation_fail_then_abandon(self, monkeypatch):
        config = {"memory_store": {"enabled": True}}
        inputs = iter(
            [
                "model",  # save审批模式
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
