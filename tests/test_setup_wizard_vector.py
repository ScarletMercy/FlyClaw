"""Tests for setup wizard vector sub-step."""

from __future__ import annotations

import builtins

import pytest

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

    def test_validation_fail_then_retry_succeeds(self, monkeypatch):
        """失败 → 重试 → 不重问字段，原值直接重验通过。"""
        config = {"memory_store": {"enabled": True}}
        inputs = iter(
            [
                "model",  # save审批模式
                "yes",  # 启用记忆存储
                "yes",  # 启用向量
                "text-embedding-3-small",
                "https://api.example.com",
                "sk-flaky",
                "重试",  # 验证失败 → 重试（不应再消耗任何输入）
            ]
        )
        monkeypatch.setattr(builtins, "input", lambda *a, **k: next(inputs))
        calls = []

        def fake_verify(b, k, m):
            calls.append(k)
            return (False, "HTTP 503") if len(calls) == 1 else (True, "999")

        monkeypatch.setattr("src.setup._verify_embedding_api_key", fake_verify)
        _step_memory_store(config)
        ms = config["memory_store"]
        assert ms["vector_enabled"] is True
        assert ms["vector_api_key"] == "sk-flaky"
        assert len(calls) == 2  # 重试确实重验了一次
        with pytest.raises(StopIteration):
            next(inputs)  # 重试未重问字段：输入已耗尽

    def test_validation_fail_then_reinput_succeeds(self, monkeypatch):
        """失败 → 重输 → 重收字段 → 重新验证通过。"""
        config = {"memory_store": {"enabled": True}}
        inputs = iter(
            [
                "model",  # save审批模式
                "yes",  # 启用记忆存储
                "yes",  # 启用向量
                "text-embedding-3-small",
                "https://api.example.com",
                "sk-bad",
                "重输",  # 验证失败 → 重输
                "text-embedding-3-small",  # （回车沿用）
                "https://api.example.com",  # （回车沿用）
                "sk-good",  # 换新值
            ]
        )
        monkeypatch.setattr(builtins, "input", lambda *a, **k: next(inputs))
        calls = []

        def fake_verify(b, k, m):
            calls.append(k)
            return (False, "HTTP 401") if len(calls) == 1 else (True, "999")

        monkeypatch.setattr("src.setup._verify_embedding_api_key", fake_verify)
        _step_memory_store(config)
        ms = config["memory_store"]
        assert ms["vector_enabled"] is True
        assert ms["vector_api_key"] == "sk-good"
        assert len(calls) == 2  # 重输后确实重新验证了一次

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
                "放弃",  # 验证失败 → 放弃
            ]
        )
        monkeypatch.setattr(builtins, "input", lambda *a, **k: next(inputs))
        monkeypatch.setattr(
            "src.setup._verify_embedding_api_key",
            lambda b, k, m: (False, "HTTP 401"),
        )
        _step_memory_store(config)
        assert config["memory_store"]["vector_enabled"] is False

    def test_existing_vector_keep_skips_verification(self, monkeypatch):
        """已配置向量记忆 → 保留 → 直接返回，不发验证请求。"""
        config = {
            "memory_store": {
                "enabled": True,
                "vector_enabled": True,
                "vector_model": "text-embedding-3-small",
                "vector_base_url": "https://api.example.com",
                "vector_api_key": "sk-xxx",
                "vector_dimensions": 999,
            }
        }
        inputs = iter(
            [
                "model",  # save审批模式
                "yes",  # 启用记忆存储
                "yes",  # 保留当前向量记忆设置
            ]
        )
        monkeypatch.setattr(builtins, "input", lambda *a, **k: next(inputs))

        def fail_verify(*a, **k):
            raise AssertionError("保留已有向量配置时不应发起验证请求")

        monkeypatch.setattr("src.setup._verify_embedding_api_key", fail_verify)
        _step_memory_store(config)
        ms = config["memory_store"]
        assert ms["vector_enabled"] is True
        assert ms["vector_api_key"] == "sk-xxx"
        assert ms["vector_dimensions"] == 999

    def test_existing_vector_reconfigure(self, monkeypatch):
        """已配置向量记忆 → 不保留 → 走重新配置流程。"""
        config = {
            "memory_store": {
                "enabled": True,
                "vector_enabled": True,
                "vector_model": "text-embedding-3-small",
                "vector_base_url": "https://api.example.com",
                "vector_api_key": "sk-xxx",
            }
        }
        inputs = iter(
            [
                "model",  # save审批模式
                "yes",  # 启用记忆存储
                "no",  # 保留当前向量记忆设置 → 不保留
                "no",  # 启用向量 → 关闭
            ]
        )
        monkeypatch.setattr(builtins, "input", lambda *a, **k: next(inputs))
        _step_memory_store(config)
        assert config["memory_store"]["vector_enabled"] is False
