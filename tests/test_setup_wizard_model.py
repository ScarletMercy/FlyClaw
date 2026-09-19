"""Tests for setup wizard model step verification failure handling.

失败菜单为三选项（重试/重输/放弃，对齐 chcode 连接测试失败处理）：
- 重试：原值直接重验，不重问任何字段
- 重输：从提供商选择重收全部字段后重验
- 放弃：模型为必配项，退出向导且不保存（SystemExit）
"""

from __future__ import annotations

import builtins

import pytest

from src.setup import _step_model


def _no_input_left(inputs):
    """断言输入迭代器已耗尽（证明某路径未额外提问）。"""
    with pytest.raises(StopIteration):
        next(inputs)


class TestWizardModelVerify:
    def test_verify_success_pass(self, monkeypatch):
        """首次验证通过：问上下文窗口 → 多模态 → 回退，无失败菜单。"""
        config = {}
        inputs = iter(
            [
                "custom",  # 选择提供商
                "m1",  # 模型名称
                "https://x/v1",  # 接口地址
                "sk-xxx",  # API 密钥
                "200000",  # 上下文窗口
                "no",  # 多模态
                "no",  # 添加/编辑回退模型
            ]
        )
        monkeypatch.setattr(builtins, "input", lambda *a, **k: next(inputs))
        monkeypatch.setattr("src.setup._verify_api_key", lambda p, n, b, k: (True, ""))
        _step_model(config)
        model = config["model"]
        assert model["api_key"] == "sk-xxx"
        assert model["context_window"] == 200000
        _no_input_left(inputs)

    def test_verify_fail_then_retry_succeeds(self, monkeypatch):
        """失败 → 重试 → 不重问字段，原值直接重验通过。"""
        config = {}
        inputs = iter(
            [
                "custom",  # 选择提供商
                "m1",  # 模型名称
                "https://x/v1",  # 接口地址
                "sk-flaky",  # API 密钥
                "重试",  # 验证失败 → 重试（不应再问提供商/名称/地址/密钥）
                "200000",  # 上下文窗口
                "no",  # 多模态
                "no",  # 回退
            ]
        )
        monkeypatch.setattr(builtins, "input", lambda *a, **k: next(inputs))
        calls = []

        def fake_verify(p, n, b, k):
            calls.append(k)
            return (False, "HTTP 503") if len(calls) == 1 else (True, "")

        monkeypatch.setattr("src.setup._verify_api_key", fake_verify)
        _step_model(config)
        model = config["model"]
        assert model["api_key"] == "sk-flaky"  # 重试沿用原值
        assert model["context_window"] == 200000
        assert len(calls) == 2
        _no_input_left(inputs)

    def test_verify_fail_then_reinput_succeeds(self, monkeypatch):
        """失败 → 重输 → 从提供商选择重收全部字段 → 重验通过。"""
        config = {}
        inputs = iter(
            [
                "custom",  # 选择提供商
                "m1",  # 模型名称
                "https://x/v1",  # 接口地址
                "sk-bad",  # API 密钥
                "重输",  # 验证失败 → 重输
                "custom",  # 选择提供商（重收）
                "m1",  # 模型名称（默认已填）
                "https://x/v1",  # 接口地址（默认已填）
                "sk-good",  # API 密钥（换新值）
                "300000",  # 上下文窗口
                "no",  # 多模态
                "no",  # 回退
            ]
        )
        monkeypatch.setattr(builtins, "input", lambda *a, **k: next(inputs))
        calls = []

        def fake_verify(p, n, b, k):
            calls.append(k)
            return (False, "HTTP 401") if len(calls) == 1 else (True, "")

        monkeypatch.setattr("src.setup._verify_api_key", fake_verify)
        _step_model(config)
        model = config["model"]
        assert model["api_key"] == "sk-good"
        assert model["context_window"] == 300000
        assert len(calls) == 2  # 重输后确实重新验证了一次

    def test_verify_fail_then_abandon_exits(self, monkeypatch):
        """失败 → 放弃 → 退出向导且不保存（模型为必配项）。"""
        config = {}
        inputs = iter(
            [
                "custom",  # 选择提供商
                "m1",  # 模型名称
                "https://x/v1",  # 接口地址
                "sk-bad",  # API 密钥
                "放弃",  # 验证失败 → 放弃
            ]
        )
        monkeypatch.setattr(builtins, "input", lambda *a, **k: next(inputs))
        monkeypatch.setattr("src.setup._verify_api_key", lambda p, n, b, k: (False, "HTTP 401"))
        with pytest.raises(SystemExit) as exc_info:
            _step_model(config)
        assert exc_info.value.code == 0
        _no_input_left(inputs)  # 放弃后不再问多模态/回退

    def test_ollama_preset_skips_verification(self, monkeypatch):
        """无密钥预设（ollama）不验证，直接问上下文窗口。"""
        config = {}
        inputs = iter(
            [
                "ollama",  # 选择提供商
                "",  # 模型名称（默认 llama3）
                "",  # 接口地址（默认 localhost:11434）
                "",  # 上下文窗口（默认 1000000）
                "no",  # 多模态
                "no",  # 回退
            ]
        )
        monkeypatch.setattr(builtins, "input", lambda *a, **k: next(inputs))

        def fail_verify(*a, **k):
            raise AssertionError("无密钥预设不应发起验证请求")

        monkeypatch.setattr("src.setup._verify_api_key", fail_verify)
        _step_model(config)
        model = config["model"]
        assert model["name"] == "llama3"
        assert "api_key" not in model
        assert model["context_window"] == 1000000

    def test_existing_config_keep_skips_verification(self, monkeypatch):
        """已配置 → 保留 → 直接进回退配置，不发验证请求。"""
        config = {
            "model": {
                "provider": "openai",
                "name": "m1",
                "api_key": "sk-xxx",
                "context_window": 200000,
            }
        }
        inputs = iter(
            [
                "yes",  # 保留当前模型设置
                "no",  # 添加/编辑回退模型
            ]
        )
        monkeypatch.setattr(builtins, "input", lambda *a, **k: next(inputs))

        def fail_verify(*a, **k):
            raise AssertionError("保留已有配置时不应发起验证请求")

        monkeypatch.setattr("src.setup._verify_api_key", fail_verify)
        _step_model(config)
        assert config["model"]["api_key"] == "sk-xxx"
        _no_input_left(inputs)
