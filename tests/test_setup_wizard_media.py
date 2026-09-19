"""Tests for setup wizard media-understanding sub-step.

镜像 test_setup_wizard_vector.py：验证启用/验证通过/失败→重试/失败→重输/失败→放弃
路径。失败菜单为三选项（重试/重输/放弃，对齐 chcode）：重试原值重验不重问字段，
重输重收字段后重验，放弃禁用媒体理解。
"""

from __future__ import annotations

import builtins

import pytest

from src.setup import _step_media_understanding


class TestWizardMedia:
    def test_disabled_skips_media(self, monkeypatch):
        config = {"tools": {}}
        # 启用媒体理解=no
        inputs = iter(["no"])
        monkeypatch.setattr(builtins, "input", lambda *a, **k: next(inputs))
        _step_media_understanding(config)
        assert config["tools"]["media_understanding"]["enabled"] is False

    def test_enabled_with_validated_inputs(self, monkeypatch):
        config = {"tools": {}}
        inputs = iter(
            [
                "yes",  # 启用媒体理解
                "openai",  # provider（回车沿用默认）
                "gpt-4o-mini",  # name
                "",  # base_url
                "sk-xxx",  # api_key
            ]
        )
        monkeypatch.setattr(builtins, "input", lambda *a, **k: next(inputs))
        monkeypatch.setattr(
            "src.setup._verify_api_key",
            lambda p, n, b, k: (True, ""),
        )
        _step_media_understanding(config)
        mu = config["tools"]["media_understanding"]
        assert mu["enabled"] is True
        assert mu["name"] == "gpt-4o-mini"
        assert mu["api_key"] == "sk-xxx"

    def test_validation_fail_then_retry_succeeds(self, monkeypatch):
        """失败 → 重试 → 不重问字段，原值直接重验通过。"""
        config = {"tools": {}}
        inputs = iter(
            [
                "yes",  # 启用媒体理解
                "openai",  # provider
                "gpt-4o-mini",  # name
                "",  # base_url
                "sk-flaky",  # api_key
                "重试",  # 验证失败 → 重试（不应再消耗任何输入）
            ]
        )
        monkeypatch.setattr(builtins, "input", lambda *a, **k: next(inputs))
        calls = []

        def fake_verify(p, n, b, k):
            calls.append(k)
            return (False, "HTTP 503") if len(calls) == 1 else (True, "")

        monkeypatch.setattr("src.setup._verify_api_key", fake_verify)
        _step_media_understanding(config)
        mu = config["tools"]["media_understanding"]
        assert mu["enabled"] is True
        assert len(calls) == 2  # 重试确实重验了一次
        with pytest.raises(StopIteration):
            next(inputs)  # 重试未重问字段：输入已耗尽

    def test_validation_fail_then_reinput_succeeds(self, monkeypatch):
        """失败 → 重输 → 重收字段 → 重新验证通过。"""
        config = {"tools": {}}
        inputs = iter(
            [
                "yes",  # 启用媒体理解
                "openai",  # provider
                "gpt-4o-mini",  # name
                "",  # base_url
                "sk-bad",  # api_key
                "重输",  # 验证失败 → 重输
                "openai",  # provider（回车沿用）
                "gpt-4o-mini",  # name（回车沿用）
                "",  # base_url（回车沿用）
                "sk-good",  # api_key（换新值）
            ]
        )
        monkeypatch.setattr(builtins, "input", lambda *a, **k: next(inputs))
        calls = []

        def fake_verify(p, n, b, k):
            calls.append(k)
            return (False, "HTTP 401") if len(calls) == 1 else (True, "")

        monkeypatch.setattr("src.setup._verify_api_key", fake_verify)
        _step_media_understanding(config)
        mu = config["tools"]["media_understanding"]
        assert mu["enabled"] is True
        assert mu["api_key"] == "sk-good"
        assert len(calls) == 2  # 重输后确实重新验证了一次

    def test_validation_fail_then_abandon(self, monkeypatch):
        """失败 → 放弃 → 禁用媒体理解。"""
        config = {"tools": {}}
        inputs = iter(
            [
                "yes",  # 启用媒体理解
                "openai",  # provider
                "gpt-4o-mini",  # name
                "",  # base_url
                "sk-bad",  # api_key
                "放弃",  # 验证失败 → 放弃
            ]
        )
        monkeypatch.setattr(builtins, "input", lambda *a, **k: next(inputs))
        monkeypatch.setattr(
            "src.setup._verify_api_key",
            lambda p, n, b, k: (False, "HTTP 401"),
        )
        _step_media_understanding(config)
        assert config["tools"]["media_understanding"]["enabled"] is False

    def test_existing_config_keep_skips_verification(self, monkeypatch):
        """已配置 → 保留 → 直接返回，不发验证请求。"""
        config = {
            "tools": {
                "media_understanding": {
                    "enabled": True,
                    "provider": "openai",
                    "name": "gpt-4o-mini",
                    "base_url": "",
                    "api_key": "sk-xxx",
                }
            }
        }
        inputs = iter(["yes"])  # 保留当前媒体理解设置
        monkeypatch.setattr(builtins, "input", lambda *a, **k: next(inputs))

        def fail_verify(*a, **k):
            raise AssertionError("保留已有配置时不应发起验证请求")

        monkeypatch.setattr("src.setup._verify_api_key", fail_verify)
        _step_media_understanding(config)
        mu = config["tools"]["media_understanding"]
        assert mu["enabled"] is True
        assert mu["api_key"] == "sk-xxx"

    def test_existing_config_reconfigure(self, monkeypatch):
        """已配置 → 不保留 → 走重新配置流程。"""
        config = {
            "tools": {
                "media_understanding": {
                    "enabled": True,
                    "provider": "openai",
                    "name": "gpt-4o-mini",
                    "base_url": "",
                    "api_key": "sk-xxx",
                }
            }
        }
        inputs = iter(
            [
                "no",  # 保留当前媒体理解设置 → 不保留
                "no",  # 启用媒体理解 → 关闭
            ]
        )
        monkeypatch.setattr(builtins, "input", lambda *a, **k: next(inputs))
        _step_media_understanding(config)
        assert config["tools"]["media_understanding"]["enabled"] is False
