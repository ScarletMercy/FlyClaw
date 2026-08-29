"""Tests for setup wizard media-understanding sub-step.

镜像 test_setup_wizard_vector.py：验证启用/验证通过/失败→重输/失败→放弃四条路径，
重点锁定失败后与向量记忆对齐的"1=重新输入, 2=放弃"流程（重新输入会重新验证）。
"""

from __future__ import annotations

import builtins


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
        """失败 → 选重新输入 → 回车沿用字段 → 重新验证通过。"""
        config = {"tools": {}}
        inputs = iter(
            [
                "yes",  # 启用媒体理解
                "openai",  # provider
                "gpt-4o-mini",  # name
                "",  # base_url
                "sk-bad",  # api_key
                "1",  # 验证失败 → 重新输入
                "openai",  # provider（回车沿用）
                "gpt-4o-mini",  # name（回车沿用）
                "",  # base_url（回车沿用）
                "sk-bad",  # api_key（回车沿用）
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
        assert len(calls) == 2  # 重新输入后确实重新验证了一次

    def test_validation_fail_then_abandon(self, monkeypatch):
        """失败 → 选放弃 → 禁用媒体理解。"""
        config = {"tools": {}}
        inputs = iter(
            [
                "yes",  # 启用媒体理解
                "openai",  # provider
                "gpt-4o-mini",  # name
                "",  # base_url
                "sk-bad",  # api_key
                "2",  # 验证失败 → 放弃
            ]
        )
        monkeypatch.setattr(builtins, "input", lambda *a, **k: next(inputs))
        monkeypatch.setattr(
            "src.setup._verify_api_key",
            lambda p, n, b, k: (False, "HTTP 401"),
        )
        _step_media_understanding(config)
        assert config["tools"]["media_understanding"]["enabled"] is False
