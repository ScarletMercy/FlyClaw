"""Tests for setup wizard — memory save approval mode configuration."""

from __future__ import annotations

import src.setup as setup_mod


def test_configure_save_approval_sets_manual(monkeypatch):
    monkeypatch.setattr(setup_mod, "_ask_choice", lambda prompt, choices, default="": "manual")
    ms = {}
    setup_mod._configure_save_approval(ms)
    assert ms["save_approval_mode"] == "manual"


def test_configure_save_approval_defaults_to_model(monkeypatch):
    monkeypatch.setattr(setup_mod, "_ask_choice", lambda prompt, choices, default="": default)
    ms = {}
    setup_mod._configure_save_approval(ms)
    assert ms["save_approval_mode"] == "model"


def test_configure_save_approval_uses_existing_as_default(monkeypatch):
    """已配置的值作为 default 传给 _ask_choice。"""
    captured = {}

    def fake(prompt, choices, default=""):
        captured["default"] = default
        return default

    monkeypatch.setattr(setup_mod, "_ask_choice", fake)
    ms = {"save_approval_mode": "manual"}
    setup_mod._configure_save_approval(ms)
    assert captured["default"] == "manual"


def test_step_memory_store_asks_save_approval_regardless_of_enabled(monkeypatch):
    """save 审批模式是记忆配置首项，不依赖 memory_store.enabled（与归档开关无关）。"""
    monkeypatch.setattr(setup_mod, "_ask_yn", lambda p, default=True: False)  # 禁用记忆存储
    monkeypatch.setattr(setup_mod, "_configure_vector_memory", lambda ms: None)
    monkeypatch.setattr(setup_mod, "_configure_save_approval", lambda ms: ms.update(save_approval_mode="manual"))

    config = {}
    setup_mod._step_memory_store(config)

    assert config["memory_store"]["save_approval_mode"] == "manual"  # 即使禁用归档也问了
    assert config["memory_store"]["enabled"] is False
