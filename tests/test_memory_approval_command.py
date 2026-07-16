"""Tests for /memory-approval command (runtime save approval mode switch)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.config import AppConfig


def _make_registered(monkeypatch, config):
    """注册所有内置命令，返回 {name: cmd} 与 save_config spy。"""
    from src.commands.register import register_builtin_commands

    class _Dispatcher:
        def __init__(self):
            self._commands = {}

        def register_builtin(self, name, fn):
            self._commands[name] = fn

    container = MagicMock()
    container.config = config
    dispatcher = _Dispatcher()
    save_spy = MagicMock()
    monkeypatch.setattr("src.config.save_config", save_spy)
    register_builtin_commands(dispatcher, container, tools=[], skills=[])
    return dispatcher._commands, save_spy


@pytest.mark.asyncio
async def test_memory_approval_sets_manual(monkeypatch):
    config = AppConfig()
    config.memory_store.save_approval_mode = "model"
    config.agents.language = "zh"
    registered, save_spy = _make_registered(monkeypatch, config)

    result = await registered["memory-approval"]("manual", {})
    assert config.memory_store.save_approval_mode == "manual"
    assert save_spy.called
    assert "manual" in result


@pytest.mark.asyncio
async def test_memory_approval_invalid_arg_keeps_current(monkeypatch):
    config = AppConfig()
    config.memory_store.save_approval_mode = "model"
    config.agents.language = "zh"
    registered, save_spy = _make_registered(monkeypatch, config)

    await registered["memory-approval"]("bogus", {})
    assert config.memory_store.save_approval_mode == "model"  # 没变
    assert not save_spy.called  # 非法值不存盘


@pytest.mark.asyncio
async def test_memory_approval_no_arg_shows_current(monkeypatch):
    config = AppConfig()
    config.memory_store.save_approval_mode = "manual"
    config.agents.language = "zh"
    registered, _ = _make_registered(monkeypatch, config)

    result = await registered["memory-approval"]("", {})
    assert "manual" in result  # 显示当前模式
