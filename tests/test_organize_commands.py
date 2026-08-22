"""Tests for /organize-session and /organize-memory commands."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import AppConfig


def _make_registered(monkeypatch, config):
    """注册所有内置命令，返回 {name: cmd}。"""
    from src.commands.register import register_builtin_commands

    class _Dispatcher:
        def __init__(self):
            self._commands = {}

        def register_builtin(self, name, fn):
            self._commands[name] = fn

    container = MagicMock()
    container.config = config
    dispatcher = _Dispatcher()
    register_builtin_commands(dispatcher, container, tools=[], skills=[])
    return dispatcher._commands, container


@pytest.mark.asyncio
async def test_organize_session_registered_and_dispatches(monkeypatch):
    """命令已注册，调用 run_session_organize，回复含统计。"""
    config = AppConfig()
    config.agents.language = "zh"
    registered, container = _make_registered(monkeypatch, config)
    assert "organize-session" in registered

    run_mock = AsyncMock(
        return_value={
            "sessions_processed": 3,
            "sessions_skipped": 1,
            "errors": [],
            "since_ts": 1000.0,
            "last_session_organize_at": 2000.0,
        }
    )
    with patch("src.services.consolidation_state.run_session_organize", run_mock):
        result = await registered["organize-session"]("", {})
    run_mock.assert_awaited_once_with(container, since_ts=None)
    assert "3" in result and "1" in result  # 处理 3，跳过 1


@pytest.mark.asyncio
async def test_organize_memory_registered_and_dispatches(monkeypatch):
    """命令已注册，调用 run_memory_organize，回复含统计。"""
    config = AppConfig()
    config.agents.language = "zh"
    registered, container = _make_registered(monkeypatch, config)
    assert "organize-memory" in registered

    run_mock = AsyncMock(
        return_value={
            "total_memories": 10,
            "merged": 2,
            "deleted": 1,
            "kept": 7,
            "errors": [],
            "last_memory_organize_at": 2000.0,
        }
    )
    with patch("src.services.consolidation_state.run_memory_organize", run_mock):
        result = await registered["organize-memory"]("", {})
    run_mock.assert_awaited_once_with(container)
    assert "10" in result and "2" in result  # 总数 10，合并 2


@pytest.mark.asyncio
async def test_organize_session_all_passes_since_zero(monkeypatch):
    """/organize-session all → since_ts=0 传给 run_session_organize。"""
    config = AppConfig()
    config.agents.language = "zh"
    registered, container = _make_registered(monkeypatch, config)

    run_mock = AsyncMock(
        return_value={
            "sessions_processed": 5,
            "sessions_skipped": 0,
            "errors": [],
            "since_ts": 0,
            "last_session_organize_at": 3000.0,
        }
    )
    with patch("src.services.consolidation_state.run_session_organize", run_mock):
        result = await registered["organize-session"]("all", {})
    assert run_mock.call_args.kwargs["since_ts"] == 0
    assert "全部" in result
