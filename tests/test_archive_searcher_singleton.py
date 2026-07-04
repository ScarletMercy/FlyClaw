"""Tests for archive searcher singleton accessors."""

from __future__ import annotations

import pytest

from src.tools.memory_tools import (
    get_memory_archive_searcher,
    reset_memory_archive_searcher,
    set_memory_archive_searcher,
)


class FakeSearcher:
    pass


@pytest.mark.asyncio
async def test_get_returns_none_when_not_set():
    await reset_memory_archive_searcher()
    assert await get_memory_archive_searcher("p2p") is None
    assert await get_memory_archive_searcher("group") is None


@pytest.mark.asyncio
async def test_set_and_get_by_chat_type():
    await reset_memory_archive_searcher()
    dm = FakeSearcher()
    grp = FakeSearcher()
    await set_memory_archive_searcher(dm, "p2p")
    await set_memory_archive_searcher(grp, "group")
    assert await get_memory_archive_searcher("p2p") is dm
    assert await get_memory_archive_searcher("group") is grp


@pytest.mark.asyncio
async def test_reset_clears_singleton():
    await set_memory_archive_searcher(FakeSearcher(), "p2p")
    await reset_memory_archive_searcher()
    assert await get_memory_archive_searcher("p2p") is None


@pytest.mark.asyncio
async def test_set_closes_previous_searcher_on_overwrite():
    """同 chat_type 重新注册 searcher 时，旧 searcher 应被 close。

    回归 bug #3 根源：set_memory_archive_searcher 直接覆盖不 close 旧实例。
    config_reload 在部分失败后无条件重跑 _setup_memory_archive 会覆盖 DM 单例，
    旧 DM store 的 SQLite/LanceDB 句柄泄漏。
    """

    class TrackingSearcher:
        def __init__(self, name):
            self.name = name
            self.closed = False

        async def close(self):
            self.closed = True

    await reset_memory_archive_searcher()
    old = TrackingSearcher("old")
    new = TrackingSearcher("new")
    try:
        await set_memory_archive_searcher(old, "p2p")
        await set_memory_archive_searcher(new, "p2p")  # 覆盖
        assert old.closed, "覆盖注册时旧 searcher 应被 close，避免 SQLite/LanceDB 句柄泄漏"
    finally:
        await reset_memory_archive_searcher()
