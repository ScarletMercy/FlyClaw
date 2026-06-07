"""Tests for memory_tools get_memory_store double-checked locking and reset_memory_store."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools import memory_tools


@pytest.fixture(autouse=True)
def _clean_globals():
    """Reset module-level globals before and after each test."""
    original_store = memory_tools.store
    original_init = memory_tools._store_initialized
    memory_tools.store = None
    memory_tools._store_initialized = False
    # Re-create lock to avoid cross-test contamination
    memory_tools._store_lock = asyncio.Lock()
    yield
    memory_tools.store = None
    memory_tools._store_initialized = False
    memory_tools._store_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# get_memory_store — double-checked locking
# ---------------------------------------------------------------------------


class TestGetMemoryStoreDoubleCheck:
    @pytest.mark.asyncio
    async def test_creates_store_on_first_call(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        mock_store = AsyncMock()
        mock_store.initialize = AsyncMock()

        with patch("src.tools.memory_tools.MemoryStore", return_value=mock_store):
            result = await memory_tools.get_memory_store(db_path)

        assert result is mock_store
        mock_store.initialize.assert_awaited_once()
        assert memory_tools._store_initialized is True

    @pytest.mark.asyncio
    async def test_returns_existing_on_second_call(self, tmp_path):
        mock_store = AsyncMock()
        mock_store.initialize = AsyncMock()

        with patch("src.tools.memory_tools.MemoryStore", return_value=mock_store):
            result1 = await memory_tools.get_memory_store(str(tmp_path / "test.db"))
            result2 = await memory_tools.get_memory_store(str(tmp_path / "other.db"))

        assert result1 is result2
        # initialize should only be called once
        mock_store.initialize.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_reset_allows_new_store(self, tmp_path):
        mock_store1 = AsyncMock()
        mock_store1.initialize = AsyncMock()
        mock_store1.close = AsyncMock()

        mock_store2 = AsyncMock()
        mock_store2.initialize = AsyncMock()

        with patch("src.tools.memory_tools.MemoryStore", side_effect=[mock_store1, mock_store2]):
            result1 = await memory_tools.get_memory_store(str(tmp_path / "a.db"))
            await memory_tools.reset_memory_store()
            result2 = await memory_tools.get_memory_store(str(tmp_path / "b.db"))

        assert result1 is mock_store1
        assert result2 is mock_store2
        assert result1 is not result2
        mock_store1.close.assert_awaited_once()
        mock_store2.initialize.assert_awaited_once()


# ---------------------------------------------------------------------------
# reset_memory_store
# ---------------------------------------------------------------------------


class TestResetMemoryStore:
    @pytest.mark.asyncio
    async def test_resets_initialized_flag(self):
        memory_tools._store_initialized = True
        memory_tools.store = AsyncMock()
        memory_tools.store.close = AsyncMock()

        await memory_tools.reset_memory_store()

        assert memory_tools._store_initialized is False
        assert memory_tools.store is None

    @pytest.mark.asyncio
    async def test_closes_existing_store(self):
        mock_store = AsyncMock()
        mock_store.close = AsyncMock()
        memory_tools.store = mock_store

        await memory_tools.reset_memory_store()

        mock_store.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_handles_close_exception_gracefully(self):
        mock_store = AsyncMock()
        mock_store.close = AsyncMock(side_effect=RuntimeError("close failed"))
        memory_tools.store = mock_store

        # Should not raise
        await memory_tools.reset_memory_store()

        assert memory_tools.store is None
        assert memory_tools._store_initialized is False
