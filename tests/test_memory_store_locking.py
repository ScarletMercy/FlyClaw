"""Tests for memory_tools get_memory_store double-checked locking and reset_memory_store."""

import asyncio
from unittest.mock import AsyncMock, patch

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


# ---------------------------------------------------------------------------
# Concurrent access — the real reason the lock exists
# ---------------------------------------------------------------------------


class TestGetMemoryStoreConcurrency:
    """Verify double-checked locking under actual concurrent coroutines."""

    @pytest.mark.asyncio
    async def test_concurrent_calls_return_same_store(self, tmp_path):
        db_path = str(tmp_path / "concurrent.db")
        results = await asyncio.gather(*[memory_tools.get_memory_store(db_path) for _ in range(20)])
        # All callers must get the exact same object
        assert all(s is results[0] for s in results)
        # Store is functional
        await results[0].remember("concurrent test", key="c1")
        await results[0].close()

    @pytest.mark.asyncio
    async def test_concurrent_initialize_called_once(self, tmp_path):
        init_count = 0

        async def slow_initialize():
            nonlocal init_count
            init_count += 1
            await asyncio.sleep(0.05)

        mock_store = AsyncMock()
        mock_store.initialize = slow_initialize

        with patch("src.tools.memory_tools.MemoryStore", return_value=mock_store):
            results = await asyncio.gather(
                *[memory_tools.get_memory_store(str(tmp_path / "init.db")) for _ in range(10)]
            )

        assert init_count == 1, f"initialize called {init_count} times, expected 1"
        assert all(s is results[0] for s in results)

    @pytest.mark.asyncio
    async def test_concurrent_reset_and_get(self, tmp_path):
        db1 = str(tmp_path / "first.db")
        db2 = str(tmp_path / "second.db")

        store1 = await memory_tools.get_memory_store(db1)
        assert store1 is not None

        # Concurrently reset and get — ordering is non-deterministic
        await asyncio.gather(
            memory_tools.reset_memory_store(),
            memory_tools.get_memory_store(db2),
        )

        # Regardless of interleaving, a subsequent get must succeed
        store_final = await memory_tools.get_memory_store(db2)
        assert store_final is not None
        assert memory_tools._store_initialized is True
        await store_final.close()

    @pytest.mark.asyncio
    async def test_concurrent_reset_calls(self):
        memory_tools.store = AsyncMock()
        memory_tools.store.close = AsyncMock()
        memory_tools._store_initialized = True

        await asyncio.gather(*[memory_tools.reset_memory_store() for _ in range(5)])

        assert memory_tools.store is None
        assert memory_tools._store_initialized is False


# ---------------------------------------------------------------------------
# Real MemoryStore integration (tmp_path SQLite, no mocking)
# ---------------------------------------------------------------------------


class TestGetMemoryStoreRealStore:
    """Integration tests using a real MemoryStore backed by tmp_path SQLite."""

    @pytest.mark.asyncio
    async def test_real_store_initialize_and_use(self, tmp_path):
        db_path = str(tmp_path / "real.db")
        store = await memory_tools.get_memory_store(db_path)
        assert store is not None
        assert memory_tools._store_initialized is True

        result = await store.remember("real integration test", key="rk1")
        assert '"ok"' in result or '"key"' in result
        await memory_tools.reset_memory_store()

    @pytest.mark.asyncio
    async def test_real_store_reset_and_reinit(self, tmp_path):
        db1 = str(tmp_path / "db1.db")
        store1 = await memory_tools.get_memory_store(db1)
        await store1.remember("first db data", key="k1")

        await memory_tools.reset_memory_store()
        assert memory_tools.store is None
        assert memory_tools._store_initialized is False

        # New store with different db_path — should start empty
        db2 = str(tmp_path / "db2.db")
        store2 = await memory_tools.get_memory_store(db2)
        assert store2 is not store1
        all_mems = await store2.list_all()
        assert len(all_mems) == 0
        await memory_tools.reset_memory_store()

    @pytest.mark.asyncio
    async def test_real_store_idempotent_get(self, tmp_path):
        db_path = str(tmp_path / "idem.db")
        s1 = await memory_tools.get_memory_store(db_path)
        s2 = await memory_tools.get_memory_store(db_path)
        # Second call ignores db_path — returns same singleton
        assert s1 is s2
        await memory_tools.reset_memory_store()
