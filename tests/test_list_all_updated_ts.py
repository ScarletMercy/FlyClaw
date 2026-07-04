"""Tests that list_all returns updated_ts field."""

from __future__ import annotations

import pytest

from src.tools.memory_tools import MemoryStore


@pytest.fixture
async def store(tmp_path):
    s = MemoryStore(db_path=str(tmp_path / "t.db"))
    await s.initialize()
    yield s
    await s.close()


class TestListAllUpdatedTs:
    @pytest.mark.asyncio
    async def test_list_all_no_query_returns_updated_ts(self, store):
        await store.remember("hello world", key="k1")
        items = await store.list_all()
        assert len(items) == 1
        assert "updated_ts" in items[0]
        assert items[0]["updated_ts"] > 0

    @pytest.mark.asyncio
    async def test_list_all_with_query_returns_updated_ts(self, store):
        await store.remember("邮箱是 a@b.com", key="email1")
        items = await store.list_all("邮箱")
        assert len(items) >= 1
        assert "updated_ts" in items[0]
        assert items[0]["updated_ts"] > 0
