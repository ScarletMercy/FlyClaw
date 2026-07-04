"""Tests for config_reload vec searcher hot reload.

验证 _do_reload_memory 在 memory reload 时：
1. 旧 vec searcher 被 close（通过 reset_memory_archive_searcher）
2. app.memory_archive_searchers 置 None
3. 真实 _setup_memory_archive 重建新 searcher（不 spy，跑真实 LanceMemoryStore 构造）

主 memory 系统走真实 sqlite backend（tmp_path 隔离），vec 走真实 LanceMemoryStore。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.app import ServiceContainer
from src.config import MemoryConfig, MemoryStoreConfig
from src.config_reload import ReloadExecutor
from src.tools.memory_tools import (
    reset_memory_archive_searcher,
    set_memory_archive_searcher,
)


class FakeSearcher:
    """追踪 close 调用的假 searcher。"""

    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


class TestConfigReloadVec:
    @pytest.mark.asyncio
    async def test_reload_closes_old_and_rebuilds(self, tmp_path, monkeypatch):
        """reload 流程：close 旧 vec searcher → reset 单例 → 真实 _setup_memory_archive 重建新 searcher。

        不 spy _setup_memory_archive——跑真实 LanceMemoryStore 构造，断言新 searcher 是真的
        MemorySearcher 实例（不是 Fake，不是 None），且 app.memory_archive_searchers 是新元组。
        """
        import src.instance
        from src.memory.search import MemorySearcher

        # group vec 路径用 data_dir()，重定向到 tmp_path
        monkeypatch.setattr(src.instance, "data_dir", lambda: tmp_path)

        # 注册旧 vec searcher 到模块单例 + container
        old_dm = FakeSearcher()
        old_group = FakeSearcher()
        await set_memory_archive_searcher(old_dm, "p2p")
        await set_memory_archive_searcher(old_group, "group")

        app = ServiceContainer.__new__(ServiceContainer)
        app.memory_searcher = None
        app.memory_archive_searchers = (old_dm, old_group)
        app.agent_loop = None
        # 不 spy _setup_memory_archive —— 让真实方法跑

        app.config = SimpleNamespace(
            memory_store=MemoryStoreConfig(
                enabled=True,
                vector_enabled=True,
                vector_model="text-embedding-3-small",
                vector_base_url="https://api.example.com",
                vector_api_key="sk-test",
                vector_dimensions=4,
                vector_db_path=str(tmp_path / "dm_vec.db"),
            ),
            # 主 memory 走 sqlite backend（无需 lancedb），embeddings 留空
            memory=MemoryConfig(
                enabled=True,
                backend="sqlite",
                db_path=str(tmp_path / "mem.db"),
                api_key="",
            ),
            model=SimpleNamespace(api_key="", base_url="https://api.example.com"),
        )

        executor = ReloadExecutor(app)
        await executor._do_reload_memory()

        # 旧 vec searcher 被 reset_memory_archive_searcher close 掉
        assert old_dm.closed, "old DM searcher was not closed"
        assert old_group.closed, "old group searcher was not closed"

        # 真实重建：app.memory_archive_searchers 是新元组，元素是真实 MemorySearcher
        assert app.memory_archive_searchers is not None, "memory_archive_searchers should be rebuilt"
        assert len(app.memory_archive_searchers) == 2
        new_dm, new_group = app.memory_archive_searchers
        assert isinstance(new_dm, MemorySearcher), f"expected MemorySearcher, got {type(new_dm)}"
        assert isinstance(new_group, MemorySearcher), f"expected MemorySearcher, got {type(new_group)}"
        # 新 searcher 不是旧的 Fake
        assert new_dm is not old_dm
        assert new_group is not old_group

        # 清理：close 新建的 searcher + memory_searcher
        for s in app.memory_archive_searchers:
            try:
                await s.close()
            except Exception:
                pass
        if app.memory_searcher:
            try:
                await app.memory_searcher.close()
            except Exception:
                pass
        await reset_memory_archive_searcher()

    @pytest.mark.asyncio
    async def test_reload_skips_vec_when_no_existing_searchers(self, tmp_path, monkeypatch):
        """无旧 vec searcher 时 reload 不调 reset_memory_archive_searcher，但仍跑 _setup_memory_archive 重建。"""
        import src.instance
        from src.memory.search import MemorySearcher

        monkeypatch.setattr(src.instance, "data_dir", lambda: tmp_path)
        await reset_memory_archive_searcher()  # 确保模块单例干净

        app = ServiceContainer.__new__(ServiceContainer)
        app.memory_searcher = None
        app.memory_archive_searchers = None  # 无旧 searcher
        app.agent_loop = None

        app.config = SimpleNamespace(
            memory_store=MemoryStoreConfig(
                enabled=True,
                vector_enabled=True,
                vector_model="text-embedding-3-small",
                vector_base_url="https://api.example.com",
                vector_api_key="sk-test",
                vector_dimensions=4,
                vector_db_path=str(tmp_path / "dm_vec.db"),
            ),
            memory=MemoryConfig(
                enabled=True,
                backend="sqlite",
                db_path=str(tmp_path / "mem.db"),
                api_key="",
            ),
            model=SimpleNamespace(api_key="", base_url="https://api.example.com"),
        )

        executor = ReloadExecutor(app)
        await executor._do_reload_memory()

        # 无旧 searcher → reset 块跳过，但 _setup_memory_archive 仍跑重建
        assert app.memory_archive_searchers is not None, "should rebuild even without old searchers"
        assert len(app.memory_archive_searchers) == 2
        assert isinstance(app.memory_archive_searchers[0], MemorySearcher)

        for s in app.memory_archive_searchers:
            try:
                await s.close()
            except Exception:
                pass
        if app.memory_searcher:
            try:
                await app.memory_searcher.close()
            except Exception:
                pass
        await reset_memory_archive_searcher()
