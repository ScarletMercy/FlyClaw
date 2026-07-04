"""Tests for app._setup_memory_archive wiring."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.app import ServiceContainer


class TestSetupMemoryArchive:
    @pytest.mark.asyncio
    async def test_skips_when_memory_store_disabled(self):
        """memory_store.enabled=False → 不构造 archive searcher。"""
        app = ServiceContainer.__new__(ServiceContainer)
        app.config = SimpleNamespace(
            memory_store=SimpleNamespace(enabled=False, vector_enabled=True),
            model=SimpleNamespace(api_key="", base_url=""),
            memory=SimpleNamespace(min_score=0.35, max_results=6),
        )
        app.memory_archive_searchers = None
        await app._setup_memory_archive()
        assert app.memory_archive_searchers is None

    @pytest.mark.asyncio
    async def test_vector_on_constructs_lance_searchers(self, tmp_path, monkeypatch):
        """vector_enabled=True → LanceMemoryStore + EmbeddingProvider，构造 DM + group searcher。"""
        from src.config import MemoryStoreConfig, MemoryConfig
        from src.tools.memory_tools import reset_memory_archive_searcher, get_memory_archive_searcher
        from src.memory.lance_store import LanceMemoryStore
        from src.memory.search import MemorySearcher
        import src.instance

        await reset_memory_archive_searcher()
        monkeypatch.setattr(src.instance, "data_dir", lambda: tmp_path)

        app = ServiceContainer.__new__(ServiceContainer)
        ms = MemoryStoreConfig(
            enabled=True,
            vector_enabled=True,
            vector_model="text-embedding-3-small",
            vector_base_url="https://api.example.com",
            vector_api_key="sk-test",
            vector_dimensions=4,
            vector_db_path=str(tmp_path / "dm_vec.db"),
        )
        app.config = SimpleNamespace(
            memory_store=ms,
            model=SimpleNamespace(api_key="sk-test", base_url="https://api.example.com"),
            memory=MemoryConfig(),
        )
        app.memory_archive_searchers = None

        await app._setup_memory_archive()

        assert app.memory_archive_searchers is not None
        assert len(app.memory_archive_searchers) == 2
        dm, group = app.memory_archive_searchers
        assert isinstance(dm, MemorySearcher)
        assert isinstance(group, MemorySearcher)
        # vector on → LanceMemoryStore + embeddings 非 None
        assert isinstance(dm.store, LanceMemoryStore)
        assert dm.embeddings is not None
        # 模块单例已注册
        assert await get_memory_archive_searcher("p2p") is dm
        assert await get_memory_archive_searcher("group") is group

        await reset_memory_archive_searcher()

    @pytest.mark.asyncio
    async def test_vector_off_uses_memory_store_fts5_only(self, tmp_path, monkeypatch):
        """vector_enabled=False → MemoryStore（FTS5-only），embeddings=None，仍构造 DM + group。

        关键：archive 不依赖 vector_enabled——FTS5-only 也能归档。
        """
        from src.config import MemoryStoreConfig, MemoryConfig
        from src.tools.memory_tools import reset_memory_archive_searcher, get_memory_archive_searcher
        from src.memory.store import MemoryStore
        from src.memory.lance_store import LanceMemoryStore
        from src.memory.search import MemorySearcher
        import src.instance

        await reset_memory_archive_searcher()
        monkeypatch.setattr(src.instance, "data_dir", lambda: tmp_path)

        app = ServiceContainer.__new__(ServiceContainer)
        ms = MemoryStoreConfig(
            enabled=True,
            vector_enabled=False,  # 关键：vector off
            vector_db_path=str(tmp_path / "dm_archive.db"),
        )
        app.config = SimpleNamespace(
            memory_store=ms,
            model=SimpleNamespace(api_key="", base_url=""),
            memory=MemoryConfig(),
        )
        app.memory_archive_searchers = None

        await app._setup_memory_archive()

        assert app.memory_archive_searchers is not None
        assert len(app.memory_archive_searchers) == 2
        dm, group = app.memory_archive_searchers
        # vector off → MemoryStore（不是 LanceMemoryStore），embeddings=None
        assert isinstance(dm.store, MemoryStore), f"expected MemoryStore, got {type(dm.store)}"
        assert not isinstance(dm.store, LanceMemoryStore)
        assert dm.embeddings is None, "vector off should have no embeddings"
        assert group.embeddings is None

        await reset_memory_archive_searcher()

    @pytest.mark.asyncio
    async def test_shutdown_closes_archive_searchers(self, monkeypatch):
        """on_shutdown 关闭 archive searcher 并置 None。"""
        import src.memory.watcher as _mw
        import src.agent.tool_cache as _tc
        import src.events as _ev

        async def _noop_async(*a, **kw):
            pass

        def _noop(*a, **kw):
            pass

        monkeypatch.setattr(_mw, "stop_memory_watcher", _noop_async)
        monkeypatch.setattr(_tc, "clear_all_caches", _noop)
        monkeypatch.setattr(_ev, "emit_async", _noop_async)

        class _FakeHookMgr:
            def unload_all(self):
                pass

        monkeypatch.setattr(_ev, "get_hook_manager", lambda: _FakeHookMgr())

        closed = []

        class FakeSearcher:
            async def close(self):
                closed.append(True)

        class _FakeSessionTracker:
            active_count = 0

            async def stop(self):
                pass

        app = ServiceContainer.__new__(ServiceContainer)
        app._consolidation_scheduler = None
        app._config_watcher = None
        app.memory_searcher = None
        app.memory_archive_searchers = (FakeSearcher(), FakeSearcher())
        app.config = SimpleNamespace(
            memory_store=SimpleNamespace(enabled=False),
            task=SimpleNamespace(enabled=False),
            canvas=SimpleNamespace(enabled=False, live_reload=False),
        )
        app.session_tracker = _FakeSessionTracker()
        app.rbac = None
        app.qq = None
        app.weixin = None
        app.state_store = None
        app.agent_loop = None
        app.session_index = None
        app.browser_manager = None
        app.cron_service = None

        await app.on_shutdown()

        assert len(closed) == 2
        assert app.memory_archive_searchers is None
