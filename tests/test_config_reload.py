import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import AppConfig, ModelConfig
from src.config_watcher import ReloadAction, ReloadPlan
from src.config_reload import ReloadExecutor


class TestReloadExecutor:
    @pytest.mark.asyncio
    async def test_reload_model_rebuilds_client(self):
        app = MagicMock()
        app.agent_loop = MagicMock()
        app.agent_loop._client = MagicMock()
        app.config = AppConfig(model=ModelConfig(name="gpt-4o-mini"))
        executor = ReloadExecutor(app)
        plan = ReloadPlan(actions=[ReloadAction(action="reload_model")])
        with patch("src.agent.client.create_chain") as mock_chain:
            mock_chain.return_value = MagicMock()
            await executor.execute(plan)
        mock_chain.assert_called_once()
        assert app.agent_loop._client == mock_chain.return_value

    @pytest.mark.asyncio
    async def test_reload_cron_restarts_service(self, tmp_path):
        app = MagicMock()
        old_service = AsyncMock()
        app.cron_service = old_service
        app.config = AppConfig()
        app.config.cron.store_path = str(tmp_path / "cron.db")
        app.qq = MagicMock()
        executor = ReloadExecutor(app)
        plan = ReloadPlan(actions=[ReloadAction(action="reload_cron")])
        with patch("src.cron.service.CronService") as MockCron:
            mock_instance = AsyncMock()
            MockCron.return_value = mock_instance
            await executor.execute(plan)
        old_service.stop.assert_called_once()
        MockCron.assert_called_once()
        from pathlib import Path

        assert MockCron.call_args[0][0].db_path == Path(app.config.cron.store_path)

    @pytest.mark.asyncio
    async def test_reload_tools(self):
        app = MagicMock()
        app.agent_loop = MagicMock()
        app.config = AppConfig()
        executor = ReloadExecutor(app)
        plan = ReloadPlan(actions=[ReloadAction(action="reload_tools")])

        mock_tool = MagicMock()
        mock_tool.name = "test"
        app._collect_builtin_tools.return_value = [mock_tool]
        app.tool_registry = MagicMock()

        await executor.execute(plan)
        assert list(app.agent_loop._tools) == [mock_tool]
        assert app.agent_loop._tool_map == {"test": mock_tool}

    @pytest.mark.asyncio
    async def test_unknown_action_logs_warning(self):
        app = MagicMock()
        app.config = AppConfig()
        executor = ReloadExecutor(app)
        plan = ReloadPlan(actions=[ReloadAction(action="nonexistent")])
        await executor.execute(plan)


# ---------------------------------------------------------------------------
# Bug-fix regression tests
# ---------------------------------------------------------------------------


class TestReloadSecurityResetsCache:
    """Bug #4: _do_reload_security must reset url_safety cache."""

    @pytest.mark.asyncio
    async def test_resets_url_safety_cache(self):
        app = MagicMock()
        app.config = AppConfig()
        executor = ReloadExecutor(app)

        with patch("src.security.url_safety.reset_cache") as mock_reset:
            await executor._do_reload_security()
            mock_reset.assert_called_once()


class TestReloadMemoryResetsSingleton:
    """Bug #2: _do_reload_memory must reset memory_tools module-level singleton."""

    @pytest.mark.asyncio
    async def test_resets_memory_tools_store(self):
        app = MagicMock()
        app.memory_searcher = None
        app.agent_loop = MagicMock()
        config = AppConfig()
        config.memory.enabled = False  # 禁用以跳过初始化逻辑
        app.config = config

        executor = ReloadExecutor(app)

        with patch("src.tools.memory_tools.reset_memory_store", new_callable=AsyncMock) as mock_reset:
            await executor._do_reload_memory()
            mock_reset.assert_awaited_once()


class TestReloadMemoryInvalidatesAgentCache:
    """Bug #3: _do_reload_memory must invalidate agent_loop's memory summary cache."""

    @pytest.mark.asyncio
    async def test_invalidates_agent_loop_cache(self):
        app = MagicMock()
        app.memory_searcher = None
        app.agent_loop = MagicMock()
        config = AppConfig()
        config.memory.enabled = True
        config.memory.db_path = ":memory:"
        config.memory.fts_tokenizer = "unicode61"
        config.model.api_key = ""
        app.config = config

        executor = ReloadExecutor(app)

        # Patch 掉重量级依赖，只测 invalidate 是否被调用
        with (
            patch("src.memory.search.MemorySearcher") as MockSearcher,
            patch("src.memory.store.MemoryStore") as MockStore,
        ):
            mock_store_instance = AsyncMock()
            MockStore.return_value = mock_store_instance

            await executor._do_reload_memory()

            app.agent_loop.invalidate_memory_cache.assert_called_once()


class TestPartialHandlerFailureDoesNotRaise:
    """Bug #1: on_config_reload must not raise on partial handler failure."""

    @pytest.mark.asyncio
    async def test_no_exception_on_partial_failure(self):
        """部分 handler 失败不应抛异常，config 应已更新，无 split-brain。"""
        old_config = AppConfig()
        new_config = AppConfig(model=ModelConfig(name="new-model"))

        app = MagicMock()
        app.agent_loop = MagicMock()
        app.dispatcher = MagicMock()
        app.qq = MagicMock()
        app._reload_executor = MagicMock()
        app._reload_executor.execute = AsyncMock(
            return_value={"succeeded": ["reload_model"], "failed": ["reload_cron"]}
        )

        # 直接调用真实方法（unbound，传 app 作为 self）
        from src.app import ServiceContainer

        await ServiceContainer.on_config_reload(app, old_config, new_config, MagicMock())

        # config 应该已更新为 new_config
        assert app.config is new_config
        # agent_loop._config 也应更新
        assert app.agent_loop._config is new_config
