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
    async def test_reload_cron_restarts_service(self):
        app = MagicMock()
        old_service = AsyncMock()
        app.cron_service = old_service
        app.config = AppConfig()
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
        with patch("src.tools.registry.get_tool_registry") as mock_reg:
            mock_tool = MagicMock()
            mock_tool.name = "test"
            mock_reg.return_value.collect.return_value = [mock_tool]
            await executor.execute(plan)
        assert app.agent_loop._tools == [mock_tool]

    @pytest.mark.asyncio
    async def test_unknown_action_logs_warning(self):
        app = MagicMock()
        app.config = AppConfig()
        executor = ReloadExecutor(app)
        plan = ReloadPlan(actions=[ReloadAction(action="nonexistent")])
        await executor.execute(plan)
