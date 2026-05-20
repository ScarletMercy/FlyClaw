"""Tests for complete config reload flow — all action handlers."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import AppConfig
from src.config_watcher import ReloadAction, ReloadPlan
from src.config_reload import ReloadExecutor


def _make_app():
    app = MagicMock()
    app.config = AppConfig()
    app.agent_loop = MagicMock()
    app.agent_loop._client = MagicMock()
    app.agent_loop._tools = []
    app.agent_loop._tool_map = {}
    app.agent_loop._skills_prompt = ""
    app.qq = MagicMock()
    app.cron_service = None
    app.skills_cache = []
    return app


class TestReloadModel:
    @pytest.mark.asyncio
    async def test_reload_model_creates_new_client(self):
        app = _make_app()
        executor = ReloadExecutor(app)
        plan = ReloadPlan(actions=[ReloadAction(action="reload_model")])
        with patch("src.agent.client.create_chain") as mock_chain:
            mock_chain.return_value = MagicMock()
            await executor.execute(plan)
        mock_chain.assert_called_once_with(app.config)
        assert app.agent_loop._client == mock_chain.return_value


class TestReloadCron:
    @pytest.mark.asyncio
    async def test_reload_cron_stops_old_starts_new(self):
        app = _make_app()
        old_svc = AsyncMock()
        app.cron_service = old_svc
        executor = ReloadExecutor(app)
        plan = ReloadPlan(actions=[ReloadAction(action="reload_cron")])

        with patch("src.cron.service.CronService") as MockCron:
            new_svc = AsyncMock()
            MockCron.return_value = new_svc
            await executor.execute(plan)

        old_svc.stop.assert_awaited_once()
        MockCron.assert_called_once()
        new_svc.start.assert_awaited_once()
        assert app.cron_service == new_svc


class TestReloadTools:
    @pytest.mark.asyncio
    async def test_reload_tools_updates_loop(self):
        app = _make_app()
        executor = ReloadExecutor(app)
        plan = ReloadPlan(actions=[ReloadAction(action="reload_tools")])

        with patch("src.tools.registry.get_tool_registry") as mock_reg:
            t1 = MagicMock()
            t1.name = "tool_a"
            t2 = MagicMock()
            t2.name = "tool_b"
            mock_reg.return_value.collect.return_value = [t1, t2]
            await executor.execute(plan)

        assert len(app.agent_loop._tools) == 2
        assert "tool_a" in app.agent_loop._tool_map
        assert "tool_b" in app.agent_loop._tool_map


class TestReloadAuth:
    @pytest.mark.asyncio
    async def test_reload_auth_creates_new_rbac(self):
        app = _make_app()
        executor = ReloadExecutor(app)
        plan = ReloadPlan(actions=[ReloadAction(action="reload_auth")])

        with patch("src.auth.rbac.RBAC") as MockRBAC, \
             patch("src.auth.store.AuthStore") as MockStore:
            MockStore.return_value = MagicMock()
            MockRBAC.return_value = MagicMock()
            await executor.execute(plan)

        MockRBAC.assert_called_once()
        assert app.rbac == MockRBAC.return_value


class TestRequiresRestart:
    @pytest.mark.asyncio
    async def test_requires_restart_skips_all_actions(self):
        app = _make_app()
        executor = ReloadExecutor(app)
        plan = ReloadPlan(
            actions=[
                ReloadAction(action="reload_model"),
                ReloadAction(action="reload_tools"),
            ],
            requires_restart=True,
        )

        with patch("src.agent.client.create_chain") as mock_chain, \
             patch("src.tools.registry.get_tool_registry") as mock_reg:
            await executor.execute(plan)

        mock_chain.assert_not_called()
        mock_reg.assert_not_called()
