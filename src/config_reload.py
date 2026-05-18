from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.config_watcher import ReloadPlan

if TYPE_CHECKING:
    from src.app import ServiceContainer

logger = logging.getLogger("myclaw.config_reload")


class ReloadExecutor:
    def __init__(self, app: ServiceContainer):
        self._app = app

    async def execute(self, plan: ReloadPlan) -> None:
        if plan.requires_restart:
            logger.warning(
                "Config change requires gateway restart — skipping hot-reload for: %s",
                [a.action for a in plan.actions],
            )
            return

        for action in plan.actions:
            handler = getattr(self, f"_do_{action.action}", None)
            if handler:
                try:
                    await handler()
                    logger.info("Reload action '%s' applied", action.action)
                except Exception as e:
                    logger.error("Reload action '%s' failed: %s", action.action, e, exc_info=True)
            else:
                logger.warning("No handler for reload action '%s'", action.action)

    async def _do_reload_model(self):
        from src.agent.client import create_chain
        new_client = create_chain(self._app.config)
        if self._app.agent_loop:
            self._app.agent_loop._client = new_client
        else:
            logger.warning("agent_loop not initialized, model reload deferred")

    async def _do_reload_cron(self):
        from src.cron.service import CronService
        from src.cron.store import CronStore
        from src.cron.executor import execute_cron_job
        if self._app.cron_service:
            await self._app.cron_service.stop()
        store = CronStore(self._app.config.cron.store_path)
        app = self._app

        async def cron_execute(job):
            return await execute_cron_job(job, app.agent_loop, app.config, app.qq)

        self._app.cron_service = CronService(store, cron_execute, config=self._app.config, channel=self._app.qq)
        await self._app.cron_service.start()

    async def _do_reload_tools(self):
        from src.tools.registry import get_tool_registry
        from src.tools.exec import reset_config_cache
        reset_config_cache()
        tools = list(get_tool_registry().collect())
        if self._app.agent_loop:
            self._app.agent_loop._tools = tools
            self._app.agent_loop._tool_map = {t.name: t for t in tools}

    async def _do_reload_skills(self):
        self._app.skills_cache = []
        await self._do_reload_tools()
        if self._app.agent_loop:
            from src.skills.loader import discover_skills
            from src.skills.prompt import build_skills_prompt

            dirs = self._app._build_skill_directories()
            skills = discover_skills(dirs, self._app.config)
            self._app.skills_cache = skills
            self._app.agent_loop._skills_prompt = build_skills_prompt(skills)

            # Update CommandDispatcher with new skills
            if hasattr(self._app, 'dispatcher'):
                self._app.dispatcher._reload_skills(skills)

    async def _do_reload_memory(self):
        pass

    async def _do_reload_mcp(self):
        try:
            from src.mcp.manager import get_mcp_manager
            mgr = get_mcp_manager()
            if mgr and hasattr(mgr, 'reload'):
                await mgr.reload(self._app.config.mcp)
        except Exception as e:
            logger.warning("MCP reload failed: %s", e)

    async def _do_reload_auth(self):
        try:
            from src.auth.rbac import RBAC
            from src.auth.store import AuthStore
            store = AuthStore(self._app.config.auth.db_path)
            self._app.rbac = RBAC(store, self._app.config)
        except Exception as e:
            logger.warning("Auth reload failed: %s", e)

    async def _do_reload_security(self):
        pass
