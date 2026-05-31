from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.config_watcher import ReloadPlan

if TYPE_CHECKING:
    from src.app import ServiceContainer

logger = logging.getLogger("flyclaw.config_reload")


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

        app = self._app
        old = app.cron_service
        old_path = str(old.store.db_path) if old else ""
        new_path = app.config.cron.store_path

        if old and old_path == new_path:
            # Store path unchanged — lightweight reload
            logger.info("Cron reload: reusing store, rescheduling")
            if old._scheduler:
                old._scheduler.shutdown(wait=False)
                old._scheduler = None
            await old._drain_running_tasks()
            # Update closure references (agent_loop/config may have changed)
            async def cron_execute(job):
                return await execute_cron_job(job, app.agent_loop, app.config, app.qq)

            old.execute_fn = cron_execute
            old._config = app.config
            old._channel = app.qq
            await old.reschedule()
        else:
            # Store path changed or first init — full restart
            logger.info("Cron reload: full restart")
            if old:
                await old.stop()
            store = CronStore(new_path)

            async def cron_execute(job):
                return await execute_cron_job(job, app.agent_loop, app.config, app.qq)

            app.cron_service = CronService(store, cron_execute, config=app.config, channel=app.qq)
            await app.cron_service.start()

    async def _do_reload_tools(self):
        from src.tools.exec import reset_config_cache

        reset_config_cache()
        tools = self._app._collect_builtin_tools()
        registry = self._app.tool_registry
        if registry is not None:
            registry._tools = list(tools)
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

            from src.prompt import _build_skills_section

            self._app.agent_loop._prompt_skills = (
                "\n".join(_build_skills_section(self._app.agent_loop._skills_prompt))
                if self._app.agent_loop._skills_prompt
                else ""
            )

            # Update CommandDispatcher with new skills
            if hasattr(self._app, "dispatcher"):
                self._app.dispatcher._reload_skills(skills)

    async def _do_reload_auth(self):
        try:
            from src.auth.rbac import RBAC
            from src.auth.store import AuthStore

            store = AuthStore(self._app.config.auth.db_path)
            self._app.rbac = RBAC(store, self._app.config)
        except Exception as e:
            logger.warning("Auth reload failed: %s", e)
