from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from src.config_watcher import ReloadPlan

if TYPE_CHECKING:
    from src.app import ServiceContainer

logger = logging.getLogger("flyclaw.config_reload")


class ReloadExecutor:
    def __init__(self, app: ServiceContainer):
        self._app = app

    async def execute(self, plan: ReloadPlan) -> dict:
        if plan.requires_restart:
            logger.warning(
                "Config change requires gateway restart — hot-reloadable changes will still be applied: %s",
                [a.action for a in plan.actions],
            )

        succeeded: list[str] = []
        failed: list[str] = []

        for action in plan.actions:
            handler = getattr(self, f"_do_{action.action}", None)
            if handler:
                try:
                    await handler()
                    succeeded.append(action.action)
                    logger.info("Reload action '%s' applied", action.action)
                except Exception as e:
                    failed.append(action.action)
                    logger.error("Reload action '%s' failed: %s", action.action, e, exc_info=True)
            else:
                logger.warning("No handler for reload action '%s'", action.action)

        return {"succeeded": succeeded, "failed": failed}

    async def _do_reload_model(self):
        from src.agent.client import create_chain

        new_client = create_chain(self._app.config)
        if self._app.agent_loop:
            old_client = self._app.agent_loop._client
            self._app.agent_loop.swap_client(new_client)
            # 关闭旧客户端连接池，避免资源泄漏
            if hasattr(old_client, "close"):
                try:
                    await old_client.close()
                except Exception:
                    pass
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
        from src.tools import web_tools

        reset_config_cache()
        web_tools._cached_api_key = None

        # Invalidate skill reference directory cache so newly discovered skills'
        # references/ dirs become accessible after hot-reload.
        import src.tools.file_tools as _ft

        _ft._skill_ref_dirs_cache = None
        old_client = web_tools._tavily_client
        web_tools._tavily_client = None
        if old_client is not None:
            try:
                await old_client.close()
            except Exception:
                pass
        tools = self._app._collect_builtin_tools()
        registry = self._app.tool_registry
        if registry is not None:
            registry._tools = list(tools)
        if self._app.agent_loop:
            self._app.agent_loop._tools = tools
            self._app.agent_loop._tool_map = {t.name: t for t in tools}
            self._app.agent_loop._cache_prompt_sections(tools, self._app.agent_loop._skills_prompt)

    async def _do_reload_skills(self):
        self._app.skills_cache = []
        await self._do_reload_tools()
        if self._app.agent_loop:
            from src.skills.loader import discover_skills
            from src.skills.prompt import build_skills_prompt

            dirs = self._app._build_skill_directories()
            skills = await discover_skills(dirs, self._app.config)
            self._app.skills_cache = skills
            self._app.agent_loop._skills_prompt = build_skills_prompt(skills)

            from src.prompt import _build_skills_section

            hub_on = getattr(self._app.config.skills.hub, "enabled", True)
            self._app.agent_loop._prompt_skills = "\n".join(
                _build_skills_section(self._app.agent_loop._skills_prompt, hub_enabled=hub_on)
            )

            # Update CommandDispatcher with new skills
            if hasattr(self._app, "dispatcher"):
                self._app.dispatcher._reload_skills(skills)

    async def _do_reload_auth(self):
        try:
            from src.auth.rbac import RBAC
            from src.auth.store import AuthStore

            old_store = self._app.rbac.store if self._app.rbac else None
            store = AuthStore(self._app.config.auth.db_path)
            self._app.rbac = RBAC(store, self._app.config)
            if old_store:
                await old_store.close()
                logger.info("Old AuthStore connection closed on reload")
        except Exception as e:
            logger.warning("Auth reload failed: %s", e)

    async def _do_reload_memory(self):
        app = self._app
        config = app.config

        if not (getattr(config, "memory", None) and config.memory.enabled):
            # Memory disabled — clean up old components first, then reset singleton
            if app.memory_searcher:
                try:
                    await app.memory_searcher.close()
                except Exception:
                    pass
                app.memory_searcher = None
            try:
                from src.tools.memory_tools import reset_memory_store

                await reset_memory_store()
            except Exception:
                pass
            logger.info("Memory system disabled after reload")
            return

        # --- Build new components BEFORE tearing down old ones ---
        store = None
        new_searcher = None
        try:
            from src.memory.search import MemorySearcher

            backend = getattr(config.memory, "backend", "sqlite")
            dimensions = getattr(config.memory, "embedding_dimensions", 1536)

            if backend == "sqlite":
                from src.memory.store import MemoryStore

                store = MemoryStore(
                    config.memory.db_path,
                    dimensions=dimensions,
                    fts_tokenizer=config.memory.fts_tokenizer,
                )
            else:
                from src.memory.factory import make_vector_store

                from src.instance import data_dir

                lancedb_uri = getattr(config.memory, "lancedb_uri", str(data_dir() / "memory_lancedb"))
                store = make_vector_store(
                    backend=backend,
                    db_path=config.memory.db_path,
                    dimensions=dimensions,
                    fts_tokenizer=config.memory.fts_tokenizer,
                    lancedb_uri=lancedb_uri,
                )
            await store.initialize()

            embeddings = None
            if getattr(config.memory, "api_key", "") or config.model.api_key:
                try:
                    from src.memory.embeddings import EmbeddingProvider

                    embeddings = EmbeddingProvider(config.memory, config.model)
                except Exception as e:
                    logger.warning("Embedding provider init failed, using FTS5-only: %s", e)

            new_searcher = MemorySearcher(store, embeddings, config.memory)
            store = None  # 所有权已转移给 MemorySearcher，不再需要清理

            # --- Atomic swap: install new, tear down old ---
            old_searcher = app.memory_searcher
            app.memory_searcher = new_searcher
            new_searcher = None  # ownership transferred

            # Reset module-level singleton so future get_memory_store() creates fresh instance
            try:
                from src.tools.memory_tools import reset_memory_store

                await reset_memory_store()
            except Exception:
                pass

            # Close old searcher AFTER swap
            if old_searcher:
                try:
                    await old_searcher.close()
                except Exception:
                    pass

            # 清空 agent_loop 的记忆摘要缓存
            if app.agent_loop:
                app.agent_loop.invalidate_memory_cache()

            # KV archive searcher 热重载：先 close + reset 模块单例，再按新 config 重建
            if app.memory_archive_searchers:
                from src.tools.memory_tools import reset_memory_archive_searcher

                await reset_memory_archive_searcher()
                app.memory_archive_searchers = None
            try:
                await app._setup_memory_archive()
            except Exception as e:
                logger.warning("memory archive searcher reload failed: %s", e)

            logger.info("Memory system reloaded")
        except Exception as e:
            # 清理已初始化但未移交的组件
            if new_searcher is not None:
                try:
                    await new_searcher.close()
                except Exception:
                    pass
            if store is not None:
                try:
                    await store.close()
                except Exception:
                    pass
            logger.error("Memory reload failed: %s", e, exc_info=True)
            raise

    async def _do_reload_security(self):
        # 重置 url_safety 模块级缓存，使 allow_private_urls 变更立即生效
        from src.security.url_safety import reset_cache

        reset_cache()

        config = self._app.config.security
        logger.info(
            "Security config reloaded: enabled=%s, audit_on_startup=%s, allow_private_urls=%s",
            config.enabled,
            config.audit_on_startup,
            config.allow_private_urls,
        )
