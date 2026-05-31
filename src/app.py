from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from src.agent.client import create_client, create_chain
from src.agent.loop import AgentLoop
from src.agent.state import StateStore
from src.agent.tooldef import ToolDef
from src.channels.qq import QQChannel
from src.channels.weixin import WeixinChannel
from src.config import load_config
from src.cron.executor import execute_cron_job
from src.cron.service import CronService
from src.cron.store import CronStore
from src.session import SessionTracker, SessionRegistry
from src.skills.loader import discover_skills
from src.skills.prompt import build_skills_prompt
from src.commands.dispatcher import CommandDispatcher

logger = logging.getLogger("flyclaw")

_FLYCLAW_DATA_DIR = Path.home() / ".flyclaw" / "data"


def _ensure_flyclaw_data_dir() -> Path:
    _FLYCLAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return _FLYCLAW_DATA_DIR


class ServiceContainer:
    def __init__(self, config=None):
        self.config = config or load_config()
        _ensure_flyclaw_data_dir()
        self.skills_cache: list = []
        self.agent_loop: AgentLoop | None = None
        self.state_store: StateStore | None = None
        self.model_ref = None
        self.qq: QQChannel | None = None
        self.weixin: WeixinChannel | None = None
        self.session_tracker: SessionTracker | None = None
        self.session_registry: SessionRegistry | None = None
        self.dispatcher: CommandDispatcher | None = None
        self.cron_service: CronService | None = None
        self.api = None
        self.memory_store = None
        self.memory_searcher = None
        self.rbac = None
        self.session_index = None
        self.plugin_registry = None
        self.agent_registry = None
        self.run_registry = None
        self.tool_registry = None
        self.approval_manager = None
        self.browser_manager = None
        self.media_understanding_runner = None
        self.background_tasks: set = set()
        self._qq_mu_runner = None
        self._config_path: str = str(Path.home() / ".flyclaw" / "config.yaml")
        self._config_watcher = None
        self._reload_executor = None
        self._startup_sync_task = None

    # ── Skill directories & loading ──────────────────────────────────

    def _build_skill_directories(self) -> list[tuple[str, Path]]:
        dirs: list[tuple[str, Path]] = []
        workspace = Path(self.config.agents.workspace).expanduser().resolve()

        user_skills = Path.home() / ".flyclaw" / "skills"
        user_skills.mkdir(parents=True, exist_ok=True)
        dirs.append(("user", user_skills))

        workspace_skills = workspace / "skills"
        if workspace_skills.exists():
            dirs.append(("workspace", workspace_skills))

        agents_skills = workspace / ".agents" / "skills"
        if agents_skills.exists():
            dirs.append(("agents-project", agents_skills))

        for extra in self.config.skills.extra_dirs or []:
            p = Path(extra).expanduser().resolve()
            if p.exists():
                dirs.append(("extra", p))

        return dirs

    def _reload_skills(self) -> list:
        from src.skills.types import Skill

        dirs = self._build_skill_directories()
        self.skills_cache = discover_skills(dirs, self.config)
        active = [s for s in self.skills_cache if not s.metadata.disable_model_invocation]
        if active:
            logger.info("技能已加载: %d 个活跃, 共 %d 个", len(active), len(self.skills_cache))
            for s in active:
                logger.info("  - %s: %s (%s)", s.name, s.description[:60], s.source)
        else:
            logger.info("未找到技能")

        if self.agent_loop:
            self.agent_loop._skills_prompt = build_skills_prompt(self.skills_cache)
            from src.prompt import _build_skills_section

            self.agent_loop._prompt_skills = (
                "\n".join(_build_skills_section(self.agent_loop._skills_prompt))
                if self.agent_loop._skills_prompt
                else ""
            )
        if self.dispatcher:
            self.dispatcher._reload_skills(self.skills_cache)

        return self.skills_cache

    # ── Tool collection ──────────────────────────────────────────────

    def _collect_builtin_tools(self) -> list[ToolDef]:
        tools: list[ToolDef] = []
        tool_modules = [
            "src.tools.exec",
            "src.tools.file_tools",
            "src.tools.chat_tools",
            "src.tools.ai_tools",
            "src.tools.cron_tools",
            "src.tools.media_understanding_tools",
            "src.tools.session_search_tools",
            "src.tools.web_tools",
            "src.tools.tts_tools",
            "src.tools.memory_tools",
            "src.tools.task_tools",
            "src.skills.manager",
            "src.agents.delegate",
        ]
        if getattr(self.config.tools, "browser", None) and self.config.tools.browser.enabled:
            tool_modules.append("src.tools.browser.tools")
        if getattr(self.config, "canvas", None) and self.config.canvas.enabled:
            tool_modules.append("src.canvas.tool")
        if (
            sys.platform == "win32"
            and getattr(self.config.tools, "windows_use", None)
            and self.config.tools.windows_use.enabled
        ):
            tool_modules.append("src.tools.windows")
        for mod_name in tool_modules:
            try:
                import importlib

                mod = importlib.import_module(mod_name)
                if hasattr(mod, "get_tools"):
                    tools.extend(mod.get_tools())
            except Exception as e:
                logger.debug("Skipping tool module %s: %s", mod_name, e)

        if self.config.plugins.enabled:
            try:
                from src.plugins.registry import get_plugin_registry

                reg = get_plugin_registry()
                tools.extend(reg.collect_tools())
            except Exception:
                pass

        return tools

    # ── Setup: independent subsystems ────────────────────────────────

    def _setup_plugins(self):
        if not self.config.plugins.enabled:
            return
        from src.plugins.registry import PluginRegistry, discover_plugins

        self.plugin_registry = PluginRegistry()
        records = discover_plugins(self.config.plugins.extra_dirs)
        for record in records:
            self.plugin_registry.register_plugin(record)
        logger.info("插件: 已加载 %d 个, %d 个工具", self.plugin_registry.plugin_count, self.plugin_registry.tool_count)

    def _setup_agents(self):
        if not self.config.agents.subagents:
            return
        from src.agents.registry import AgentRegistry
        from src.agents.run_registry import RunRegistry

        self.agent_registry = AgentRegistry()
        subagents = getattr(self.config.agents, "subagents", None)
        if subagents:
            for name, cfg in subagents.items():
                if isinstance(cfg, dict):
                    from src.config import AgentSubconfig

                    cfg = AgentSubconfig(**cfg)
                self.agent_registry.register(name, cfg)
        self.run_registry = RunRegistry()
        logger.info("子代理: 已注册 %d 个", self.agent_registry.count)

    async def _setup_memory(self):
        if not (getattr(self.config, "memory", None) and self.config.memory.enabled):
            return
        try:
            from src.memory.embeddings import EmbeddingProvider
            from src.memory.search import MemorySearcher

            backend = getattr(self.config.memory, "backend", "sqlite")
            dimensions = getattr(self.config.memory, "embedding_dimensions", 1536)
            if backend == "lancedb":
                from src.memory.lance_store import LanceMemoryStore

                lancedb_uri = getattr(
                    self.config.memory, "lancedb_uri", str(Path.home() / ".flyclaw" / "data" / "memory_lancedb")
                )
                store = LanceMemoryStore(
                    self.config.memory.db_path,
                    dimensions=dimensions,
                    fts_tokenizer=self.config.memory.fts_tokenizer,
                    lancedb_uri=lancedb_uri,
                )
            else:
                from src.memory.store import MemoryStore

                store = MemoryStore(
                    self.config.memory.db_path,
                    dimensions=dimensions,
                    fts_tokenizer=self.config.memory.fts_tokenizer,
                )
            await store.initialize()
            self.memory_store = store

            embeddings = None
            if getattr(self.config.memory, "api_key", "") or self.config.model.api_key:
                try:
                    from src.memory.embeddings import EmbeddingProvider

                    embeddings = EmbeddingProvider(self.config.memory, self.config.model)
                except Exception as e:
                    logger.warning("Embedding provider init failed, using FTS5-only: %s", e)

            searcher = MemorySearcher(store, embeddings, self.config.memory)
            self.memory_searcher = searcher

            indexed = 0
            for path_str in self.config.memory.extra_paths:
                p = Path(path_str).expanduser()
                if p.exists() and p.is_file():
                    content = p.read_text(encoding="utf-8")
                    n = await searcher.index_document(str(p), content)
                    indexed += n
                    logger.info("记忆索引: %s (%d 个分块)", p, n)
            logger.info("记忆系统已初始化 (额外路径: %d 个分块)", indexed)
        except Exception as e:
            logger.warning("记忆系统初始化失败: %s", e)

    def _setup_auth(self):
        if not self.config.auth.enabled:
            return
        from src.auth.store import AuthStore
        from src.auth.rbac import RBAC

        auth_store = AuthStore(self.config.auth.db_path)
        self.rbac = RBAC(auth_store, self.config)
        logger.info(
            "RBAC 已初始化 (默认角色=%s, 配对=%s)",
            self.config.auth.default_role,
            self.config.auth.pairing_enabled,
        )

    async def _setup_session_search(self):
        if not self.config.session_search.enabled:
            return
        from src.session_index.store import SessionIndexStore

        store = await SessionIndexStore.create(self.config.session_search.index_path)
        self.session_index = store
        logger.info("会话搜索索引已初始化: %s", self.config.session_search.index_path)
        self._startup_sync_task = asyncio.create_task(self._run_startup_sync(store))

    def _setup_channels_and_sessions(self, skills: list):
        self.session_tracker = SessionTracker(
            idle_reset_minutes=self.config.session.idle_reset_minutes,
        )
        self.session_registry = SessionRegistry()
        self.session_registry.init(str(_FLYCLAW_DATA_DIR / "sessions.json"))
        self.dispatcher = CommandDispatcher(skills if skills else [], config=self.config)

    def _setup_registries(self):
        from src.tools.registry import ToolRegistry
        from src.tools.approval import ApprovalManager

        self.tool_registry = ToolRegistry()
        self.approval_manager = ApprovalManager(data_dir=str(_FLYCLAW_DATA_DIR))

    def _setup_media_understanding(self):
        if not self.config.tools.media_understanding.enabled:
            return
        if self.config.channels.qq.enabled:
            try:
                from src.media_understanding.runner import MediaUnderstandingRunner

                self._qq_mu_runner = MediaUnderstandingRunner(
                    self.config.tools.media_understanding,
                    fallback_api_key=self.config.model.api_key or "",
                )
                self.media_understanding_runner = self._qq_mu_runner
            except Exception as e:
                logger.warning("QQ 多媒体理解初始化失败: %s", e)

    def _setup_browser(self):
        if not self.config.tools.browser.enabled:
            return
        try:
            from src.tools.browser.manager import BrowserManager

            self.browser_manager = BrowserManager()
            logger.info("浏览器管理器已初始化")
        except Exception as e:
            logger.warning("浏览器管理器初始化失败: %s", e)

    def _setup_cron(self):
        if not self.config.cron.enabled:
            return
        cron_store = CronStore(self.config.cron.store_path)

        def _get_cron_channel():
            return self.qq or self.weixin

        async def cron_execute(job):
            return await execute_cron_job(job, self.agent_loop, self.config, _get_cron_channel())

        self.cron_service = CronService(cron_store, cron_execute, config=self.config, channel=None)
        logger.info("定时任务服务已初始化")

    async def _setup_workspace(self):
        from src.tools.file_tools import set_workspace

        workspace_path = str(Path(self.config.agents.workspace).expanduser().resolve())
        Path(workspace_path).mkdir(parents=True, exist_ok=True)
        set_workspace(workspace_path)

        if getattr(self.config, "memory_store", None) and self.config.memory_store.enabled:
            from src.tools.memory_tools import get_memory_store

            await get_memory_store(self.config.memory_store.db_path)

        if getattr(self.config, "task", None) and self.config.task.enabled:
            from src.task.store import get_task_store

            store = get_task_store(self.config.task.db_path)
            await store.initialize()

    def _setup_qq_channel(self):
        self.qq = QQChannel(self.config.channels.qq)
        if self._qq_mu_runner:
            self.qq.set_media_understanding_runner(self._qq_mu_runner)
            logger.info("QQ 频道多媒体理解已初始化")

    def _setup_weixin_channel(self):
        self.weixin = WeixinChannel(self.config.channels.weixin)
        logger.info("微信频道已初始化")

    def _setup_gateway(self):
        from src.gateway import create_gateway
        import src.gateway as _gw_mod

        self.api = create_gateway(self.config, self.agent_loop, self.cron_service)
        _gw_mod._app_ref = self

    # ── Setup: main orchestrator ─────────────────────────────────────

    async def setup(self):
        from src._container import set_container

        set_container(self)

        logger.info("flyclaw 0.1.0 启动中...")
        logger.info("模型: %s/%s", self.config.model.provider, self.config.model.name)
        if not self.config.gateway.auth_token:
            logger.warning("网关认证令牌为空 — 所有认证已禁用")

        # Phase 1: independent subsystems
        self._setup_plugins()
        self._setup_agents()
        await self._setup_memory()
        self._setup_auth()

        # Phase 2: client + tools + skills
        if self.config.model.fallbacks:
            client = create_chain(self.config)
            logger.info("模型链: 主模型 + %d 个回退模型", len(self.config.model.fallbacks))
        else:
            client = create_client(
                self.config.model.provider,
                self.config.model.name,
                self.config.model.temperature,
                base_url=self.config.model.base_url,
                api_key=self.config.model.api_key,
            )

        tools = self._collect_builtin_tools()

        skills: list = []
        if self.config.skills.enabled:
            skills = self._reload_skills()

        skills_prompt = ""
        if skills:
            skills_prompt = build_skills_prompt(skills)

        # Phase 3: agent loop + state
        cp_path = Path(self.config.checkpointer.path).expanduser().resolve()
        cp_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_store = StateStore(str(cp_path))

        self.agent_loop = AgentLoop(
            client=client,
            tools=tools,
            state_store=self.state_store,
            config=self.config,
            skills_prompt=skills_prompt,
            context_window_tokens=self.config.model.context_window,
        )
        logger.info("AgentLoop 已创建: %d 个工具, %d 个技能", len(tools), len(skills))

        # Phase 4: remaining subsystems
        await self._setup_session_search()
        self._setup_channels_and_sessions(skills)
        self._setup_registries()
        for t in tools:
            self.tool_registry.register(t)
        logger.info("ToolRegistry: %d tools registered", len(tools))
        self._setup_media_understanding()
        self._setup_browser()

        self._setup_cron()
        await self._setup_workspace()
        if self.config.channels.qq.enabled:
            self._setup_qq_channel()
        if self.config.channels.weixin.enabled:
            self._setup_weixin_channel()
        self._setup_gateway()

        from src.tools.exec import reset_config_cache

        reset_config_cache()

        return tools, skills

    # ── Startup helpers ──────────────────────────────────────────────

    async def _run_startup_sync(self, store):
        try:
            await asyncio.sleep(2)
            from src.session_index.sync import startup_sync

            await startup_sync(
                store,
                self.state_store,
                tool_max_chars=self.config.session_search.tool_content_max_chars,
            )
        except Exception as e:
            logger.warning("启动同步失败: %s", e)

    async def _start_events(self):
        from src.events import emit_async, get_hook_manager

        await emit_async(
            "app.startup",
            model=f"{self.config.model.provider}/{self.config.model.name}",
            tools_count=len(self.agent_loop._tools) if self.agent_loop else 0,
            skills_count=len(self.skills_cache),
            channels_enabled=[
                ch for ch in ["qq", "weixin"] if getattr(getattr(self.config.channels, ch, None), "enabled", False)
            ],
        )

        try:
            from src.analytics.audit_store import subscribe_audit_to_events

            subscribe_audit_to_events()
            logger.info("审计存储已订阅工具事件")
        except Exception as e:
            logger.warning("审计存储订阅失败: %s", e)

        try:
            from src.events import subscribe_async
            from src.agent.learning import LearningLoop

            learning_loop = LearningLoop(self.config)

            async def _on_session_reset(event, **ctx):
                tid = ctx.get("thread_id", "")
                if not tid:
                    return
                if self.agent_loop:
                    self.agent_loop.invalidate_memory_cache()
                try:
                    state = await self.state_store.aload(tid)
                except Exception:
                    state = None
                if state and state.messages:
                    try:
                        result = await learning_loop.on_session_end(state.messages)
                        if result.get("memories_extracted", 0) > 0:
                            logger.info("学习循环: 从 %s 提取了 %d 条记忆", result["memories_extracted"], tid)
                    except Exception as e:
                        logger.debug("学习循环会话结束处理失败: %s", e)

            subscribe_async("session.reset", _on_session_reset, priority=10)
            logger.info("学习循环已订阅 session.reset 事件")
        except Exception as e:
            logger.warning("学习循环订阅失败: %s", e)

        try:
            hook_mgr = get_hook_manager()
            hook_count = hook_mgr.load_from_config(self.config)
            if hook_count > 0:
                logger.info("已加载 %d 个用户自定义钩子", hook_count)
        except Exception as e:
            logger.warning("加载用户钩子失败: %s", e)

    def _start_security_audit(self):
        if not (getattr(self.config, "security", None) and self.config.security.audit_on_startup):
            return
        try:
            from src.security import run_security_audit

            run_security_audit(self.config)
        except Exception as e:
            logger.warning("安全审计失败: %s", e)

    async def _maybe_run_curator(self):
        curator_cfg = getattr(self.config.skills, "curator", None)
        if not curator_cfg or not curator_cfg.enabled:
            return
        if not self.config.skills.enabled:
            return
        try:
            from src.skills.curator import SkillCurator

            curator = SkillCurator(
                review_interval_days=curator_cfg.interval_hours // 24 if curator_cfg.interval_hours >= 24 else 1,
                stale_after_days=curator_cfg.stale_after_days,
                archive_after_days=curator_cfg.archive_after_days,
            )
            if curator.days_since_last_review() >= curator.review_interval_days:
                result = await curator.review_skills()
                if result.get("changes"):
                    logger.info("Curator auto-review: %s", result["changes"])
                self._reload_skills()
        except Exception as e:
            logger.warning("Curator auto-review failed: %s", e)

    async def _start_canvas(self):
        if not (getattr(self.config, "canvas", None) and self.config.canvas.enabled and self.config.canvas.root):
            return
        from pathlib import Path as _Path
        from src.canvas.server import init_canvas

        canvas_root = _Path(self.config.canvas.root)
        init_canvas(canvas_root)
        if self.config.canvas.live_reload:
            from src.canvas.live_reload import start_canvas_watcher

            await start_canvas_watcher(canvas_root)

    async def _start_session_maintenance(self):
        await self.session_tracker.start_periodic_cleanup(self.state_store)
        if self.config.session.auto_prune and self.state_store:
            try:
                from src.session.pruner import should_prune_now, prune_sessions, vacuum_database

                cp_path = self.config.checkpointer.path
                si_path = self.config.session_search.index_path if self.config.session_search.enabled else None
                if should_prune_now(cp_path, self.config.session.min_interval_hours):
                    logger.info(
                        "自动清理: 检查超过 %d 天的会话",
                        self.config.session.retention_days,
                    )
                    stats = prune_sessions(
                        cp_path,
                        older_than_days=self.config.session.retention_days,
                        session_index_path=si_path,
                    )
                    if stats["sessions_removed"] > 0 and self.config.session.vacuum_after_prune:
                        vacuum_database(cp_path)
                        logger.info(
                            "自动清理完成: 已移除 %d 个会话",
                            stats["sessions_removed"],
                        )
                else:
                    logger.debug("自动清理: 跳过 (在最小间隔时间内)")
            except Exception as e:
                logger.warning("自动清理失败: %s", e)

    async def _start_memory_watcher(self):
        if not (self.memory_searcher and getattr(self.config.memory, "watch", False)):
            return
        try:
            from src.memory.watcher import start_memory_watcher

            await start_memory_watcher(
                self.config.memory.extra_paths,
                lambda path, content: asyncio.ensure_future(self.memory_searcher.index_document(path, content)),
            )
        except Exception as e:
            logger.warning("记忆监视器启动失败: %s", e)

    async def _start_channels(self):
        if self.qq:
            await self.qq.start()
        if self.weixin:
            await self.weixin.start()
        logger.info(
            "网关已就绪: http://%s:%d",
            self.config.gateway.host,
            self.config.gateway.port,
        )
        logger.info("OpenAI 兼容接口: POST /v1/chat/completions")
        logger.info("WebSocket:        ws://%s:%d/ws", self.config.gateway.host, self.config.gateway.port)
        logger.info("健康检查:          GET /healthz")
        if self.cron_service:
            logger.info("定时任务 API:     GET /api/cron/status")

    # ── Startup: main orchestrator ───────────────────────────────────

    async def on_startup(self):
        await self._start_events()
        self._start_security_audit()
        await self._maybe_run_curator()
        if self.cron_service:
            await self.cron_service.start()
        await self._start_canvas()
        await self._start_session_maintenance()
        await self._start_memory_watcher()
        await self._start_channels()

        from src.config_watcher import ConfigWatcher
        from src.config_reload import ReloadExecutor

        self._reload_executor = ReloadExecutor(self)
        self._config_watcher = ConfigWatcher(
            path=self._config_path,
            on_reload=self.on_config_reload,
        )
        await self._config_watcher.start()

    # ── Shutdown ─────────────────────────────────────────────────────

    async def on_shutdown(self):
        try:
            if self._config_watcher:
                await self._config_watcher.stop()
            if self.memory_store:
                await self.memory_store.close()
            if getattr(self.config, "memory_store", None) and self.config.memory_store.enabled:
                try:
                    from src.tools.memory_tools import get_memory_store

                    mem_store = await get_memory_store()
                    await mem_store.close()
                except Exception:
                    pass
            if getattr(self.config, "task", None) and self.config.task.enabled:
                try:
                    from src.task.store import get_task_store

                    task_store = get_task_store()
                    await task_store.close()
                except Exception:
                    pass
            from src.memory.watcher import stop_memory_watcher

            await stop_memory_watcher()
            if getattr(self.config, "canvas", None) and self.config.canvas.enabled and self.config.canvas.live_reload:
                from src.canvas.live_reload import stop_canvas_watcher

                await stop_canvas_watcher()
            if self.cron_service:
                await self.cron_service.stop()
            await self.session_tracker.stop()
            if self.qq:
                await self.qq.stop()
            if self.weixin:
                await self.weixin.stop()
            if self.state_store:
                self.state_store.close()
            if self.session_index:
                await self.session_index.close()
                self.session_index = None
            if self.browser_manager:
                await self.browser_manager.close_all()
            try:
                from src.events import emit_async

                await emit_async(
                    "app.shutdown",
                    active_sessions=self.session_tracker.active_count if self.session_tracker else 0,
                )
            except Exception:
                pass
            try:
                from src.events import get_hook_manager

                get_hook_manager().unload_all()
            except Exception:
                pass
            logger.info("flyclaw 已停止")
        except Exception as e:
            logger.error("关闭过程中出错: %s", e, exc_info=True)

    async def on_config_reload(self, old_config, new_config, plan):
        self.config = new_config
        await self._reload_executor.execute(plan)
