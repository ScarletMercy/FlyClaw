from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from src.agent.client import create_client, create_chain
from src.agent.loop import AgentLoop
from src.agent.state import StateStore
from src.agent.tooldef import ToolDef
from src.channels.qq import QQChannel
from src.config import load_config
from src.cron.executor import execute_cron_job
from src.cron.service import CronService
from src.cron.store import CronStore
from src.session import SessionTracker, SessionRegistry
from src.skills.loader import discover_skills
from src.skills.prompt import build_skills_prompt
from src.commands.dispatcher import CommandDispatcher

if TYPE_CHECKING:
    from src.skills.types import Skill

logger = logging.getLogger("myclaw")

_MYCLAW_DATA_DIR = Path.home() / ".myclaw" / "data"


def _ensure_myclaw_data_dir() -> Path:
    _MYCLAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    old_data = Path("data")
    if old_data.exists() and not (old_data / ".migrated").exists():
        migrated_count = 0
        for f in old_data.glob("*"):
            if f.is_file() and not f.name.startswith("."):
                try:
                    import shutil
                    shutil.copy2(f, _MYCLAW_DATA_DIR / f.name)
                    migrated_count += 1
                except Exception as e:
                    logger.warning("Failed to migrate %s: %s", f.name, e)
        if migrated_count > 0:
            (old_data / ".migrated").touch()
            logger.info("Migrated %d files from project data/ to %s", migrated_count, _MYCLAW_DATA_DIR)
    return _MYCLAW_DATA_DIR


class ServiceContainer:
    def __init__(self, config=None):
        self.config = config or load_config()
        _ensure_myclaw_data_dir()
        self.skills_cache: list = []
        self.agent_loop: AgentLoop | None = None
        self.state_store: StateStore | None = None
        self.model_ref = None
        self.qq: QQChannel | None = None
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
        self.mcp_manager = None
        self.tool_registry = None
        self.approval_manager = None
        self.browser_manager = None
        self.media_understanding_runner = None
        self.process_supervisor = None
        self.background_tasks: set = set()
        self._qq_mu_runner = None
        self._config_path: str = "config.yaml"
        self._config_watcher = None
        self._reload_executor = None
        self._startup_sync_task = None

    # ── Skill directories & loading ──────────────────────────────────

    def _build_skill_directories(self) -> list[tuple[str, Path]]:
        dirs: list[tuple[str, Path]] = []
        workspace = Path(self.config.agents.workspace).expanduser().resolve()

        user_skills = Path.home() / ".myclaw" / "skills"
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
            logger.info("Skills loaded: %d active, %d total", len(active), len(self.skills_cache))
            for s in active:
                logger.info("  - %s: %s (%s)", s.name, s.description[:60], s.source)
        else:
            logger.info("No skills found")

        if self.agent_loop:
            self.agent_loop._skills_prompt = build_skills_prompt(self.skills_cache)
        if self.dispatcher:
            self.dispatcher._reload_skills(self.skills_cache)

        return self.skills_cache

    # ── Tool collection ──────────────────────────────────────────────

    def _collect_builtin_tools(self) -> list[ToolDef]:
        tools: list[ToolDef] = []
        tool_modules = [
            "src.tools.exec",
            "src.tools.file_tools",
            "src.tools.qq_tools",
            "src.tools.ai_tools",
            "src.tools.cron_tools",
            "src.tools.media_tools",
            "src.tools.media_understanding_tools",
            "src.tools.session_search_tools",
            "src.tools.web_tools",
            "src.tools.tts_tools",
            "src.tools.memory_tools",
            "src.tools.task_tools",
            "src.tools.mcp_tools",
            "src.memory.memory_sync",
            "src.skills.manager",
            "src.skills.curator",
            "src.agent.learning",
            "src.agents.delegate",
            "src.memory.procedures",
        ]
        if getattr(self.config.tools, "browser", None) and self.config.tools.browser.enabled:
            tool_modules.append("src.tools.browser.tools")
        if getattr(self.config, "canvas", None) and self.config.canvas.enabled:
            tool_modules.append("src.canvas.tool")
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

        if getattr(self.config, "mcp", None) and self.config.mcp.enabled:
            try:
                from src.mcp.adapter import get_mcp_tools
                tools.extend(get_mcp_tools())
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
        logger.info("Plugins: %d loaded, %d tools", self.plugin_registry.plugin_count, self.plugin_registry.tool_count)

    async def _setup_mcp(self):
        from src.mcp.manager import MCPManager
        self.mcp_manager = MCPManager()
        if getattr(self.config, "mcp", None) and self.config.mcp.enabled and self.config.mcp.servers:
            self.mcp_manager.load_config(self.config.mcp.servers)
            logger.info("MCP: %d servers configured", len(self.config.mcp.servers))
            # Eagerly connect all servers so tools are available at startup
            for name in list(self.config.mcp.servers):
                try:
                    await self.mcp_manager.ensure_connected(name)
                except Exception as e:
                    logger.warning("MCP server '%s' failed at startup: %s", name, e)
        else:
            logger.info("MCP: manager initialized, no servers configured")

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
        logger.info("Sub-agents: %d registered", self.agent_registry.count)

    async def _setup_memory(self):
        if not (getattr(self.config, "memory", None) and self.config.memory.enabled):
            return
        try:
            from src.memory.embeddings import EmbeddingProvider
            from src.memory.search import MemorySearcher

            backend = getattr(self.config.memory, "backend", "sqlite")
            if backend == "lancedb":
                from src.memory.lance_store import LanceMemoryStore
                store = LanceMemoryStore(
                    self.config.memory.db_path,
                    dimensions=self.config.memory.embedding_dimensions,
                    fts_tokenizer=self.config.memory.fts_tokenizer,
                    lancedb_uri=getattr(self.config.memory, "lancedb_uri", str(Path.home() / ".myclaw" / "data" / "memory_lancedb")),
                )
            else:
                from src.memory.store import MemoryStore
                store = MemoryStore(
                    self.config.memory.db_path,
                    dimensions=self.config.memory.embedding_dimensions,
                    fts_tokenizer=self.config.memory.fts_tokenizer,
                )
            await store.initialize()
            self.memory_store = store

            embeddings = EmbeddingProvider(self.config.memory, self.config.model)
            searcher = MemorySearcher(store, embeddings, self.config.memory)
            self.memory_searcher = searcher

            indexed = 0
            for path_str in self.config.memory.extra_paths:
                p = Path(path_str).expanduser()
                if p.exists() and p.is_file():
                    content = p.read_text(encoding="utf-8")
                    n = await searcher.index_document(str(p), content)
                    indexed += n
                    logger.info("Memory indexed: %s (%d chunks)", p, n)
            logger.info("Memory system initialized (extra paths: %d chunks)", indexed)
        except Exception as e:
            logger.warning("Failed to initialize memory system: %s", e)

    def _setup_auth(self):
        if not self.config.auth.enabled:
            return
        from src.auth.store import AuthStore
        from src.auth.rbac import RBAC
        auth_store = AuthStore(self.config.auth.db_path)
        self.rbac = RBAC(auth_store, self.config)
        logger.info(
            "RBAC initialized (default_role=%s, pairing=%s)",
            self.config.auth.default_role,
            self.config.auth.pairing_enabled,
        )

    def _setup_session_search(self):
        if not self.config.session_search.enabled:
            return
        from src.session_index.store import SessionIndexStore
        store = SessionIndexStore(self.config.session_search.index_path)
        self.session_index = store
        logger.info("Session search index initialized: %s", self.config.session_search.index_path)
        self._startup_sync_task = asyncio.create_task(self._run_startup_sync(store))

    def _setup_channels_and_sessions(self, skills: list):
        self.session_tracker = SessionTracker(
            idle_reset_minutes=self.config.session.idle_reset_minutes,
        )
        self.session_registry = SessionRegistry()
        self.session_registry.init(str(_MYCLAW_DATA_DIR / "sessions.json"))
        self.dispatcher = CommandDispatcher(skills if skills else [], config=self.config)

    def _setup_registries(self):
        from src.tools.registry import ToolRegistry
        from src.tools.approval import ApprovalManager
        self.tool_registry = ToolRegistry()
        self.approval_manager = ApprovalManager(data_dir=str(_MYCLAW_DATA_DIR))

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
                logger.warning("Failed to init QQ media understanding: %s", e)

    def _setup_browser(self):
        if not self.config.tools.browser.enabled:
            return
        try:
            from src.tools.browser.manager import BrowserManager
            self.browser_manager = BrowserManager()
            logger.info("Browser manager initialized")
        except Exception as e:
            logger.warning("Failed to init browser manager: %s", e)

    def _setup_cron(self):
        if not self.config.cron.enabled:
            return
        cron_store = CronStore(self.config.cron.store_path)

        async def cron_execute(job):
            return await execute_cron_job(job, self.agent_loop, self.config, self.qq)

        self.cron_service = CronService(cron_store, cron_execute, config=self.config, channel=self.qq)
        logger.info("Cron service initialized")

    async def _setup_workspace(self):
        from src.tools.file_tools import set_workspace
        workspace_path = str(Path(self.config.agents.workspace).expanduser().resolve())
        Path(workspace_path).mkdir(parents=True, exist_ok=True)
        set_workspace(workspace_path)

        if getattr(self.config, "memory_store", None) and self.config.memory_store.enabled:
            from src.tools.memory_tools import get_memory_store
            store = get_memory_store(self.config.memory_store.db_path)
            await store.initialize()

        if getattr(self.config, "task", None) and self.config.task.enabled:
            from src.task.store import get_task_store
            store = get_task_store(self.config.task.db_path)
            await store.initialize()

    def _setup_qq_channel(self):
        self.qq = QQChannel(self.config.channels.qq)
        if self._qq_mu_runner:
            self.qq.set_media_understanding_runner(self._qq_mu_runner)
            logger.info("Media understanding runner initialized for QQ channel")

    def _setup_gateway(self):
        from src.gateway import create_gateway
        import src.gateway as _gw_mod
        self.api = create_gateway(self.config, self.agent_loop, self.cron_service)
        _gw_mod._app_ref = self

    # ── Setup: main orchestrator ─────────────────────────────────────

    async def setup(self):
        from src._container import set_container
        set_container(self)

        logger.info("MyClaw 0.1.0 starting...")
        logger.info("Model: %s/%s", self.config.model.provider, self.config.model.name)
        if not self.config.gateway.auth_token:
            logger.warning("Gateway auth_token is empty — all authentication is DISABLED")

        # Phase 1: independent subsystems
        self._setup_plugins()
        await self._setup_mcp()
        self._setup_agents()
        await self._setup_memory()
        self._setup_auth()

        # Phase 2: client + tools + skills
        if self.config.model.fallbacks:
            client = create_chain(self.config)
            logger.info("Model chain: primary + %d fallbacks", len(self.config.model.fallbacks))
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
        logger.info("AgentLoop created with %d tools, %d skills", len(tools), len(skills))

        # Phase 4: remaining subsystems
        self._setup_session_search()
        self._setup_channels_and_sessions(skills)
        self._setup_registries()
        self._setup_media_understanding()
        self._setup_browser()

        from src.tools.process import ProcessSupervisor
        self.process_supervisor = ProcessSupervisor()

        self._setup_cron()
        await self._setup_workspace()
        self._setup_qq_channel()
        self._setup_gateway()

        from src.tools.exec import reset_config_cache
        reset_config_cache()

        if self.config.procedural_memory.enabled and self.config.procedural_memory.auto_learn:
            try:
                from src.memory.procedures import register_extraction_listener
                register_extraction_listener()
            except Exception as e:
                logger.warning("Failed to register procedure extraction listener: %s", e)

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
            logger.warning("Startup sync failed: %s", e)

    async def _start_events(self):
        from src.events import emit_async, get_hook_manager

        await emit_async(
            "app.startup",
            model=f"{self.config.model.provider}/{self.config.model.name}",
            tools_count=len(self.agent_loop._tools) if self.agent_loop else 0,
            skills_count=len(self.skills_cache),
            channels_enabled=[ch for ch in ["qq"] if getattr(getattr(self.config.channels, ch, None), "enabled", False)],
        )

        try:
            from src.analytics.audit_store import subscribe_audit_to_events
            subscribe_audit_to_events()
            logger.info("Audit store subscribed to tool events")
        except Exception as e:
            logger.warning("Failed to subscribe audit store: %s", e)

        try:
            from src.events import subscribe_async
            from src.agent.learning import LearningLoop
            learning_loop = LearningLoop(self.config)

            async def _on_session_reset(event, **ctx):
                tid = ctx.get("thread_id", "")
                if not tid:
                    return
                try:
                    state = await self.state_store.aload(tid)
                except Exception:
                    state = None
                if state and state.messages:
                    try:
                        result = await learning_loop.on_session_end(state.messages)
                        if result.get("memories_extracted", 0) > 0:
                            logger.info("Learning loop: extracted %d memories from %s", result["memories_extracted"], tid)
                    except Exception as e:
                        logger.debug("Learning loop session-end failed: %s", e)

            subscribe_async("session.reset", _on_session_reset, priority=10)
            logger.info("Learning loop subscribed to session.reset events")
        except Exception as e:
            logger.warning("Failed to subscribe learning loop: %s", e)

        try:
            hook_mgr = get_hook_manager()
            hook_count = hook_mgr.load_from_config(self.config)
            if hook_count > 0:
                logger.info("Loaded %d user-defined hooks", hook_count)
        except Exception as e:
            logger.warning("Failed to load user hooks: %s", e)

    def _start_security_audit(self):
        if not (getattr(self.config, "security", None) and self.config.security.audit_on_startup):
            return
        try:
            from src.security import run_security_audit
            run_security_audit(self.config)
        except Exception as e:
            logger.warning("Security audit failed: %s", e)

    async def _start_skills_watcher(self):
        if not (self.config.skills.enabled and self.config.skills.watch):
            return
        from src.skills.watcher import start_skills_watcher
        await start_skills_watcher(
            self._build_skill_directories(),
            lambda: self._reload_skills(),
        )

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
                        "Auto-prune: checking for sessions older than %d days",
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
                            "Auto-prune complete: removed %d sessions",
                            stats["sessions_removed"],
                        )
                else:
                    logger.debug("Auto-prune: skipping (within min_interval_hours)")
            except Exception as e:
                logger.warning("Auto-prune failed: %s", e)

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
            logger.warning("Failed to start memory watcher: %s", e)

    async def _start_channels(self):
        await self.qq.start()
        logger.info(
            "Gateway ready: http://%s:%d",
            self.config.gateway.host,
            self.config.gateway.port,
        )
        logger.info("OpenAI compat: POST /v1/chat/completions")
        logger.info("WebSocket:     ws://%s:%d/ws", self.config.gateway.host, self.config.gateway.port)
        logger.info("Health:        GET /healthz")
        if self.cron_service:
            logger.info("Cron API:      GET /api/cron/status")

    # ── Startup: main orchestrator ───────────────────────────────────

    async def on_startup(self):
        await self._start_events()
        self._start_security_audit()
        await self._start_skills_watcher()
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
                    mem_store = get_memory_store()
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
            if self.config.skills.enabled and self.config.skills.watch:
                from src.skills.watcher import stop_skills_watcher
                await stop_skills_watcher()
            from src.memory.watcher import stop_memory_watcher
            await stop_memory_watcher()
            if getattr(self.config, "canvas", None) and self.config.canvas.enabled and self.config.canvas.live_reload:
                from src.canvas.live_reload import stop_canvas_watcher
                await stop_canvas_watcher()
            if self.cron_service:
                await self.cron_service.stop()
            await self.session_tracker.stop()
            await self.qq.stop()
            try:
                from src.mcp.manager import get_mcp_manager
                mcp_mgr = get_mcp_manager()
                await mcp_mgr.disconnect_all()
            except Exception:
                pass
            if self.state_store:
                self.state_store.close()
            if self.session_index:
                self.session_index.close()
                self.session_index = None
            if self.browser_manager:
                await self.browser_manager.close_all()
            try:
                from src.memory.procedures import reset_procedure_store
                await reset_procedure_store()
            except Exception:
                pass
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
            logger.info("MyClaw stopped")
        except Exception as e:
            logger.error("Error during shutdown: %s", e, exc_info=True)

    async def on_config_reload(self, old_config, new_config, plan):
        self.config = new_config
        await self._reload_executor.execute(plan)
