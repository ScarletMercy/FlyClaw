from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from src.agent.client import create_client, create_chain
from src.agent.loop import AgentLoop
from src.agent.state import StateStore
from src.agent.tooldef import ToolDef
from src.channels.feishu import FeishuChannel
from src.channels.qq import QQChannel
from src.channels.typing import TypingIndicator
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
    """Create ~/.myclaw/data/ and migrate from project data/ if needed."""
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
        self.feishu: FeishuChannel | None = None
        self.qq: QQChannel | None = None
        self.session_tracker: SessionTracker | None = None
        self.session_registry: SessionRegistry | None = None
        self.typing: TypingIndicator | None = None
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
        self.card_callback_registry = None
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

    def _build_skill_directories(self) -> list[tuple[str, Path]]:
        dirs: list[tuple[str, Path]] = []
        workspace = Path(self.config.agents.workspace).expanduser().resolve()

        # 1. user skills (~/.myclaw/skills/) - primary user location
        user_skills = Path.home() / ".myclaw" / "skills"
        user_skills.mkdir(parents=True, exist_ok=True)
        dirs.append(("user", user_skills))

        # 2. workspace skills (~/.myclaw/workspace/skills/)
        workspace_skills = workspace / "skills"
        if workspace_skills.exists():
            dirs.append(("workspace", workspace_skills))

        # 3. agents-project skills (~/.myclaw/workspace/.agents/skills/)
        agents_skills = workspace / ".agents" / "skills"
        if agents_skills.exists():
            dirs.append(("agents-project", agents_skills))

        # 4. extra dirs (from config)
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

    def _collect_builtin_tools(self) -> list[ToolDef]:
        tools: list[ToolDef] = []
        tool_modules = [
            "src.tools.exec",
            "src.tools.file_tools",
            "src.tools.feishu_tools",
            "src.tools.qq_tools",
            "src.tools.ai_tools",
            "src.tools.cron_tools",
            "src.tools.media_tools",
            "src.tools.media_understanding_tools",
            "src.tools.session_search_tools",
            "src.tools.web_tools",
            "src.tools.beads_tools",
            "src.memory.beads_sync",
            "src.tools.skills_tools",
            "src.skills.manager",
            "src.skills.curator",
            "src.agent.learning",
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

    async def setup(self):
        from src._container import set_container

        set_container(self)

        logger.info("MyClaw 0.1.0 starting...")
        logger.info("Model: %s/%s", self.config.model.provider, self.config.model.name)
        if not self.config.gateway.auth_token:
            logger.warning("Gateway auth_token is empty — all authentication is DISABLED")

        if self.config.plugins.enabled:
            from src.plugins.registry import PluginRegistry, discover_plugins

            self.plugin_registry = PluginRegistry()
            records = discover_plugins(self.config.plugins.extra_dirs)
            for record in records:
                self.plugin_registry.register_plugin(record)
            logger.info("Plugins: %d loaded, %d tools", self.plugin_registry.plugin_count, self.plugin_registry.tool_count)

        if getattr(self.config, "mcp", None) and self.config.mcp.enabled and self.config.mcp.servers is not None:
            from src.mcp.manager import MCPManager

            self.mcp_manager = MCPManager()
            self.mcp_manager.load_config(self.config.mcp.servers)
            logger.info("MCP: %d servers configured", len(self.config.mcp.servers))

        if self.config.agents.subagents:
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

        if getattr(self.config, "memory", None) and self.config.memory.enabled:
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

        if self.config.auth.enabled:
            from src.auth.store import AuthStore
            from src.auth.rbac import RBAC

            auth_store = AuthStore(self.config.auth.db_path)
            self.rbac = RBAC(auth_store, self.config)
            logger.info(
                "RBAC initialized (default_role=%s, pairing=%s)",
                self.config.auth.default_role,
                self.config.auth.pairing_enabled,
            )

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

        cp_path = Path(self.config.checkpointer.path)
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

        if self.config.session_search.enabled:
            from src.session_index.store import SessionIndexStore

            store = SessionIndexStore(self.config.session_search.index_path)
            self.session_index = store
            logger.info("Session search index initialized: %s", self.config.session_search.index_path)

            self._startup_sync_task = asyncio.create_task(
                self._run_startup_sync(store)
            )

        self.feishu = FeishuChannel(self.config.channels.feishu)
        session_scope = self.config.session.scope
        self.session_tracker = SessionTracker(
            idle_reset_minutes=self.config.session.idle_reset_minutes,
            beads_config=self.config.beads if self.config.beads.enabled else None,
        )
        from src.session import SessionRegistry

        self.session_registry = SessionRegistry()
        self.session_registry.init(str(_MYCLAW_DATA_DIR / "sessions.json"))

        from src.channels.typing import TypingIndicator

        self.typing = TypingIndicator(self.feishu.client, enabled=self.config.channels.feishu.typing_indicator)

        self.dispatcher = CommandDispatcher(skills if skills else [], config=self.config)

        from src.tools.registry import ToolRegistry
        from src.channels.cards import CardCallbackRegistry
        from src.tools.approval import ApprovalManager

        self.tool_registry = ToolRegistry()
        self.card_callback_registry = CardCallbackRegistry()
        self.approval_manager = ApprovalManager(data_dir=str(_MYCLAW_DATA_DIR))

        if self.config.tools.media_understanding.enabled:
            try:
                from src.media_understanding.runner import MediaUnderstandingRunner
                from src.channels.feishu import set_media_understanding_runner

                mu_runner = MediaUnderstandingRunner(
                    self.config.tools.media_understanding,
                    fallback_api_key=self.config.model.api_key or "",
                )
                self.media_understanding_runner = mu_runner
                set_media_understanding_runner(mu_runner)
                logger.info("Media understanding runner initialized for Feishu channel")
            except Exception as e:
                logger.warning("Failed to init media understanding: %s", e)

        if self.config.channels.qq.enabled and self.config.tools.media_understanding.enabled:
            try:
                from src.media_understanding.runner import MediaUnderstandingRunner

                qq_runner = MediaUnderstandingRunner(
                    self.config.tools.media_understanding,
                    fallback_api_key=self.config.model.api_key or "",
                )
                self._qq_mu_runner = qq_runner
            except Exception as e:
                logger.warning("Failed to init QQ media understanding: %s", e)

        if self.config.tools.browser.enabled:
            try:
                from src.tools.browser.manager import BrowserManager

                self.browser_manager = BrowserManager()
                logger.info("Browser manager initialized")
            except Exception as e:
                logger.warning("Failed to init browser manager: %s", e)

        from src.tools.process import ProcessSupervisor

        self.process_supervisor = ProcessSupervisor()

        if self.config.cron.enabled:
            cron_store = CronStore(self.config.cron.store_path)

            async def cron_execute(job):
                return await execute_cron_job(job, self.agent_loop, self.config, self.feishu)

            self.cron_service = CronService(cron_store, cron_execute, config=self.config, feishu_channel=self.feishu)
            logger.info("Cron service initialized")

        from src.tools.file_tools import set_workspace

        workspace_path = str(Path(self.config.agents.workspace).expanduser().resolve())
        Path(workspace_path).mkdir(parents=True, exist_ok=True)
        set_workspace(workspace_path)

        if getattr(self.config, "beads", None) and self.config.beads.enabled:
            from src.tools.beads_tools import set_beads_workspace

            beads_ws = self.config.beads.workspace or workspace_path
            beads_ws = str(Path(beads_ws).expanduser().resolve())
            Path(beads_ws).mkdir(parents=True, exist_ok=True)
            set_beads_workspace(beads_ws)

        self.qq = QQChannel(self.config.channels.qq)
        if self._qq_mu_runner:
            self.qq.set_media_understanding_runner(self._qq_mu_runner)
            logger.info("Media understanding runner initialized for QQ channel")

        from src.gateway import create_gateway
        import src.gateway as _gw_mod

        self.api = create_gateway(self.config, self.agent_loop, self.feishu, self.cron_service)
        _gw_mod._app_ref = self

        from src.tools.exec import reset_config_cache

        reset_config_cache()

        return tools, skills

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

    async def on_startup(self):
        if getattr(self.config, "security", None) and self.config.security.audit_on_startup:
            try:
                from src.security import run_security_audit

                run_security_audit(self.config)
            except Exception as e:
                logger.warning("Security audit failed: %s", e)

        if self.config.skills.enabled and self.config.skills.watch:
            from src.skills.watcher import start_skills_watcher

            await start_skills_watcher(
                self._build_skill_directories(),
                lambda: self._reload_skills(),
            )

        if self.cron_service:
            await self.cron_service.start()

        if getattr(self.config, "canvas", None) and self.config.canvas.enabled and self.config.canvas.root:
            from pathlib import Path as _Path
            from src.canvas.server import init_canvas

            canvas_root = _Path(self.config.canvas.root)
            init_canvas(canvas_root)
            if self.config.canvas.live_reload:
                from src.canvas.live_reload import start_canvas_watcher

                await start_canvas_watcher(canvas_root)

        await self.session_tracker.start_periodic_cleanup(self.state_store)

        if self.memory_searcher and getattr(self.config.memory, "watch", False):
            try:
                from src.memory.watcher import start_memory_watcher

                await start_memory_watcher(
                    self.config.memory.extra_paths,
                    lambda path, content: asyncio.ensure_future(self.memory_searcher.index_document(path, content)),
                )
            except Exception as e:
                logger.warning("Failed to start memory watcher: %s", e)

        await self.feishu.start()
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

        from src.config_watcher import ConfigWatcher
        from src.config_reload import ReloadExecutor

        self._reload_executor = ReloadExecutor(self)
        self._config_watcher = ConfigWatcher(
            path=self._config_path,
            on_reload=self.on_config_reload,
        )
        await self._config_watcher.start()

    async def on_shutdown(self):
        try:
            if self._config_watcher:
                await self._config_watcher.stop()
            if self.memory_store:
                await self.memory_store.close()
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
            await self.typing.stop_all()
            await self.feishu.stop()
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
            logger.info("MyClaw stopped")
        except Exception as e:
            logger.error("Error during shutdown: %s", e, exc_info=True)

    async def on_config_reload(self, old_config, new_config, plan):
        self.config = new_config
        await self._reload_executor.execute(plan)
