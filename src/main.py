from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import uvicorn

from langchain_core.messages import AIMessage, HumanMessage

from src.config import load_config
from src.graph import collect_tools, create_agent_graph, create_model
from src.channels.feishu import FeishuChannel
from src.cron.store import CronStore
from src.cron.service import CronService
from src.cron.executor import execute_cron_job
from src.skills.loader import discover_skills
from src.skills.types import Skill
from src.skills.prompt import build_skill_commands
from src.session import SessionTracker
from src.commands.dispatcher import CommandDispatcher, build_builtin_help

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
# Suppress lark_oapi "processor not found" warnings for unhandled event types
_lark_logger = logging.getLogger("Lark")
_lark_logger.setLevel(logging.CRITICAL)
_lark_logger.handlers.clear()
logger = logging.getLogger("myclaw")


class Application:
    """Main application class that encapsulates the MyClaw lifecycle.

    This class manages initialization, message handling, WebSocket callbacks,
    and the overall application startup/shutdown sequence.
    """

    def __init__(self, config=None):
        self.config = config or load_config()
        self._skills_cache: list[Skill] = []
        self.compiled_graph = None
        self.checkpointer = None
        self.graph = None

        # Components initialized during setup
        self.feishu = None
        self.session_tracker = None
        self.typing = None
        self.dispatcher = None
        self.cron_service = None
        self.api = None
        self._checkpointer_ctx = None
        self._memory_store = None
        self._memory_searcher = None

    def _resolve_session_key(self, sender_id: str, chat_type: str, chat_id: str, scope: str) -> str:
        if scope == "global":
            return "global"
        if chat_type == "p2p":
            return f"user:{sender_id}"
        return f"group:{chat_id}"

    def _build_skill_directories(self) -> list[tuple[str, Path]]:
        dirs: list[tuple[str, Path]] = []
        workspace = Path(self.config.agents.workspace).resolve()

        bundled_skills = Path(__file__).parent.parent / "skills"
        if bundled_skills.exists():
            dirs.append(("bundled", bundled_skills))

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

    def _reload_skills(self) -> list[Skill]:
        dirs = self._build_skill_directories()
        self._skills_cache = discover_skills(dirs)
        active = [s for s in self._skills_cache if not s.metadata.disable_model_invocation]
        if active:
            logger.info("Skills loaded: %d active, %d total", len(active), len(self._skills_cache))
            for s in active:
                logger.info("  - %s: %s (%s)", s.name, s.description[:60], s.source)
        else:
            logger.info("No skills found")
        return self._skills_cache

    async def _handle_approval_interrupt(
        self,
        run_config: dict,
        chat_id: str,
        interrupt_value: dict,
    ):
        from src.tools.approval import get_approval_manager

        mgr = get_approval_manager()
        request_id = interrupt_value.get("request_id", "")
        command_preview = interrupt_value.get("command_preview", "")
        denylisted = interrupt_value.get("denylisted", False)

        if not request_id:
            return

        await self.feishu.send_approval_card(
            chat_id,
            request_id,
            command_preview,
            denylisted=denylisted,
        )

        decision = await mgr.await_approval(request_id)

        from langgraph.types import Command

        await self.compiled_graph.aupdate_state(run_config, Command(resume=decision))

        async for event in self.compiled_graph.astream_events(None, run_config, version="v2"):
            kind = event.get("event", "")
            if kind == "on_chain_error":
                err = event.get("data", {}).get("error")
                logger.error("Post-approval graph error: %s", err)

    async def setup(self):
        """Initialize all application components."""
        logger.info("MyClaw 0.1.0 starting...")
        logger.info("Model: %s/%s", self.config.model.provider, self.config.model.name)
        if not self.config.gateway.auth_token:
            logger.warning("Gateway auth_token is empty — all authentication is DISABLED")

        # Initialize plugins
        if self.config.plugins.enabled:
            from src.plugins.registry import init_plugin_registry

            registry = init_plugin_registry(self.config.plugins.extra_dirs)
            logger.info("Plugins: %d loaded, %d tools", registry.plugin_count, registry.tool_count)

        # Initialize sub-agents
        if self.config.agents.subagents:
            from src.agents.registry import init_agent_registry
            from src.agents.run_registry import init_run_registry

            agent_reg = init_agent_registry(self.config)
            init_run_registry()
            logger.info("Sub-agents: %d registered", agent_reg.count)

        # Initialize memory/RAG system
        if getattr(self.config, "memory", None) and self.config.memory.enabled:
            try:
                from src.memory.store import MemoryStore
                from src.memory.embeddings import EmbeddingProvider
                from src.memory.search import MemorySearcher
                from src.tools.memory_tools import set_memory_searcher

                store = MemoryStore(
                    self.config.memory.db_path,
                    dimensions=self.config.memory.embedding_dimensions,
                    fts_tokenizer=self.config.memory.fts_tokenizer,
                )
                await store.initialize()
                self._memory_store = store

                embeddings = EmbeddingProvider(self.config.memory, self.config.model)
                searcher = MemorySearcher(store, embeddings, self.config.memory)
                set_memory_searcher(searcher)
                self._memory_searcher = searcher

                # Index extra paths
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

        # Create model and collect tools
        model = create_model(self.config)
        tools = collect_tools(self.config)

        # Load skills
        skills: list[Skill] = []
        if self.config.skills.enabled:
            skills = self._reload_skills()

        # Build the agent graph
        self.graph = create_agent_graph(
            model,
            tools,
            self.config.agents.system_prompt,
            skills=skills if skills else None,
            skills_budget=self.config.skills.budget_chars,
            config=self.config,
        )

        # Compile graph with checkpointer
        if self.config.checkpointer.type == "sqlite":
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            db_path = Path(self.config.checkpointer.path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._checkpointer_ctx = AsyncSqliteSaver.from_conn_string(str(db_path))
            self.checkpointer = await self._checkpointer_ctx.__aenter__()
            self.compiled_graph = self.graph.compile(checkpointer=self.checkpointer)
        elif self.config.checkpointer.type == "memory":
            from langgraph.checkpoint.memory import InMemorySaver

            self.compiled_graph = self.graph.compile(checkpointer=InMemorySaver())

        logger.info("Graph compiled with %d tools, %d skills", len(tools), len(skills))

        # Initialize Feishu channel
        self.feishu = FeishuChannel(self.config.channels.feishu)
        session_scope = self.config.session.scope
        self.session_tracker = SessionTracker(idle_reset_minutes=self.config.session.idle_reset_minutes)

        from src.channels.typing import TypingIndicator

        self.typing = TypingIndicator(
            self.feishu.client, enabled=self.config.channels.feishu.typing_indicator
        )

        # Initialize command dispatcher
        self.dispatcher = CommandDispatcher(skills if skills else [], config=self.config)

        from src.commands.dispatcher import set_dispatcher

        set_dispatcher(self.dispatcher)

        # Register built-in commands
        self._register_builtin_commands(tools, skills)

        # Initialize media understanding runner for Feishu auto-processing
        if self.config.tools.media_understanding.enabled:
            try:
                from src.media_understanding.runner import MediaUnderstandingRunner
                from src.channels.feishu import set_media_understanding_runner

                mu_runner = MediaUnderstandingRunner(
                    self.config.tools.media_understanding,
                    fallback_api_key=self.config.model.api_key or "",
                )
                set_media_understanding_runner(mu_runner)
                logger.info("Media understanding runner initialized for Feishu channel")
            except Exception as e:
                logger.warning("Failed to init media understanding: %s", e)

        # Initialize cron service
        if self.config.cron.enabled:
            cron_store = CronStore(self.config.cron.store_path)

            async def cron_execute(job):
                return await execute_cron_job(job, self.compiled_graph, self.config, self.feishu)

            self.cron_service = CronService(
                cron_store, cron_execute, config=self.config, feishu_channel=self.feishu
            )
            logger.info("Cron service initialized")
            from src.tools.cron_tools import set_cron_service
            set_cron_service(self.cron_service)

        # Register message callback
        self.feishu.set_message_callback(self._create_message_callback(session_scope))

        # Create FastAPI gateway
        from src.gateway import create_gateway

        self.api = create_gateway(
            self.config, self.compiled_graph, self.feishu, self.cron_service
        )

        # Register startup/shutdown handlers
        self.api.on_event("startup")(self._on_startup)
        self.api.on_event("shutdown")(self._on_shutdown)

    def _register_builtin_commands(self, tools, skills):
        async def cmd_help(args: str, ctx: dict) -> str:
            commands = self.dispatcher.list_commands()
            return build_builtin_help(commands)

        async def cmd_reset(args: str, ctx: dict) -> str:
            thread_id = ctx.get("thread_id", "")
            if thread_id:
                try:
                    run_config = {"configurable": {"thread_id": thread_id}}
                    await self.compiled_graph.aupdate_state(run_config, {"messages": []})
                    return "Session reset."
                except Exception as e:
                    return f"Reset failed: {e}"
            return "No session to reset."

        async def cmd_status(args: str, ctx: dict) -> str:
            lines = [
                f"Model: {self.config.model.provider}/{self.config.model.name}",
                f"Tools: {len(tools)}",
                f"Skills: {len(skills)}",
                f"Sessions: {self.session_tracker.active_count}",
            ]
            if self.cron_service:
                s = self.cron_service.status()
                lines.append(f"Cron: {s['enabled_jobs']}/{s['total_jobs']} jobs")
            try:
                from src.plugins.registry import get_plugin_registry

                reg = get_plugin_registry()
                lines.append(f"Plugins: {reg.plugin_count} ({reg.tool_count} tools)")
            except Exception:
                pass
            return "\n".join(lines)

        async def cmd_skills(args: str, ctx: dict) -> str:
            if not skills:
                return "No skills loaded."
            lines = []
            for s in skills:
                invocable = "📋" if s.metadata.user_invocable else "🔒"
                lines.append(f"{invocable} {s.name}: {s.description[:80]}")
            return "\n".join(lines)

        self.dispatcher.register_builtin("help", cmd_help)
        self.dispatcher.register_builtin("reset", cmd_reset)
        self.dispatcher.register_builtin("status", cmd_status)
        self.dispatcher.register_builtin("skills", cmd_skills)
        logger.info("Commands: %d skill + 4 built-in", len(self.dispatcher._commands))

    def _create_message_callback(self, session_scope: str):
        async def on_feishu_message(
            text: str,
            sender_id: str,
            chat_id: str,
            chat_type: str,
            message_id: str,
            reply_fn,
            stream_fn,
        ):
            session_key = self._resolve_session_key(sender_id, chat_type, chat_id, session_scope)
            thread_id = f"feishu:{session_key}"
            run_config = {"configurable": {"thread_id": thread_id}}

            self.session_tracker.touch(thread_id)

            cmd_match = self.dispatcher.match(text)
            if cmd_match is not None:
                cmd_name, cmd_args = cmd_match
                logger.info("Slash command: /%s %.50s", cmd_name, cmd_args)
                result = await self.dispatcher.dispatch(
                    cmd_name,
                    cmd_args,
                    context={"thread_id": thread_id, "sender_id": sender_id, "chat_id": chat_id},
                )
                await reply_fn(result)
                return

            await self.typing.start(message_id)

            from src.tools.cron_tools import set_current_chat_id
            set_current_chat_id(chat_id)

            from src.graph import create_agent_state

            input_state = create_agent_state(
                sender_id=sender_id,
                chat_id=chat_id,
                message_text=text,
                chat_type=chat_type,
                message_id=message_id,
                system_prompt=self.config.agents.system_prompt,
            )

            assistant_text = None
            try:
                async for event in self.compiled_graph.astream_events(input_state, run_config, version="v2"):
                    kind = event.get("event", "")

                    if kind == "on_chain_error":
                        err = event.get("data", {}).get("error")
                        from langgraph.errors import GraphInterrupt
                        if err and isinstance(err, GraphInterrupt):
                            interrupts = getattr(err, "interrupts", [])
                            if interrupts:
                                interrupt_value = (
                                    interrupts[0].value
                                    if hasattr(interrupts[0], "value")
                                    else interrupts[0]
                                )
                                if (
                                    isinstance(interrupt_value, dict)
                                    and interrupt_value.get("type") == "approval_request"
                                ):
                                    await self._handle_approval_interrupt(
                                        run_config,
                                        chat_id,
                                        interrupt_value,
                                    )
                        continue

                state = await self.compiled_graph.aget_state(run_config)
                tasks = state.tasks
                for task in tasks:
                    if hasattr(task, "interrupts") and task.interrupts:
                        for intr in task.interrupts:
                            iv = intr.value if hasattr(intr, "value") else intr
                            if isinstance(iv, dict) and iv.get("type") == "approval_request":
                                await self._handle_approval_interrupt(
                                    run_config,
                                    chat_id,
                                    iv,
                                )

                state = await self.compiled_graph.aget_state(run_config)
                for msg in reversed(state.values.get("messages", [])):
                    if isinstance(msg, AIMessage) and msg.content:
                        assistant_text = msg.content
                        break
            except Exception as e:
                logger.error("Agent error: %s", e, exc_info=True)
                assistant_text = f"[error] {type(e).__name__}: {e}"

            if assistant_text:
                # Append link previews if configured
                if getattr(self.config, "link_understanding", None) and self.config.link_understanding.enabled:
                    try:
                        from src.link_understanding import detect_and_preview_links

                        preview = await detect_and_preview_links(text, max_previews=self.config.link_understanding.max_previews)
                        if preview:
                            assistant_text += "\n" + preview
                    except Exception:
                        pass

                # TTS processing
                tts_text = ""
                if getattr(self.config, "tts", None) and self.config.tts.enabled and self.config.tts.auto_mode != "off":
                    try:
                        await self._handle_tts(assistant_text, chat_id)
                        if self.config.tts.auto_mode == "tagged":
                            from src.tts.directives import strip_tts_directives
                            tts_text = strip_tts_directives(assistant_text)
                    except Exception as e:
                        logger.warning("TTS processing failed: %s", e)

                display_text = tts_text if tts_text else assistant_text
                await reply_fn(display_text)
                await self.typing.stop(message_id)
                logger.info("Reply to %s: %.100s", session_key, display_text)

                # Session memory: auto-write Q&A to memory
                if getattr(self, '_memory_searcher', None) and self.config.memory.enabled and getattr(self.config.memory, 'auto_session_memory', False):
                    try:
                        await self._memory_searcher.store.add_document(
                            f"session:{session_key}",
                            f"Q: {text}\nA: {display_text}",
                        )
                    except Exception:
                        pass

        return on_feishu_message

    async def _handle_tts(self, assistant_text: str, chat_id: str):
        """Process TTS for assistant text based on auto_mode."""
        if not self.config.tts.enabled:
            return

        tts_config = self.config.tts
        from src.tts.provider import TtsProvider
        from src.tts.directives import parse_tts_directives

        provider = TtsProvider(tts_config, self.config.model)

        if tts_config.auto_mode == "always":
            audio = await provider.synthesize(assistant_text)
            if audio:
                await self.feishu.send_audio(chat_id, audio)
        elif tts_config.auto_mode == "tagged":
            directives = parse_tts_directives(assistant_text)
            for directive in directives:
                audio = await provider.synthesize(directive.text)
                if audio:
                    await self.feishu.send_audio(chat_id, audio)

    async def _on_startup(self):
        # Run security audit on startup
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
                lambda s: self._reload_skills(),
            )

        if self.cron_service:
            await self.cron_service.start()

        await self.session_tracker.start_periodic_cleanup(self.compiled_graph)

        # Start memory watcher for auto-reindexing
        if self._memory_searcher and getattr(self.config.memory, 'watch', False):
            try:
                from src.memory.watcher import start_memory_watcher

                await start_memory_watcher(
                    self.config.memory.extra_paths,
                    lambda path, content: asyncio.ensure_future(
                        self._memory_searcher.index_document(path, content)
                    ),
                )
            except Exception as e:
                logger.warning("Failed to start memory watcher: %s", e)

        await self.feishu.start()
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

    async def _on_shutdown(self):
        try:
            if hasattr(self, '_memory_store') and self._memory_store:
                await self._memory_store.close()
            if self.config.skills.enabled and self.config.skills.watch:
                from src.skills.watcher import stop_skills_watcher

                await stop_skills_watcher()
            from src.memory.watcher import stop_memory_watcher
            await stop_memory_watcher()
            if self.cron_service:
                await self.cron_service.stop()
            await self.session_tracker.stop()
            await self.typing.stop_all()
            await self.feishu.stop()
            if self._checkpointer_ctx:
                await self._checkpointer_ctx.__aexit__(None, None, None)
            logger.info("MyClaw stopped")
        except Exception as e:
            logger.error("Error during shutdown: %s", e, exc_info=True)
            if hasattr(self, '_checkpointer_ctx') and self._checkpointer_ctx:
                try:
                    await self._checkpointer_ctx.__aexit__(None, None, None)
                except Exception:
                    pass

    async def run_async(self):
        """Run the application using the current event loop (non-blocking setup)."""
        config = uvicorn.Config(
            self.api,
            host=self.config.gateway.host,
            port=self.config.gateway.port,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        await server.serve()


def main():
    """Sync entry point for console script (myclaw)."""
    app = Application()

    async def _run():
        await app.setup()
        await app.run_async()

    asyncio.run(_run())


if __name__ == "__main__":
    main()
