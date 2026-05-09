from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import uvicorn

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.config import load_config
from src.graph import collect_tools, create_agent_graph, create_model
from src.channels.feishu import FeishuChannel
from src.channels.qq import QQChannel
from src.cron.store import CronStore
from src.cron.service import CronService
from src.cron.executor import execute_cron_job
from src.skills.loader import discover_skills
from src.skills.types import Skill
from src.skills.prompt import build_skill_commands
from src.session import SessionTracker
from src.commands.dispatcher import CommandDispatcher, build_builtin_help
from src.auth.store import AuthStore
from src.auth.rbac import RBAC
from src.auth.models import UserRole
from src.auth.rbac import set_rbac

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
        self.model_ref = None

        # Components initialized during setup
        self.feishu = None
        self.qq = None
        self.session_tracker = None
        self.session_registry = None
        self.typing = None
        self.dispatcher = None
        self.cron_service = None
        self.api = None
        self._checkpointer_ctx = None
        self._memory_store = None
        self._memory_searcher = None
        self._rbac: RBAC | None = None

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

        # Send approval card via appropriate channel
        if chat_id.startswith(("c2c:", "group:", "channel:", "dm:")):
            # QQ channel: send as text message (no interactive cards)
            warn = "DANGEROUS" if denylisted else "requires approval"
            await self.qq.send_text(
                chat_id,
                f"**Approval Required** ({warn})\n```\n{command_preview}\n```\n"
                f"Reply 'yes' to allow or 'no' to deny. (request: {request_id})",
            )
            # Auto-approve since QQ has no interactive buttons
            from langgraph.types import Command

            decision = "allow_once"
            await self.compiled_graph.aupdate_state(run_config, Command(resume=decision))
            return
        else:
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

        # Initialize MCP subsystem
        if getattr(self.config, "mcp", None) and self.config.mcp.enabled and self.config.mcp.servers is not None:
            from src.mcp.manager import get_mcp_manager

            mcp_manager = get_mcp_manager()
            mcp_manager.load_config(self.config.mcp.servers)
            logger.info("MCP: %d servers configured", len(self.config.mcp.servers))

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
                from src.memory.embeddings import EmbeddingProvider
                from src.memory.search import MemorySearcher
                from src.tools.ai_tools import set_memory_searcher

                # Choose backend
                backend = getattr(self.config.memory, "backend", "sqlite")
                if backend == "lancedb":
                    from src.memory.lance_store import LanceMemoryStore

                    store = LanceMemoryStore(
                        self.config.memory.db_path,
                        dimensions=self.config.memory.embedding_dimensions,
                        fts_tokenizer=self.config.memory.fts_tokenizer,
                        lancedb_uri=getattr(self.config.memory, "lancedb_uri", "data/memory_lancedb"),
                    )
                else:
                    from src.memory.store import MemoryStore

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

        # Initialize RBAC
        if self.config.auth.enabled:
            auth_store = AuthStore(self.config.auth.db_path)
            self._rbac = RBAC(auth_store, self.config)
            set_rbac(self._rbac)
            logger.info(
                "RBAC initialized (default_role=%s, pairing=%s)",
                self.config.auth.default_role,
                self.config.auth.pairing_enabled,
            )

        # Create model and collect tools
        if self.config.model.fallbacks:
            from src.graph import create_model_chain

            model = create_model_chain(self.config)
            logger.info("Model chain: primary + %d fallbacks", len(self.config.model.fallbacks))
        else:
            model = create_model(self.config)
        tools = collect_tools(self.config)

        # Load skills
        skills: list[Skill] = []
        if self.config.skills.enabled:
            skills = self._reload_skills()

        # Build the agent graph
        self.graph, self.model_ref = create_agent_graph(
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
        from src.session import SessionRegistry
        self.session_registry = SessionRegistry()
        self.session_registry.init("data/sessions.json")

        from src.channels.typing import TypingIndicator

        self.typing = TypingIndicator(self.feishu.client, enabled=self.config.channels.feishu.typing_indicator)

        # Initialize command dispatcher
        self.dispatcher = CommandDispatcher(skills if skills else [], config=self.config)

        from src.commands.dispatcher import set_dispatcher

        set_dispatcher(self.dispatcher)

        # Register built-in commands
        self._register_builtin_commands(tools, skills)

        # Initialize media understanding runner for channel auto-processing
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

        # Initialize QQ channel (media runner)
        if self.config.channels.qq.enabled and self.config.tools.media_understanding.enabled:
            try:
                from src.media_understanding.runner import MediaUnderstandingRunner
                qq_runner = MediaUnderstandingRunner(
                    self.config.tools.media_understanding,
                    fallback_api_key=self.config.model.api_key or "",
                )
                # Will be applied after QQChannel is created
                self._qq_mu_runner = qq_runner
            except Exception as e:
                logger.warning("Failed to init QQ media understanding: %s", e)

        # Initialize cron service
        if self.config.cron.enabled:
            cron_store = CronStore(self.config.cron.store_path)

            async def cron_execute(job):
                return await execute_cron_job(job, self.compiled_graph, self.config, self.feishu)

            self.cron_service = CronService(cron_store, cron_execute, config=self.config, feishu_channel=self.feishu)
            logger.info("Cron service initialized")
            from src.tools.cron_tools import set_cron_service

            set_cron_service(self.cron_service)

        # Initialize file tools workspace from config
        from src.tools.file_tools import set_workspace

        set_workspace(self.config.agents.workspace)

        # Initialize beads workspace
        if getattr(self.config, "beads", None) and self.config.beads.enabled:
            from src.tools.beads_tools import set_beads_workspace
            beads_ws = self.config.beads.workspace or str(Path(self.config.agents.workspace).resolve())
            set_beads_workspace(beads_ws)

        # Register message callback
        self.feishu.set_message_callback(self._create_message_callback(session_scope))

        # Initialize QQ channel
        self.qq = QQChannel(self.config.channels.qq)
        self.qq.set_message_callback(self._create_message_callback(session_scope, channel_prefix="qq"))
        if hasattr(self, "_qq_mu_runner") and self._qq_mu_runner:
            self.qq.set_media_understanding_runner(self._qq_mu_runner)
            logger.info("Media understanding runner initialized for QQ channel")

        # Create FastAPI gateway
        from src.gateway import create_gateway

        self.api = create_gateway(self.config, self.compiled_graph, self.feishu, self.cron_service)

        # Register startup/shutdown handlers
        self.api.on_event("startup")(self._on_startup)
        self.api.on_event("shutdown")(self._on_shutdown)

        # Register dashboard
        from src.dashboard.routes import register_dashboard

        register_dashboard(self.api, self)

        # Reset exec tool config cache so it picks up the loaded config
        from src.tools.exec import reset_config_cache

        reset_config_cache()

    def _register_auth_commands(self):
        """Register /pair, /role, /whoami commands."""
        rbac = self._rbac
        store = rbac.store

        async def cmd_pair(args: str, ctx: dict) -> str:
            if not self.config.auth.pairing_enabled:
                return "Pairing is not enabled."
            sender_id = ctx.get("sender_id", "")
            if not sender_id:
                return "Cannot determine your identity."
            pairing = store.create_pairing_code(
                user_id=sender_id,
                ttl_seconds=self.config.auth.pairing_ttl_seconds,
            )
            return (
                f"Your pairing code: `{pairing.code}`\n"
                f"Valid for {self.config.auth.pairing_ttl_seconds // 60} minutes.\n"
                f"Submit it at the Dashboard or via API to complete pairing."
            )

        async def cmd_whoami(args: str, ctx: dict) -> str:
            sender_id = ctx.get("sender_id", "")
            if not sender_id:
                return "Unknown identity."
            user = rbac.resolve_user(sender_id)
            lines = [
                f"User ID: {user.user_id}",
                f"Role: {user.role.value}",
                f"Display: {user.display_name or '(not set)'}",
            ]
            devices = store.list_user_devices(sender_id)
            if devices:
                lines.append(f"Devices: {len(devices)} ({sum(1 for d in devices if d.trusted)} trusted)")
            return "\n".join(lines)

        async def cmd_role(args: str, ctx: dict) -> str:
            """Admin: change a user's role. Usage: /role <user_id> <role>"""
            sender_id = ctx.get("sender_id", "")
            caller = rbac.resolve_user(sender_id)
            if not rbac.check_admin_access(caller):
                return "Permission denied. Admin access required."
            parts = args.strip().split()
            if len(parts) < 2:
                return "Usage: /role <user_id> <owner|admin|user|guest>"
            target_id, role_str = parts[0], parts[1]
            try:
                target_role = UserRole(role_str)
            except ValueError:
                return f"Invalid role: {role_str}. Use: owner, admin, user, guest"
            # Non-owner cannot assign owner role
            if target_role == UserRole.owner and not caller.is_owner:
                return "Only owners can assign the owner role."
            if store.update_user_role(target_id, target_role):
                return f"User {target_id} role updated to {target_role.value}"
            return f"User {target_id} not found."

        self.dispatcher.register_builtin("pair", cmd_pair)
        self.dispatcher.register_builtin("whoami", cmd_whoami)
        self.dispatcher.register_builtin("role", cmd_role)

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
            try:
                from src.mcp.manager import get_mcp_manager

                mcp_mgr = get_mcp_manager()
                servers = await mcp_mgr.list_servers()
                connected = sum(1 for s in servers if s.connected)
                lines.append(f"MCP: {connected}/{len(servers)} servers connected")
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

        # Session management commands
        async def cmd_new(args: str, ctx: dict) -> str:
            user_key = ctx.get("user_key", "")
            channel_prefix = ctx.get("channel_prefix", "feishu")
            if not user_key:
                return "Cannot determine session."
            # Extract user hash from user_key (e.g. "qq:user:ABC123" -> "ABC123")
            user_hash = user_key.split(":")[-1] if user_key else "unknown"
            sid = self.session_registry.new_session(user_key, channel_prefix, user_hash)
            return f"New session started: {sid}\nSend messages to begin. Use /old to list sessions, /re <id> to switch."

        async def cmd_old(args: str, ctx: dict) -> str:
            user_key = ctx.get("user_key", "")
            if not user_key:
                return "Cannot determine session."
            from src.session import get_threads_for_user
            cp_path = self.config.checkpointer.path if self.config.checkpointer else ""
            reg_sessions = self.session_registry.list_sessions(user_key)
            db_threads = get_threads_for_user(cp_path, user_key)
            current_override = self.session_registry.get_current(user_key)

            lines = []

            # Legacy (default) session
            has_legacy = False
            for t in db_threads:
                if t["thread_id"] == user_key:
                    has_legacy = True
                    legacy_summary = ""
                    try:
                        cfg = {"configurable": {"thread_id": user_key}}
                        state = await self.compiled_graph.aget_state(cfg)
                        msgs = state.values.get("messages", [])
                        for m in msgs:
                            if isinstance(m, HumanMessage):
                                legacy_summary = str(m.content)[:50]
                                break
                    except Exception:
                        pass
                    current_marker = " [current]" if current_override is None else ""
                    lines.append(f"[default] {legacy_summary or '(empty)'}{current_marker}")
                    break

            if not has_legacy:
                current_marker = " [current]" if current_override is None else ""
                lines.append(f"[default] (no history){current_marker}")

            # Registry sessions
            for s in reg_sessions:
                summary = s["summary"]
                if summary in ("(new)", ""):
                    try:
                        cfg = {"configurable": {"thread_id": s["thread_id"]}}
                        state = await self.compiled_graph.aget_state(cfg)
                        msgs = state.values.get("messages", [])
                        for m in msgs:
                            if isinstance(m, HumanMessage):
                                summary = str(m.content)[:50]
                                break
                        if not summary:
                            summary = "(new)"
                    except Exception:
                        summary = "(new)"
                    self.session_registry.update_summary(user_key, s["thread_id"], summary)
                current_marker = " [current]" if s["is_current"] else ""
                dt = time.strftime("%m-%d %H:%M", time.localtime(s["created_at"]))
                lines.append(f"[{s['session_id']}] {summary}{current_marker} ({dt})")

            if len(lines) == 1 and not reg_sessions:
                return "No sessions yet.\n/new - create new session"

            result = "Your sessions:\n" + "\n".join(lines)
            result += "\n\n/re <id> - switch session (e.g. /re default, /re s1)\n/new - create new session"
            return result

        async def cmd_re(args: str, ctx: dict) -> str:
            user_key = ctx.get("user_key", "")
            session_id = args.strip()
            if not user_key:
                return "Cannot determine session."
            if not session_id:
                return "Usage: /re <session_id>\nUse /old to list sessions."
            tid = self.session_registry.switch_to(user_key, session_id)
            if tid == "default":
                return "Switched back to default session."
            if tid:
                return f"Resumed session: {session_id}\nThread: {tid}"
            return f"Session not found: {session_id}\nUse /old to list available sessions."

        self.dispatcher.register_builtin("new", cmd_new)
        self.dispatcher.register_builtin("old", cmd_old)
        self.dispatcher.register_builtin("re", cmd_re)

        # Auth commands
        if self.config.auth.enabled and self._rbac:
            self._register_auth_commands()

        logger.info("Commands: %d skill + built-in", len(self.dispatcher._commands))

    def _create_message_callback(self, session_scope: str, channel_prefix: str = "feishu"):
        async def on_message(
            text: str,
            sender_id: str,
            chat_id: str,
            chat_type: str,
            message_id: str,
            reply_fn,
            stream_fn,
        ):
            session_key = self._resolve_session_key(sender_id, chat_type, chat_id, session_scope)
            legacy_thread_id = f"{channel_prefix}:{session_key}"
            # Check if user has switched to a multi-session
            override = self.session_registry.get_current(legacy_thread_id)
            thread_id = override or legacy_thread_id
            # recursion_limit must be in config dict (LangGraph ignores kwarg form)
            _recursion_limit = (self.config.agents.max_tool_rounds or 50) * 2 + 10
            run_config = {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": _recursion_limit,
            }

            self.session_tracker.touch(thread_id)

            cmd_match = self.dispatcher.match(text)
            if cmd_match is not None:
                cmd_name, cmd_args = cmd_match
                logger.info("Slash command: /%s %.50s", cmd_name, cmd_args)
                result = await self.dispatcher.dispatch(
                    cmd_name,
                    cmd_args,
                    context={"thread_id": thread_id, "sender_id": sender_id, "chat_id": chat_id, "user_key": legacy_thread_id, "channel_prefix": channel_prefix},
                )
                await reply_fn(result)
                return

            if channel_prefix == "feishu":
                await self.typing.start(message_id)

            from src.tools.cron_tools import set_current_chat_id

            set_current_chat_id(chat_id)
            from src.tools.media_tools import set_current_channel
            set_current_channel(channel_prefix)

            from src.graph import create_agent_state

            input_state = create_agent_state(
                sender_id=sender_id,
                chat_id=chat_id,
                message_text=text,
                chat_type=chat_type,
                message_id=message_id,
                system_prompt=self.config.agents.system_prompt,
                channel=channel_prefix,
            )

            assistant_text = None
            try:
                logger.debug("[flow] graph ainvoke start, state has %d messages",
                             len(input_state.get("messages", [])))
                async for event in self.compiled_graph.astream_events(input_state, run_config, version="v2"):
                    kind = event.get("event", "")
                    if kind == "on_chain_error":
                        err = event.get("data", {}).get("error")
                        logger.error("Graph chain error: %s", err)
                        continue
                logger.debug("[flow] graph ainvoke done")

                # Check for pending interrupts (e.g. approval requests)
                state = await self.compiled_graph.aget_state(run_config)
                for task in state.tasks:
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
                messages = state.values.get("messages", [])
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage) and msg.content:
                        assistant_text = msg.content
                        break

                # Detect IDENTITY.md writes for memory notification (deduplicated)
                identity_written = False
                for i, msg in enumerate(messages):
                    if not isinstance(msg, ToolMessage):
                        continue
                    tool_out = msg.content if isinstance(msg.content, str) else ""
                    if "IDENTITY.md" not in tool_out:
                        continue
                    identity_written = True
                    break
            except Exception as e:
                logger.error("Agent error: %s", e, exc_info=True)
                assistant_text = f"[error] {type(e).__name__}: {e}"

            if assistant_text:
                # Append link previews if configured
                if getattr(self.config, "link_understanding", None) and self.config.link_understanding.enabled:
                    try:
                        from src.link_understanding import detect_and_preview_links

                        preview = await detect_and_preview_links(
                            text, max_previews=self.config.link_understanding.max_previews
                        )
                        if preview:
                            assistant_text += "\n" + preview
                    except Exception:
                        pass

                # TTS processing
                tts_text = ""
                if getattr(self.config, "tts", None) and self.config.tts.enabled and self.config.tts.auto_mode != "off":
                    try:
                        await self._handle_tts(assistant_text, chat_id, channel_prefix)
                        if self.config.tts.auto_mode == "tagged":
                            from src.tts.directives import strip_tts_directives

                            tts_text = strip_tts_directives(assistant_text)
                    except Exception as e:
                        logger.warning("TTS processing failed: %s", e)

                display_text = tts_text if tts_text else assistant_text

                # Media tag delivery: <media>path</media> → send file
                try:
                    from src.media_delivery import deliver_media
                    ch = self.qq if channel_prefix == "qq" else self.feishu if channel_prefix == "feishu" else None
                    if ch:
                        display_text = await deliver_media(display_text, chat_id, channel_prefix, ch)
                except Exception as e:
                    logger.warning("Media delivery failed: %s", e)

                try:
                    logger.debug("[flow] sending reply, len=%d", len(display_text))
                    await reply_fn(display_text)
                finally:
                    if channel_prefix == "feishu":
                        await self.typing.stop(message_id)
                logger.info("Reply to %s: %.100s", session_key, display_text)

                # Memory notifications (sent as separate messages)
                # 1. IDENTITY.md write
                if identity_written:
                    try:
                        await reply_fn("💾 update memory: 已更新身份记忆")
                    except Exception:
                        pass

                # 2. Auto session memory
                if (
                    getattr(self, "_memory_searcher", None)
                    and self.config.memory.enabled
                    and getattr(self.config.memory, "auto_session_memory", False)
                ):
                    try:
                        await self._memory_searcher.store.add_document(
                            f"session:{session_key}",
                            f"Q: {text}\nA: {display_text}",
                        )
                        # Silent save — no notification for auto session memory
                    except Exception:
                        pass

                # 3. Beads passive memory
                if getattr(self.config, "beads", None) and self.config.beads.enabled:
                    try:
                        from src.tools.beads_tools import auto_extract_memory, save_memory
                        extracted = auto_extract_memory(text, display_text)
                        if extracted:
                            content, category = extracted
                            await save_memory(content)
                            # Silent save — no notification for regex-based extraction
                        elif self.config.beads.memory_judge_model:
                            asyncio.create_task(self._beads_llm_judge(
                                text, display_text, reply_fn,
                            ))
                    except Exception:
                        pass

                    # Notify if LLM manually called bd_remember
                    try:
                        for msg in messages:
                            if isinstance(msg, ToolMessage) and getattr(msg, "name", "") == "bd_remember":
                                await reply_fn("💾 update memory: 已保存到 beads")
                                break
                    except Exception:
                        pass

        return on_message

    async def _beads_llm_judge(self, user_input: str, ai_response: str, reply_fn):
        """Background task: use a small model to judge if conversation is worth remembering."""
        try:
            from src.tools.beads_tools import judge_memory_with_llm, save_memory

            model_name = self.config.beads.memory_judge_model
            base_url = self.config.beads.memory_judge_base_url or self.config.model.base_url
            api_key = self.config.beads.memory_judge_api_key or self.config.model.api_key

            if not model_name or not base_url or not api_key:
                return

            content = await judge_memory_with_llm(
                user_input, ai_response, model_name, base_url, api_key,
            )
            if content:
                await save_memory(content)
                await reply_fn(f"💾 update memory: {content[:50]}")
        except Exception:
            logger.debug("Beads LLM judge failed", exc_info=True)

    async def _handle_tts(self, assistant_text: str, chat_id: str, channel_prefix: str = "feishu"):
        """Process TTS for assistant text based on auto_mode."""
        if not self.config.tts.enabled:
            return

        tts_config = self.config.tts
        from src.tts.provider import TtsProvider
        from src.tts.directives import parse_tts_directives

        provider = TtsProvider(tts_config, self.config.model)

        async def _send_audio(audio_bytes: bytes):
            if channel_prefix == "feishu" and self.feishu:
                await self.feishu.send_audio(chat_id, audio_bytes)
            elif channel_prefix == "qq" and self.qq:
                await self.qq.send_audio(chat_id, audio_bytes)

        if tts_config.auto_mode == "always":
            audio = await provider.synthesize(assistant_text)
            if audio:
                await _send_audio(audio)
        elif tts_config.auto_mode == "tagged":
            directives = parse_tts_directives(assistant_text)
            for directive in directives:
                audio = await provider.synthesize(directive.text)
                if audio:
                    await _send_audio(audio)

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
        if self._memory_searcher and getattr(self.config.memory, "watch", False):
            try:
                from src.memory.watcher import start_memory_watcher

                await start_memory_watcher(
                    self.config.memory.extra_paths,
                    lambda path, content: asyncio.ensure_future(self._memory_searcher.index_document(path, content)),
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

    async def _on_shutdown(self):
        try:
            if hasattr(self, "_memory_store") and self._memory_store:
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
            await self.qq.stop()
            # Shutdown MCP connections
            try:
                from src.mcp.manager import get_mcp_manager

                mcp_mgr = get_mcp_manager()
                await mcp_mgr.disconnect_all()
            except Exception:
                pass
            if self._checkpointer_ctx:
                await self._checkpointer_ctx.__aexit__(None, None, None)
            logger.info("MyClaw stopped")
        except Exception as e:
            logger.error("Error during shutdown: %s", e, exc_info=True)
            if hasattr(self, "_checkpointer_ctx") and self._checkpointer_ctx:
                try:
                    await self._checkpointer_ctx.__aexit__(None, None, None)
                except Exception:
                    pass

    async def run_async(self):
        """Run the application."""
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
    import os

    # On Windows, Python's signal.signal() doesn't fire inside asyncio's
    # ProactorEventLoop because the main thread is blocked in a C extension.
    # Use the native Windows SetConsoleCtrlHandler instead, which runs in its
    # own thread at the OS level and bypasses Python entirely.
    if sys.platform == "win32":
        import ctypes

        _HANDLER = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_uint)

        def _ctrl_handler(ctrl_type):
            # CTRL_C_EVENT=0, CTRL_CLOSE_EVENT=2, CTRL_LOGOFF_EVENT=5, CTRL_SHUTDOWN_EVENT=6
            if ctrl_type in (0, 2, 5, 6):
                os._exit(0)
            return False

        ctypes.windll.kernel32.SetConsoleCtrlHandler(_HANDLER(_ctrl_handler), True)

    app = Application()

    async def _run():
        await app.setup()
        await app.run_async()

    asyncio.run(_run())


if __name__ == "__main__":
    main()
