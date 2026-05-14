from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import uvicorn

from src.agent.client import create_client, create_chain
from src.agent.loop import AgentLoop, ApprovalPending
from src.agent.state import StateStore
from src.agent.tooldef import ToolDef
from src.config import load_config
from src.channels.feishu import FeishuChannel
from src.channels.qq import QQChannel
from src.cron.store import CronStore
from src.cron.service import CronService
from src.cron.executor import execute_cron_job
from src.skills.loader import discover_skills
from src.skills.types import Skill
from src.skills.prompt import build_skills_prompt
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
_lark_logger = logging.getLogger("Lark")
_lark_logger.setLevel(logging.CRITICAL)
_lark_logger.handlers.clear()
logger = logging.getLogger("myclaw")


def _collect_builtin_tools(config) -> list[ToolDef]:
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
    ]
    if getattr(config.tools, "browser", None) and config.tools.browser.enabled:
        tool_modules.append("src.tools.browser.tools")
    for mod_name in tool_modules:
        try:
            import importlib
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "get_tools"):
                tools.extend(mod.get_tools())
        except Exception as e:
            logger.debug("Skipping tool module %s: %s", mod_name, e)

    if config.plugins.enabled:
        try:
            from src.plugins.registry import get_plugin_registry
            reg = get_plugin_registry()
            tools.extend(reg.collect_tools())
        except Exception:
            pass

    if getattr(config, "mcp", None) and config.mcp.enabled:
        try:
            from src.mcp.adapter import get_mcp_tools
            tools.extend(get_mcp_tools())
        except Exception:
            pass

    return tools


class Application:
    def __init__(self, config=None):
        self.config = config or load_config()
        self._skills_cache: list[Skill] = []
        self.agent_loop: AgentLoop | None = None
        self.state_store: StateStore | None = None
        self._model_ref = None

        self.feishu = None
        self.qq = None
        self.session_tracker = None
        self.session_registry = None
        self.typing = None
        self.dispatcher = None
        self.cron_service = None
        self.api = None
        self._memory_store = None
        self._memory_searcher = None
        self._rbac: RBAC | None = None
        self._background_tasks: set = set()

    def _resolve_session_key(self, sender_id: str, chat_type: str, chat_id: str, scope: str) -> str:
        if scope == "global":
            return "global"
        if chat_type == "p2p":
            return f"user:{sender_id}"
        return f"group:{chat_id}"

    def _build_skill_directories(self) -> list[tuple[str, Path]]:
        dirs: list[tuple[str, Path]] = []
        workspace = Path(self.config.agents.workspace).expanduser().resolve()

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

    async def _handle_approval_pending(
        self,
        exc: ApprovalPending,
        chat_id: str,
    ):
        from src.tools.approval import get_approval_manager

        mgr = get_approval_manager()

        if chat_id.startswith(("c2c:", "group:", "channel:", "dm:")):
            mgr.request_approval(
                "exec_command",
                exc.command_preview,
                chat_id=chat_id,
                timeout_seconds=120,
            )
            warn = "DANGEROUS" if exc.denylisted else "requires approval"
            await self.qq.send_text(
                chat_id,
                f"**Approval Required** ({warn})\n```\n{exc.command_preview}\n```\n"
                f"Reply 'yes' to allow or 'no' to deny. (request: {exc.request_id})",
            )
            decision = await mgr.await_approval(exc.request_id, timeout=120)
            result_state = await self.agent_loop.resume(exc.thread_id, decision)
            assistant_text = ""
            for msg in reversed(result_state.messages):
                if msg.get("role") == "assistant" and msg.get("content"):
                    assistant_text = msg["content"]
                    break
            if assistant_text:
                await self.qq.send_text(chat_id, assistant_text)
            return
        else:
            await self.feishu.send_approval_card(
                chat_id,
                exc.request_id,
                exc.command_preview,
                denylisted=exc.denylisted,
            )

        decision = await mgr.await_approval(exc.request_id)
        result_state = await self.agent_loop.resume(exc.thread_id, decision)
        assistant_text = ""
        for msg in reversed(result_state.messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                assistant_text = msg["content"]
                break
        if assistant_text:
            await self.feishu.send_text(chat_id, assistant_text)

    async def setup(self):
        logger.info("MyClaw 0.1.0 starting...")
        logger.info("Model: %s/%s", self.config.model.provider, self.config.model.name)
        if not self.config.gateway.auth_token:
            logger.warning("Gateway auth_token is empty — all authentication is DISABLED")

        if self.config.plugins.enabled:
            from src.plugins.registry import init_plugin_registry

            registry = init_plugin_registry(self.config.plugins.extra_dirs)
            logger.info("Plugins: %d loaded, %d tools", registry.plugin_count, registry.tool_count)

        if getattr(self.config, "mcp", None) and self.config.mcp.enabled and self.config.mcp.servers is not None:
            from src.mcp.manager import get_mcp_manager

            mcp_manager = get_mcp_manager()
            mcp_manager.load_config(self.config.mcp.servers)
            logger.info("MCP: %d servers configured", len(self.config.mcp.servers))

        if self.config.agents.subagents:
            from src.agents.registry import init_agent_registry
            from src.agents.run_registry import init_run_registry

            agent_reg = init_agent_registry(self.config)
            init_run_registry()
            logger.info("Sub-agents: %d registered", agent_reg.count)

        if getattr(self.config, "memory", None) and self.config.memory.enabled:
            try:
                from src.memory.embeddings import EmbeddingProvider
                from src.memory.search import MemorySearcher
                from src.tools.ai_tools import set_memory_searcher

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
            auth_store = AuthStore(self.config.auth.db_path)
            self._rbac = RBAC(auth_store, self.config)
            set_rbac(self._rbac)
            logger.info(
                "RBAC initialized (default_role=%s, pairing=%s)",
                self.config.auth.default_role,
                self.config.auth.pairing_enabled,
            )

        # Build model client
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

        # Collect tools
        tools = _collect_builtin_tools(self.config)

        # Load skills
        skills: list[Skill] = []
        if self.config.skills.enabled:
            skills = self._reload_skills()

        skills_prompt = ""
        if skills:
            skills_prompt = build_skills_prompt(skills)

        # Create state store
        cp_path = Path(self.config.checkpointer.path)
        cp_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_store = StateStore(str(cp_path))

        # Create agent loop
        self.agent_loop = AgentLoop(
            client=client,
            tools=tools,
            state_store=self.state_store,
            config=self.config,
            skills_prompt=skills_prompt,
            context_window_tokens=self.config.model.context_window,
        )

        logger.info("AgentLoop created with %d tools, %d skills", len(tools), len(skills))

        # Initialize session search index
        if self.config.session_search.enabled:
            from src.session_index.store import SessionIndexStore, set_session_index

            store = SessionIndexStore(self.config.session_search.index_path)
            set_session_index(store)
            logger.info("Session search index initialized: %s", self.config.session_search.index_path)

            self._session_index_store = store
            self._startup_sync_task = asyncio.create_task(
                self._run_startup_sync(store)
            )

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

        self._register_builtin_commands(tools, skills)

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

        # Initialize cron service
        if self.config.cron.enabled:
            cron_store = CronStore(self.config.cron.store_path)

            async def cron_execute(job):
                return await execute_cron_job(job, self.agent_loop, self.config, self.feishu)

            self.cron_service = CronService(cron_store, cron_execute, config=self.config, feishu_channel=self.feishu)
            logger.info("Cron service initialized")
            from src.tools.cron_tools import set_cron_service

            set_cron_service(self.cron_service)

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

        self.feishu.set_message_callback(self._create_message_callback(session_scope))

        self.qq = QQChannel(self.config.channels.qq)
        self.qq.set_message_callback(self._create_message_callback(session_scope, channel_prefix="qq"))
        if hasattr(self, "_qq_mu_runner") and self._qq_mu_runner:
            self.qq.set_media_understanding_runner(self._qq_mu_runner)
            logger.info("Media understanding runner initialized for QQ channel")

        from src.gateway import create_gateway

        self.api = create_gateway(self.config, self.agent_loop, self.feishu, self.cron_service)

        self.api.on_event("startup")(self._on_startup)
        self.api.on_event("shutdown")(self._on_shutdown)

        from src.dashboard.routes import register_dashboard

        register_dashboard(self.api, self)

        from src.tools.exec import reset_config_cache

        reset_config_cache()

    def _register_auth_commands(self):
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
                    state = self.state_store.load(thread_id)
                    if state:
                        state.messages = []
                        await self.state_store.save(thread_id, state)
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
                invocable = "\U0001f4cb" if s.metadata.user_invocable else "\U0001f512"
                lines.append(f"{invocable} {s.name}: {s.description[:80]}")
            return "\n".join(lines)

        self.dispatcher.register_builtin("help", cmd_help)
        self.dispatcher.register_builtin("reset", cmd_reset)
        self.dispatcher.register_builtin("status", cmd_status)
        self.dispatcher.register_builtin("skills", cmd_skills)

        async def cmd_search(args: str, ctx: dict) -> str:
            from src.session_index.store import get_session_index
            from src.tools.session_search_tools import _format_results, _try_llm_search

            store = get_session_index()
            if not store:
                return "Search not enabled"

            limit = self.config.session_search.max_results

            if not args.strip():
                results = store.search("", limit=limit)
                return _format_results(results) if results else "No sessions"

            results = await _try_llm_search(store, args, limit)
            if results is not None:
                return _format_results(results) if results else "No results found"

            results = store.search(args, limit=limit)
            return _format_results(results) if results else "No results found"

        self.dispatcher.register_builtin("search", cmd_search)

        async def cmd_new(args: str, ctx: dict) -> str:
            user_key = ctx.get("user_key", "")
            channel_prefix = ctx.get("channel_prefix", "feishu")
            if not user_key:
                return "Cannot determine session."
            user_hash = user_key.split(":")[-1] if user_key else "unknown"
            sid = self.session_registry.new_session(user_key, channel_prefix, user_hash)
            return f"New session started: {sid}\nSend messages to begin. Use /old to list sessions, /re <id> to switch."

        async def cmd_old(args: str, ctx: dict) -> str:
            user_key = ctx.get("user_key", "")
            if not user_key:
                return "Cannot determine session."
            reg_sessions = self.session_registry.list_sessions(user_key)
            current_override = self.session_registry.get_current(user_key)

            lines = []

            # Default session
            default_state = self.state_store.load(user_key)
            has_default = default_state is not None
            default_summary = ""
            if has_default:
                for m in default_state.messages:
                    if m.get("role") == "user":
                        default_summary = str(m.get("content", ""))[:50]
                        break
            current_marker = " [current]" if current_override is None else ""
            if has_default:
                lines.append(f"[default] {default_summary or '(empty)'}{current_marker}")
            else:
                lines.append(f"[default] (no history){current_marker}")

            for s in reg_sessions:
                summary = s["summary"]
                if summary in ("(new)", ""):
                    try:
                        st = self.state_store.load(s["thread_id"])
                        if st:
                            for m in st.messages:
                                if m.get("role") == "user":
                                    summary = str(m.get("content", ""))[:50]
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
            override = self.session_registry.get_current(legacy_thread_id)
            thread_id = override or legacy_thread_id

            self.session_tracker.touch(thread_id)

            if channel_prefix == "qq" and text.strip().lower() in ("yes", "no"):
                from src.tools.approval import get_approval_manager
                mgr = get_approval_manager()
                pending_list = mgr.list_pending()
                for req in pending_list:
                    if req.chat_id == chat_id:
                        decision = "allow_once" if text.strip().lower() == "yes" else "deny"
                        mgr.resolve(req.id, decision)
                        await reply_fn(f"Approval {'granted' if decision == 'allow_once' else 'denied'}.")
                        return

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
            from src.tools.browser.tools import set_browser_session
            set_browser_session(chat_id)

            from src.agent.state import AgentState

            input_state = AgentState(
                messages=[{"role": "user", "content": text}],
                system_prompt=self.config.agents.system_prompt,
                sender_id=sender_id,
                chat_id=chat_id,
                chat_type=chat_type,
                message_id=message_id,
                channel=channel_prefix,
            )

            # Load existing state messages (history) and prepend them
            existing = self.state_store.load(thread_id)
            if existing:
                if existing.pending_approval:
                    await reply_fn("⏳ 有待审批的操作，请先回复审批后再发新消息。")
                    if channel_prefix == "feishu":
                        await self.typing.stop(message_id)
                    return
                input_state.messages = existing.messages + input_state.messages

            assistant_text = None
            identity_written = False
            pre_msg_count = len(input_state.messages)
            try:
                logger.debug("[flow] agent_loop run start, state has %d messages", pre_msg_count)

                try:
                    result_state = await self.agent_loop.run(input_state, thread_id)
                except ApprovalPending as exc:
                    asyncio.create_task(self._handle_approval_pending(exc, chat_id))
                    result_state = self.state_store.load(thread_id) or input_state

                logger.debug("[flow] agent_loop run done")

                messages = result_state.messages

                if self.config.session_search.enabled and self.config.session_search.auto_sync:
                    try:
                        from src.session_index.store import get_session_index
                        from src.session_index.sync import sync_messages

                        idx = get_session_index()
                        if idx:
                            sync_messages(
                                store=idx,
                                thread_id=thread_id,
                                messages=messages,
                                channel=channel_prefix,
                                sender_id=sender_id,
                                chat_id=chat_id,
                                chat_type=chat_type,
                                tool_max_chars=self.config.session_search.tool_content_max_chars,
                            )
                    except Exception as e:
                        logger.warning("Session index sync failed: %s", e)

                for msg in reversed(messages):
                    if msg.get("role") == "assistant" and msg.get("content"):
                        assistant_text = msg["content"]
                        break

                for msg in messages[pre_msg_count:]:
                    if msg.get("role") != "tool":
                        continue
                    tool_out = msg.get("content", "")
                    if "IDENTITY.md" not in tool_out:
                        continue
                    identity_written = True
                    break
            except Exception as e:
                logger.error("Agent error: %s", e, exc_info=True)
                assistant_text = f"[error] {type(e).__name__}: {e}"

            if assistant_text:
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

                if identity_written:
                    try:
                        await reply_fn("\U0001f4be update memory: 已更新身份记忆")
                    except Exception:
                        pass

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
                    except Exception:
                        pass

                if getattr(self.config, "beads", None) and self.config.beads.enabled:
                    try:
                        from src.tools.beads_tools import auto_extract_memory, save_memory
                        extracted = auto_extract_memory(text, display_text)
                        if extracted:
                            content, category = extracted
                            await save_memory(content)
                        elif self.config.beads.memory_judge_model:
                            task = asyncio.create_task(self._beads_llm_judge(
                                text, display_text, reply_fn,
                            ))
                            self._background_tasks.add(task)
                            task.add_done_callback(self._background_tasks.discard)
                    except Exception:
                        pass

                    try:
                        call_id_to_name = {}
                        for m in messages:
                            if m.get("role") == "assistant" and m.get("tool_calls"):
                                for tc in m["tool_calls"]:
                                    fn_info = tc.get("function", {})
                                    if isinstance(fn_info, dict) and tc.get("id"):
                                        call_id_to_name[tc["id"]] = fn_info.get("name", "")
                        for msg in messages[pre_msg_count:]:
                            if msg.get("role") == "tool":
                                tc_name = call_id_to_name.get(msg.get("tool_call_id", ""), "")
                                if "bd_remember" in tc_name:
                                    await reply_fn("\U0001f4be update memory: 已保存到 beads")
                                    break
                    except Exception:
                        pass

        return on_message

    async def _beads_llm_judge(self, user_input: str, ai_response: str, reply_fn):
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
                await reply_fn(f"\U0001f4be update memory: {content[:50]}")
        except Exception:
            logger.debug("Beads LLM judge failed", exc_info=True)

    async def _handle_tts(self, assistant_text: str, chat_id: str, channel_prefix: str = "feishu"):
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

    async def _on_startup(self):
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

        await self.session_tracker.start_periodic_cleanup(self.state_store)

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
            try:
                from src.mcp.manager import get_mcp_manager

                mcp_mgr = get_mcp_manager()
                await mcp_mgr.disconnect_all()
            except Exception:
                pass
            if self.state_store:
                self.state_store.close()
            try:
                from src.session_index.store import get_session_index, set_session_index

                idx = get_session_index()
                if idx:
                    idx.close()
                    set_session_index(None)
            except Exception:
                pass
            try:
                if self.config.tools.browser.enabled:
                    from src.tools.browser.manager import get_browser_manager
                    await get_browser_manager().close_all()
            except Exception:
                pass
            logger.info("MyClaw stopped")
        except Exception as e:
            logger.error("Error during shutdown: %s", e, exc_info=True)

    async def run_async(self):
        config = uvicorn.Config(
            self.api,
            host=self.config.gateway.host,
            port=self.config.gateway.port,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        await server.serve()


def main():
    import os

    if sys.platform == "win32":
        import ctypes

        _HANDLER = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_uint)

        def _ctrl_handler(ctrl_type):
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
