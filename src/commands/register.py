from __future__ import annotations

import logging
import time

from src.auth.models import UserRole
from src.commands.dispatcher import build_builtin_help

logger = logging.getLogger("myclaw")


def register_auth_commands(dispatcher, container):
    rbac = container.rbac
    store = rbac.store

    async def cmd_pair(args: str, ctx: dict) -> str:
        if not container.config.auth.pairing_enabled:
            return "Pairing is not enabled."
        sender_id = ctx.get("sender_id", "")
        if not sender_id:
            return "Cannot determine your identity."
        pairing = store.create_pairing_code(
            user_id=sender_id,
            ttl_seconds=container.config.auth.pairing_ttl_seconds,
        )
        return (
            f"Your pairing code: `{pairing.code}`\n"
            f"Valid for {container.config.auth.pairing_ttl_seconds // 60} minutes.\n"
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

    dispatcher.register_builtin("pair", cmd_pair)
    dispatcher.register_builtin("whoami", cmd_whoami)
    dispatcher.register_builtin("role", cmd_role)


def register_builtin_commands(dispatcher, container, tools, skills):
    async def cmd_help(args: str, ctx: dict) -> str:
        commands = dispatcher.list_commands()
        return build_builtin_help(commands)

    async def cmd_reset(args: str, ctx: dict) -> str:
        thread_id = ctx.get("thread_id", "")
        if thread_id:
            try:
                state = await container.state_store.aload(thread_id)
                if state:
                    state.messages = []
                    await container.state_store.save(thread_id, state)
                return "Session reset."
            except Exception as e:
                return f"Reset failed: {e}"
        return "No session to reset."

    async def cmd_status(args: str, ctx: dict) -> str:
        lines = [
            f"Model: {container.config.model.provider}/{container.config.model.name}",
            f"Tools: {len(tools)}",
            f"Skills: {len(skills)}",
            f"Sessions: {container.session_tracker.active_count}",
        ]
        if container.cron_service:
            s = container.cron_service.status()
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

    dispatcher.register_builtin("help", cmd_help)
    dispatcher.register_builtin("reset", cmd_reset)
    dispatcher.register_builtin("status", cmd_status)
    dispatcher.register_builtin("skills", cmd_skills)

    async def cmd_search(args: str, ctx: dict) -> str:
        from src.session_index.store import get_session_index
        from src.tools.session_search_tools import _format_results, _try_llm_search

        store = get_session_index()
        if not store:
            return "Search not enabled"

        limit = container.config.session_search.max_results

        if not args.strip():
            results = store.search("", limit=limit)
            return _format_results(results) if results else "No sessions"

        results = await _try_llm_search(store, args, limit)
        if results is not None:
            return _format_results(results) if results else "No results found"

        results = store.search(args, limit=limit)
        return _format_results(results) if results else "No results found"

    dispatcher.register_builtin("search", cmd_search)

    async def cmd_new(args: str, ctx: dict) -> str:
        user_key = ctx.get("user_key", "")
        channel_prefix = ctx.get("channel_prefix", "feishu")
        if not user_key:
            return "Cannot determine session."
        user_hash = user_key.split(":")[-1] if user_key else "unknown"
        sid = container.session_registry.new_session(user_key, channel_prefix, user_hash)
        return f"New session started: {sid}\nSend messages to begin. Use /old to list sessions, /re <id> to switch."

    async def cmd_old(args: str, ctx: dict) -> str:
        user_key = ctx.get("user_key", "")
        if not user_key:
            return "Cannot determine session."
        reg_sessions = container.session_registry.list_sessions(user_key)
        current_override = container.session_registry.get_current(user_key)

        lines = []

        default_state = await container.state_store.aload(user_key)
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
                    st = await container.state_store.aload(s["thread_id"])
                    if st:
                        for m in st.messages:
                            if m.get("role") == "user":
                                summary = str(m.get("content", ""))[:50]
                                break
                    if not summary:
                        summary = "(new)"
                except Exception:
                    summary = "(new)"
                container.session_registry.update_summary(user_key, s["thread_id"], summary)
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
        tid = container.session_registry.switch_to(user_key, session_id)
        if tid == "default":
            return "Switched back to default session."
        if tid:
            return f"Resumed session: {session_id}\nThread: {tid}"
        return f"Session not found: {session_id}\nUse /old to list available sessions."

    dispatcher.register_builtin("new", cmd_new)
    dispatcher.register_builtin("old", cmd_old)
    dispatcher.register_builtin("re", cmd_re)

    if container.config.auth.enabled and container.rbac:
        register_auth_commands(dispatcher, container)

    logger.info("Commands: %d skill + built-in", len(dispatcher._commands))
