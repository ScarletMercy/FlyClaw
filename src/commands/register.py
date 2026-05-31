from __future__ import annotations

import asyncio
import logging
import time

from src.auth.models import UserRole

logger = logging.getLogger("flyclaw")


def register_auth_commands(dispatcher, container):
    rbac = container.rbac
    store = rbac.store

    async def cmd_pair(args: str, ctx: dict) -> str:
        zh = container.config.agents.language == "zh"
        if not container.config.auth.pairing_enabled:
            return "配对功能未启用。" if zh else "Pairing is not enabled."
        sender_id = ctx.get("sender_id", "")
        if not sender_id:
            return "无法确定身份。" if zh else "Cannot determine your identity."
        pairing = store.create_pairing_code(
            user_id=sender_id,
            ttl_seconds=container.config.auth.pairing_ttl_seconds,
        )
        minutes = container.config.auth.pairing_ttl_seconds // 60
        if zh:
            return f"配对码：`{pairing.code}`\n有效期 {minutes} 分钟。\n请在 Dashboard 或通过 API 提交以完成配对。"
        return (
            f"Your pairing code: `{pairing.code}`\n"
            f"Valid for {minutes} minutes.\n"
            f"Submit it at the Dashboard or via API to complete pairing."
        )

    async def cmd_whoami(args: str, ctx: dict) -> str:
        zh = container.config.agents.language == "zh"
        sender_id = ctx.get("sender_id", "")
        if not sender_id:
            return "未知身份。" if zh else "Unknown identity."
        user = rbac.resolve_user(sender_id)
        if zh:
            lines = [
                f"用户ID: {user.user_id}",
                f"角色: {user.role.value}",
                f"显示名: {user.display_name or '(未设置)'}",
            ]
            devices = store.list_user_devices(sender_id)
            if devices:
                trusted = sum(1 for d in devices if d.trusted)
                lines.append(f"设备: {len(devices)} 个（{trusted} 个已信任）")
        else:
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
        zh = container.config.agents.language == "zh"
        sender_id = ctx.get("sender_id", "")
        caller = rbac.resolve_user(sender_id)
        if not rbac.check_admin_access(caller):
            return "权限不足，需要管理员权限。" if zh else "Permission denied. Admin access required."
        parts = args.strip().split()
        if len(parts) < 2:
            return (
                "用法: /role <用户ID> <owner|admin|user|guest>"
                if zh
                else "Usage: /role <user_id> <owner|admin|user|guest>"
            )
        target_id, role_str = parts[0], parts[1]
        try:
            target_role = UserRole(role_str)
        except ValueError:
            if zh:
                return f"无效角色: {role_str}，可用: owner, admin, user, guest"
            return f"Invalid role: {role_str}. Use: owner, admin, user, guest"
        if target_role == UserRole.owner and not caller.is_owner:
            return "只有 owner 可以分配 owner 角色。" if zh else "Only owners can assign the owner role."
        if store.update_user_role(target_id, target_role):
            if zh:
                return f"用户 {target_id} 角色已更新为 {target_role.value}"
            return f"User {target_id} role updated to {target_role.value}"
        return f"用户 {target_id} 不存在。" if zh else f"User {target_id} not found."

    dispatcher.register_builtin("pair", cmd_pair)
    dispatcher.register_builtin("whoami", cmd_whoami)
    dispatcher.register_builtin("role", cmd_role)


def register_builtin_commands(dispatcher, container, tools, skills):
    async def cmd_help(args: str, ctx: dict) -> str:
        zh = container.config.agents.language == "zh"

        if zh:
            lines = [
                "== 系统命令 ==",
                "/help    — 显示此帮助",
                "/status  — 系统状态",
                "/reset   — 重置当前会话",
                "",
                "== 会话命令 ==",
                "/new     — 新建会话",
                "/old     — 列出历史会话",
                "/re <id> — 切换会话",
                "",
                "== 配置命令 ==",
                "/model [list|switch|temp|name]  — 查看/设置模型",
                "/sandbox on|off                 — sandbox 开关",
                "/approval off|ask|always        — 审批模式",
                "/rounds <n>                     — 最大工具轮数",
                "/compress on|off                — 压缩开关",
                "/progress on|off                — 工具进度通知开关",
                "/timezone <tz>                  — 时区设置",
                "/lang zh|en                     — 中英文切换",
                "/voice                          — 语音模式开关与设置",
                "",
                "== 快照 ==",
                "/snapshots [id]  — 查看/列出快照",
                "/rollback <id> [文件] — 回滚到快照",
                "",
                "== 子 Agent ==",
                "/agents [stop <id>] — 查看/管理子 Agent",
                "",
                "== 后台进程 ==",
                "/ps [kill <id>] — 后台进程管理",
                "",
                "== 其他 ==",
                "/skills  — 技能列表",
                "/search  — 搜索会话",
                "/prune   — 清理旧会话",
            ]
        else:
            lines = [
                "== System ==",
                "/help    — Show this help",
                "/status  — System status",
                "/reset   — Reset current session",
                "",
                "== Session ==",
                "/new     — New session",
                "/old     — List sessions",
                "/re <id> — Switch session",
                "",
                "== Config ==",
                "/model [list|switch|temp|name]  — View/set model",
                "/sandbox on|off                 — Sandbox toggle",
                "/approval off|ask|always        — Approval mode",
                "/rounds <n>                     — Max tool rounds",
                "/compress on|off                — Compression toggle",
                "/progress on|off                — Tool progress notifications toggle",
                "/timezone <tz>                  — Timezone",
                "/lang zh|en                     — Language switch",
                "/voice                          — Voice mode toggle & settings",
                "",
                "== Snapshots ==",
                "/snapshots [id]  — List/view snapshot diff",
                "/rollback <id> [file] — Rollback to snapshot",
                "",
                "== Sub-agents ==",
                "/agents [stop <id>] — View/manage sub-agents",
                "",
                "== Background ==",
                "/ps [kill <id>] — Background process management",
                "",
                "== Other ==",
                "/skills  — Skill list",
                "/search  — Search sessions",
                "/prune   — Prune old sessions",
            ]

        commands = dispatcher.list_commands()
        skill_cmds = [c for c in commands if not c.get("builtin")]
        if skill_cmds:
            lines.append("")
            lines.append("== 技能命令 ==" if zh else "== Skills ==")
            for c in skill_cmds:
                desc = c.get("description", "")[:40]
                lines.append(f"/{c['name']:<12} — {desc}")

        auth_cmds = [c for c in commands if c.get("builtin") and c["name"] in ("pair", "whoami", "role")]
        if auth_cmds:
            lines.append("")
            lines.append("== 认证命令 ==" if zh else "== Auth ==")
            if zh:
                lines.append("/pair   — 生成配对码")
                lines.append("/whoami — 查看身份")
                lines.append("/role   — 修改用户角色")
            else:
                lines.append("/pair   — Generate pairing code")
                lines.append("/whoami — View identity")
                lines.append("/role   — Change user role")

        return "\n".join(lines)

    async def cmd_reset(args: str, ctx: dict) -> str:
        zh = container.config.agents.language == "zh"
        thread_id = ctx.get("thread_id", "")
        if thread_id:
            try:
                state = await container.state_store.aload(thread_id)
                if state:
                    state.messages = []
                    await container.state_store.save(thread_id, state)
                if container.agent_loop:
                    container.agent_loop.invalidate_memory_cache()
                return "会话已重置。" if zh else "Session reset."
            except Exception as e:
                return f"重置失败: {e}" if zh else f"Reset failed: {e}"
        return "没有可重置的会话。" if zh else "No session to reset."

    async def cmd_status(args: str, ctx: dict) -> str:
        zh = container.config.agents.language == "zh"
        if zh:
            lines = [
                f"模型: {container.config.model.provider}/{container.config.model.name}",
                f"工具: {len(tools)}",
                f"技能: {len(skills)}",
                f"会话: {container.session_tracker.active_count}",
            ]
        else:
            lines = [
                f"Model: {container.config.model.provider}/{container.config.model.name}",
                f"Tools: {len(tools)}",
                f"Skills: {len(skills)}",
                f"Sessions: {container.session_tracker.active_count}",
            ]
        if container.cron_service:
            s = container.cron_service.status()
            if zh:
                lines.append(f"定时任务: {s['enabled_jobs']}/{s['total_jobs']} 个")
            else:
                lines.append(f"Cron: {s['enabled_jobs']}/{s['total_jobs']} jobs")
        try:
            from src.plugins.registry import get_plugin_registry

            reg = get_plugin_registry()
            if zh:
                lines.append(f"插件: {reg.plugin_count} 个（{reg.tool_count} 个工具）")
            else:
                lines.append(f"Plugins: {reg.plugin_count} ({reg.tool_count} tools)")
        except Exception:
            pass
        return "\n".join(lines)

    async def cmd_skills(args: str, ctx: dict) -> str:
        zh = container.config.agents.language == "zh"
        if not skills:
            return "没有已加载的技能。" if zh else "No skills loaded."
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
        zh = container.config.agents.language == "zh"
        from src.session_index.store import get_session_index
        from src.tools.session_search_tools import _format_results

        store = get_session_index()
        if not store:
            return "搜索功能未启用" if zh else "Search not enabled"

        limit = container.config.session_search.max_results

        results = await store.search(args, limit=limit)
        if results:
            return _format_results(results)
        return "暂无会话" if zh else "No sessions" if not args.strip() else ("未找到结果" if zh else "No results found")

    dispatcher.register_builtin("search", cmd_search)

    async def cmd_new(args: str, ctx: dict) -> str:
        zh = container.config.agents.language == "zh"
        user_key = ctx.get("user_key", "")
        channel_prefix = ctx.get("channel_prefix", "qq")
        if not user_key:
            return "无法确定会话。" if zh else "Cannot determine session."
        user_hash = user_key.split(":")[-1] if user_key else "unknown"
        sid = container.session_registry.new_session(user_key, channel_prefix, user_hash)
        if container.agent_loop:
            container.agent_loop.invalidate_memory_cache()
        if zh:
            return f"新会话已创建: {sid}\n发送消息即可开始。/old 查看会话列表，/re <id> 切换会话。"
        return f"New session started: {sid}\nSend messages to begin. Use /old to list sessions, /re <id> to switch."

    async def cmd_old(args: str, ctx: dict) -> str:
        zh = container.config.agents.language == "zh"
        user_key = ctx.get("user_key", "")
        if not user_key:
            return "无法确定会话。" if zh else "Cannot determine session."
        reg_sessions = container.session_registry.list_sessions(user_key)

        if not reg_sessions:
            orphaned = container.session_registry.find_orphaned_threads(
                user_key,
                container.state_store,
            )
            if not orphaned:
                orphaned = container.session_registry.find_all_channel_threads(
                    user_key,
                    container.state_store,
                )
            if orphaned:
                recovered = container.session_registry.recover_sessions(
                    user_key,
                    orphaned,
                )
                if recovered:
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
        current_marker = (
            " [当前]"
            if zh and current_override is None
            else (" [current]" if not zh and current_override is None else "")
        )
        empty_label = "(空)" if zh else "(empty)"
        no_history = "(无历史)" if zh else "(no history)"
        if has_default:
            lines.append(f"[default] {default_summary or empty_label}{current_marker}")
        else:
            lines.append(f"[default] {no_history}{current_marker}")

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
                        summary = "(新)" if zh else "(new)"
                except Exception:
                    summary = "(新)" if zh else "(new)"
                container.session_registry.update_summary(user_key, s["thread_id"], summary)
            cur = " [当前]" if zh and s["is_current"] else (" [current]" if not zh and s["is_current"] else "")
            dt = time.strftime("%m-%d %H:%M", time.localtime(s["created_at"]))
            lines.append(f"[{s['session_id']}] {summary}{cur} ({dt})")

        if len(lines) == 1 and not reg_sessions:
            if zh:
                return "暂无会话。\n/new - 创建新会话"
            return "No sessions yet.\n/new - create new session"

        if zh:
            result = "你的会话:\n" + "\n".join(lines)
            result += "\n\n/re <id> - 切换会话 (如 /re default, /re s1)\n/new - 创建新会话"
        else:
            result = "Your sessions:\n" + "\n".join(lines)
            result += "\n\n/re <id> - switch session (e.g. /re default, /re s1)\n/new - create new session"
        return result

    async def cmd_re(args: str, ctx: dict) -> str:
        zh = container.config.agents.language == "zh"
        user_key = ctx.get("user_key", "")
        session_id = args.strip()
        if not user_key:
            return "无法确定会话。" if zh else "Cannot determine session."
        if not session_id:
            if zh:
                return "用法: /re <会话ID>\n使用 /old 查看会话列表。"
            return "Usage: /re <session_id>\nUse /old to list sessions."
        tid = container.session_registry.switch_to(user_key, session_id)
        if tid and container.agent_loop:
            container.agent_loop.invalidate_memory_cache()
        if tid == "default":
            return "已切换回默认会话。" if zh else "Switched back to default session."
        if tid:
            if zh:
                return f"已恢复会话: {session_id}\n线程: {tid}"
            return f"Resumed session: {session_id}\nThread: {tid}"
        if zh:
            return f"会话不存在: {session_id}\n使用 /old 查看可用会话。"
        return f"Session not found: {session_id}\nUse /old to list available sessions."

    dispatcher.register_builtin("new", cmd_new)
    dispatcher.register_builtin("old", cmd_old)
    dispatcher.register_builtin("re", cmd_re)

    async def cmd_prune(args: str, ctx: dict) -> str:
        zh = container.config.agents.language == "zh"
        from src.session.pruner import prune_sessions, vacuum_database

        parts = args.strip().split()
        older_than_days = container.config.session.retention_days
        do_vacuum = container.config.session.vacuum_after_prune

        i = 0
        while i < len(parts):
            if parts[i] == "--older-than" and i + 1 < len(parts):
                try:
                    older_than_days = int(parts[i + 1])
                    i += 2
                    continue
                except ValueError:
                    pass
            if parts[i] == "--vacuum":
                do_vacuum = True
                i += 1
                continue
            if parts[i] == "--no-vacuum":
                do_vacuum = False
                i += 1
                continue
            i += 1

        cp_path = container.config.checkpointer.path
        si_path = container.config.session_search.index_path if container.config.session_search.enabled else None
        stats = prune_sessions(cp_path, older_than_days=older_than_days, session_index_path=si_path)

        if zh:
            lines = [
                f"清理完成（超过 {older_than_days} 天）:",
                f"  总会话数: {stats['total_sessions']}",
                f"  已删除: {stats['sessions_removed']}",
            ]
            if stats.get("index_sessions_removed", 0) > 0:
                lines.append(f"  索引已删除: {stats['index_sessions_removed']}")
            if do_vacuum and stats["sessions_removed"] > 0:
                new_size = vacuum_database(cp_path)
                lines.append(f"  清理后数据库大小: {new_size} 字节")
        else:
            lines = [
                "Prune complete (older than " + str(older_than_days) + " days):",
                "  Total sessions: " + str(stats["total_sessions"]),
                "  Checkpoints removed: " + str(stats["sessions_removed"]),
            ]
            if stats.get("index_sessions_removed", 0) > 0:
                lines.append("  Index removed: " + str(stats["index_sessions_removed"]))
            if do_vacuum and stats["sessions_removed"] > 0:
                new_size = vacuum_database(cp_path)
                lines.append("  Database size after cleanup: " + str(new_size) + " bytes")

        return "\n".join(lines)

    dispatcher.register_builtin("prune", cmd_prune)

    async def cmd_sandbox(args: str, ctx: dict) -> str:
        from src.tools.exec import set_sandbox_enabled

        zh = container.config.agents.language == "zh"
        cfg = container.config
        current = getattr(cfg.tools.exec, "sandbox_enabled", True)
        arg = args.strip().lower()

        if arg in ("on", "enable", "1", "true"):
            set_sandbox_enabled(True)
            return (
                "Sandbox 已开启 — 工作目录限制已启用。"
                if zh
                else "Sandbox ON — working directory restrictions enabled."
            )
        if arg in ("off", "disable", "0", "false"):
            set_sandbox_enabled(False)
            return "Sandbox 已关闭 — 无目录限制。" if zh else "Sandbox OFF — no directory restrictions."
        if zh:
            return f"Sandbox 状态: {'开启' if current else '关闭'}。用法: /sandbox on|off"
        return f"Sandbox is {'ON' if current else 'OFF'}. Usage: /sandbox on|off"

    dispatcher.register_builtin("sandbox", cmd_sandbox)

    # ── Config helper ──

    def _save_config():
        try:
            from src.config import save_config

            save_config(container.config)
        except Exception as e:
            logger.warning("Failed to persist config: %s", e)

    async def cmd_model(args: str, ctx: dict) -> str:
        from src.agent.client import FallbackChain, ChatClient

        zh = container.config.agents.language == "zh"
        cfg = container.config
        parts = args.strip().split(None, 1)
        sub = parts[0].lower() if parts else ""
        val = parts[1].strip() if len(parts) > 1 else ""

        # Build model list from config
        model_list = [
            {
                "provider": cfg.model.provider,
                "name": cfg.model.name,
                "base_url": cfg.model.base_url,
                "api_key": cfg.model.api_key,
                "context_window": cfg.model.context_window,
            },
        ]
        for fb in cfg.model.fallbacks or []:
            model_list.append(
                {
                    "provider": fb.provider,
                    "name": fb.name,
                    "base_url": fb.base_url or cfg.model.base_url,
                    "api_key": fb.api_key or cfg.model.api_key,
                    "context_window": fb.context_window,
                }
            )

        client = container.agent_loop._client if container.agent_loop else None
        active_idx = client._active_idx if isinstance(client, FallbackChain) else 0

        # /model list
        if sub == "list":
            if zh:
                lines = [f"可用模型 ({len(model_list)}):"]
            else:
                lines = [f"Available models ({len(model_list)}):"]
            for i, m in enumerate(model_list):
                marker = " *" if i == active_idx else ""
                key_status = (
                    "(有密钥)"
                    if zh and m.get("api_key")
                    else ("(has key)" if m.get("api_key") else "(无密钥)" if zh else "(no key)")
                )
                lines.append(f"  [{i}] {m['provider']}/{m['name']} {key_status}{marker}")
            lines.append("")
            if zh:
                lines.append("用法: /model switch <id>")
            else:
                lines.append("Usage: /model switch <id>")
            return "\n".join(lines)

        # /model switch <idx>
        if sub == "switch":
            if not val:
                if zh:
                    return "用法: /model switch <id>\n使用 /model list 查看 ID。"
                return "Usage: /model switch <id>\nUse /model list to see IDs."
            try:
                idx = int(val)
            except ValueError:
                if zh:
                    return f"无效 ID: {val}。使用 /model list 查看。"
                return f"Invalid ID: {val}. Use /model list to see IDs."
            if idx < 0 or idx >= len(model_list):
                if zh:
                    return f"ID 超出范围 (0-{len(model_list) - 1})。使用 /model list 查看。"
                return f"ID out of range (0-{len(model_list) - 1}). Use /model list."
            if isinstance(client, FallbackChain):
                client.switch_to(idx)
            m = model_list[idx]
            if zh:
                return f"已切换到 [{idx}] {m['provider']}/{m['name']}"
            return f"Switched to [{idx}] {m['provider']}/{m['name']}"

        # /model temp <value>
        if sub in ("temp", "temperature"):
            if not val:
                return f"温度: {cfg.model.temperature}" if zh else f"Temperature: {cfg.model.temperature}"
            try:
                t = float(val)
                assert 0 <= t <= 2
            except (ValueError, AssertionError):
                return "用法: /model temp <0-2>" if zh else "Usage: /model temp <0-2>"
            cfg.model.temperature = t
            if isinstance(client, FallbackChain):
                client.active.temperature = t
            elif isinstance(client, ChatClient):
                client.temperature = t
            _save_config()
            return f"温度已设为 {t}" if zh else f"Temperature set to {t}"

        # /model name <name>
        if sub == "name":
            if not val:
                return (
                    f"模型: {cfg.model.provider}/{cfg.model.name}"
                    if zh
                    else f"Model: {cfg.model.provider}/{cfg.model.name}"
                )
            cfg.model.name = val
            if isinstance(client, FallbackChain):
                client._all[0].model = val
            elif isinstance(client, ChatClient):
                client.model = val
            _save_config()
            return f"模型名称已设为 {val}" if zh else f"Model name set to {val}"

        # /model test
        if sub == "test":
            try:
                m = model_list[active_idx] if active_idx < len(model_list) else model_list[0]
                test_client = ChatClient(
                    base_url=m.get("base_url") or "",
                    api_key=m.get("api_key") or "",
                    model=m["name"],
                )
                resp = await test_client.chat_simple([{"role": "user", "content": "你好"}])
                if zh:
                    return f"[通过] 模型验证成功\n响应: {resp[:100]}..."
                return f"[OK] Model verification passed\nResponse: {resp[:100]}..."
            except Exception as e:
                if zh:
                    return f"[错误] 模型验证失败: {e}"
                return f"[Error] Model verification failed: {e}"

        # /model — show current status
        m = model_list[active_idx] if active_idx < len(model_list) else model_list[0]
        if zh:
            lines = [
                f"当前: [{active_idx}] {m['provider']}/{m['name']}",
                f"温度: {cfg.model.temperature}",
                f"上下文: {cfg.model.context_window}",
                f"已加载模型: {len(model_list)}",
                "",
                "用法:",
                "  /model list              — 查看所有模型",
                "  /model switch <id>       — 切换模型",
                "  /model temp <0-2>        — 设置温度",
                "  /model name <名称>       — 修改主模型名称",
                "  /model test              — 测试当前模型",
            ]
        else:
            lines = [
                f"Active: [{active_idx}] {m['provider']}/{m['name']}",
                f"Temperature: {cfg.model.temperature}",
                f"Context: {cfg.model.context_window}",
                f"Models loaded: {len(model_list)}",
                "",
                "Usage:",
                "  /model list              — show all models",
                "  /model switch <id>       — switch model",
                "  /model temp <0-2>        — set temperature",
                "  /model name <name>       — change primary model",
                "  /model test              — test current model",
            ]
        return "\n".join(lines)

    dispatcher.register_builtin("model", cmd_model)

    async def cmd_approval(args: str, ctx: dict) -> str:
        zh = container.config.agents.language == "zh"
        cfg = container.config
        arg = args.strip().lower()

        valid = ("off", "ask", "on_denylist_miss", "always")
        if arg in valid:
            cfg.tools.exec.approval_mode = arg
            from src.tools.exec import reset_config_cache

            reset_config_cache()
            _save_config()
            if zh:
                return f"审批模式已设为: {arg}"
            return f"Approval mode set to: {arg}"
        current = getattr(cfg.tools.exec, "approval_mode", "off")
        if zh:
            return f"审批模式: {current}\n用法: /approval off|ask|on_denylist_miss|always"
        return f"Approval mode: {current}\nUsage: /approval off|ask|on_denylist_miss|always"

    dispatcher.register_builtin("approval", cmd_approval)

    async def cmd_rounds(args: str, ctx: dict) -> str:
        zh = container.config.agents.language == "zh"
        cfg = container.config
        arg = args.strip()
        if arg:
            try:
                n = int(arg)
                assert n > 0
            except (ValueError, AssertionError):
                return "用法: /rounds <正整数>" if zh else "Usage: /rounds <positive integer>"
            cfg.agents.max_tool_rounds = n
            _save_config()
            return f"最大工具轮数已设为 {n}" if zh else f"Max tool rounds set to {n}"
        return f"最大工具轮数: {cfg.agents.max_tool_rounds}" if zh else f"Max tool rounds: {cfg.agents.max_tool_rounds}"

    dispatcher.register_builtin("rounds", cmd_rounds)

    async def cmd_compress(args: str, ctx: dict) -> str:
        zh = container.config.agents.language == "zh"
        cfg = container.config
        arg = args.strip().lower()

        if arg in ("on", "enable", "1", "true"):
            cfg.compression.enabled = True
            _save_config()
            return "压缩已开启。" if zh else "Compression ON."
        if arg in ("off", "disable", "0", "false"):
            cfg.compression.enabled = False
            _save_config()
            return "压缩已关闭。" if zh else "Compression OFF."
        current = getattr(cfg.compression, "enabled", True)
        if zh:
            return f"压缩状态: {'开启' if current else '关闭'}。用法: /compress on|off"
        return f"Compression is {'ON' if current else 'OFF'}. Usage: /compress on|off"

    dispatcher.register_builtin("compress", cmd_compress)

    async def cmd_progress(args: str, ctx: dict) -> str:
        zh = container.config.agents.language == "zh"
        cfg = container.config
        arg = args.strip().lower()

        if arg in ("on", "enable", "1", "true"):
            cfg.agents.tool_progress_notifications = True
            _save_config()
            return "工具进度通知已开启。" if zh else "Tool progress notifications ON."
        if arg in ("off", "disable", "0", "false"):
            cfg.agents.tool_progress_notifications = False
            _save_config()
            return "工具进度通知已关闭。" if zh else "Tool progress notifications OFF."
        current = cfg.agents.tool_progress_notifications
        if zh:
            return f"工具进度通知: {'开启' if current else '关闭'}。用法: /progress on|off"
        return f"Tool progress notifications are {'ON' if current else 'OFF'}. Usage: /progress on|off"

    dispatcher.register_builtin("progress", cmd_progress)

    async def cmd_timezone(args: str, ctx: dict) -> str:
        zh = container.config.agents.language == "zh"
        cfg = container.config
        arg = args.strip()
        if arg:
            try:
                import zoneinfo

                zoneinfo.ZoneInfo(arg)
            except Exception:
                return f"无效时区: {arg}" if zh else f"Invalid timezone: {arg}"
            cfg.agents.timezone = arg
            _save_config()
            return f"时区已设为 {arg}" if zh else f"Timezone set to {arg}"
        return f"时区: {cfg.agents.timezone}" if zh else f"Timezone: {cfg.agents.timezone}"

    dispatcher.register_builtin("timezone", cmd_timezone)

    async def cmd_lang(args: str, ctx: dict) -> str:
        cfg = container.config
        arg = args.strip().lower()
        if arg in ("zh", "cn", "chinese"):
            cfg.agents.language = "zh"
            _save_config()
            return "语言已切换为中文。"
        if arg in ("en", "english"):
            cfg.agents.language = "en"
            _save_config()
            return "Language switched to English."
        current = cfg.agents.language
        return f"语言: {'中文 (zh)' if current == 'zh' else 'English (en)'}\n用法: /lang zh|en"

    dispatcher.register_builtin("lang", cmd_lang)

    # ── Snapshot commands ──

    async def cmd_snapshots(args: str, ctx: dict) -> str:
        zh = container.config.agents.language == "zh"
        from src.tools.snapshot import get_snapshot_manager

        mgr = get_snapshot_manager()
        if mgr is None:
            return "快照功能未启用。" if zh else "Snapshots not enabled."

        import os
        from src.tools.file_tools import _BASE_DIR

        work_dir = _BASE_DIR

        arg = args.strip()
        if arg:
            # Show diff for a specific snapshot
            diff_text = mgr.diff(work_dir, arg)
            if diff_text.startswith("Error"):
                return diff_text
            header = f"快照 {arg} 与当前差异:" if zh else f"Diff from snapshot {arg}:"
            return f"{header}\n{diff_text}"

        snapshots = mgr.list_snapshots(work_dir)
        if not snapshots:
            return "暂无快照。" if zh else "No snapshots yet."

        lines = []
        if zh:
            lines.append(f"快照列表 ({len(snapshots)}):")
        else:
            lines.append(f"Snapshots ({len(snapshots)}):")
        for s in snapshots:
            lines.append(f"  {s['id']}  {s['date']}  {s['message']}")
        if zh:
            lines.append("")
            lines.append("用法:")
            lines.append("  /snapshots <id>  — 查看快照差异")
            lines.append("  /rollback <id>   — 回滚到快照")
            lines.append("  /rollback <id> <文件> — 回滚单个文件")
        else:
            lines.append("")
            lines.append("Usage:")
            lines.append("  /snapshots <id>  — view snapshot diff")
            lines.append("  /rollback <id>   — rollback to snapshot")
            lines.append("  /rollback <id> <file> — rollback single file")
        return "\n".join(lines)

    dispatcher.register_builtin("snapshots", cmd_snapshots)

    async def cmd_rollback(args: str, ctx: dict) -> str:
        zh = container.config.agents.language == "zh"
        parts = args.strip().split(None, 1)
        if not parts:
            if zh:
                return "用法: /rollback <快照ID> [文件路径]"
            return "Usage: /rollback <snapshot_id> [file_path]"

        from src.tools.snapshot import get_snapshot_manager

        mgr = get_snapshot_manager()
        if mgr is None:
            return "快照功能未启用。" if zh else "Snapshots not enabled."

        import os
        from src.tools.file_tools import _BASE_DIR

        work_dir = _BASE_DIR

        snap_id = parts[0]
        file_path = parts[1].strip() if len(parts) > 1 else ""

        if file_path:
            result = mgr.restore_file(work_dir, snap_id, file_path)
        else:
            result = mgr.restore(work_dir, snap_id)

        if result.startswith("Restore failed") or result.startswith("No snapshot"):
            return result
        if file_path:
            if zh:
                return f"已从快照 {snap_id} 恢复文件 {file_path}"
            return result
        if zh:
            return f"已回滚到快照 {snap_id}"
        return result

    dispatcher.register_builtin("rollback", cmd_rollback)

    # ── Sub-agent commands ──

    async def cmd_agents(args: str, ctx: dict) -> str:
        zh = container.config.agents.language == "zh"
        from src.agents.run_registry import get_run_registry

        run_registry = get_run_registry()

        parts = args.strip().split(None, 1)
        if parts and parts[0].lower() == "stop" and len(parts) > 1:
            target_id = parts[1].strip()
            ok = await run_registry.request_interrupt(target_id)
            if ok:
                return f"已请求中断子 Agent {target_id}。" if zh else f"Interrupt requested for sub-agent {target_id}."
            return f"未找到运行中的子 Agent: {target_id}" if zh else f"No running sub-agent found: {target_id}"

        active = await run_registry.get_active_tree()
        recent = await run_registry.list_runs(limit=5)

        lines = []
        if zh:
            lines.append("== 子 Agent 状态 ==")
        else:
            lines.append("== Sub-agents ==")

        if active:
            for a in active:
                status_icon = "\U0001f504" if a["status"] == "running" else "⏳"
                interrupt_tag = " [中断中]" if a.get("interrupt_requested") else ""
                lines.append(
                    f"  {status_icon} [{a['id']}] {a['agent_name']} — "
                    f'"{a["task"][:50]}" ({a["elapsed"]}s){interrupt_tag}'
                )
        else:
            lines.append("  " + ("（无运行中）" if zh else "(none running)"))

        # Show recent completed
        done = [r for r in recent if r["status"] in ("completed", "error", "timeout", "interrupted")]
        if done:
            lines.append("")
            if zh:
                lines.append("最近完成:")
            else:
                lines.append("Recent:")
            for r in done[:5]:
                icon = {"completed": "✅", "error": "❌", "timeout": "⏰", "interrupted": "⚡"}.get(r["status"], "?")
                dur = round((r.get("completed_at") or 0) - r["started_at"], 1) if r.get("completed_at") else "?"
                result_preview = (r.get("result") or "")[:40]
                lines.append(f"  {icon} [{r['id']}] {r['agent_name']} ({dur}s) {result_preview}")

        if active:
            lines.append("")
            if zh:
                lines.append("用法: /agents stop <id> — 中断子 Agent")
            else:
                lines.append("Usage: /agents stop <id> — interrupt sub-agent")

        return "\n".join(lines)

    dispatcher.register_builtin("agents", cmd_agents)

    # ── Background process commands ──

    async def cmd_ps(args: str, ctx: dict) -> str:
        zh = container.config.agents.language == "zh"
        from src.tools.process import get_process_registry

        registry = get_process_registry()

        parts = args.strip().split(None, 1)

        # /ps kill <id>
        if parts and parts[0].lower() == "kill" and len(parts) > 1:
            result = await registry.kill(parts[1].strip())
            return result

        # /ps <id> — show detail
        if parts and parts[0].lower() not in ("", "list"):
            sid = parts[0].strip()
            info = await registry.poll(sid)
            if info.get("status") == "not_found":
                return f"进程 {sid} 不存在。" if zh else f"Process {sid} not found."
            lines = []
            if zh:
                lines.append(f"进程 [{sid}]")
                lines.append(f"  命令: {info.get('command', '')}")
                lines.append(f"  状态: {info['status']}")
                if info.get("exit_code") is not None:
                    lines.append(f"  退出码: {info['exit_code']}")
                lines.append(f"  运行时间: {info['elapsed']}s")
            else:
                lines.append(f"Process [{sid}]")
                lines.append(f"  Command: {info.get('command', '')}")
                lines.append(f"  Status: {info['status']}")
                if info.get("exit_code") is not None:
                    lines.append(f"  Exit code: {info['exit_code']}")
                lines.append(f"  Elapsed: {info['elapsed']}s")
            tail = info.get("output_tail", "")
            if tail:
                lines.append("")
                lines.append("--- output tail ---")
                lines.append(tail[-2000:])
            return "\n".join(lines)

        # /ps — list all
        sessions = registry.list_sessions()
        if not sessions:
            return "没有后台进程。" if zh else "No background processes."

        lines = []
        if zh:
            lines.append("== 后台进程 ==")
        else:
            lines.append("== Background Processes ==")
        for s in sessions:
            if s["status"] == "running":
                icon = "\U0001f504"
            elif s.get("exit_code") == 0:
                icon = "✅"
            else:
                icon = "❌"
            lines.append(f"  {icon} [{s['id']}] pid={s['pid']} ({s['elapsed']}s) {s['command']}")
        lines.append("")
        if zh:
            lines.append("用法: /ps <id> — 查看详情 | /ps kill <id> — 终止")
        else:
            lines.append("Usage: /ps <id> — detail | /ps kill <id> — terminate")
        return "\n".join(lines)

    dispatcher.register_builtin("ps", cmd_ps)

    async def cmd_auto(args: str, ctx: dict) -> str:
        zh = container.config.agents.language == "zh"
        arg = args.strip().lower()
        if arg in ("on", "开启", "开"):
            container.config.task.enabled = True
            return (
                "✅ 自主工作模式已开启。我会为复杂任务制定计划并自动检查进度。" if zh else "✅ Autonomous mode enabled."
            )
        elif arg in ("off", "关闭", "关"):
            container.config.task.enabled = False
            from src.task.store import get_task_store

            store = get_task_store(container.config.task.db_path)
            runs = await store.list_by_status("running", "planning")
            cancelled = 0
            for r in runs:
                r.status = "cancelled"
                for cp in r.checkpoints:
                    if cp.cron_job_id and container.cron_service:
                        try:
                            await container.cron_service.remove_job(cp.cron_job_id)
                        except Exception:
                            pass
                await store.save(r)
                cancelled += 1
            msg = (
                f"❌ 自主工作模式已关闭。已取消 {cancelled} 个活跃任务。"
                if zh
                else f"❌ Autonomous mode disabled. {cancelled} tasks cancelled."
            )
            return msg
        else:
            status = "已开启" if container.config.task.enabled else "已关闭"
            return (
                f"自主工作模式当前: {status}。用法: /auto on 或 /auto off"
                if zh
                else f"Autonomous mode: {status}. Usage: /auto on or /auto off"
            )

    dispatcher.register_builtin("auto", cmd_auto)

    async def cmd_interrupt(args: str, ctx: dict) -> str:
        zh = container.config.agents.language == "zh"
        thread_id = ctx.get("thread_id", "")
        flag = container.state_store.get_interrupt_flag(thread_id)
        msg = args.strip() if args.strip() else None
        flag.interrupt(msg)
        if zh:
            return "⚡ 已发送中断信号" + (f"，附带消息: {msg[:80]}" if msg else "")
        return "⚡ Interrupt signal sent" + (f", message: {msg[:80]}" if msg else "")

    dispatcher.register_builtin("interrupt", cmd_interrupt)

    async def cmd_steer(args: str, ctx: dict) -> str:
        zh = container.config.agents.language == "zh"
        thread_id = ctx.get("thread_id", "")
        flag = container.state_store.get_interrupt_flag(thread_id)
        if flag.steer(args):
            return "📡 引导消息已发送" if zh else "📡 Steer message sent"
        if zh:
            return "引导消息为空或 agent 正在中断中"
        return "Steer text empty or agent is interrupting"

    dispatcher.register_builtin("steer", cmd_steer)

    # ── /voice ──────────────────────────────────────────────
    async def cmd_voice(args: str, ctx: dict) -> str:
        zh = container.config.agents.language == "zh"
        cfg = container.config.voice
        parts = args.strip().split(None, 1)
        sub = parts[0].lower() if parts else ""
        val = parts[1].strip() if len(parts) > 1 else ""

        VOICE_OPTIONS = [
            ("zh-CN-YunxiNeural", "男声，自然"),
            ("zh-CN-XiaoxiaoNeural", "女声，温柔"),
            ("zh-CN-YunjianNeural", "男声，沉稳"),
            ("zh-CN-XiaoyiNeural", "女声，活泼"),
        ]
        VOICE_OPTIONS_EN = [
            ("zh-CN-YunxiNeural", "Male, natural"),
            ("zh-CN-XiaoxiaoNeural", "Female, gentle"),
            ("zh-CN-YunjianNeural", "Male, steady"),
            ("zh-CN-XiaoyiNeural", "Female, lively"),
        ]
        opts = VOICE_OPTIONS if zh else VOICE_OPTIONS_EN

        # /voice on
        if sub in ("on", "enable", "开启"):
            cfg.enabled = True
            _save_config()
            return "语音模式已开启。" if zh else "Voice mode enabled."

        # /voice off
        if sub in ("off", "disable", "关闭"):
            cfg.enabled = False
            _save_config()
            return "语音模式已关闭。" if zh else "Voice mode disabled."

        # /voice list
        if sub == "list":
            lines = ["可用音色:" if zh else "Available voices:"]
            for i, (voice_id, desc) in enumerate(opts, 1):
                marker = " (当前)" if zh and cfg.voice == voice_id else (" (current)" if cfg.voice == voice_id else "")
                lines.append(f"  {i}. {voice_id} — {desc}{marker}")
            return "\n".join(lines)

        # /voice set <1-4>
        if sub == "set":
            if not val:
                return "用法: /voice set <1-4>" if zh else "Usage: /voice set <1-4>"
            try:
                idx = int(val) - 1
                if idx < 0 or idx >= len(opts):
                    raise ValueError
            except ValueError:
                return f"无效选项: {val}。请输入 1-4。" if zh else f"Invalid option: {val}. Use 1-4."
            cfg.voice = opts[idx][0]
            _save_config()
            return f"音色已设为: {opts[idx][0]}" if zh else f"Voice set to: {opts[idx][0]}"

        # /voice threshold <n>
        if sub in ("threshold", "阈值"):
            if not val:
                return f"当前阈值: {cfg.threshold} 字" if zh else f"Current threshold: {cfg.threshold} chars"
            try:
                n = int(val)
                assert n > 0
            except (ValueError, AssertionError):
                return "用法: /voice threshold <正整数>" if zh else "Usage: /voice threshold <positive integer>"
            cfg.threshold = n
            _save_config()
            return f"字数阈值已设为 {n} 字" if zh else f"Threshold set to {n} chars"

        # /voice (无参数) — 显示帮助和当前状态
        if zh:
            status = "开启" if cfg.enabled else "关闭"
            voice_name = cfg.voice
            lines = [
                "== 语音模式 ==",
                "/voice on             — 开启语音模式",
                "/voice off            — 关闭语音模式",
                "/voice list           — 列出可用音色",
                "/voice set <1-4>      — 设置音色",
                "/voice threshold <n>  — 设置字数阈值",
                "",
                f"当前状态: {status}",
                f"当前音色: {voice_name}",
                f"当前阈值: {cfg.threshold} 字",
            ]
        else:
            status = "ON" if cfg.enabled else "OFF"
            voice_name = cfg.voice
            lines = [
                "== Voice Mode ==",
                "/voice on             — Enable voice mode",
                "/voice off            — Disable voice mode",
                "/voice list           — List available voices",
                "/voice set <1-4>      — Set voice",
                "/voice threshold <n>  — Set character threshold",
                "",
                f"Status: {status}",
                f"Voice: {voice_name}",
                f"Threshold: {cfg.threshold} chars",
            ]
        return "\n".join(lines)

    dispatcher.register_builtin("voice", cmd_voice)

    async def cmd_restart(args: str, ctx: dict) -> str:
        zh = container.config.agents.language == "zh"
        logger.info("Restart requested via /restart command")

        chat_id = ctx.get("chat_id", "")
        channel_prefix = ctx.get("channel_prefix", "")
        ch = None
        if channel_prefix == "qq" and container.qq:
            ch = container.qq
        elif channel_prefix == "weixin" and container.weixin:
            ch = container.weixin
        if ch and chat_id:
            try:
                await ch.send_text(chat_id, "🔄 正在重启..." if zh else "🔄 Restarting...")
            except Exception:
                pass

        await asyncio.sleep(0.5)

        import os
        import subprocess

        subprocess.Popen(["flyclaw"], close_fds=True)
        os._exit(0)

    dispatcher.register_builtin("restart", cmd_restart)

    if container.config.auth.enabled and container.rbac:
        register_auth_commands(dispatcher, container)

    logger.info("Commands: %d skill + built-in", len(dispatcher._commands))
