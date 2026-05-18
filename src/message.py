from __future__ import annotations

import asyncio
import json
import logging
import re

from src.agent.loop import ApprovalPending

logger = logging.getLogger("myclaw")

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n([\s\S]*?)\n\s*```", re.IGNORECASE)
_MAX_FORMAT_LEN = 8000


def _format_json_value(val, indent=0):
    prefix = "  " * indent
    if isinstance(val, dict):
        if not val:
            return "{}"
        lines = []
        for k, v in val.items():
            formatted = _format_json_value(v, indent + 1)
            lines.append(f"{prefix}  **{k}**: {formatted}")
        return "\n".join(lines)
    elif isinstance(val, list):
        if not val:
            return "[]"
        if val and isinstance(val[0], dict):
            lines = []
            for i, item in enumerate(val):
                lines.append(f"{prefix}{i + 1}.")
                for k, v in item.items():
                    formatted = _format_json_value(v, indent + 1)
                    lines.append(f"{prefix}  - **{k}**: {formatted}")
            return "\n".join(lines)
        return ", ".join(str(x) for x in val)
    elif isinstance(val, bool):
        return "true" if val else "false"
    elif isinstance(val, (int, float)):
        return str(val)
    elif val is None:
        return "null"
    return str(val)


def _format_display(text: str) -> str:
    if not text or len(text) > _MAX_FORMAT_LEN:
        return text

    def _replacer(m):
        raw = m.group(1).strip()
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return m.group(0)
        formatted = _format_json_value(data)
        return f"```\n{formatted}\n```"

    return _JSON_FENCE_RE.sub(_replacer, text)


_SILENT_TOOLS = frozenset({
    "text_to_speech", "send_image_to_chat", "send_file_to_chat", "send_voice",
    "qq_send_image", "qq_send_file",
    "skill_manage", "curate_skills", "curator_status",
})


class MessageHandler:
    def __init__(self, container):
        self._container = container

    @staticmethod
    def _resolve_session_key(sender_id: str, chat_type: str, chat_id: str, scope: str) -> str:
        if scope == "global":
            return "global"
        if chat_type == "p2p":
            return f"user:{sender_id}"
        return f"group:{chat_id}"

    def create_callback(self, session_scope: str, channel_prefix: str = "qq"):
        async def on_message(
            text: str,
            sender_id: str,
            chat_id: str,
            chat_type: str,
            message_id: str,
            reply_fn,
            stream_fn,
        ):
            from src.events import emit_async

            session_key = self._resolve_session_key(sender_id, chat_type, chat_id, session_scope)
            legacy_thread_id = f"{channel_prefix}:{session_key}"
            override = self._container.session_registry.get_current(legacy_thread_id)
            thread_id = override or legacy_thread_id

            is_command = text.strip().startswith("/")

            await emit_async(
                "message.received",
                text=text[:200],
                sender_id=sender_id,
                chat_id=chat_id,
                chat_type=chat_type,
                message_id=message_id,
                channel=channel_prefix,
                thread_id=thread_id,
                is_command=is_command,
            )

            self._container.session_tracker.touch(thread_id)

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

            cmd_match = self._container.dispatcher.match(text)
            if cmd_match is not None:
                cmd_name, cmd_args = cmd_match
                logger.info("Slash command: /%s %.50s", cmd_name, cmd_args)
                result = await self._container.dispatcher.dispatch(
                    cmd_name,
                    cmd_args,
                    context={"thread_id": thread_id, "sender_id": sender_id, "chat_id": chat_id, "user_key": legacy_thread_id, "channel_prefix": channel_prefix},
                )
                await reply_fn(result)
                return

            # "/" prefix but not a registered command — reply directly, skip model
            if text.strip().startswith("/"):
                await reply_fn("未知命令。输入 /help 查看可用命令。")
                return

            from src.tools.cron_tools import set_current_chat_id
            set_current_chat_id(chat_id)
            from src.tools.media_tools import set_current_channel
            set_current_channel(channel_prefix)
            from src.tools.browser.tools import set_browser_session
            set_browser_session(chat_id)

            from src.agent.state import AgentState

            input_state = AgentState(
                messages=[{"role": "user", "content": text}],
                system_prompt=self._container.config.agents.system_prompt,
                sender_id=sender_id,
                chat_id=chat_id,
                chat_type=chat_type,
                message_id=message_id,
                channel=channel_prefix,
            )

            existing = await self._container.state_store.aload(thread_id)
            if existing:
                if existing.pending_approval:
                    await reply_fn("⏳ 有待审批的操作，请先回复审批后再发新消息。")
                    return
                input_state.messages = existing.messages + input_state.messages

            assistant_text = None
            identity_written = False
            pre_msg_count = len(input_state.messages)

            # Subscribe to tool events for progress reporting
            zh = self._container.config.agents.language == "zh"
            show_progress = self._container.config.agents.tool_progress_notifications
            active_chat_id = chat_id
            channel = channel_prefix
            _progress_unsub = None

            async def _on_tool_event(event: str, **kwargs):
                if not show_progress:
                    return
                tid = kwargs.get("thread_id", "")
                if tid != thread_id:
                    return
                try:
                    if event == "tool.exec_started":
                        tool = kwargs.get("tool_name", "")
                        if tool in _SILENT_TOOLS:
                            return
                        raw_args = kwargs.get("args_preview", "")
                        try:
                            parsed = json.loads(raw_args)
                            if isinstance(parsed, dict):
                                vals = list(parsed.values())
                                args_preview = str(vals[0])[:80] if len(parsed) == 1 else ", ".join(f"{k}={v}" for k, v in parsed.items())[:80]
                            else:
                                args_preview = raw_args[:80]
                        except (json.JSONDecodeError, ValueError):
                            args_preview = raw_args[:80]
                        if zh:
                            msg = f"🔧 {tool}: {args_preview}" if args_preview else f"🔧 执行 {tool}..."
                        else:
                            msg = f"🔧 {tool}: {args_preview}" if args_preview else f"🔧 Running {tool}..."
                        if channel == "qq" and self._container.qq:
                            await self._container.qq.send_text(active_chat_id, msg)
                    elif event == "tool.exec_failed":
                        tool = kwargs.get("tool_name", "")
                        err = kwargs.get("error", "")[:80]
                        if zh:
                            msg = f"❌ {tool} 失败: {err}"
                        else:
                            msg = f"❌ {tool} failed: {err}"
                        if channel == "qq" and self._container.qq:
                            await self._container.qq.send_text(active_chat_id, msg)
                except Exception:
                    pass

            from src.events import subscribe_async, unsubscribe
            _progress_unsub = subscribe_async("tool.*", _on_tool_event)

            try:
                logger.debug("[flow] agent_loop run start, state has %d messages", pre_msg_count)

                try:
                    result_state = await self._container.agent_loop.run(input_state, thread_id)
                except ApprovalPending as exc:
                    asyncio.create_task(self._handle_approval_pending(exc, chat_id))
                    result_state = await self._container.state_store.aload(thread_id) or input_state

                logger.debug("[flow] agent_loop run done")

                messages = result_state.messages

                if self._container.config.session_search.enabled and self._container.config.session_search.auto_sync:
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
                                tool_max_chars=self._container.config.session_search.tool_content_max_chars,
                            )
                    except Exception as e:
                        logger.warning("Session index sync failed: %s", e)

                for msg in reversed(messages):
                    if msg.get("role") == "assistant" and msg.get("content"):
                        assistant_text = msg["content"].lstrip("\n")
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

                from src.events import emit_async
                await emit_async(
                    "agent.error",
                    thread_id=thread_id,
                    error=str(e),
                    error_type=type(e).__name__,
                    channel=channel_prefix,
                )
            finally:
                if _progress_unsub:
                    unsubscribe("tool.*", _on_tool_event)

            if assistant_text:
                if getattr(self._container.config, "link_understanding", None) and self._container.config.link_understanding.enabled:
                    try:
                        from src.link_understanding import detect_and_preview_links

                        preview = await detect_and_preview_links(
                            text, max_previews=self._container.config.link_understanding.max_previews
                        )
                        if preview:
                            assistant_text += "\n" + preview
                    except Exception:
                        pass

                display_text = _format_display(assistant_text)

                try:
                    from src.media_delivery import deliver_media
                    ch = self._container.qq if channel_prefix == "qq" else None
                    if ch:
                        display_text = await deliver_media(display_text, chat_id, channel_prefix, ch)
                except Exception as e:
                    logger.warning("Media delivery failed: %s", e)

                try:
                    logger.debug("[flow] sending reply, len=%d", len(display_text))
                    await reply_fn(display_text)
                    logger.info("Reply to %s: %.100s", session_key, display_text)
                except Exception as e:
                    logger.error("Reply failed: %s", e)

                from src.events import emit_async
                await emit_async(
                    "message.replied",
                    thread_id=thread_id,
                    reply_length=len(display_text),
                    channel=channel_prefix,
                    session_key=session_key,
                )

                if identity_written:
                    try:
                        await reply_fn("\U0001f4be update memory: 已更新身份记忆")
                    except Exception:
                        pass

                if (
                    getattr(self._container, "memory_searcher", None)
                    and self._container.config.memory.enabled
                    and getattr(self._container.config.memory, "auto_session_memory", False)
                ):
                    try:
                        await self._container.memory_searcher.store.add_document(
                            f"session:{session_key}",
                            f"Q: {text}\nA: {display_text}",
                        )
                    except Exception:
                        pass

                if getattr(self._container.config, "beads", None) and self._container.config.beads.enabled:
                    try:
                        from src.tools.beads_tools import auto_extract_memory, save_memory
                        extracted = auto_extract_memory(text, display_text)
                        if extracted:
                            content, category = extracted
                            await save_memory(content)
                        elif self._container.config.beads.memory_judge_model:
                            task = asyncio.create_task(self._beads_llm_judge(
                                text, display_text, reply_fn,
                            ))
                            self._container.background_tasks.add(task)
                            task.add_done_callback(self._container.background_tasks.discard)
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

    async def _handle_approval_pending(
        self,
        exc,
        chat_id: str,
    ):
        from src.tools.approval import get_approval_manager

        try:
            mgr = get_approval_manager()
            approval_timeout = getattr(exc, "timeout", None) or 120
            auto_deny = getattr(exc, "auto_deny", False)
            zh = self._container.config.agents.language == "zh"

            if chat_id.startswith(("c2c:", "group:", "channel:", "dm:")):
                warn = "DANGEROUS" if exc.denylisted else "requires approval"
                await self._container.qq.send_text(
                    chat_id,
                    f"**Approval Required** ({warn})\n```\n{exc.command_preview}\n```\n"
                    f"Reply 'yes' to allow or 'no' to deny. (request: {exc.request_id})",
                )
                decision = await mgr.await_approval(exc.request_id, timeout=approval_timeout)
                if decision == "timeout":
                    if auto_deny:
                        msg = "操作超时，已自动拒绝。" if zh else "Operation timed out, auto-denied."
                        await self._container.qq.send_text(chat_id, msg)
                    decision = "deny"
                result_state = await self._container.agent_loop.resume(exc.thread_id, decision)
                assistant_text = ""
                for msg in reversed(result_state.messages):
                    if msg.get("role") == "assistant" and msg.get("content"):
                        assistant_text = msg["content"]
                        break
                if assistant_text:
                    await self._container.qq.send_text(chat_id, assistant_text)
                return
        except Exception:
            logger.exception("Error in _handle_approval_pending for request %s", exc.request_id)

    async def _beads_llm_judge(self, user_input: str, ai_response: str, reply_fn):
        try:
            from src.tools.beads_tools import judge_memory_with_llm, save_memory

            model_name = self._container.config.beads.memory_judge_model
            base_url = self._container.config.beads.memory_judge_base_url or self._container.config.model.base_url
            api_key = self._container.config.beads.memory_judge_api_key or self._container.config.model.api_key

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
