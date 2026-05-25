from __future__ import annotations

import asyncio
import json
import logging
import re

from src.agent.loop import ApprovalPending

logger = logging.getLogger("flyclaw")

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
    "weixin_send_image", "weixin_send_file", "weixin_send_voice",
    "skills_list", "skill_view", "skill_manage", "skill_hub",
})


_MAX_INTERRUPT_DEPTH = 5
_MAX_PENDING_QUEUE = 10


class MessageHandler:
    def __init__(self, container):
        self._container = container
        self._approval_handler_threads: set[str] = set()
        self._pending_queue: dict[str, list[str]] = {}
        self._interrupted_threads: dict[str, str] = {}

    def _get_channel(self, channel_prefix: str):
        if channel_prefix == "qq":
            return self._container.qq
        if channel_prefix == "weixin":
            return self._container.weixin
        return None

    def _get_channel_for_chat_id(self, chat_id: str):
        if chat_id.startswith(("c2c:", "group:", "channel:", "dm:")):
            return self._container.qq
        return self._container.weixin or self._container.qq

    def _enqueue(self, thread_id: str, text: str) -> None:
        q = self._pending_queue.setdefault(thread_id, [])
        q.append(text)
        if len(q) > _MAX_PENDING_QUEUE:
            self._pending_queue[thread_id] = q[-_MAX_PENDING_QUEUE:]

    def _dequeue(self, thread_id: str) -> str | None:
        q = self._pending_queue.get(thread_id)
        if not q:
            return None
        text = q.pop(0)
        if not q:
            del self._pending_queue[thread_id]
        return text

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

            from src.tools.exec import _current_thread_id
            _tid_token = _current_thread_id.set(thread_id)
            try:
                await _on_message_inner(
                    text, sender_id, chat_id, chat_type, message_id,
                    reply_fn, stream_fn, thread_id, is_command, channel_prefix,
                    session_scope, legacy_thread_id, session_key,
                )
            finally:
                _current_thread_id.reset(_tid_token)

        async def _on_message_inner(
                text, sender_id, chat_id, chat_type, message_id,
                reply_fn, stream_fn, thread_id, is_command, channel_prefix,
                session_scope, legacy_thread_id, session_key,
        ):
            if channel_prefix in ("qq", "weixin"):
                from src.tools.approval import get_approval_manager
                mgr = get_approval_manager()
                pending_list = mgr.list_pending()
                for req in pending_list:
                    if req.chat_id == chat_id:
                        is_y = text.strip().lower() == "/y"
                        zh = self._container.config.agents.language == "zh"
                        if is_y:
                            mgr.resolve(req.id, "allow_once")
                            if req.tool_name in ("memory_delete", "memory"):
                                await reply_fn("✅ 已确认删除记忆")
                            else:
                                await reply_fn("Approval granted." if not zh else "✅ 已批准执行。")
                            if req.thread_id not in self._approval_handler_threads:
                                asyncio.create_task(self._resume_and_reply(req.thread_id, "allow_once", chat_id))
                            return
                        else:
                            mgr.resolve(req.id, "deny")
                            if req.thread_id not in self._approval_handler_threads:
                                asyncio.create_task(self._resume_and_reply(req.thread_id, "deny", chat_id))
                            break

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

            from src.tools.task_tools import set_task_context
            set_task_context(chat_id=chat_id, sender_id=sender_id, thread_id=thread_id)

            from src.agent.state import AgentState

            system_prompt = self._container.config.agents.system_prompt
            if getattr(self._container.config, "task", None) and self._container.config.task.enabled:
                system_prompt += (
                    "\n\n## 自主工作模式\n"
                    "自主工作模式已开启。如果用户提出复杂任务（研究、开发、调研、写作等），"
                    "请先调用 task_manage(action=\"plan\") 工具制定执行计划，包含步骤和检查点，然后再开始执行。"
                    "在检查点触发时，调用 task_manage(action=\"status\") 查看进度，然后继续执行下一步。"
                    "完成任务后调用 task_manage(action=\"advance\") 标记步骤完成。"
                )

            input_state = AgentState(
                messages=[{"role": "user", "content": text}],
                system_prompt=system_prompt,
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

            # Handle busy agent: interrupt / queue / steer
            # Also treat as busy when drain_pending is still processing an
            # interrupted message or queued messages — prevents new messages
            # from racing in and grabbing the lock before drain runs.
            if (self._container.agent_loop.is_thread_busy(thread_id)
                    or thread_id in self._interrupted_threads
                    or thread_id in self._pending_queue):
                await self._handle_busy_message(
                    text, thread_id, channel_prefix, reply_fn,
                )
                return

            await self._run_agent_turn(
                input_state=input_state,
                thread_id=thread_id,
                session_key=session_key,
                chat_id=chat_id,
                sender_id=sender_id,
                chat_type=chat_type,
                channel_prefix=channel_prefix,
                reply_fn=reply_fn,
                stream_fn=stream_fn,
                system_prompt=system_prompt,
                original_text=text,
            )

        return on_message

    async def _try_send_voice(
        self, text: str, chat_id: str, channel_prefix: str, voice: str
    ) -> bool:
        """Try to convert text to speech and send as voice message. Returns True if sent."""
        import edge_tts
        import tempfile
        from pathlib import Path

        try:
            communicate = edge_tts.Communicate(text, voice=voice)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name
            await communicate.save(tmp_path)
            audio_bytes = Path(tmp_path).read_bytes()
            Path(tmp_path).unlink(missing_ok=True)
        except Exception as e:
            logger.error("edge-tts synthesis failed: %s", e)
            return False

        if not audio_bytes:
            return False

        channel = self._get_channel(channel_prefix)
        if not channel:
            return False

        try:
            if channel_prefix == "qq":
                return await channel.send_audio(chat_id, audio_bytes)
            elif channel_prefix == "weixin":
                import tempfile as _tf
                with _tf.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                    f.write(audio_bytes)
                    tmp_file = f.name
                try:
                    return await channel.send_voice(chat_id, tmp_file)
                finally:
                    Path(tmp_file).unlink(missing_ok=True)
        except Exception as e:
            logger.error("Voice send failed: %s", e)

        return False

    async def _handle_busy_message(
        self, text: str, thread_id: str, channel_prefix: str, reply_fn,
    ) -> None:
        mode = self._container.config.agents.busy_input_mode
        flag = self._container.state_store.get_interrupt_flag(thread_id)
        zh = self._container.config.agents.language == "zh"

        if text.strip().startswith("/"):
            await reply_fn("⏳ 当前忙碌，请稍后重试命令。" if zh else "⏳ Busy, please retry the command later.")
            return

        if mode == "queue":
            self._enqueue(thread_id, text)
            await reply_fn("⏳ 已排队，当前任务完成后处理。" if zh else "⏳ Queued for processing after current task.")
            return

        if mode == "steer":
            accepted = flag.steer(text)
            if accepted:
                return
            self._enqueue(thread_id, text)
            await reply_fn("⏳ 消息已排队。" if zh else "⏳ Message queued.")
            return

        if thread_id not in self._interrupted_threads:
            self._interrupted_threads[thread_id] = text
            flag.interrupt(text)
            await reply_fn("⚡ 已中断当前任务，正在处理你的新消息.." if zh else
                           "⚡ Interrupted current task, processing your message...")
        else:
            self._enqueue(thread_id, text)
            await reply_fn("⏳ 消息已排队，将在当前中断任务完成后处理。" if zh else
                           "⏳ Message queued, will process after the current interrupted task.")

    async def _run_agent_turn(
        self,
        input_state,
        thread_id: str,
        session_key: str,
        chat_id: str,
        sender_id: str,
        chat_type: str,
        channel_prefix: str,
        reply_fn,
        stream_fn,
        system_prompt: str,
        original_text: str,
        depth: int = 0,
    ):
        from src.agent.loop import ApprovalPending as AP

        if depth > _MAX_INTERRUPT_DEPTH:
            logger.warning("Interrupt depth %d reached for %s, draining queue", depth, thread_id)
            self._pending_queue.pop(thread_id, None)
            self._interrupted_threads.pop(thread_id, None)
            return

        assistant_text = None
        identity_written = False
        pre_msg_count = len(input_state.messages)
        messages = []

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
                progress_ch = self._get_channel(channel)
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
                    if progress_ch:
                        await progress_ch.send_text(active_chat_id, msg)
                elif event == "tool.exec_failed":
                    tool = kwargs.get("tool_name", "")
                    err = kwargs.get("error", "")[:80]
                    if zh:
                        msg = f"❌ {tool} 失败: {err}"
                    else:
                        msg = f"❌ {tool} failed: {err}"
                    if progress_ch:
                        await progress_ch.send_text(active_chat_id, msg)
            except Exception:
                pass

        from src.events import subscribe_async, unsubscribe
        _progress_unsub = subscribe_async("tool.*", _on_tool_event)

        _assistant_msg_unsub = None

        async def _on_assistant_message(event: str, **kwargs):
            tid = kwargs.get("thread_id", "")
            if tid != thread_id:
                return
            content = kwargs.get("content", "")
            if not content:
                return
            try:
                formatted = _format_display(content)
                await reply_fn(formatted)
            except Exception:
                pass

        _assistant_msg_unsub = subscribe_async("agent_loop.assistant_message", _on_assistant_message)

        try:
            logger.debug("[flow] agent_loop run start, state has %d messages, depth=%d", pre_msg_count, depth)

            try:
                result_state = await self._container.agent_loop.run(input_state, thread_id)
            except AP as exc:
                asyncio.create_task(self._handle_approval_pending(exc, chat_id, channel_prefix))
                return

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

            for msg in reversed(messages[pre_msg_count:]):
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
            if _assistant_msg_unsub:
                unsubscribe("agent_loop.assistant_message", _on_assistant_message)

        if assistant_text:
            if getattr(self._container.config, "link_understanding", None) and self._container.config.link_understanding.enabled:
                try:
                    from src.link_understanding import detect_and_preview_links

                    preview = await detect_and_preview_links(
                        original_text, max_previews=self._container.config.link_understanding.max_previews
                    )
                    if preview:
                        assistant_text += "\n" + preview
                except Exception:
                    pass

            display_text = _format_display(assistant_text)

            try:
                from src.media_delivery import deliver_media
                ch = self._get_channel(channel_prefix)
                if ch:
                    display_text = await deliver_media(display_text, chat_id, channel_prefix, ch)
            except Exception as e:
                logger.warning("Media delivery failed: %s", e)

            # Voice mode: auto-convert short replies to voice
            voice_sent = False
            voice_cfg = getattr(self._container.config, "voice", None)
            if (
                voice_cfg
                and voice_cfg.enabled
                and not display_text.startswith("[error]")
                and "<" not in display_text  # skip if contains media tags
                and len(display_text.strip()) < voice_cfg.threshold
            ):
                try:
                    voice_sent = await self._try_send_voice(
                        display_text, chat_id, channel_prefix, voice_cfg.voice
                    )
                except Exception as e:
                    logger.warning("Voice mode TTS failed: %s", e)

            if not voice_sent:
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
                        f"Q: {original_text}\nA: {display_text}",
                    )
                except Exception:
                    pass

            if getattr(self._container.config, "memory_store", None) and self._container.config.memory_store.enabled:
                try:
                    from src.tools.memory_tools import auto_extract_memory, save_memory
                    extracted = auto_extract_memory(original_text, display_text)
                    if extracted:
                        content, category = extracted
                        await save_memory(content, category=category)
                    elif self._container.config.memory_store.memory_judge_model:
                        task = asyncio.create_task(self._memory_llm_judge(
                            original_text, display_text, reply_fn,
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
                            if tc_name == "memory" and "save" in msg.get("content", ""):
                                await reply_fn("\U0001f4be update memory: 已保存到记忆")
                                break
                except Exception:
                    pass

        await self._drain_pending(
            thread_id=thread_id,
            session_key=session_key,
            chat_id=chat_id,
            sender_id=sender_id,
            chat_type=chat_type,
            channel_prefix=channel_prefix,
            reply_fn=reply_fn,
            stream_fn=stream_fn,
            system_prompt=system_prompt,
            depth=depth,
        )

    async def _drain_pending(
        self,
        thread_id: str,
        session_key: str,
        chat_id: str,
        sender_id: str,
        chat_type: str,
        channel_prefix: str,
        reply_fn,
        stream_fn,
        system_prompt: str,
        depth: int,
    ):
        from src.agent.state import AgentState

        # Priority 1: re-run agent for interrupted message
        interrupt_msg = self._interrupted_threads.pop(thread_id, None)
        if interrupt_msg:
            latest = await self._container.state_store.aload(thread_id)
            if latest:
                await self._run_agent_turn(
                    input_state=latest,
                    thread_id=thread_id,
                    session_key=session_key,
                    chat_id=chat_id,
                    sender_id=sender_id,
                    chat_type=chat_type,
                    channel_prefix=channel_prefix,
                    reply_fn=reply_fn,
                    stream_fn=stream_fn,
                    system_prompt=system_prompt,
                    original_text=interrupt_msg,
                    depth=depth + 1,
                )
            else:
                logger.warning("drain_pending: no state for %s, interrupt message lost", thread_id)
            return

        # Priority 2: dequeue next pending message
        pending = self._dequeue(thread_id)
        if not pending:
            return

        next_state = AgentState(
            messages=[{"role": "user", "content": pending}],
            system_prompt=system_prompt,
            sender_id=sender_id,
            chat_id=chat_id,
            chat_type=chat_type,
            message_id="",
            channel=channel_prefix,
        )
        existing = await self._container.state_store.aload(thread_id)
        if existing:
            next_state.messages = existing.messages + next_state.messages

        await self._run_agent_turn(
            input_state=next_state,
            thread_id=thread_id,
            session_key=session_key,
            chat_id=chat_id,
            sender_id=sender_id,
            chat_type=chat_type,
            channel_prefix=channel_prefix,
            reply_fn=reply_fn,
            stream_fn=stream_fn,
            system_prompt=system_prompt,
            original_text=pending,
            depth=depth + 1,
        )

    async def _handle_approval_pending(
        self,
        exc,
        chat_id: str,
        channel_prefix: str = "qq",
    ):
        from src.tools.approval import get_approval_manager
        from src.agent.loop import ApprovalPending

        thread_id = exc.thread_id
        self._approval_handler_threads.add(thread_id)
        try:
            mgr = get_approval_manager()
            zh = self._container.config.agents.language == "zh"
            current_exc = exc
            consecutive_denies = 0

            while True:
                approval_timeout = getattr(current_exc, "timeout", None) or 120
                auto_deny = getattr(current_exc, "auto_deny", False)
                tool_name = getattr(current_exc, "tool_name", "exec_command")

                approval_ch = self._get_channel(channel_prefix)
                if not approval_ch:
                    return

                if tool_name in ("memory_delete", "memory"):
                    keys = getattr(current_exc, "keys", [])
                    preview = current_exc.command_preview
                    msg_text = (
                        f"🗑️ 以下 {len(keys)} 条记忆将被删除：\n\n"
                        f"{preview}\n\n"
                        f"发送 /y 确认，其它任何消息自动取消（{approval_timeout}秒超时）"
                    )
                else:
                    warn = "DANGEROUS" if current_exc.denylisted else "requires approval"
                    msg_text = (
                        f"**Approval Required** ({warn})\n```\n{current_exc.command_preview}\n```\n"
                        f"Send /y to confirm. Any other message will cancel. (request: {current_exc.request_id})"
                    )

                await approval_ch.send_text(chat_id, msg_text)
                decision, user_response = await mgr.await_approval(current_exc.request_id, timeout=approval_timeout)
                if decision == "timeout":
                    if tool_name in ("memory_delete", "memory") or auto_deny:
                        timeout_msg = "⏰ 操作超时，记忆删除已取消。" if tool_name in ("memory_delete", "memory") else ("操作超时，已自动拒绝。" if zh else "Operation timed out, auto-denied.")
                        await approval_ch.send_text(chat_id, timeout_msg)
                    decision = "deny"

                if decision == "deny":
                    consecutive_denies += 1
                else:
                    consecutive_denies = 0

                try:
                    result_state = await self._container.agent_loop.resume(current_exc.thread_id, decision)
                    assistant_text = ""
                    for msg in reversed(result_state.messages):
                        if msg.get("role") == "assistant" and msg.get("content"):
                            assistant_text = msg["content"]
                            break
                    if assistant_text:
                        await approval_ch.send_text(chat_id, assistant_text)
                    return
                except ApprovalPending as next_exc:
                    if consecutive_denies >= 3:
                        state = await self._container.state_store.aload(current_exc.thread_id)
                        if state:
                            state.pending_approval = None
                            await self._container.state_store.save(current_exc.thread_id, state)
                        await approval_ch.send_text(chat_id, "⚠️ 连续多次被拒绝后仍在重试，已终止操作。")
                        return
                    current_exc = next_exc
                    continue
        except Exception:
            logger.exception("Error in _handle_approval_pending for request %s", exc.request_id)
            try:
                state = await self._container.state_store.aload(exc.thread_id)
                if state and state.pending_approval:
                    state.pending_approval = None
                    await self._container.state_store.save(exc.thread_id, state)
                    logger.info("Cleared orphaned pending_approval for thread %s", exc.thread_id)
            except Exception:
                pass
        finally:
            self._approval_handler_threads.discard(thread_id)

    async def _resume_and_reply(self, thread_id: str, decision: str, chat_id: str):
        """Fallback resume when _handle_approval_pending is not running.

        Called when user sends /y but the original _handle_approval_pending
        task has already exited (e.g. due to an exception or chained ApprovalPending).
        If _handle_approval_pending is still running, resume() will fail with
        a lock conflict and we silently ignore it.
        """
        try:
            state = await self._container.state_store.aload(thread_id)
            if not state or not state.pending_approval:
                logger.debug("_resume_and_reply: no pending_approval for thread %s, skipping", thread_id)
                return
            result_state = await self._container.agent_loop.resume(thread_id, decision)
            assistant_text = ""
            for msg in reversed(result_state.messages):
                if msg.get("role") == "assistant" and msg.get("content"):
                    assistant_text = msg["content"]
                    break
            ch = self._get_channel_for_chat_id(chat_id)
            if assistant_text and ch:
                await ch.send_text(chat_id, assistant_text)
        except ApprovalPending as exc:
            _rp = "weixin" if not chat_id.startswith(("c2c:", "group:", "channel:", "dm:")) else "qq"
            asyncio.create_task(self._handle_approval_pending(exc, chat_id, _rp))
        except RuntimeError as e:
            if "busy" in str(e).lower() or "lock" in str(e).lower():
                logger.debug("_resume_and_reply: thread %s is busy, _handle_approval_pending is still running", thread_id)
            else:
                logger.warning("_resume_and_reply failed: %s", e)
        except Exception:
            logger.exception("_resume_and_reply failed for thread %s", thread_id)

    async def _memory_llm_judge(self, user_input: str, ai_response: str, reply_fn):
        try:
            from src.tools.memory_tools import judge_memory_with_llm, save_memory

            model_name = self._container.config.memory_store.memory_judge_model
            base_url = self._container.config.memory_store.memory_judge_base_url or self._container.config.model.base_url
            api_key = self._container.config.memory_store.memory_judge_api_key or self._container.config.model.api_key

            if not model_name or not base_url or not api_key:
                return

            result = await judge_memory_with_llm(
                user_input, ai_response, model_name, base_url, api_key,
            )
            if result:
                content, category = result
                await save_memory(content, category=category)
                await reply_fn(f"\U0001f4be update memory: {content[:50]}")
        except Exception:
            logger.debug("Memory LLM judge failed", exc_info=True)
