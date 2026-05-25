"""Self-contained agent loop.

The loop is a simple while-cycle:
  call model → check tool_calls → execute tools → append results → loop
Terminates when the model responds without tool calls (or max rounds hit).
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from typing import Any

from src.agent.client import ChatClient, ChatResponse, FallbackChain
from src.agent.guardrails import ToolLoopGuardrails
from src.agent.state import AgentState, StateStore
from src.agent.tooldef import ToolDef
from src.skills.manager import _SKILL_TOOL_NAMES

logger = logging.getLogger("myclaw.agent.loop")

_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")

_CHARS_PER_TOKEN = 4


def _escape_control_in_json_strings(s: str) -> str:
    out: list[str] = []
    in_str = False
    esc = False
    for ch in s:
        if esc:
            out.append(ch)
            esc = False
            continue
        if ch == "\\":
            out.append(ch)
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            out.append(ch)
            continue
        if in_str and ord(ch) < 0x20:
            out.append(f"\\u{ord(ch):04x}")
            continue
        out.append(ch)
    return "".join(out)


def _repair_tool_args(args_str: str) -> str:
    if not args_str:
        return "{}"

    try:
        json.loads(args_str, strict=False)
        return args_str
    except (json.JSONDecodeError, TypeError):
        pass

    s = args_str.strip()

    for i in range(min(len(s), len(s) - 1)):
        try:
            json.loads(s[:len(s) - i], strict=False)
            return s[:len(s) - i]
        except (json.JSONDecodeError, TypeError):
            continue

    s = re.sub(r",\s*([}\]])", r"\1", s)

    opens_brace = s.count("{") - s.count("}")
    opens_bracket = s.count("[") - s.count("]")
    if opens_brace > 0:
        s += "}" * opens_brace
    if opens_bracket > 0:
        s += "]" * opens_bracket
    try:
        json.loads(s, strict=False)
        return s
    except (json.JSONDecodeError, TypeError):
        pass

    for _ in range(min(len(s), 50)):
        if s.endswith("}") or s.endswith("]"):
            trimmed = s[:-1]
            try:
                json.loads(trimmed, strict=False)
                return trimmed
            except (json.JSONDecodeError, TypeError):
                s = trimmed
        else:
            break

    s = _escape_control_in_json_strings(args_str)
    try:
        json.loads(s, strict=False)
        return s
    except (json.JSONDecodeError, TypeError):
        pass

    return "{}"

_PARALLEL_SAFE_TOOLS = frozenset({
    "read_file", "list_dir", "search_files",
    "web_search", "web_fetch", "session_search",
    "describe_media",
    "memory", "cronjob", "task_manage", "skills_list", "skill_view", "skill_manage", "skill_hub",
})

_PATH_SCOPED_TOOLS = frozenset({
    "read_file", "write_file", "edit_file", "search_files",
})


def _estimate_tokens_simple(messages: list[dict]) -> int:
    """Fast token estimate without per-message overhead."""
    total = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            total += len(content) // _CHARS_PER_TOKEN
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    total += len(part.get("text", "")) // _CHARS_PER_TOKEN
        total += 10
    return total


class ApprovalPending(Exception):
    """Raised when a tool needs user approval. Loop pauses, caller resumes."""

    def __init__(self, thread_id: str, request_id: str, tool_name: str, command_preview: str, denylisted: bool = False, timeout: int | None = None, auto_deny: bool = False, keys: list[str] | None = None):
        self.thread_id = thread_id
        self.request_id = request_id
        self.tool_name = tool_name
        self.command_preview = command_preview
        self.denylisted = denylisted
        self.timeout = timeout
        self.auto_deny = auto_deny
        self.keys = keys or []
        super().__init__(f"Approval needed: {tool_name} — {command_preview[:80]}")


class AgentLoop:
    def __init__(
        self,
        client: ChatClient | FallbackChain,
        tools: list[ToolDef],
        state_store: StateStore,
        config: Any = None,
        skills_prompt: str = "",
        context_window_tokens: int = 100000,
    ):
        self._client = client
        self._tools = tools
        self._store = state_store
        self._config = config
        self._skills_prompt = skills_prompt
        self._ctx_window_tokens = context_window_tokens
        self._context_files: list[dict] = []

        # Build tool name → ToolDef lookup
        self._tool_map: dict[str, ToolDef] = {t.name: t for t in tools}

        # Compressor - always create, even without config
        from src.compressor.compressor import ContextCompressor
        from src.config import CompressionConfig
        compression_config = config.compression if config else CompressionConfig()
        self._compressor = ContextCompressor(compression_config, main_config=config, client=client)

        if config:
            from src.bootstrap import load_bootstrap_files
            agents_cfg = getattr(config, "agents", None)
            if agents_cfg:
                extra = getattr(agents_cfg, "bootstrap_files", None)
                self._context_files = load_bootstrap_files(
                    agents_cfg.workspace, extra_names=extra
                )

        self._cache_prompt_sections(tools, skills_prompt)

        self._memory_summary_cache: str = ""
        self._memory_summary_ts: float = 0

        # Cache: avoids re-building identical messages across tool rounds
        self._cached_history: list[dict] | None = None
        self._cached_history_count: int = 0

        self._guardrails = ToolLoopGuardrails()

        self._iters_since_skill = 0
        self._skill_nudge_interval = 0
        if config and hasattr(config, "skills"):
            raw = getattr(config.skills, "creation_nudge_interval", 0)
            if isinstance(raw, int):
                self._skill_nudge_interval = raw

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, state: AgentState, thread_id: str, max_rounds: int = 50) -> AgentState:
        """Execute agent loop with per-thread locking."""
        from src.events import emit_async

        await emit_async(
            "agent_loop.started",
            thread_id=thread_id,
            max_rounds=max_rounds,
            message_count=len(state.messages),
            channel=getattr(state, 'channel', ''),
            sender_id=getattr(state, 'sender_id', ''),
        )

        lock = await self._store.acquire_thread(thread_id)
        timeout = (
            getattr(self._config.agents, "lock_timeout", 30.0)
            if self._config and getattr(self._config, "agents", None)
            else 30.0
        )
        try:
            await asyncio.wait_for(lock.acquire(), timeout=timeout)
        except asyncio.TimeoutError:
            raise RuntimeError(f"Thread {thread_id} is busy (lock timeout {timeout}s)")
        try:
            return await self._run_inner(state, thread_id, max_rounds)
        finally:
            lock.release()

    async def resume(self, thread_id: str, decision: str) -> AgentState:
        """Resume after approval with per-thread locking."""
        lock = await self._store.acquire_thread(thread_id)
        timeout = (
            getattr(self._config.agents, "lock_timeout", 30.0)
            if self._config and getattr(self._config, "agents", None)
            else 30.0
        )
        try:
            await asyncio.wait_for(lock.acquire(), timeout=timeout)
        except asyncio.TimeoutError:
            raise RuntimeError(f"Thread {thread_id} is busy (lock timeout {timeout}s)")
        try:
            return await self._resume_inner(thread_id, decision)
        finally:
            lock.release()

    def get_store(self) -> StateStore:
        return self._store

    def is_thread_busy(self, thread_id: str) -> bool:
        lock = self._store._locks.get(thread_id)
        return lock is not None and lock.locked()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _check_interrupt(self, state: AgentState, thread_id: str, start_ts: float, tool_round: int) -> bool:
        """Check interrupt flag. If interrupted, append message and return True."""
        flag = self._store.get_interrupt_flag(thread_id)
        is_interrupted, interrupt_msg = flag.check()
        if not is_interrupted:
            return False
        flag.clear()
        if interrupt_msg:
            state.append_message({"role": "user", "content": interrupt_msg})
        import time as _time
        from src.events import emit_async
        duration_ms = (_time.monotonic() - start_ts) * 1000
        await emit_async(
            "agent_loop.interrupted",
            thread_id=thread_id,
            interrupt_message=interrupt_msg,
            duration_ms=duration_ms,
            total_rounds=tool_round,
        )
        await self._store.save(thread_id, state)
        self._maybe_trigger_skill_review(state, thread_id)
        return True

    async def _run_inner(self, state: AgentState, thread_id: str, max_rounds: int) -> AgentState:
        """Internal run logic, called with thread lock held.

        Uses proactive compression (hermes-style): check token budget BEFORE
        each model call and compress if over threshold.
        """
        import time as _time
        from src.events import emit_async

        start_ts = _time.monotonic()
        max_tool_rounds = self._get_max_tool_rounds()
        tool_round = 0

        self._guardrails.reset()

        for _ in range(max_rounds):
            # 0. Check interrupt
            if await self._check_interrupt(state, thread_id, start_ts, tool_round):
                return state

            # 1. Prepare messages — proactive compression if over budget
            messages = await self._prepare_messages(state, thread_id)

            # 2. Build tool list
            active_tools = self._filter_tools(state)
            openai_tools = [t.to_openai_tool() for t in active_tools] if active_tools else None

            # 3. Call model (with retry)
            response = await self._call_model_with_retry(messages, tools=openai_tools)

            # 4. Append assistant message
            assistant_msg = self._build_assistant_msg(response)
            state.append_message(assistant_msg)

            # Emit event for intermediate assistant messages (has tool_calls + content)
            if response.tool_calls and assistant_msg.get("content"):
                await emit_async(
                    "agent_loop.assistant_message",
                    thread_id=thread_id,
                    content=assistant_msg["content"],
                    has_tool_calls=True,
                    message_count=len(state.messages),
                )

            # 5. No tool calls → done
            if not response.tool_calls:
                self._redact_last_assistant(state)
                await self._store.save(thread_id, state)
                try:
                    from src.tools.approval import get_approval_manager
                    get_approval_manager().clear_session(thread_id)
                except Exception:
                    pass
                self._store.get_interrupt_flag(thread_id).clear()
                duration_ms = (_time.monotonic() - start_ts) * 1000
                await emit_async(
                    "agent_loop.completed",
                    thread_id=thread_id,
                    final_message_count=len(state.messages),
                    duration_ms=duration_ms,
                    total_rounds=tool_round,
                )
                self._maybe_trigger_skill_review(state, thread_id)
                return state

            # Checkpoint: save assistant message (with tool_calls) before execution
            await self._store.save(thread_id, state)

            # 6. Execute tool calls (parallel when safe)
            tool_round += 1
            if self._should_parallelize(response.tool_calls):
                parallel_results = await self._execute_tools_parallel(response.tool_calls, state, thread_id)
                for tc_id, result in parallel_results:
                    state.append_message({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": result,
                    })
                await self._store.save(thread_id, state)
                if await self._check_interrupt(state, thread_id, start_ts, tool_round):
                    return state
            else:
                for tc in response.tool_calls:
                    if await self._check_interrupt(state, thread_id, start_ts, tool_round):
                        return state
                    tool_result = await self._execute_tool(tc, state, thread_id)
                    state.append_message({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result,
                    })
                    await self._store.save(thread_id, state)

            # Nudge counter: increment after every tool round
            if self._skill_nudge_interval > 0 and "skill_manage" in self._tool_map:
                self._iters_since_skill += 1

            # 7. Inject pending steer text into last tool result
            steer_text = self._store.get_interrupt_flag(thread_id).drain_steer()
            if steer_text:
                for i in range(len(state.messages) - 1, -1, -1):
                    if state.messages[i].get("role") == "tool":
                        state.messages[i]["content"] += f"\n\nUser guidance: {steer_text}"
                        logger.info("Steer injected after tool batch (%d chars): %s", len(steer_text), steer_text[:120])
                        break
                await self._store.save(thread_id, state)

        duration_ms = (_time.monotonic() - start_ts) * 1000
        await emit_async(
            "agent_loop.completed",
            thread_id=thread_id,
            final_message_count=len(state.messages),
            duration_ms=duration_ms,
            total_rounds=tool_round,
        )

        # Grace period: budget exhausted, let the model summarize
        state.append_message({
            "role": "user",
            "content": "工具调用预算已用完。请总结当前进度和结果，不要再调用工具。",
        })
        try:
            summary_resp = await self._call_model_with_retry(state.messages, tools=None)
            state.append_message({
                "role": "assistant",
                "content": summary_resp.content,
            })
            await self._store.save(thread_id, state)
        except Exception as e:
            logger.warning("Grace period summary failed: %s", e)

        self._maybe_trigger_skill_review(state, thread_id)
        return state

    async def _resume_inner(self, thread_id: str, decision: str) -> AgentState:
        """Internal resume logic, called with thread lock held."""
        state = await self._store.aload(thread_id)
        if state is None:
            raise RuntimeError(f"No saved state for thread {thread_id}")

        pending = state.pending_approval or {}
        pending_tc_id = pending.get("tool_call_id", "")

        # Find the assistant message that contains the pending tool_call
        # and collect ALL tool_call ids from it
        assistant_msg_idx = None
        all_tc_ids: list[str] = []
        for i in range(len(state.messages) - 1, -1, -1):
            msg = state.messages[i]
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                assistant_msg_idx = i
                for tc in msg["tool_calls"]:
                    tc_id = tc.get("id", "")
                    if tc_id:
                        all_tc_ids.append(tc_id)
                break

        # Determine which tool_calls already have results
        existing_results: set[str] = set()
        if assistant_msg_idx is not None:
            for msg in state.messages[assistant_msg_idx + 1:]:
                if msg.get("role") == "tool":
                    existing_results.add(msg.get("tool_call_id", ""))

        # Handle the pending tool call
        if decision == "allow_once":
            pending_tool = pending.get("tool_name", "exec_command")
            command_preview = pending.get("command_preview", "")
            memory_keys = pending.get("memory_keys")

            from src.tools.approval import get_approval_manager
            from src.tools.exec import _current_thread_id
            try:
                _approval_mgr = get_approval_manager()
                _approval_mgr.approve_session(thread_id, pending_tool, command_preview)
            except Exception:
                pass
            _tid_token = _current_thread_id.set(thread_id)

            try:
                if pending_tool in ("memory_delete", "memory") and memory_keys:
                    deleted = []
                    try:
                        from src.tools.memory_tools import get_memory_store
                        mem_store = get_memory_store()
                        for k in memory_keys:
                            await mem_store.forget(k)
                            deleted.append(k)
                    except Exception as exc:
                        import logging as _log
                        _log.getLogger("myclaw.loop").warning("memory delete resume failed: %s", exc)
                    result_content = json.dumps(
                        {"ok": True, "deleted": deleted, "count": len(deleted)},
                        ensure_ascii=False,
                    )
                    state.append_message({
                        "role": "tool",
                        "tool_call_id": pending_tc_id,
                        "content": result_content,
                    })
                    existing_results.add(pending_tc_id)
                elif pending_tc_id and assistant_msg_idx is not None:
                    assistant_msg = state.messages[assistant_msg_idx]
                    for tc in assistant_msg["tool_calls"]:
                        if tc.get("id") == pending_tc_id:
                            result = await self._execute_tool(tc, state, thread_id)
                            state.append_message({
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "content": result,
                            })
                            break
                    existing_results.add(pending_tc_id)

                    # Execute any other tool_calls from the same message that lack results
                    for tc in assistant_msg["tool_calls"]:
                        tc_id = tc.get("id", "")
                        if tc_id and tc_id not in existing_results:
                            try:
                                result = await self._execute_tool(tc, state, thread_id)
                            except ApprovalPending:
                                for t in assistant_msg["tool_calls"]:
                                    tid = t.get("id", "")
                                    if tid and tid not in existing_results and tid != tc_id:
                                        state.append_message({
                                            "role": "tool",
                                            "tool_call_id": tid,
                                            "content": "[skipped] Execution paused due to pending approval.",
                                        })
                                        existing_results.add(tid)
                                await self._store.save(thread_id, state)
                                raise
                            state.append_message({
                                "role": "tool",
                                "tool_call_id": tc_id,
                                "content": result,
                            })
                            existing_results.add(tc_id)
            finally:
                _current_thread_id.reset(_tid_token)
        else:
            if pending_tc_id:
                pending_tool = pending.get("tool_name", "exec_command")
                deny_msg = "[denied] 记忆删除已取消。" if pending_tool in ("memory_delete", "memory") else "[denied] Command execution was denied by user."
                state.append_message({
                    "role": "tool",
                    "tool_call_id": pending_tc_id,
                    "content": deny_msg,
                })
                existing_results.add(pending_tc_id)

            # Deny remaining unexecuted tool calls too
            if assistant_msg_idx is not None:
                assistant_msg = state.messages[assistant_msg_idx]
                for tc in assistant_msg["tool_calls"]:
                    tc_id = tc.get("id", "")
                    if tc_id and tc_id not in existing_results:
                        state.append_message({
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": "[denied] Skipped due to associated denial.",
                        })
                        existing_results.add(tc_id)

        state.pending_approval = None
        await self._store.save(thread_id, state)

        return await self._run_inner(state, thread_id, max_rounds=self._get_max_tool_rounds())

    def _get_max_tool_rounds(self) -> int:
        if not self._config:
            return 50
        agents_cfg = getattr(self._config, "agents", None)
        if agents_cfg is None:
            return 50
        return getattr(agents_cfg, "max_tool_rounds", 0) or 50

    async def _call_model_with_retry(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_retries: int = 3,
    ) -> ChatResponse:
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                return await self._client.chat(messages, tools=tools)
            except Exception as e:
                last_exc = e
                if attempt >= max_retries:
                    break
                err_str = str(e).lower()
                status = getattr(getattr(e, "status_code", None), "value", getattr(e, "status_code", None))
                if status in (401, 403) or "auth" in err_str:
                    raise
                if "context" in err_str and ("length" in err_str or "token" in err_str or "overflow" in err_str):
                    messages = await self._compressor.compress(messages, self._ctx_window_tokens)
                    continue
                if status == 400 and "context" not in err_str:
                    raise
                base = 5.0
                if status == 429 or "rate" in err_str:
                    base = 10.0
                elif status in (503, 529) or "overload" in err_str:
                    base = 15.0
                delay = min(base * (2 ** attempt) + random.uniform(0, base), 60.0)
                logger.warning(
                    "Model API error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, max_retries, delay, e,
                )
                await asyncio.sleep(delay)
        raise last_exc

    def _maybe_trigger_skill_review(self, state: AgentState, thread_id: str) -> None:
        """Fire-and-forget background skill review if nudge threshold met."""
        if self._skill_nudge_interval <= 0:
            return
        if self._iters_since_skill < self._skill_nudge_interval:
            return
        if "skill_manage" not in self._tool_map:
            return

        self._iters_since_skill = 0
        messages_snapshot = list(state.messages)

        async def _bg_review():
            from src.skills.review import spawn_background_review
            summary = await spawn_background_review(
                client=self._client,
                tools=self._tools,
                config=self._config,
                messages_snapshot=messages_snapshot,
                review_skills=True,
                review_memory=False,
            )
            if summary:
                logger.info("💾 Self-improvement review: %s", summary)

        try:
            task = asyncio.create_task(_bg_review())
            logger.debug("Background skill review task spawned after %d iterations", self._skill_nudge_interval)
        except RuntimeError:
            logger.debug("No event loop, skipping background skill review")

    # ------------------------------------------------------------------
    # Message preparation — two modes
    # ------------------------------------------------------------------

    async def _prepare_messages(self, state: AgentState, thread_id: str = "") -> list[dict]:
        """Prepare messages for model call with proactive compression.

        hermes-style: check token budget BEFORE calling the model.
        If over threshold, compress first; otherwise send raw history.
        Large tool outputs are always truncated to keep payload small.
        """
        self._truncate_large_outputs(state.messages, thread_id)

        # Clean orphan tool_call/result pairs BEFORE compression
        history = self._sanitize_api_messages(state.messages)

        # Proactive compression check
        if self._compressor.should_compress(history, self._ctx_window_tokens):
            history = await self._compressor.compress(history, self._ctx_window_tokens)
            try:
                from src.tools.file_tools import reset_read_dedup
                reset_read_dedup(thread_id)
            except Exception:
                pass

        memory_summary = await self._fetch_memory_summary()
        system_text = self._build_system_prompt(state, self._get_active_tool_defs(state), memory_summary)
        history = self._sanitize_api_messages(history)
        history = self._sanitize_surrogates(history)
        return [{"role": "system", "content": system_text}] + list(history)

    def _truncate_large_outputs(self, messages: list[dict], thread_id: str = "") -> None:
        """Truncate large tool outputs in-place, caching full content to temp files.

        Modifies ``state.messages`` directly so that the ``_truncated`` flag
        and truncated content persist across loop iterations.
        """
        from src.agent.tool_cache import cache_large_output

        for m in messages:
            if m.get("role") == "tool" and not m.get("_truncated"):
                content = m.get("content", "")
                if isinstance(content, str) and len(content) > 8000:
                    truncated, _ = cache_large_output(content, thread_id)
                    m["content"] = truncated
                    m["_truncated"] = True


    # ------------------------------------------------------------------
    # Sanitization helpers
    # ------------------------------------------------------------------

    def _sanitize_api_messages(self, messages: list[dict]) -> list[dict]:
        """Fix orphan tool_call/result pairs after compression.

        1. If a tool_call has no matching tool result → insert stub result
        2. If a tool result has no matching tool_call → remove it
        """
        tc_ids: list[str] = []
        for m in messages:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    tid = tc.get("id", "")
                    if tid:
                        tc_ids.append(tid)

        result_ids: set[str] = set()
        result_indices: list[int] = []
        for i, m in enumerate(messages):
            if m.get("role") == "tool":
                tid = m.get("tool_call_id", "")
                result_ids.add(tid)
                result_indices.append(i)

        sanitized = [{k: v for k, v in m.items() if k != "_truncated"} for m in messages]

        for tid in tc_ids:
            if tid not in result_ids:
                # 正常情况下不应该走到这里，因为压缩时已确保tool result不丢失
                logger.warning("Missing tool result for tool_call_id=%s", tid)
                stub = {"role": "tool", "tool_call_id": tid, "content": "[tool result unavailable]"}
                sanitized.append(stub)

        for i in reversed(result_indices):
            tid = messages[i].get("tool_call_id", "")
            if tid not in tc_ids:
                sanitized.pop(i)

        return sanitized

    def _sanitize_surrogates(self, messages: list[dict]) -> list[dict]:
        """Remove lone surrogate characters (U+D800-U+DFFF) that crash json.dumps.

        Ollama and some local models emit lone surrogates.
        """
        def _clean(obj: Any) -> Any:
            if isinstance(obj, str):
                return _SURROGATE_RE.sub("\ufffd", obj)
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_clean(v) for v in obj]
            return obj

        return _clean(messages)

    # ------------------------------------------------------------------
    # Tool handling
    # ------------------------------------------------------------------

    def _get_active_tool_defs(self, state: AgentState) -> list[ToolDef]:
        return self._filter_tools(state)

    def _filter_tools(self, state: AgentState) -> list[ToolDef]:
        tools = self._tools

        if self._config:
            from src.tools.policy import apply_tool_policy
            sender_id = state.sender_id
            user = self._resolve_user(sender_id)
            tools = apply_tool_policy(tools, sender_id, self._config, user=user)

        channel = state.channel
        if channel == "qq":
            tools = [t for t in tools if not t.name.startswith("weixin_")]
        elif channel == "weixin":
            tools = [t for t in tools if not t.name.startswith("qq_")]

        max_rounds = self._get_max_tool_rounds()
        if max_rounds > 0:
            turn_msgs = state.messages
            for i in range(len(state.messages) - 1, -1, -1):
                if state.messages[i].get("role") == "user":
                    turn_msgs = state.messages[i:]
                    break
            tool_count = sum(1 for m in turn_msgs if m.get("role") == "tool")
            if tool_count >= max_rounds:
                logger.info("Max tool rounds (%d) reached, disabling tools", max_rounds)
                return []

        return tools

    def _resolve_user(self, sender_id: str):
        if not self._config or not getattr(self._config, "auth", None) or not self._config.auth.enabled:
            return None
        try:
            from src.auth.rbac import get_rbac
            rbac = get_rbac()
            if rbac:
                return rbac.resolve_user(sender_id)
        except Exception as exc:
            logger.debug("resolve_user failed: %s", exc)
        return None

    def _cache_prompt_sections(self, tools: list[ToolDef], skills_prompt: str) -> None:
        from pathlib import Path
        from src.prompt import (
            _load_soul_md, _build_environment_hints, _build_tooling_rules,
            _build_tool_guidance, _build_safety, _build_skills_section,
            _build_workspace, _build_bootstrap_context,
        )

        workspace_dir = "."
        if self._config:
            agents_cfg = getattr(self._config, "agents", None)
            if agents_cfg:
                raw_ws = getattr(agents_cfg, "workspace", ".") or "."
                workspace_dir = str(Path(raw_ws).expanduser().resolve())

        self._prompt_soul = _load_soul_md()
        self._prompt_env = "\n".join(_build_environment_hints())
        self._prompt_tool_rules = "\n".join(_build_tooling_rules())
        self._prompt_tool_guidance = "\n".join(_build_tool_guidance(tools))
        self._prompt_safety = "\n".join(_build_safety())
        self._prompt_skills = "\n".join(_build_skills_section(skills_prompt)) if skills_prompt else ""
        self._prompt_workspace = "\n".join(_build_workspace(workspace_dir))
        self._prompt_bootstrap = "\n".join(_build_bootstrap_context(self._context_files)) if self._context_files else ""
        self._prompt_platform_cache: dict[str, str] = {}

        logger.info(
            "System prompt sections cached (soul=%d, env=%d, guidance=%d, skills=%d, bootstrap=%d chars)",
            len(self._prompt_soul), len(self._prompt_env),
            len(self._prompt_tool_guidance), len(self._prompt_skills),
            len(self._prompt_bootstrap),
        )

    async def _fetch_memory_summary(self) -> str:
        import time
        now = time.monotonic()
        if self._memory_summary_cache and now - self._memory_summary_ts < 300:
            return self._memory_summary_cache
        try:
            from src.tools.memory_tools import get_memory_store
            store = get_memory_store()
            items = await store.list_all(limit=20)
            if not items:
                self._memory_summary_ts = now
                return ""
            lines = [f"## 已知记忆（{len(items)} 条）"]
            for item in items:
                cat = item.get("category", "fact")
                content = item.get("content", "")[:80]
                lines.append(f"- [{cat}] {content}")
            lines.append('以上是已加载的主要部分的记忆，直接基于这些信息回答即可，除非用户表示不足或要求更多回忆，否则无需再次搜索。如果需要修改或补充，使用 memory 工具。')
            result = "\n".join(lines)
            self._memory_summary_cache = result
            self._memory_summary_ts = now
            return result
        except Exception:
            return self._memory_summary_cache or ""

    def _build_system_prompt(self, state: AgentState, active_tools: list[ToolDef], memory_summary: str = "") -> str:
        from src.prompt import _build_platform_hints

        parts = [self._prompt_soul, ""]

        extra = state.system_prompt.strip()
        if extra:
            parts.extend([extra, ""])

        if memory_summary:
            parts.extend([memory_summary, ""])

        parts.append(self._prompt_env)

        channel = state.channel
        if channel not in self._prompt_platform_cache:
            self._prompt_platform_cache[channel] = "\n".join(_build_platform_hints(channel))
        platform = self._prompt_platform_cache[channel]
        if platform:
            parts.append(platform)

        parts.append(self._prompt_tool_rules)
        if self._prompt_tool_guidance:
            parts.append(self._prompt_tool_guidance)
        parts.append(self._prompt_safety)
        if self._prompt_skills:
            parts.append(self._prompt_skills)
        parts.append(self._prompt_workspace)
        if self._prompt_bootstrap:
            parts.append(self._prompt_bootstrap)

        return "\n".join(parts)

    def _build_assistant_msg(self, response: ChatResponse) -> dict:
        msg: dict[str, Any] = {"role": "assistant", "content": response.content}
        if response.tool_calls:
            fixed_calls = []
            for tc in response.tool_calls:
                args_str = tc.function.arguments
                repaired = _repair_tool_args(args_str)
                if repaired != args_str:
                    logger.warning("Repaired arguments for %s", tc.function.name)
                fixed_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": repaired,
                    },
                })
            msg["tool_calls"] = fixed_calls
        return msg

    def _redact_last_assistant(self, state: AgentState) -> None:
        if not state.messages:
            return
        last = state.messages[-1]
        if last.get("role") == "assistant" and isinstance(last.get("content"), str):
            try:
                from src.security.redact import redact
                last["content"] = redact(last["content"])
            except Exception as exc:
                logger.debug("redact failed: %s", exc)

    async def _execute_tool(self, tc: Any, state: AgentState, thread_id: str) -> str:
        """Execute a single tool call. Returns result string."""
        from src.events import emit_async

        if isinstance(tc, dict):
            tool_name = tc.get("function", {}).get("name", "")
        else:
            tool_name = tc.function.name
        try:
            if isinstance(tc, dict):
                args_str = tc.get("function", {}).get("arguments", "")
            else:
                args_str = tc.function.arguments
            args = json.loads(args_str) if args_str else {}
        except json.JSONDecodeError:
            args = {}

        tool_def = self._tool_map.get(tool_name)
        if tool_def is None:
            return f"[error] Unknown tool: {tool_name}"

        guard = self._guardrails.check(tool_name, args)
        if guard is not None:
            return guard.synthetic_result

        import time as _time
        start = _time.monotonic()
        args_preview = json.dumps(args, ensure_ascii=False)[:200] if args else ""

        await emit_async(
            "tool.exec_started",
            thread_id=thread_id,
            tool_name=tool_name,
            tool_call_id=getattr(tc, 'id', '') if not isinstance(tc, dict) else tc.get("id", ""),
            args_preview=args_preview,
        )

        try:
            result = await tool_def.execute(args)
            duration_ms = (_time.monotonic() - start) * 1000
            try:
                from src.security.redact import redact
                result = redact(result)
            except Exception:
                pass
            self._guardrails.record(tool_name, args, success=True, result=result if isinstance(result, str) else "")

            if tool_name in _SKILL_TOOL_NAMES and self._skill_nudge_interval > 0:
                self._iters_since_skill = 0

            await emit_async(
                "tool.exec_completed",
                thread_id=thread_id,
                tool_name=tool_name,
                success=True,
                duration_ms=duration_ms,
                result_length=len(result) if isinstance(result, str) else 0,
                args_preview=args_preview,
                sender_id=getattr(state, 'sender_id', ''),
                channel=getattr(state, 'channel', ''),
            )
            return result
        except Exception as e:
            duration_ms = (_time.monotonic() - start) * 1000
            from src.tools.exec import ApprovalNeededError
            from src.tools.memory_tools import MemoryDeleteNeedsApproval
            if isinstance(e, (ApprovalNeededError, MemoryDeleteNeedsApproval)):
                await emit_async(
                    "tool.approval_pending",
                    thread_id=thread_id,
                    tool_name=getattr(e, "tool_name", tool_name),
                    command_preview=getattr(e, "command_preview", getattr(e, "command", "")[:200]),
                    denylisted=getattr(e, "denylisted", False),
                )
                return await self._handle_approval(e, tc, state, thread_id)
            logger.error("Tool %s failed: %s", tool_name, e)
            self._guardrails.record(tool_name, args, success=False)

            await emit_async(
                "tool.exec_failed",
                thread_id=thread_id,
                tool_name=tool_name,
                success=False,
                duration_ms=duration_ms,
                error=str(e)[:500],
                error_type=type(e).__name__,
                args_preview=args_preview,
                sender_id=getattr(state, 'sender_id', ''),
                channel=getattr(state, 'channel', ''),
            )
            return f"[error] {type(e).__name__}: {e}"

    def _should_parallelize(self, tool_calls: list[Any]) -> bool:
        if len(tool_calls) <= 1:
            return False
        for tc in tool_calls:
            name = tc.function.name if hasattr(tc, "function") else tc.get("function", {}).get("name", "")
            if name not in _PARALLEL_SAFE_TOOLS:
                return False
        paths: list[str] = []
        for tc in tool_calls:
            name = tc.function.name if hasattr(tc, "function") else tc.get("function", {}).get("name", "")
            if name not in _PATH_SCOPED_TOOLS:
                continue
            try:
                args_str = tc.function.arguments if hasattr(tc, "function") else tc.get("function", {}).get("arguments", "")
                args = json.loads(args_str) if args_str else {}
                p = args.get("path", args.get("file_path", args.get("directory", "")))
                if p:
                    paths.append(str(p).lower().rstrip("/\\"))
            except (json.JSONDecodeError, TypeError):
                return False
        if paths and len(paths) > 1:
            seen: set[str] = set()
            for p in paths:
                for s in seen:
                    if p.startswith(s) or s.startswith(p):
                        return False
                seen.add(p)
        return True

    async def _execute_tools_parallel(self, tool_calls: list[Any], state: AgentState, thread_id: str) -> list[tuple[str, str]]:
        results: list[tuple[str, str] | BaseException] = [NotImplemented] * len(tool_calls)

        async def _run_one(idx: int, tc: Any) -> None:
            tc_id = tc.id if hasattr(tc, "id") else tc.get("id", "")
            try:
                result = await self._execute_tool(tc, state, thread_id)
                results[idx] = (tc_id, result)
            except BaseException as exc:
                results[idx] = exc

        tasks = [asyncio.create_task(_run_one(i, tc)) for i, tc in enumerate(tool_calls)]
        await asyncio.gather(*tasks, return_exceptions=True)

        final: list[tuple[str, str]] = []
        for i, r in enumerate(results):
            tc_id = tool_calls[i].id if hasattr(tool_calls[i], "id") else tool_calls[i].get("id", "")
            if isinstance(r, ApprovalPending):
                for j in range(i + 1, len(results)):
                    if results[j] is NotImplemented:
                        tc_j = tool_calls[j]
                        tc_j_id = tc_j.id if hasattr(tc_j, "id") else tc_j.get("id", "")
                        final.append((tc_j_id, "[skipped] Execution paused due to pending approval."))
                    elif not isinstance(results[j], BaseException):
                        final.append(results[j])
                raise r
            if isinstance(r, BaseException):
                final.append((tc_id, f"[error] {type(r).__name__}: {r}"))
            else:
                final.append(r)
        return final

    async def _handle_approval(self, error: Exception, tc: Any, state: AgentState, thread_id: str) -> str:
        """Handle approval-needed error by saving state and raising ApprovalPending."""
        from src.tools.approval import get_approval_manager
        mgr = get_approval_manager()
        cmd = getattr(error, "command_preview", "") or getattr(error, "command", "")
        denylisted = getattr(error, "denylisted", False)
        tool_name = getattr(error, "tool_name", "exec_command")

        if isinstance(tc, dict):
            tc_id = tc.get("id", "")
        else:
            tc_id = tc.id

        req = mgr.request_approval(
            tool_name=tool_name,
            args=cmd,
            sender_id=state.sender_id,
            chat_id=state.chat_id,
            message_id=state.message_id,
            thread_id=thread_id,
        )

        pending_data = {
            "request_id": req.id,
            "tool_name": tool_name,
            "command_preview": cmd[:200],
            "tool_call_id": tc_id,
        }
        memory_keys = getattr(error, "keys", None)
        if memory_keys:
            pending_data["memory_keys"] = memory_keys

        state.pending_approval = pending_data
        await self._store.save(thread_id, state)

        raise ApprovalPending(
            thread_id=thread_id,
            request_id=req.id,
            tool_name=tool_name,
            command_preview=cmd[:200],
            denylisted=denylisted,
            timeout=getattr(error, "timeout", None),
            auto_deny=getattr(error, "auto_deny", False),
            keys=getattr(error, "keys", None),
        )
