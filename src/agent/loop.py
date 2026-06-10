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
from typing import Any, ClassVar

from src.agent.client import ChatClient, ChatResponse, FallbackChain
from src.agent.guardrails import ToolLoopGuardrails
from src.agent.state import AgentState, StateStore
from src.agent.tooldef import ToolDef
from src.skills.manager import _SKILL_TOOL_NAMES

logger = logging.getLogger("flyclaw.agent.loop")

_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


async def interruptible(event: asyncio.Event, coro):
    task = asyncio.ensure_future(coro)
    wait_task = asyncio.ensure_future(event.wait())
    done, pending = await asyncio.wait(
        {task, wait_task},
        return_when=asyncio.FIRST_COMPLETED,
    )
    for p in pending:
        p.cancel()
        try:
            await p
        except (asyncio.CancelledError, Exception):
            pass
    if wait_task in done:
        return None
    if task in done:
        return task.result()
    return None


class ApprovalPending(Exception):
    """Raised when a tool needs user approval. Loop pauses, caller resumes."""

    def __init__(
        self,
        thread_id: str,
        request_id: str,
        tool_name: str,
        command_preview: str,
        tc_id: str,
        denylisted: bool = False,
        timeout: int | None = None,
        auto_deny: bool = False,
        keys: list[str] | None = None,
        partial_results: list[tuple[str, str]] | None = None,
    ):
        self.thread_id = thread_id
        self.request_id = request_id
        self.tool_name = tool_name
        self.command_preview = command_preview
        self.tc_id = tc_id
        self.denylisted = denylisted
        self.timeout = timeout
        self.auto_deny = auto_deny
        self.keys = keys or []
        self.partial_results: list[tuple[str, str]] = list(partial_results) if partial_results else []
        super().__init__(f"需要审批: {tool_name} — {command_preview[:80]}")

    def to_pending_data(self) -> dict:
        pending_data: dict[str, Any] = {
            "request_id": self.request_id,
            "tool_name": self.tool_name,
            "command_preview": self.command_preview[:200],
            "tool_call_id": self.tc_id,
        }
        if self.keys:
            pending_data["memory_keys"] = self.keys
        return pending_data


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
        self._compressor = ContextCompressor(compression_config, client=client)

        if config:
            from src.bootstrap import load_bootstrap_files

            agents_cfg = getattr(config, "agents", None)
            if agents_cfg:
                extra = getattr(agents_cfg, "bootstrap_files", None)
                self._context_files = load_bootstrap_files(agents_cfg.workspace, extra_names=extra)

        self._cache_prompt_sections(tools, skills_prompt)

        # Frozen system prompt per session — built once, reused across rounds.
        # Keyed by thread_id so each session gets its own frozen prompt.
        self._system_prompt_cache: dict[str, str] = {}

        self._memory_summary_cache: str = ""
        self._memory_summary_queried: bool = False

        gr_cfg = None
        if config and hasattr(config, "tools"):
            gr_cfg = getattr(config.tools, "guardrails", None)
        if gr_cfg and getattr(gr_cfg, "enabled", False):
            self._guardrails_defaults = dict(
                repeat_fail_block=getattr(gr_cfg, "repeat_fail_block", 5),
                storm_block=getattr(gr_cfg, "storm_block", 8),
            )
        else:
            self._guardrails_defaults = {}
        self._guardrails_map: dict[str, ToolLoopGuardrails] = {}

        self._auto_deny_approval: bool = False

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
            channel=getattr(state, "channel", ""),
            sender_id=getattr(state, "sender_id", ""),
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
        _approval_paused = False
        try:
            return await self._run_inner(state, thread_id, max_rounds)
        except ApprovalPending:
            _approval_paused = True
            raise
        finally:
            if not _approval_paused:
                self._cleanup_guardrails(thread_id)
                self._system_prompt_cache.pop(thread_id, None)
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
            self._cleanup_guardrails(thread_id)
            self._system_prompt_cache.pop(thread_id, None)
            lock.release()

    def get_store(self) -> StateStore:
        return self._store

    async def close(self) -> None:
        """关闭底层客户端连接池。"""
        if hasattr(self._client, "close"):
            await self._client.close()

    def invalidate_memory_cache(self) -> None:
        self._memory_summary_cache = ""
        self._memory_summary_queried = False

    def is_thread_busy(self, thread_id: str) -> bool:
        lock = self._store._locks.get(thread_id)
        return lock is not None and lock.locked()

    def _get_guardrails(self, thread_id: str) -> ToolLoopGuardrails:
        """Return per-thread guardrails, creating on first access."""
        gr = self._guardrails_map.get(thread_id)
        if gr is None:
            gr = ToolLoopGuardrails(**self._guardrails_defaults)
            self._guardrails_map[thread_id] = gr
        return gr

    def _cleanup_guardrails(self, thread_id: str) -> None:
        """Remove guardrails for a finished thread."""
        self._guardrails_map.pop(thread_id, None)

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

    async def _run_inner(self, state: AgentState, thread_id: str, max_rounds: int = 50) -> AgentState:
        """Internal run logic, called with thread lock held.

        Uses proactive compression (hermes-style): check token budget BEFORE
        each model call and compress if over threshold.
        """
        import time as _time
        from src.events import emit_async

        start_ts = _time.monotonic()
        tool_round = 0
        ie = self._store.get_interrupt_flag(thread_id).get_event()

        self._get_guardrails(thread_id).reset()

        for _ in range(max_rounds):
            self._repair_orphan_tool_results(state)

            # 0. Check interrupt
            if await self._check_interrupt(state, thread_id, start_ts, tool_round):
                return state

            # 1. Prepare messages — proactive compression if over budget
            _t_prepare = _time.monotonic()
            messages = await interruptible(ie, self._prepare_messages(state, thread_id))
            _prepare_ms = (_time.monotonic() - _t_prepare) * 1000
            if messages is None:
                if await self._check_interrupt(state, thread_id, start_ts, tool_round):
                    return state
                continue

            # 2. Build tool list
            _t_tools = _time.monotonic()
            active_tools = await self._filter_tools(state)
            openai_tools = [t.to_openai_tool() for t in active_tools] if active_tools else None
            _tools_ms = (_time.monotonic() - _t_tools) * 1000

            # 3. Call model (with retry)
            _t_model = _time.monotonic()
            response = await interruptible(ie, self._call_model_with_retry(messages, tools=openai_tools))
            _model_ms = (_time.monotonic() - _t_model) * 1000
            logger.info(
                "[perf] round=%d prepare=%.0fms tools=%.0fms model=%.0fms total=%.0fms msgs=%d",
                tool_round,
                _prepare_ms,
                _tools_ms,
                _model_ms,
                (_time.monotonic() - start_ts) * 1000,
                len(messages),
            )
            if response is None:
                if await self._check_interrupt(state, thread_id, start_ts, tool_round):
                    return state
                continue

            # 4. Append assistant message
            assistant_msg = self._build_assistant_msg(response)
            state.append_message(assistant_msg)

            # Emit event for intermediate assistant messages (tool_calls present)
            if response.tool_calls:
                await emit_async(
                    "agent_loop.assistant_message",
                    thread_id=thread_id,
                    content=assistant_msg.get("content", ""),
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
                    from src.agent.tool_cache import clear_thread_cache

                    clear_thread_cache(thread_id)
                except Exception:
                    pass
                flag = self._store.get_interrupt_flag(thread_id)
                if not flag.check()[0]:
                    flag.clear()
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

            # 6. Execute tool calls (always parallel)
            tool_round += 1
            parallel_results = await self._execute_tools_parallel(
                response.tool_calls, state, thread_id, interrupt_event=ie
            )
            for tc_id, result in parallel_results:
                state.append_message(
                    {
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": result,
                    }
                )
            await self._store.save(thread_id, state)
            if await self._check_interrupt(state, thread_id, start_ts, tool_round):
                return state

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
        state.append_message(
            {
                "role": "user",
                "content": "工具调用预算已用完。请总结当前进度和结果，不要再调用工具。",
            }
        )
        try:
            summary_resp = await interruptible(ie, self._call_model_with_retry(state.messages, tools=None))
            if summary_resp is not None:
                state.append_message(
                    {
                        "role": "assistant",
                        "content": summary_resp.content,
                    }
                )
                await self._store.save(thread_id, state)
        except Exception as e:
            logger.warning("Grace period summary failed: %s", e)

        try:
            from src.agent.tool_cache import clear_thread_cache

            clear_thread_cache(thread_id)
        except Exception:
            pass

        self._maybe_trigger_skill_review(state, thread_id)
        return state

    async def _resume_inner(self, thread_id: str, decision: str) -> AgentState:
        """Internal resume logic, called with thread lock held."""
        state = await self._store.load(thread_id)
        if state is None:
            raise RuntimeError(f"No saved state for thread {thread_id}")

        pending = state.pending_approval or {}
        pending_tc_id = pending.get("tool_call_id", "")

        # Find the assistant message that contains the pending tool_call
        assistant_msg_idx = None
        for i in range(len(state.messages) - 1, -1, -1):
            msg = state.messages[i]
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                assistant_msg_idx = i
                break

        # Determine which tool_calls already have results
        existing_results: set[str] = set()
        if assistant_msg_idx is not None:
            for msg in state.messages[assistant_msg_idx + 1 :]:
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

                        mem_store = await get_memory_store()
                        for k in memory_keys:
                            await mem_store.forget(k)
                            deleted.append(k)
                    except Exception as exc:
                        import logging as _log

                        _log.getLogger("flyclaw.loop").warning("memory delete resume failed: %s", exc)
                    if deleted:
                        self._memory_summary_cache = ""
                        self._memory_summary_queried = False
                    result_content = json.dumps(
                        {"ok": True, "deleted": deleted, "count": len(deleted)},
                        ensure_ascii=False,
                    )
                    state.append_message(
                        {
                            "role": "tool",
                            "tool_call_id": pending_tc_id,
                            "content": result_content,
                        }
                    )
                    existing_results.add(pending_tc_id)
                elif pending_tc_id and assistant_msg_idx is not None:
                    assistant_msg = state.messages[assistant_msg_idx]
                    for tc in assistant_msg["tool_calls"]:
                        if tc.get("id") == pending_tc_id:
                            try:
                                result = await self._execute_tool(tc, state, thread_id)
                            except ApprovalPending as ap:
                                state.pending_approval = ap.to_pending_data()
                                await self._store.save(thread_id, state)
                                raise
                            if not result:
                                result = "[interrupted] 工具执行被打断"
                            state.append_message(
                                {
                                    "role": "tool",
                                    "tool_call_id": tc["id"],
                                    "content": result,
                                }
                            )
                            break
                    existing_results.add(pending_tc_id)

                    # Execute any other tool_calls from the same message that lack results
                    for tc in assistant_msg["tool_calls"]:
                        tc_id = tc.get("id", "")
                        if tc_id and tc_id not in existing_results:
                            try:
                                result = await self._execute_tool(tc, state, thread_id)
                            except ApprovalPending as ap:
                                state.pending_approval = ap.to_pending_data()
                                # Remaining tools without results will be
                                # re-executed in the next resume cycle —
                                # do NOT add fake results here, or
                                # _resume_inner will skip them forever.
                                await self._store.save(thread_id, state)
                                raise
                            if not result:
                                result = "[interrupted] 工具执行被打断"
                            state.append_message(
                                {
                                    "role": "tool",
                                    "tool_call_id": tc_id,
                                    "content": result,
                                }
                            )
                            existing_results.add(tc_id)
            finally:
                _current_thread_id.reset(_tid_token)
        else:
            if pending_tc_id:
                pending_tool = pending.get("tool_name", "exec_command")
                deny_msg = (
                    "[denied] 记忆删除已取消。"
                    if pending_tool in ("memory_delete", "memory")
                    else "[denied] Command execution was denied by user."
                )
                state.append_message(
                    {
                        "role": "tool",
                        "tool_call_id": pending_tc_id,
                        "content": deny_msg,
                    }
                )
                existing_results.add(pending_tc_id)

            # Deny remaining unexecuted tool calls too
            if assistant_msg_idx is not None:
                assistant_msg = state.messages[assistant_msg_idx]
                for tc in assistant_msg["tool_calls"]:
                    tc_id = tc.get("id", "")
                    if tc_id and tc_id not in existing_results:
                        state.append_message(
                            {
                                "role": "tool",
                                "tool_call_id": tc_id,
                                "content": "[denied] Skipped due to associated denial.",
                            }
                        )
                        existing_results.add(tc_id)

        state.pending_approval = None
        await self._store.save(thread_id, state)

        return await self._run_inner(state, thread_id)

    async def _call_model_with_retry(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_retries: int = 3,
    ) -> ChatResponse:
        import time as _time

        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            t0 = _time.monotonic()
            try:
                resp = await self._client.chat(messages, tools=tools)
                logger.info("[perf] model API attempt=%d took %.0fms", attempt, (_time.monotonic() - t0) * 1000)
                return resp
            except Exception as e:
                logger.warning(
                    "[perf] model API attempt=%d failed after %.0fms: %s",
                    attempt,
                    (_time.monotonic() - t0) * 1000,
                    e,
                )
                last_exc = e
                if attempt >= max_retries:
                    break
                # All models in cooldown — retrying is pointless
                if isinstance(e, RuntimeError) and "No models available" in str(e):
                    raise
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
                delay = min(base * (2**attempt) + random.uniform(0, base), 60.0)
                logger.warning(
                    "Model API error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1,
                    max_retries,
                    delay,
                    e,
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
            try:
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
            except Exception as e:
                logger.warning("Background skill review failed: %s", e)

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
        import time as _time

        _t0 = _time.monotonic()

        await self._truncate_large_outputs(state.messages, thread_id)

        history = list(state.messages)

        _t_compress_start = _time.monotonic()
        if self._compressor.should_compress(history, self._ctx_window_tokens):
            history = await self._compressor.compress(history, self._ctx_window_tokens)
        _compress_ms = (_time.monotonic() - _t_compress_start) * 1000

        # Frozen system prompt: memory → DB → build
        if thread_id in self._system_prompt_cache:
            system_text = self._system_prompt_cache[thread_id]
            _mem_ms = _prompt_ms = 0.0
        elif state.frozen_system_prompt:
            system_text = state.frozen_system_prompt
            self._system_prompt_cache[thread_id] = system_text
            _mem_ms = _prompt_ms = 0.0
        else:
            _t_mem_start = _time.monotonic()
            memory_summary = await self._fetch_memory_summary()
            _mem_ms = (_time.monotonic() - _t_mem_start) * 1000

            _t_prompt_start = _time.monotonic()
            system_text = self._build_system_prompt(state, await self._get_active_tool_defs(state), memory_summary)
            state.frozen_system_prompt = system_text
            self._system_prompt_cache[thread_id] = system_text
            _prompt_ms = (_time.monotonic() - _t_prompt_start) * 1000

        history = self._sanitize_surrogates(history)
        _total_ms = (_time.monotonic() - _t0) * 1000
        logger.debug(
            "[perf] _prepare_messages: total=%.0fms compress=%.0fms memory=%.0fms prompt=%.0fms",
            _total_ms,
            _compress_ms,
            _mem_ms,
            _prompt_ms,
        )
        return [{"role": "system", "content": system_text}] + list(history)

    _NO_TRUNCATE_TOOLS: ClassVar[frozenset[str]] = frozenset({"skill_view"})

    async def _truncate_large_outputs(self, messages: list[dict], thread_id: str = "") -> None:
        """Truncate large tool outputs in-place, but preserve skill_view results."""
        tc_id_to_name: dict[str, str] = {}
        for m in messages:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    if isinstance(tc, dict):
                        tid = tc.get("id", "")
                        fname = tc.get("function", {}).get("name", "")
                    else:
                        tid = getattr(tc, "id", "")
                        fname = getattr(tc.function, "name", "")
                    if tid:
                        tc_id_to_name[tid] = fname

        for m in messages:
            if m.get("role") == "tool":
                tc_id = m.get("tool_call_id", "")
                if tc_id in tc_id_to_name and tc_id_to_name[tc_id] in self._NO_TRUNCATE_TOOLS:
                    continue
                content = m.get("content", "")
                _config = getattr(self, "_config", None)
                threshold = _config.agents.tool_output_cache_chars if _config else 8000
                if isinstance(content, str) and len(content) > threshold and not m.get("_truncated"):
                    from src.agent.tool_cache import cache_large_output

                    truncated, _path = await cache_large_output(
                        content, thread_id, max_chars=threshold, preview=threshold
                    )
                    m["content"] = truncated
                    m["_truncated"] = True

    # ------------------------------------------------------------------
    # Sanitization helpers
    # ------------------------------------------------------------------

    def _repair_orphan_tool_results(self, state: AgentState) -> None:
        tc_ids: set[str] = set()
        for m in state.messages:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    tid = tc.get("id", "")
                    if tid:
                        tc_ids.add(tid)

        result_ids: set[str] = set()
        for m in state.messages:
            if m.get("role") == "tool":
                tid = m.get("tool_call_id", "")
                if tid:
                    result_ids.add(tid)

        missing_ids = tc_ids - result_ids

        # Don't repair tool_calls that are waiting for user approval
        pending = state.pending_approval or {}
        pending_tc_id = pending.get("tool_call_id", "")
        if pending_tc_id:
            missing_ids.discard(pending_tc_id)

        orphan_result_indices = [
            i
            for i, m in enumerate(state.messages)
            if m.get("role") == "tool" and m.get("tool_call_id", "") not in tc_ids
        ]

        if not missing_ids and not orphan_result_indices:
            return

        for i in reversed(orphan_result_indices):
            removed = state.messages.pop(i)
            logger.warning("Repaired orphan tool result: removed %s", removed.get("tool_call_id", ""))

        if missing_ids:
            missing_by_assistant: dict[int, list[str]] = {}
            for tid in missing_ids:
                for i, m in enumerate(state.messages):
                    if m.get("role") == "assistant" and m.get("tool_calls"):
                        if any(tc.get("id") == tid for tc in m["tool_calls"]):
                            missing_by_assistant.setdefault(i, []).append(tid)
                            break
            offset = 0
            for assistant_idx in sorted(missing_by_assistant):
                tids = missing_by_assistant[assistant_idx]
                insert_pos = assistant_idx + 1 + offset
                for tid in reversed(tids):
                    stub = {"role": "tool", "tool_call_id": tid, "content": "[工具被用户打断]"}
                    state.messages.insert(insert_pos, stub)
                    logger.warning("Repaired orphan tool_call: inserted stub for %s", tid)
                offset += len(tids)

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

    async def _get_active_tool_defs(self, state: AgentState) -> list[ToolDef]:
        return await self._filter_tools(state)

    async def _filter_tools(self, state: AgentState) -> list[ToolDef]:
        tools = self._tools

        if self._config:
            from src.tools.policy import apply_tool_policy

            sender_id = state.sender_id
            user = await self._resolve_user(sender_id)
            tools = apply_tool_policy(tools, sender_id, self._config, user=user)

        return tools

    async def _resolve_user(self, sender_id: str):
        if not self._config or not getattr(self._config, "auth", None) or not self._config.auth.enabled:
            return None
        try:
            from src.auth.rbac import get_rbac

            rbac = get_rbac()
            if rbac:
                return await rbac.resolve_user(sender_id)
        except Exception as exc:
            logger.debug("resolve_user failed: %s", exc)
        return None

    def _cache_prompt_sections(self, tools: list[ToolDef], skills_prompt: str) -> None:
        from pathlib import Path
        from src.prompt import (
            _load_soul_md,
            _build_environment_hints,
            _build_tooling_rules,
            _build_tool_index,
            _build_safety,
            _build_skills_section,
            _build_workspace,
            _build_bootstrap_context,
        )

        workspace_dir = "."
        if self._config:
            agents_cfg = getattr(self._config, "agents", None)
            if agents_cfg:
                raw_ws = getattr(agents_cfg, "workspace", ".") or "."
                workspace_dir = str(Path(raw_ws).expanduser().resolve())

        self._prompt_soul = _load_soul_md()
        self._prompt_env = "\n".join(_build_environment_hints(workspace_dir))
        self._prompt_tool_rules = "\n".join(_build_tooling_rules(tools))
        self._prompt_tool_guidance = "\n".join(_build_tool_index(tools))
        self._prompt_safety = "\n".join(_build_safety())
        hub_on = bool(self._config and getattr(self._config.skills.hub, "enabled", True))
        self._prompt_skills = "\n".join(_build_skills_section(skills_prompt, hub_enabled=hub_on))
        self._prompt_workspace = "\n".join(_build_workspace(workspace_dir))
        self._prompt_bootstrap = "\n".join(_build_bootstrap_context(self._context_files)) if self._context_files else ""
        self._prompt_platform_cache: dict[str, str] = {}

        logger.info(
            "System prompt sections cached (soul=%d, env=%d, tool_index=%d, skills=%d, bootstrap=%d chars)",
            len(self._prompt_soul),
            len(self._prompt_env),
            len(self._prompt_tool_guidance),
            len(self._prompt_skills),
            len(self._prompt_bootstrap),
        )

    async def _fetch_memory_summary(self) -> str:
        if self._memory_summary_queried:
            return self._memory_summary_cache
        try:
            from src.tools.memory_tools import get_memory_store

            store = await get_memory_store()
            items = await store.list_all(limit=20)
            if not items:
                self._memory_summary_queried = True
                return ""
            lines = ["## 已知记忆"]
            for item in items:
                cat = item.get("category", "fact")
                content = item.get("content", "")[:80]
                lines.append(f"- [{cat}] {content}")
            lines.append(
                "以上是已加载的主要部分的记忆，直接基于这些信息回答即可，除非用户表示不足或要求更多回忆，否则无需再次搜索。如果需要修改或补充，使用 memory 工具。"
            )
            result = "\n".join(lines)
            self._memory_summary_cache = result
            self._memory_summary_queried = True
            return result
        except Exception:
            return self._memory_summary_cache or ""

    def _build_system_prompt(self, state: AgentState, active_tools: list[ToolDef], memory_summary: str = "") -> str:
        from src.prompt import _build_platform_hints, _build_tool_index, _build_sandbox_hints
        from src.tools.exec import is_sandbox_enabled

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

        # Sandbox hints — dynamically read current state (not cached, since /sandbox toggles at runtime)
        from pathlib import Path as _Path

        workspace_dir = "."
        if self._config:
            agents_cfg = getattr(self._config, "agents", None)
            if agents_cfg:
                raw_ws = getattr(agents_cfg, "workspace", ".") or "."
                workspace_dir = str(_Path(raw_ws).expanduser().resolve())
        sandbox_hints = _build_sandbox_hints(is_sandbox_enabled(), workspace_dir)
        if sandbox_hints:
            parts.append("\n".join(sandbox_hints))

        parts.append(self._prompt_tool_rules)
        tool_index = "\n".join(_build_tool_index(active_tools))
        if tool_index:
            parts.append(tool_index)
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
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in response.tool_calls
            ]
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
            return f"[error] Tool '{tool_name}' arguments were malformed/truncated, please retry with valid JSON."

        tool_def = self._tool_map.get(tool_name)
        if tool_def is None:
            available = sorted(self._tool_map.keys())
            logger.warning(
                "Unknown tool '%s'. Available (%d): %s",
                tool_name,
                len(available),
                available,
            )
            hint = ", ".join(available[:10])
            return f"[error] Unknown tool: {tool_name}. Available: {hint}{'...' if len(available) > 10 else ''}"

        guard = self._get_guardrails(thread_id).check(tool_name, args)
        if guard is not None:
            return guard.synthetic_result

        import time as _time

        start = _time.monotonic()
        args_preview = json.dumps(args, ensure_ascii=False)[:200] if args else ""

        await emit_async(
            "tool.exec_started",
            thread_id=thread_id,
            tool_name=tool_name,
            tool_call_id=getattr(tc, "id", "") if not isinstance(tc, dict) else tc.get("id", ""),
            args_preview=args_preview,
        )

        from src.tools.exec import _current_agent_context

        _ctx_token = _current_agent_context.set(
            {
                "sender_id": state.sender_id,
                "chat_id": state.chat_id,
                "channel": state.channel,
                "parent_thread_id": thread_id,
            }
        )
        try:
            result = await asyncio.wait_for(tool_def.execute(args), timeout=tool_def.timeout)
            duration_ms = (_time.monotonic() - start) * 1000
            return await self._record_tool_success(
                tool_name=tool_name,
                args=args,
                result=result,
                thread_id=thread_id,
                duration_ms=duration_ms,
                args_preview=args_preview,
                state=state,
            )
        except Exception as e:
            duration_ms = (_time.monotonic() - start) * 1000
            from src.tools.exec import ApprovalNeededError
            from src.tools.exceptions import ToolExecutionError
            from src.tools.memory_tools import MemoryDeleteNeedsApproval

            if isinstance(e, (ApprovalNeededError, MemoryDeleteNeedsApproval)):
                # ── 记忆删除：不可逆操作，始终需要审批 ──
                if not isinstance(e, MemoryDeleteNeedsApproval):
                    # ── 审批绕过检查：owner 等高权限用户可跳过审批（仅 exec_command 等命令执行类） ──
                    # try/except 仅包裹绕过资格判定（resolve_user / RBAC /
                    # approve_session），工具实际执行放在 try/except 之外，
                    # 避免执行阶段的异常被静默吞掉而落入正常审批流程。
                    _bypass_ctx = None
                    try:
                        bypass_user = await self._resolve_user(state.sender_id)
                        if bypass_user:
                            from src.auth.rbac import get_rbac

                            rbac = get_rbac()
                            if rbac and rbac.check_approval_bypass(bypass_user):
                                cmd = getattr(e, "command_preview", "") or getattr(e, "command", "")

                                from src.tools.approval import get_approval_manager

                                mgr = get_approval_manager()
                                mgr.approve_session(thread_id, tool_name, cmd)
                                _bypass_ctx = (bypass_user, cmd)
                    except Exception as exc:
                        logger.debug("approval bypass check failed, falling back to normal flow: %s", exc)

                    if _bypass_ctx is not None:
                        bypass_user, cmd = _bypass_ctx
                        # 直接重新执行工具（session 已批准，不会再抛
                        # ApprovalNeededError），然后走完整审计链。
                        # 注意：不能用递归 self._execute_tool()，
                        # 否则 tool.exec_started 会双重发射。
                        # 执行/审计异常自然传播到外层 except 处理。
                        result = await tool_def.execute(args)
                        duration_ms = (_time.monotonic() - start) * 1000

                        logger.info(
                            "[approval-bypass] user=%s role=%s tool=%s cmd=%.200s",
                            bypass_user.user_id,
                            bypass_user.role.value,
                            tool_name,
                            cmd,
                        )
                        return await self._record_tool_success(
                            tool_name=tool_name,
                            args=args,
                            result=result,
                            thread_id=thread_id,
                            duration_ms=duration_ms,
                            args_preview=args_preview,
                            state=state,
                        )

                # ── 正常审批流程 ──
                await emit_async(
                    "tool.approval_pending",
                    thread_id=thread_id,
                    tool_name=getattr(e, "tool_name", tool_name),
                    command_preview=getattr(e, "command_preview", getattr(e, "command", "")[:200]),
                    denylisted=getattr(e, "denylisted", False),
                )
                return await self._handle_approval(e, tc, state, thread_id)
            logger.error("Tool %s failed: %s", tool_name, e)
            self._get_guardrails(thread_id).record(tool_name, args, success=False)

            await emit_async(
                "tool.exec_failed",
                thread_id=thread_id,
                tool_name=tool_name,
                success=False,
                duration_ms=duration_ms,
                error=str(e)[:500],
                error_type=type(e).__name__,
                args_preview=args_preview,
                sender_id=getattr(state, "sender_id", ""),
                channel=getattr(state, "channel", ""),
            )
            if isinstance(e, ToolExecutionError):
                return f"[error] {e}"
            err_msg = str(e)[:200]
            return f"[error] 工具 {tool_name} 执行失败: {err_msg}"
        finally:
            _current_agent_context.reset(_ctx_token)

    async def _record_tool_success(
        self,
        *,
        tool_name: str,
        args: dict,
        result: str,
        thread_id: str,
        duration_ms: float,
        args_preview: str,
        state: AgentState,
    ) -> str:
        """审计工具执行成功：redact → guardrails → skill nudge → emit completed。

        被 _execute_tool 成功路径和 approval-bypass 路径共用，
        确保两条路径产生一致的审计事件。
        """
        from src.events import emit_async

        try:
            from src.security.redact import redact

            result = redact(result)
        except Exception:
            pass
        self._get_guardrails(thread_id).record(
            tool_name, args, success=True, result=result if isinstance(result, str) else ""
        )
        if tool_name in _SKILL_TOOL_NAMES and self._skill_nudge_interval > 0:
            self._iters_since_skill = 0
        if tool_name == "memory":
            self._memory_summary_cache = ""
            self._memory_summary_queried = False
        await emit_async(
            "tool.exec_completed",
            thread_id=thread_id,
            tool_name=tool_name,
            success=True,
            duration_ms=duration_ms,
            result_length=len(result) if isinstance(result, str) else 0,
            args_preview=args_preview,
            sender_id=getattr(state, "sender_id", ""),
            channel=getattr(state, "channel", ""),
        )
        if not result:
            result = "[interrupted] 工具执行被打断"
        return result

    async def _execute_tools_parallel(
        self, tool_calls: list[Any], state: AgentState, thread_id: str, interrupt_event: asyncio.Event | None = None
    ) -> list[tuple[str, str]]:
        results: list[tuple[str, str] | BaseException] = [NotImplemented] * len(tool_calls)

        async def _run_one(idx: int, tc: Any) -> None:
            tc_id = tc.id if hasattr(tc, "id") else tc.get("id", "")
            try:
                result = await self._execute_tool(tc, state, thread_id)
                if not result:
                    result = "[interrupted] 工具执行被打断"
                results[idx] = (tc_id, result)
            except BaseException as exc:
                results[idx] = exc

        tasks = [asyncio.create_task(_run_one(i, tc)) for i, tc in enumerate(tool_calls)]
        gather_task = asyncio.gather(*tasks, return_exceptions=True)

        if interrupt_event is not None:
            gather_result = await interruptible(interrupt_event, gather_task)
            if gather_result is None:
                for t in tasks:
                    if not t.done():
                        t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                for i, r in enumerate(results):
                    tc_id = tool_calls[i].id if hasattr(tool_calls[i], "id") else tool_calls[i].get("id", "")
                    if r is NotImplemented or isinstance(r, asyncio.CancelledError):
                        results[i] = (tc_id, "[interrupted] 工具执行被打断")
                    elif isinstance(r, ApprovalPending):
                        from src.tools.approval import get_approval_manager

                        get_approval_manager().cancel_pending(r.request_id)
                        results[i] = (tc_id, "[interrupted] 工具执行被打断（审批已取消）")
                if state.pending_approval is not None:
                    state.pending_approval = None
        else:
            await gather_task

        final: list[tuple[str, str]] = []
        for i, r in enumerate(results):
            tc_id = tool_calls[i].id if hasattr(tool_calls[i], "id") else tool_calls[i].get("id", "")
            if isinstance(r, ApprovalPending):
                from src.tools.approval import get_approval_manager

                mgr = get_approval_manager()
                for j in range(i + 1, len(results)):
                    if results[j] is NotImplemented:
                        # Not started — will be re-executed on resume
                        pass
                    elif isinstance(results[j], ApprovalPending):
                        # Also needs approval — cancel request; will
                        # re-request when _resume_inner re-executes it
                        mgr.cancel_pending(results[j].request_id)
                    elif not isinstance(results[j], BaseException):
                        final.append(results[j])
                state.pending_approval = r.to_pending_data()
                r.partial_results = list(final)
                for tc_id, result in r.partial_results:
                    state.append_message(
                        {
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": result,
                        }
                    )
                await self._store.save(thread_id, state)
                raise r
            if isinstance(r, BaseException):
                final.append((tc_id, f"[error] {type(r).__name__}: {r}"))
            else:
                final.append(r)
        return final

    async def _handle_approval(self, error: Exception, tc: Any, state: AgentState, thread_id: str) -> str:
        """Handle approval-needed error by saving state and raising ApprovalPending."""
        if getattr(self, "_auto_deny_approval", False):
            tool_name = getattr(error, "tool_name", "exec_command")
            cmd = getattr(error, "command_preview", "") or getattr(error, "command", "")
            logger.info("Auto-denying approval in sub-agent for %s: %s", tool_name, cmd[:80])
            return "[denied] 此操作需要用户审批，子代理模式下自动拒绝。请使用其他方式完成任务。"

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
            timeout_seconds=getattr(error, "timeout", None) or 120,
            approval_key=getattr(error, "approval_key", ""),
        )

        raise ApprovalPending(
            thread_id=thread_id,
            request_id=req.id,
            tool_name=tool_name,
            command_preview=cmd[:200],
            denylisted=denylisted,
            timeout=getattr(error, "timeout", None),
            auto_deny=getattr(error, "auto_deny", False),
            keys=getattr(error, "keys", None),
            tc_id=tc_id,
        )
