"""LiteGraph-backed agent loop — drop-in replacement for AgentLoop.

Public API is identical to AgentLoop:
  - run(state, thread_id, max_rounds) -> AgentState
  - resume(thread_id, decision) -> AgentState
  - get_store() -> StateStore

Internally builds a 3-node StateGraph (agent → tools → condition) and uses
LiteGraph's checkpointing + interrupt for state persistence and approval flow.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Annotated, Any, TypedDict

from litegraph import END, START, StateGraph
from litegraph.channels.messages import add_messages
from litegraph.checkpoint.sqlite import SqliteSaver

from src.agent.loop import ApprovalPending
from src.agent.state import AgentState, StateStore
from src.agent.tooldef import ToolDef

logger = logging.getLogger("myclaw.agent.litegraph_loop")

_CHARS_PER_TOKEN = 4


def _estimate_tokens_simple(messages: list[dict]) -> int:
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


class _GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    system_prompt: str
    sender_id: str
    chat_id: str
    chat_type: str
    message_id: str
    user_role: str
    channel: str
    __thread_id__: str


class LiteGraphAgentLoop:
    def __init__(
        self,
        client: Any,
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
        self._tool_map: dict[str, ToolDef] = {t.name: t for t in tools}

        self._compiled = None
        self._checkpointer = None

        self._compressor = None
        self._context_files: list[str] = []

        if config:
            from src.compressor.compressor import ContextCompressor
            self._compressor = ContextCompressor(config.compression, main_config=config)

            from src.bootstrap import load_bootstrap_files
            agents_cfg = getattr(config, "agents", None)
            if agents_cfg:
                extra = getattr(agents_cfg, "bootstrap_files", None)
                self._context_files = load_bootstrap_files(
                    agents_cfg.workspace, extra_names=extra
                )

    # ------------------------------------------------------------------
    # Public API — identical to AgentLoop
    # ------------------------------------------------------------------

    async def run(self, state: AgentState, thread_id: str, max_rounds: int = 50) -> AgentState:
        from src.events import emit_async

        await emit_async(
            "agent_loop.started",
            thread_id=thread_id,
            max_rounds=max_rounds,
            message_count=len(state.messages),
            channel=getattr(state, 'channel', ''),
            sender_id=getattr(state, 'sender_id', ''),
        )

        self._ensure_graph()

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

    def close(self) -> None:
        if self._checkpointer is not None:
            try:
                self._checkpointer.close()
            except Exception:
                pass
            self._checkpointer = None
            self._compiled = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_graph(self):
        if self._compiled is not None:
            return

        graph = StateGraph(_GraphState)
        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", self._tools_node)
        graph.add_edge(START, "agent")
        graph.add_conditional_edges(
            "agent",
            self._should_continue,
            {"tools": "tools", END: END},
        )
        graph.add_edge("tools", "agent")

        cp_path = self._config.checkpointer.path if self._config else "checkpoints.db"
        if cp_path != ":memory:":
            from pathlib import Path
            cp_path = str(Path(cp_path).expanduser().resolve())
        self._checkpointer = SqliteSaver(cp_path)
        self._compiled = graph.compile(checkpointer=self._checkpointer)
        logger.info("LiteGraph agent graph compiled with %d tools", len(self._tools))

    async def _run_inner(self, state: AgentState, thread_id: str, max_rounds: int) -> AgentState:
        from src.events import emit_async

        start_ts = time.monotonic()
        # recursion_limit counts graph steps (agent + tools = 2 per round), so multiply by 2
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": max_rounds * 2}

        # Clear old checkpoint to avoid message duplication.
        # Full history is passed via input, so old checkpoint data is redundant.
        # Checkpoints are only kept for interrupt/resume (handled by _resume_inner).
        self._checkpointer.delete_thread(thread_id)

        input_dict = self._state_to_input(state)
        input_dict["__thread_id__"] = thread_id

        try:
            result = await self._compiled.ainvoke(input_dict, config)
        except Exception as e:
            from litegraph.errors import GraphInterrupt, GraphRecursionError
            if isinstance(e, GraphRecursionError):
                logger.warning("Graph recursion limit (%d) reached", max_rounds)
                result = self._compiled.get_state(config).values
            elif isinstance(e, GraphInterrupt):
                result = None
            else:
                raise

        snap = self._compiled.get_state(config)
        if snap.interrupts:
            interrupt_val = snap.interrupts[0].value
            if isinstance(interrupt_val, dict) and interrupt_val.get("type") == "approval_request":
                tool_name = interrupt_val.get("tool_name", "exec_command")
                cmd_preview = interrupt_val.get("command_preview", "")
                denylisted = interrupt_val.get("denylisted", False)
                req_id = interrupt_val.get("request_id", "")
                approval_timeout = interrupt_val.get("timeout")
                auto_deny = interrupt_val.get("auto_deny", False)
                raise ApprovalPending(
                    thread_id=thread_id,
                    request_id=req_id,
                    tool_name=tool_name,
                    command_preview=cmd_preview,
                    denylisted=denylisted,
                    timeout=approval_timeout,
                    auto_deny=auto_deny,
                )

        if result is None:
            result = snap.values

        final_state = self._result_to_state(result, state)

        # Count tool rounds: tool messages added during this run
        input_tool_count = sum(1 for m in state.messages if m.get("role") == "tool")
        output_tool_count = sum(1 for m in final_state.messages if m.get("role") == "tool")
        total_rounds = output_tool_count - input_tool_count

        duration_ms = (time.monotonic() - start_ts) * 1000
        await emit_async(
            "agent_loop.completed",
            thread_id=thread_id,
            final_message_count=len(final_state.messages),
            duration_ms=duration_ms,
            total_rounds=total_rounds,
        )
        await self._store.save(thread_id, final_state)
        return final_state

    async def _resume_inner(self, thread_id: str, decision: str) -> AgentState:
        from litegraph.types import Command
        from src.events import emit_async

        self._ensure_graph()
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": self._get_max_tool_rounds() * 2}

        # Load pre-resume state for base fallback and round counting
        pre_resume_state = await self._store.aload(thread_id)
        pre_resume_tool_count = (
            sum(1 for m in pre_resume_state.messages if m.get("role") == "tool")
            if pre_resume_state else 0
        )

        start_ts = time.monotonic()
        try:
            result = await self._compiled.ainvoke(Command(resume=decision), config)
        except Exception as e:
            from litegraph.errors import GraphInterrupt, GraphRecursionError
            if isinstance(e, GraphRecursionError):
                logger.warning("Graph recursion limit reached on resume")
                result = self._compiled.get_state(config).values
            elif isinstance(e, GraphInterrupt):
                result = None
            else:
                raise

        snap = self._compiled.get_state(config)
        if snap.interrupts:
            interrupt_val = snap.interrupts[0].value
            if isinstance(interrupt_val, dict) and interrupt_val.get("type") == "approval_request":
                raise ApprovalPending(
                    thread_id=thread_id,
                    request_id=interrupt_val.get("request_id", ""),
                    tool_name=interrupt_val.get("tool_name", "exec_command"),
                    command_preview=interrupt_val.get("command_preview", ""),
                    denylisted=interrupt_val.get("denylisted", False),
                    timeout=interrupt_val.get("timeout"),
                    auto_deny=interrupt_val.get("auto_deny", False),
                )

        if result is None:
            result = snap.values

        final_state = self._result_to_state(result, base=pre_resume_state)
        post_resume_tool_count = sum(1 for m in final_state.messages if m.get("role") == "tool")
        total_rounds = post_resume_tool_count - pre_resume_tool_count
        duration_ms = (time.monotonic() - start_ts) * 1000
        await emit_async(
            "agent_loop.completed",
            thread_id=thread_id,
            final_message_count=len(final_state.messages),
            duration_ms=duration_ms,
            total_rounds=total_rounds,
        )
        await self._store.save(thread_id, final_state)
        return final_state

    # ------------------------------------------------------------------
    # Graph nodes
    # ------------------------------------------------------------------

    async def _agent_node(self, state: dict) -> dict:
        messages = state.get("messages", [])
        if not messages:
            return {"messages": []}

        # Truncate large tool outputs (same as old _truncate_large_outputs)
        history = self._truncate_large_outputs(messages)
        # Defensive validation (same as old _fix_tool_calls_args)
        history = self._fix_tool_calls_args(history)

        # Proactive compression (same as old _prepare_messages)
        if self._compressor is not None and self._compressor.should_compress(history):
            history = await self._compressor.compress(history, self._ctx_window_tokens)
        elif self._compressor is None:
            estimated = _estimate_tokens_simple(history)
            if estimated > int(self._ctx_window_tokens * 0.8):
                history = self._static_fallback(history)

        active_tools = self._filter_tools(state)
        openai_tools = [t.to_openai_tool() for t in active_tools] if active_tools else None

        sp = state.get("system_prompt", "")
        from src.prompt import build_system_prompt
        full_system = build_system_prompt(
            config=self._config,
            tools=active_tools,
            skills_prompt=self._skills_prompt,
            context_files=self._context_files,
            extra_system_prompt=sp,
        )

        all_messages = [{"role": "system", "content": full_system}] + list(history)

        if openai_tools:
            response = await self._client.chat(all_messages, tools=openai_tools)
        else:
            response = await self._client.chat(all_messages)

        assistant_msg = self._build_assistant_msg(response)

        # Redact credentials — only when no tool_calls (final response)
        # When there are tool_calls, redaction is deferred until the last assistant
        if not response.tool_calls:
            self._redact_assistant_content(assistant_msg)

        return {"messages": [assistant_msg]}

    async def _tools_node(self, state: dict) -> dict:
        from src.events import emit_async
        from litegraph.types import interrupt as _lg_interrupt

        # Detect resume context: on resume the scratchpad has resume values.
        # This lets us suppress duplicate events for re-executed tools.
        import litegraph.pregel._loop as _loop_mod
        _cfg = _loop_mod._current_config.get({})
        _sp = _cfg.get("configurable", {}).get("__scratchpad__")
        is_resume = _sp is not None and bool(_sp.resume)

        thread_id = state.get("__thread_id__", "")
        messages = state.get("messages", [])
        last_assistant = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                last_assistant = msg
                break

        if not last_assistant:
            return {"messages": []}

        # Find which tool_calls already have results — avoid re-execution on resume
        existing_results: set[str] = set()
        last_assistant_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i] is last_assistant or (
                messages[i].get("role") == "assistant"
                and messages[i].get("tool_calls") == last_assistant.get("tool_calls")
            ):
                last_assistant_idx = i
                break
        if last_assistant_idx is not None:
            for msg in messages[last_assistant_idx + 1:]:
                if msg.get("role") == "tool":
                    existing_results.add(msg.get("tool_call_id", ""))

        tool_messages = []
        for tc in last_assistant["tool_calls"]:
            tc_id = tc.get("id", "")

            # Skip already-executed tool calls (on resume)
            if tc_id in existing_results:
                continue

            fn_info = tc.get("function", {})
            tool_name = fn_info.get("name", "")
            args_str = fn_info.get("arguments", "")

            try:
                args = json.loads(args_str) if args_str else {}
            except json.JSONDecodeError:
                args = {}
            args_preview = json.dumps(args, ensure_ascii=False)[:200] if args else ""

            # On first pass: emit exec_started for all tools.
            # On resume: suppress events for re-executed tools before the
            # interrupted one — only emit for the approval-handled tool.
            if not is_resume:
                await emit_async(
                    "tool.exec_started",
                    thread_id=thread_id,
                    tool_name=tool_name,
                    tool_call_id=tc_id,
                    args_preview=args_preview,
                )

            start = time.monotonic()
            try:
                result = await self._execute_tool(tc)
                duration_ms = (time.monotonic() - start) * 1000

                try:
                    from src.security.redact import redact
                    result = redact(result)
                except Exception:
                    pass

                if not is_resume:
                    await emit_async(
                        "tool.exec_completed",
                        thread_id=thread_id,
                        tool_name=tool_name,
                        success=True,
                        duration_ms=duration_ms,
                        result_length=len(result) if isinstance(result, str) else 0,
                        args_preview=args_preview,
                        sender_id=state.get("sender_id", ""),
                        channel=state.get("channel", ""),
                    )

            except Exception as e:
                duration_ms = (time.monotonic() - start) * 1000
                from src.tools.exec import ApprovalNeededError

                if isinstance(e, ApprovalNeededError):
                    cmd = getattr(e, "command", "")
                    denylisted = getattr(e, "denylisted", False)
                    approval_timeout = getattr(e, "timeout", None)
                    auto_deny = getattr(e, "auto_deny", False)
                    sender_id = state.get("sender_id", "")
                    chat_id = state.get("chat_id", "")
                    message_id = state.get("message_id", "")

                    from src.tools.approval import get_approval_manager
                    mgr = get_approval_manager()

                    # Only create new approval request on first pass.
                    # On resume, interrupt() returns the decision immediately
                    # and we already have the approval info.
                    req_id = ""
                    if not is_resume:
                        req = mgr.request_approval(
                            tool_name="exec_command",
                            args=cmd,
                            sender_id=sender_id,
                            chat_id=chat_id,
                            message_id=message_id,
                            thread_id=thread_id,
                        )
                        req_id = req.id
                        await emit_async(
                            "tool.approval_pending",
                            thread_id=thread_id,
                            tool_name="exec_command",
                            command_preview=cmd[:200],
                            denylisted=denylisted,
                        )

                    decision = _lg_interrupt({
                        "type": "approval_request",
                        "tool_name": "exec_command",
                        "command_preview": cmd[:200],
                        "denylisted": denylisted,
                        "request_id": req_id,
                        "timeout": approval_timeout,
                        "auto_deny": auto_deny,
                    })

                    if decision in ("allow_once", "allow_always"):
                        # Update durable approvals
                        if decision == "allow_always":
                            mgr._durable.setdefault("exec_command", [])
                            digest = mgr._make_digest("exec_command", cmd[:200])
                            if digest not in mgr._durable["exec_command"]:
                                mgr._durable["exec_command"].append(digest)
                                mgr._save_durable()

                        # Execute the approved tool with redaction + completion event
                        approve_start = time.monotonic()
                        try:
                            tool_def = self._tool_map.get(tool_name)
                            if tool_def:
                                result = await tool_def.execute(args)
                            else:
                                result = f"[error] Unknown tool: {tool_name}"
                        except Exception as exec_err:
                            result = f"[error] {type(exec_err).__name__}: {exec_err}"

                        try:
                            from src.security.redact import redact
                            result = redact(result)
                        except Exception:
                            pass

                        approve_dur = (time.monotonic() - approve_start) * 1000
                        await emit_async(
                            "tool.exec_completed",
                            thread_id=thread_id,
                            tool_name=tool_name,
                            success=True,
                            duration_ms=approve_dur,
                            result_length=len(result) if isinstance(result, str) else 0,
                            args_preview=args_preview,
                            sender_id=sender_id,
                            channel=state.get("channel", ""),
                        )
                    else:
                        result = "[denied] Command execution was denied by user."
                        # Cascade deny to remaining unexecuted tool_calls
                        for later_tc in last_assistant["tool_calls"]:
                            later_id = later_tc.get("id", "")
                            if later_id and later_id not in existing_results and later_id != tc_id:
                                tool_messages.append({
                                    "role": "tool",
                                    "tool_call_id": later_id,
                                    "content": "[denied] Skipped due to associated denial.",
                                })
                                existing_results.add(later_id)

                    tool_messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": result,
                    })
                    continue

                logger.error("Tool %s failed: %s", tool_name, e)
                await emit_async(
                    "tool.exec_failed",
                    thread_id=thread_id,
                    tool_name=tool_name,
                    success=False,
                    duration_ms=duration_ms,
                    error=str(e)[:500],
                    error_type=type(e).__name__,
                    args_preview=args_preview,
                    sender_id=state.get("sender_id", ""),
                    channel=state.get("channel", ""),
                )
                result = f"[error] {type(e).__name__}: {e}"

            tool_messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": result,
            })

        return {"messages": tool_messages}

    def _should_continue(self, state: dict) -> str:
        messages = state.get("messages", [])
        last = messages[-1] if messages else {}
        if last.get("tool_calls"):
            return "tools"
        return END

    # ------------------------------------------------------------------
    # Helpers — reused from AgentLoop
    # ------------------------------------------------------------------

    def _truncate_large_outputs(self, messages: list[dict]) -> list[dict]:
        result = []
        for m in messages:
            if m.get("role") == "tool":
                content = m.get("content", "")
                if isinstance(content, str) and len(content) > 2000:
                    result.append({**m, "content": content[:500] + f"\n... [truncated, {len(content)} chars total]"})
                    continue
            result.append(m)
        return result

    def _fix_tool_calls_args(self, messages: list[dict]) -> list[dict]:
        for m in messages:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    fn = tc.get("function", {})
                    args_str = fn.get("arguments", "")
                    try:
                        json.loads(args_str)
                    except (json.JSONDecodeError, TypeError):
                        logger.error("Invalid arguments for %s - save-time fix may have a bug",
                                     fn.get("name", "unknown"))
        return messages

    def _filter_tools(self, state: dict) -> list[ToolDef]:
        tools = self._tools

        if self._config:
            from src.tools.policy import apply_tool_policy
            sender_id = state.get("sender_id", "")
            user = self._resolve_user(sender_id)
            tools = apply_tool_policy(tools, sender_id, self._config, user=user)

        channel = state.get("channel", "")
        if channel == "feishu":
            tools = [t for t in tools if not t.name.startswith("qq_")]
        elif channel == "qq":
            feishu_extra = frozenset({"send_image_to_chat", "send_file_to_chat"})
            tools = [t for t in tools if not t.name.startswith("feishu_") and t.name not in feishu_extra]

        max_rounds = self._get_max_tool_rounds()
        if max_rounds > 0:
            messages = state.get("messages", [])
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "user":
                    messages = messages[i:]
                    break
            tool_count = sum(1 for m in messages if m.get("role") == "tool")
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

    def _get_max_tool_rounds(self) -> int:
        if not self._config:
            return 50
        agents_cfg = getattr(self._config, "agents", None)
        if agents_cfg is None:
            return 50
        return getattr(agents_cfg, "max_tool_rounds", 0) or 50

    def _build_assistant_msg(self, response) -> dict:
        msg: dict[str, Any] = {"role": "assistant", "content": response.content}
        if response.tool_calls:
            fixed_calls = []
            for tc in response.tool_calls:
                args_str = tc.function.arguments
                try:
                    json.loads(args_str)
                except (json.JSONDecodeError, TypeError):
                    if args_str and args_str[-1] not in ("}", "]"):
                        try:
                            json.loads(args_str + "}")
                            args_str = args_str + "}"
                            logger.warning("Fixed truncated arguments for %s", tc.function.name)
                        except (json.JSONDecodeError, TypeError):
                            args_str = "{}"
                    else:
                        args_str = "{}"

                fixed_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": args_str,
                    },
                })
            msg["tool_calls"] = fixed_calls
        return msg

    def _redact_assistant_content(self, msg: dict) -> None:
        if isinstance(msg.get("content"), str) and msg["content"]:
            try:
                from src.security.redact import redact
                msg["content"] = redact(msg["content"])
            except Exception:
                pass

    async def _execute_tool(self, tc) -> str:
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

        result = await tool_def.execute(args)
        return result

    def _static_fallback(self, messages: list[dict]) -> list[dict]:
        max_tokens = int(self._ctx_window_tokens * 0.7)
        estimated = _estimate_tokens_simple(messages)
        if estimated <= max_tokens:
            return messages

        non_system = [m for m in messages if m.get("role") != "system"]
        if not non_system:
            return messages

        tail_budget = int(max_tokens * 0.5)
        tail_msgs: list[dict] = []
        tail_tokens = 0
        for m in reversed(non_system):
            content = m.get("content", "")
            msg_tokens = (len(content) if isinstance(content, str) else len(str(content))) // _CHARS_PER_TOKEN + 10
            if tail_tokens + msg_tokens > tail_budget:
                break
            tail_msgs.append(m)
            tail_tokens += msg_tokens
        tail_msgs.reverse()

        pruned = non_system[:len(non_system) - len(tail_msgs)]
        if not pruned:
            return messages

        call_id_to_name: dict[str, str] = {}
        for m in messages:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    fn = tc.get("function", {})
                    if isinstance(fn, dict) and tc.get("id"):
                        call_id_to_name[tc["id"]] = fn.get("name", "unknown")

        summaries = []
        for m in pruned:
            role = m.get("role", "")
            content = str(m.get("content", ""))[:150]
            if role == "user":
                summaries.append(f"User: {content}")
            elif role == "assistant":
                summaries.append(f"Assistant: {content}")
            elif role == "tool":
                name = call_id_to_name.get(m.get("tool_call_id", ""), "unknown")
                summaries.append(f"[Tool({name})]: {content[:150]}")

        summary = "[Earlier conversation summarized]\n" + "\n".join(summaries[-20:])
        return [{"role": "system", "content": summary}] + tail_msgs

    # ------------------------------------------------------------------
    # State conversion
    # ------------------------------------------------------------------

    def _state_to_input(self, state: AgentState) -> dict:
        return {
            "messages": list(state.messages),
            "system_prompt": state.system_prompt,
            "sender_id": state.sender_id,
            "chat_id": state.chat_id,
            "chat_type": state.chat_type,
            "message_id": state.message_id,
            "user_role": state.user_role,
            "channel": state.channel,
        }

    def _result_to_state(self, result: dict | None, base: AgentState | None = None) -> AgentState:
        r = result or {}
        return AgentState(
            messages=r.get("messages", []),
            system_prompt=r.get("system_prompt", base.system_prompt if base else ""),
            sender_id=r.get("sender_id", base.sender_id if base else ""),
            chat_id=r.get("chat_id", base.chat_id if base else ""),
            chat_type=r.get("chat_type", base.chat_type if base else "p2p"),
            message_id=r.get("message_id", base.message_id if base else ""),
            user_role=r.get("user_role", base.user_role if base else ""),
            channel=r.get("channel", base.channel if base else ""),
        )
