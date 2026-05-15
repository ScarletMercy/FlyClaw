"""Self-contained agent loop.

The loop is a simple while-cycle:
  call model → check tool_calls → execute tools → append results → loop
Terminates when the model responds without tool calls (or max rounds hit).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any

from src.agent.client import ChatClient, ChatResponse, FallbackChain
from src.agent.state import AgentState, StateStore
from src.agent.tooldef import ToolDef

logger = logging.getLogger("myclaw.agent.loop")

_CHARS_PER_TOKEN = 4


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

    def __init__(self, thread_id: str, request_id: str, tool_name: str, command_preview: str, denylisted: bool = False):
        self.thread_id = thread_id
        self.request_id = request_id
        self.tool_name = tool_name
        self.command_preview = command_preview
        self.denylisted = denylisted
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
        self._compressor = None
        self._context_files: list[str] = []

        # Build tool name → ToolDef lookup
        self._tool_map: dict[str, ToolDef] = {t.name: t for t in tools}

        # Compressor
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

        # Cache: avoids re-building identical messages across tool rounds
        self._cached_history: list[dict] | None = None
        self._cached_history_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, state: AgentState, thread_id: str, max_rounds: int = 50) -> AgentState:
        """Execute agent loop with per-thread locking."""
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

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _run_inner(self, state: AgentState, thread_id: str, max_rounds: int) -> AgentState:
        """Internal run logic, called with thread lock held.

        Uses proactive compression (hermes-style): check token budget BEFORE
        each model call and compress if over threshold.
        """
        max_tool_rounds = self._get_max_tool_rounds()
        tool_round = 0

        for _ in range(max_rounds):
            # 1. Prepare messages — proactive compression if over budget
            messages = await self._prepare_messages(state)

            # 2. Build tool list
            active_tools = self._filter_tools(state)
            openai_tools = [t.to_openai_tool() for t in active_tools] if active_tools else None

            # 3. Call model
            response = await self._client.chat(messages, tools=openai_tools)

            # 4. Append assistant message
            assistant_msg = self._build_assistant_msg(response)
            state.append_message(assistant_msg)

            # 5. No tool calls → done
            if not response.tool_calls:
                self._redact_last_assistant(state)
                await self._store.save(thread_id, state)
                return state

            # Checkpoint: save assistant message (with tool_calls) before execution
            await self._store.save(thread_id, state)

            # 6. Execute tool calls
            tool_round += 1
            for tc in response.tool_calls:
                tool_result = await self._execute_tool(tc, state, thread_id)
                state.append_message({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result,
                })
                await self._store.save(thread_id, state)

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
        if decision in ("allow_once", "allow_always"):
            if pending_tc_id and assistant_msg_idx is not None:
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
                            state.append_message({
                                "role": "tool",
                                "tool_call_id": tc_id,
                                "content": "[pending approval]",
                            })
                            state.pending_approval = {
                                "request_id": pending.get("request_id", ""),
                                "tool_name": "exec_command",
                                "command_preview": pending.get("command_preview", ""),
                                "tool_call_id": tc_id,
                            }
                            await self._store.save(thread_id, state)
                            raise
                        state.append_message({
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": result,
                        })
                        existing_results.add(tc_id)
        else:
            if pending_tc_id:
                state.append_message({
                    "role": "tool",
                    "tool_call_id": pending_tc_id,
                    "content": "[denied] Command execution was denied by user.",
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

    # ------------------------------------------------------------------
    # Message preparation — two modes
    # ------------------------------------------------------------------

    async def _prepare_messages(self, state: AgentState) -> list[dict]:
        """Prepare messages for model call with proactive compression.

        hermes-style: check token budget BEFORE calling the model.
        If over threshold, compress first; otherwise send raw history.
        Large tool outputs are always truncated to keep payload small.
        """
        history = state.messages

        # Always truncate large tool outputs (cheap, no LLM)
        history = self._truncate_large_outputs(history)

        # Proactive compression check
        if self._compressor is not None and self._compressor.should_compress(history):
            history = await self._compressor.compress(history, self._ctx_window_tokens)
        elif self._compressor is None:
            # No compressor configured — do a quick static check
            estimated = _estimate_tokens_simple(history)
            if estimated > int(self._ctx_window_tokens * 0.8):
                history = self._static_fallback(history)

        system_text = self._build_system_prompt(state, self._get_active_tool_defs(state))
        return [{"role": "system", "content": system_text}] + list(history)

    def _truncate_large_outputs(self, messages: list[dict]) -> list[dict]:
        """Truncate large tool outputs to keep payload small. Non-destructive."""
        result = []
        for m in messages:
            if m.get("role") == "tool":
                content = m.get("content", "")
                if isinstance(content, str) and len(content) > 2000:
                    result.append({**m, "content": content[:500] + f"\n... [truncated, {len(content)} chars total]"})
                    continue
            result.append(m)
        return result

    def _static_fallback(self, messages: list[dict]) -> list[dict]:
        """Emergency static truncation when no compressor is configured."""
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

        # Build tool_call_id → name mapping from assistant messages
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
        if channel == "feishu":
            tools = [t for t in tools if not t.name.startswith("qq_")]
        elif channel == "qq":
            feishu_extra = frozenset({"send_image_to_chat", "send_file_to_chat"})
            tools = [t for t in tools if not t.name.startswith("feishu_") and t.name not in feishu_extra]

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

    def _build_system_prompt(self, state: AgentState, active_tools: list[ToolDef]) -> str:
        from src.prompt import build_system_prompt
        return build_system_prompt(
            config=self._config,
            tools=active_tools,
            skills_prompt=self._skills_prompt,
            context_files=self._context_files,
            extra_system_prompt=state.system_prompt,
        )

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

        try:
            result = await tool_def.execute(args)
            try:
                from src.security.redact import redact
                result = redact(result)
            except Exception:
                pass
            return result
        except Exception as e:
            from src.tools.exec import ApprovalNeededError
            if isinstance(e, ApprovalNeededError):
                return await self._handle_approval(e, tc, state, thread_id)
            logger.error("Tool %s failed: %s", tool_name, e)
            return f"[error] {type(e).__name__}: {e}"

    async def _handle_approval(self, error: Exception, tc: Any, state: AgentState, thread_id: str) -> str:
        """Handle approval-needed error by saving state and raising ApprovalPending."""
        from src.tools.approval import get_approval_manager
        mgr = get_approval_manager()
        cmd = getattr(error, "command", "")
        denylisted = getattr(error, "denylisted", False)

        if isinstance(tc, dict):
            tc_id = tc.get("id", "")
        else:
            tc_id = tc.id

        req = mgr.request_approval(
            tool_name="exec_command",
            args=cmd,
            sender_id=state.sender_id,
            chat_id=state.chat_id,
            message_id=state.message_id,
            thread_id=thread_id,
        )

        state.pending_approval = {
            "request_id": req.id,
            "tool_name": "exec_command",
            "command_preview": cmd[:200],
            "tool_call_id": tc_id,
        }
        await self._store.save(thread_id, state)

        raise ApprovalPending(
            thread_id=thread_id,
            request_id=req.id,
            tool_name="exec_command",
            command_preview=cmd[:200],
            denylisted=denylisted,
        )
