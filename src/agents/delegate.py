from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Optional

from src.agent.tooldef import ToolDef

logger = logging.getLogger("flyclaw.agents.delegate")

_HEARTBEAT_INTERVAL = 30
_HEARTBEAT_STALE_IDLE_SECONDS = 450

_ESSENTIAL_TOOLS = frozenset([
    "write_file", "edit_file", "read_file", "grep", "glob", "list_dir",
])


def _register_builtin_tools(config) -> list[ToolDef]:
    from src._container import get_container
    container = get_container()
    if container.agent_loop and container.agent_loop._tools:
        return list(container.agent_loop._tools)
    return []


def _filter_tools(tools: list[ToolDef], config) -> list[ToolDef]:
    blocked = set(config.delegation.blocked_tools)
    return [t for t in tools if t.name not in blocked]


def _resolve_child_timeout(
    explicit_timeout: int | None,
    config,
) -> float:
    """Resolve effective child timeout with priority chain and guards.

    Priority: explicit param > env var > config value > hardcoded default.
    Floor: child_timeout_floor (default 30s).
    Ceiling: max(default * 3, 1800).
    """
    default_timeout = float(config.delegation.child_timeout_seconds)
    floor = float(getattr(config.delegation, "child_timeout_floor", 30))

    # Step 1: determine raw value from priority chain
    if explicit_timeout and explicit_timeout > 0:
        raw = float(explicit_timeout)
    else:
        env_val = os.getenv("FLYCLAW_CHILD_TIMEOUT_SECONDS")
        if env_val:
            try:
                raw = float(env_val)
            except (TypeError, ValueError):
                raw = default_timeout
        else:
            raw = default_timeout

    # Step 2: apply floor and ceiling
    ceiling = max(default_timeout * 3, 1800)
    return max(floor, min(raw, ceiling))


async def _run_single(
    agent_name: str,
    task: str,
    context: str,
    config,
    run_registry,
    depth: int,
    timeout: int | None = None,
) -> dict:
    """Run a single sub-agent. Returns result dict."""
    from src.agents.registry import get_agent_registry
    from src.agent.state import AgentState, MemoryStateStore
    from src.agent.loop import AgentLoop
    from src.agent.client import create_chain
    from src.tools.exec import _current_agent_context

    registry = get_agent_registry()
    agent_config = registry.get(agent_name)
    if agent_config is None:
        available = [a["name"] for a in registry.list_agents()]
        return {
            "agent_name": agent_name,
            "status": "error",
            "result": f"Unknown agent: '{agent_name}'. Available: {', '.join(available)}",
            "duration_seconds": 0,
        }

    parent_ctx = _current_agent_context.get({})
    parent_thread_id = parent_ctx.get("parent_thread_id", "")

    run_id = None
    agent_loop = None
    start = time.monotonic()
    effective_timeout = _resolve_child_timeout(timeout, config)

    try:
        run_id = await run_registry.start_run(agent_name, task, depth=depth)

        try:
            from src.events import emit_async
            await emit_async("delegate.child_started", agent_name=agent_name, task=task[:100], run_id=run_id)
        except Exception:
            pass

        all_tools = _filter_tools(_register_builtin_tools(config), config)

        if agent_config.tools and agent_config.tools != ["*"]:
            tool_map = {t.name: t for t in all_tools}
            filtered = [tool_map[p] for p in agent_config.tools if p in tool_map]
            filtered_names = {t.name for t in filtered}
            for name in _ESSENTIAL_TOOLS:
                if name not in filtered_names and name in tool_map:
                    filtered.append(tool_map[name])
            all_tools = filtered

        if agent_config.model:
            from src.agent.client import create_client
            client = create_client(
                config.model.provider,
                agent_config.model,
                config.model.temperature,
                base_url=config.model.base_url,
                api_key=config.model.api_key,
            )
        else:
            client = create_chain(config)

        skills_prompt = ""
        try:
            from src._container import get_container
            container = get_container()
            if container.agent_loop:
                skills_prompt = container.agent_loop._skills_prompt
        except Exception:
            pass

        state_store = MemoryStateStore()
        agent_loop = AgentLoop(
            client=client,
            tools=all_tools,
            state_store=state_store,
            config=config,
            skills_prompt=skills_prompt,
            context_window_tokens=config.model.context_window,
        )

        agent_loop._auto_deny_approval = True

        prompt = agent_config.system_prompt
        if context:
            prompt = f"{prompt}\n\nContext:\n{context}" if prompt else f"Context:\n{context}"

        state = AgentState(
            messages=[{"role": "user", "content": task}],
            system_prompt=prompt,
            sender_id=parent_ctx.get("sender_id", ""),
            chat_id=parent_ctx.get("chat_id", ""),
            channel=parent_ctx.get("channel", ""),
        )

        max_rounds = config.delegation.max_iterations
        child_thread_id = f"delegate:{agent_name}:{depth}"

        if run_id and run_registry.is_interrupt_requested(run_id):
            await run_registry.complete_run(run_id, result="[interrupted]")
            return {
                "agent_name": agent_name,
                "status": "interrupted",
                "result": "Interrupted before start.",
                "duration_seconds": 0,
            }

        # Background monitors
        monitor_task: asyncio.Task | None = None
        if run_id:
            async def _monitor(
                _run_id: str,
                _loop: AgentLoop,
                _child_tid: str,
                _interval: float = 1.5,
            ):
                last_touch = time.monotonic()
                while True:
                    await asyncio.sleep(_interval)
                    now = time.monotonic()

                    # Check interrupt
                    if run_registry.is_interrupt_requested(_run_id):
                        flag = _loop._store.get_interrupt_flag(_child_tid)
                        flag.interrupt("Interrupted by parent")
                        return

                    # Heartbeat: touch run activity timestamp every HEARTBEAT_INTERVAL
                    if now - last_touch >= _HEARTBEAT_INTERVAL:
                        run_registry.touch(_run_id)
                        last_touch = now

            monitor_task = asyncio.create_task(
                _monitor(run_id, agent_loop, child_thread_id)
            )

        # Stale detection heartbeat
        stale_task: asyncio.Task | None = None
        if run_id:
            async def _stale_detector(_run_id: str):
                while True:
                    await asyncio.sleep(_HEARTBEAT_INTERVAL)
                    run = run_registry._runs.get(_run_id)
                    if not run or run["status"] != "running":
                        return
                    last = run.get("last_activity_at") or run["started_at"]
                    idle = time.time() - last
                    if idle > _HEARTBEAT_STALE_IDLE_SECONDS:
                        logger.warning(
                            "Sub-agent run '%s' appears stale: no activity for %.0fs",
                            _run_id, idle,
                        )

            stale_task = asyncio.create_task(_stale_detector(run_id))

        # Forward key tool events at delegation layer
        event_unsubs: list = []
        if parent_thread_id:
            from src.events import subscribe_async

            async def _make_forwarder(child_tid: str, parent_tid: str):
                async def _forward(event: str, **kwargs):
                    if kwargs.get("thread_id") != child_tid:
                        return
                    from src.events import emit_async
                    await emit_async(event, **{**kwargs, "thread_id": parent_tid, "_delegated": True})
                return _forward

            for evt in ("tool.exec_started", "tool.exec_completed", "tool.exec_failed"):
                fwd = await _make_forwarder(child_thread_id, parent_thread_id)
                unsub = subscribe_async(evt, fwd)
                event_unsubs.append(unsub)

        from src.tools.exec import _current_thread_id
        _child_tid_token = _current_thread_id.set(child_thread_id)
        try:
            result_state = await asyncio.wait_for(
                agent_loop.run(state, child_thread_id, max_rounds=max_rounds),
                timeout=effective_timeout,
            )
        finally:
            _current_thread_id.reset(_child_tid_token)
            for t in (monitor_task, stale_task):
                if t:
                    t.cancel()
                    try:
                        await t
                    except (asyncio.CancelledError, Exception):
                        pass
            for unsub in event_unsubs:
                try:
                    unsub()
                except Exception:
                    pass

        # Result extraction with fallbacks
        result_text = ""
        for msg in reversed(result_state.messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                result_text = msg["content"]
                break

        if not result_text:
            for msg in reversed(result_state.messages):
                if msg.get("role") == "assistant":
                    for tc in msg.get("tool_calls", []):
                        tc_id = tc.get("id", "")
                        if tc_id:
                            for m in result_state.messages:
                                if m.get("role") == "tool" and m.get("tool_call_id") == tc_id:
                                    content = m.get("content", "")
                                    if content and not content.startswith("[error]"):
                                        result_text = content[:2000]
                                        break
                            if result_text:
                                break
                    break

        if not result_text:
            tool_parts = []
            for msg in result_state.messages:
                if msg.get("role") == "tool" and msg.get("content"):
                    tool_parts.append(msg["content"][:500])
            if tool_parts:
                result_text = "[No final summary] Tool results:\n" + "\n---\n".join(tool_parts[-3:])

        duration = round(time.monotonic() - start, 2)
        await run_registry.complete_run(run_id, result=result_text[:500])

        return {
            "agent_name": agent_name,
            "status": "completed",
            "result": result_text,
            "duration_seconds": duration,
        }

    except asyncio.TimeoutError:
        duration = round(time.monotonic() - start, 2)
        if run_id:
            await run_registry.timeout_run(run_id, error=f"Timeout after {effective_timeout}s")
        logger.warning("Sub-agent '%s' timed out after %.0fs", agent_name, effective_timeout)
        return {
            "agent_name": agent_name,
            "status": "timeout",
            "result": f"Agent '{agent_name}' exceeded {effective_timeout:.0f}s limit.",
            "duration_seconds": duration,
        }

    except Exception as e:
        duration = round(time.monotonic() - start, 2)
        logger.error("Delegate failed for '%s': %s", agent_name, e, exc_info=True)
        if run_id:
            await run_registry.fail_run(run_id, error=str(e))
        return {
            "agent_name": agent_name,
            "status": "error",
            "result": f"{type(e).__name__}: {e}",
            "duration_seconds": duration,
        }

    finally:
        try:
            from src.events import emit_async
            await emit_async("delegate.child_completed", agent_name=agent_name)
        except Exception:
            pass


async def delegate_task(agent_name: str, task: str, context: str = "",
                       timeout: int | None = None) -> str:
    """Delegate a task to a specialized sub-agent. Use for tasks requiring specific expertise.

    Args:
        agent_name: Name of the sub-agent (e.g. "research", "coder", "reviewer").
        task: Clear description of the task to delegate.
        context: Optional background information for the sub-agent.
        timeout: Optional timeout in seconds for the sub-agent run.
    """
    from src.agents.run_registry import get_current_depth, set_current_depth, get_run_registry
    from src.config import load_config

    config = load_config()
    if not config.delegation.enabled:
        return "Delegation is disabled in config."

    current_depth = get_current_depth()
    max_depth = getattr(config.agents, "subagent_max_depth", 2)
    if current_depth >= max_depth:
        return f"Sub-agent depth limit reached ({current_depth}/{max_depth}). Cannot delegate further."

    run_registry = get_run_registry()
    set_current_depth(current_depth + 1)
    try:
        result = await _run_single(agent_name, task, context, config, run_registry, current_depth + 1, timeout=timeout)
        if result["status"] == "completed":
            return result["result"]
        return f"[{result['status']}] {result['result']}"
    finally:
        set_current_depth(current_depth)


async def delegate_batch(tasks: str) -> str:
    """Delegate multiple tasks to sub-agents in parallel. Returns JSON results.

    Args:
        tasks: JSON array of task objects. Each: {"agent_name": "...", "task": "...", "context": "..."}
    """
    from src.agents.run_registry import get_current_depth, set_current_depth, get_run_registry
    from src.config import load_config

    config = load_config()
    if not config.delegation.enabled:
        return "Delegation is disabled in config."

    try:
        parsed = json.loads(tasks) if isinstance(tasks, str) else tasks
    except json.JSONDecodeError as e:
        return f"[error] Invalid JSON: {e}"

    if not isinstance(parsed, list) or not parsed:
        return "[error] tasks must be a non-empty JSON array."

    max_concurrent = config.delegation.max_concurrent
    if len(parsed) > max_concurrent:
        return f"[error] Too many tasks ({len(parsed)}). Max concurrent: {max_concurrent}."

    for i, t in enumerate(parsed):
        if not isinstance(t, dict) or not t.get("agent_name") or not t.get("task"):
            return f"[error] Task {i} must have 'agent_name' and 'task' fields."

    current_depth = get_current_depth()
    max_depth = getattr(config.agents, "subagent_max_depth", 2)
    if current_depth >= max_depth:
        return f"Sub-agent depth limit reached ({current_depth}/{max_depth}). Cannot delegate further."

    run_registry = get_run_registry()
    set_current_depth(current_depth + 1)
    overall_start = time.monotonic()

    try:
        coros = [
            _run_single(
                t["agent_name"],
                t["task"],
                t.get("context", ""),
                config,
                run_registry,
                current_depth + 1,
                timeout=t.get("timeout"),
            )
            for t in parsed
        ]
        results = await asyncio.gather(*coros, return_exceptions=True)

        final_results = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                final_results.append({
                    "agent_name": parsed[i]["agent_name"],
                    "status": "error",
                    "result": f"{type(r).__name__}: {r}",
                    "duration_seconds": 0,
                })
            else:
                final_results.append(r)

        total_duration = round(time.monotonic() - overall_start, 2)
        return json.dumps({
            "results": final_results,
            "total_duration_seconds": total_duration,
        }, ensure_ascii=False)

    finally:
        set_current_depth(current_depth)


async def _delegate_unified(agent_name: str = "", task: str = "", context: str = "",
                            tasks: str = "", timeout: int | None = None) -> str:
    if tasks:
        return await delegate_batch(tasks)
    if not agent_name or not task:
        return "[error] Provide either (agent_name + task) for single delegation or tasks JSON array for batch."
    return await delegate_task(agent_name, task, context, timeout=timeout)


def get_tools() -> list[ToolDef]:
    return [
        ToolDef.from_schema(
            name="delegate_task",
            description=(
                "将子任务委派给专业代理。支持单个和批量模式。\n\n"
                "单个任务: 提供 agent_name + task，可选 context。\n"
                "批量并行: 提供 tasks JSON 数组，每个元素: {\"agent_name\": \"...\", \"task\": \"...\", \"context\": \"...\"}\n\n"
                "可用代理: research, coder, reviewer（用 agent_name 指定）。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "description": "子代理名称，如 research、coder、reviewer（单个任务时必填）",
                    },
                    "task": {
                        "type": "string",
                        "description": "任务描述（单个任务时必填）",
                    },
                    "context": {
                        "type": "string",
                        "description": "背景信息（可选）",
                    },
                    "tasks": {
                        "type": "string",
                        "description": "批量任务的 JSON 数组，每个元素: {\"agent_name\": \"...\", \"task\": \"...\", \"context\": \"...\"}（批量模式时必填）",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "超时时间（秒），可选。不传则使用全局默认值。复杂任务可适当增大。环境变量 FLYCLAW_CHILD_TIMEOUT_SECONDS 可覆盖默认值。",
                    },
                },
                "required": [],
            },
            fn=_delegate_unified,
        ),
    ]
