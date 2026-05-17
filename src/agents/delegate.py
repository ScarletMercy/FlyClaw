from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Optional

from src.agent.tooldef import ToolDef

logger = logging.getLogger("myclaw.agents.delegate")


def _register_builtin_tools(config) -> list[ToolDef]:
    from src.tools.registry import get_tool_registry
    return list(get_tool_registry().collect())


def _filter_tools(tools: list[ToolDef], config) -> list[ToolDef]:
    blocked = set(config.delegation.blocked_tools)
    return [t for t in tools if t.name not in blocked]


async def _run_single(
    agent_name: str,
    task: str,
    context: str,
    config,
    run_registry,
    depth: int,
) -> dict:
    """Run a single sub-agent. Returns result dict."""
    from src.agents.registry import get_agent_registry
    from src.agent.state import AgentState, MemoryStateStore
    from src.agent.litegraph_loop import LiteGraphAgentLoop
    from src.agent.client import create_chain

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

    run_id = None
    agent_loop = None
    start = time.monotonic()

    try:
        run_id = await run_registry.start_run(agent_name, task, depth=depth)

        # Progress: child started
        try:
            from src.events import emit_async
            await emit_async("delegate.child_started", agent_name=agent_name, task=task[:100], run_id=run_id)
        except Exception:
            pass

        all_tools = _filter_tools(_register_builtin_tools(config), config)

        if agent_config.tools and agent_config.tools != ["*"]:
            tool_map = {t.name: t for t in all_tools}
            all_tools = [tool_map[p] for p in agent_config.tools if p in tool_map]

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

        state_store = MemoryStateStore()
        agent_loop = LiteGraphAgentLoop(
            client=client,
            tools=all_tools,
            state_store=state_store,
            config=config,
        )

        prompt = agent_config.system_prompt
        if context:
            prompt = f"{prompt}\n\nContext:\n{context}" if prompt else f"Context:\n{context}"

        state = AgentState(
            messages=[{"role": "user", "content": task}],
            system_prompt=prompt,
        )

        max_rounds = config.delegation.max_iterations
        timeout = config.delegation.child_timeout_seconds

        # Check for interrupt before starting
        if run_id and run_registry.is_interrupt_requested(run_id):
            await run_registry.complete_run(run_id, result="[interrupted]")
            return {
                "agent_name": agent_name,
                "status": "interrupted",
                "result": "Interrupted before start.",
                "duration_seconds": 0,
            }

        result_state = await asyncio.wait_for(
            agent_loop.run(state, f"delegate:{agent_name}:{depth}", max_rounds=max_rounds),
            timeout=timeout,
        )

        result_text = ""
        for msg in reversed(result_state.messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                result_text = msg["content"]
                break

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
            await run_registry.fail_run(run_id, error=f"Timeout after {timeout}s")
        logger.warning("Sub-agent '%s' timed out after %ds", agent_name, timeout)
        return {
            "agent_name": agent_name,
            "status": "timeout",
            "result": f"Agent '{agent_name}' exceeded {timeout}s limit.",
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
        if agent_loop:
            agent_loop.close()
        # Progress: child done
        try:
            from src.events import emit_async
            await emit_async("delegate.child_completed", agent_name=agent_name)
        except Exception:
            pass


async def delegate_task(agent_name: str, task: str, context: str = "") -> str:
    """Delegate a task to a specialized sub-agent. Use for tasks requiring specific expertise.

    Args:
        agent_name: Name of the sub-agent (e.g. "research", "coder", "reviewer").
        task: Clear description of the task to delegate.
        context: Optional background information for the sub-agent.
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
        result = await _run_single(agent_name, task, context, config, run_registry, current_depth + 1)
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

    # Parse tasks
    try:
        parsed = json.loads(tasks) if isinstance(tasks, str) else tasks
    except json.JSONDecodeError as e:
        return f"[error] Invalid JSON: {e}"

    if not isinstance(parsed, list) or not parsed:
        return "[error] tasks must be a non-empty JSON array."

    max_concurrent = config.delegation.max_concurrent
    if len(parsed) > max_concurrent:
        return f"[error] Too many tasks ({len(parsed)}). Max concurrent: {max_concurrent}."

    # Validate each task
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


def get_tools() -> list[ToolDef]:
    return [ToolDef.from_function(delegate_task), ToolDef.from_function(delegate_batch)]
