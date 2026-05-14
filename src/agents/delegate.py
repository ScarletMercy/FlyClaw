from __future__ import annotations

import asyncio
import logging
from typing import Optional

from src.agent.tooldef import ToolDef

logger = logging.getLogger("myclaw.agents.delegate")


def _register_builtin_tools(config) -> list[ToolDef]:
    from src.tools.registry import get_tool_registry

    return list(get_tool_registry().collect())


async def delegate_task(agent_name: str, task: str) -> str:
    """Delegate a task to a specialized sub-agent. Use this when the task requires
    specific expertise that a sub-agent provides (e.g. research, coding, review).

    Args:
        agent_name: Name of the sub-agent to delegate to (e.g. "research", "coder", "reviewer").
        task: Clear description of the task to delegate.
    """
    from src.agents.registry import get_agent_registry
    from src.agents.run_registry import (
        get_current_depth,
        set_current_depth,
        get_run_registry,
    )
    from src.agent.state import AgentState, MemoryStateStore
    from src.agent.loop import AgentLoop
    from src.agent.client import create_client, create_chain
    from src.config import load_config

    registry = get_agent_registry()
    agent_config = registry.get(agent_name)
    if agent_config is None:
        available = [a["name"] for a in registry.list_agents()]
        return f"Unknown agent: '{agent_name}'. Available agents: {', '.join(available)}"

    current_depth = get_current_depth()
    config = load_config()
    max_depth = getattr(config.agents, "subagent_max_depth", 2)
    if current_depth >= max_depth:
        return f"Sub-agent depth limit reached ({current_depth}/{max_depth}). Cannot delegate further."

    run_id = None
    run_registry = get_run_registry()
    set_current_depth(current_depth + 1)

    try:
        run_id = await run_registry.start_run(
            agent_name, task, depth=current_depth + 1
        )

        all_tools = _register_builtin_tools(config)

        if agent_config.tools and agent_config.tools != ["*"]:
            tool_map = {t.name: t for t in all_tools}
            all_tools = [tool_map[p] for p in agent_config.tools if p in tool_map]

        if agent_config.model:
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
        agent_loop = AgentLoop(
            client=client,
            tools=all_tools,
            state_store=state_store,
            config=config,
        )

        state = AgentState(
            messages=[{"role": "user", "content": task}],
            system_prompt=agent_config.system_prompt,
        )

        max_rounds = 10
        estimated_tokens_per_round = 1000
        max_total_tokens = getattr(config.model, "max_tokens", 100000)
        safe_rounds = (
            min(max_rounds, max_total_tokens // estimated_tokens_per_round)
            if max_total_tokens
            else max_rounds
        )
        if safe_rounds < 1:
            safe_rounds = 1

        logger.info(
            "Delegating to agent '%s' with max %d rounds (token limit: %d)",
            agent_name,
            safe_rounds,
            max_total_tokens,
        )

        result_state = await agent_loop.run(
            state, f"delegate:{agent_name}:{current_depth + 1}", max_rounds=safe_rounds
        )

        result_text = ""
        for msg in reversed(result_state.messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                result_text = msg["content"]
                break

        await run_registry.complete_run(run_id, result=result_text)
        return result_text

    except Exception as e:
        logger.error(
            "Delegate task failed for agent '%s': %s", agent_name, e, exc_info=True
        )
        if run_id:
            await run_registry.fail_run(run_id, error=str(e))
        return f"[error] Delegation to '{agent_name}' failed: {type(e).__name__}: {e}"
    finally:
        set_current_depth(current_depth)
