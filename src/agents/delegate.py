from __future__ import annotations

import asyncio
import logging
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, tool

logger = logging.getLogger("myclaw.agents.delegate")


@tool
async def delegate_task(agent_name: str, task: str) -> str:
    """Delegate a task to a specialized sub-agent. Use this when the task requires
    specific expertise that a sub-agent provides (e.g. research, coding, review).

    Args:
        agent_name: Name of the sub-agent to delegate to (e.g. "research", "coder", "reviewer").
        task: Clear description of the task to delegate.
    """
    from src.agents.registry import get_agent_registry

    registry = get_agent_registry()
    agent_config = registry.get(agent_name)
    if agent_config is None:
        available = [a["name"] for a in registry.list_agents()]
        return f"Unknown agent: '{agent_name}'. Available agents: {', '.join(available)}"

    try:
        from src.graph import create_model, collect_tools
        from src.config import load_config

        # Load isolated config for sub-agent (don't share main agent config)
        config = load_config()
        all_tools = collect_tools(config)

        # Token usage monitoring
        max_rounds = 10
        estimated_tokens_per_round = 1000  # Conservative estimate
        max_total_tokens = getattr(config.model, "max_tokens", 100000)
        safe_rounds = min(max_rounds, max_total_tokens // estimated_tokens_per_round) if max_total_tokens else max_rounds

        if safe_rounds < 1:
            safe_rounds = 1

        logger.info(
            "Delegating to agent '%s' with max %d rounds (token limit: %d)",
            agent_name,
            safe_rounds,
            max_total_tokens,
        )

        if agent_config.tools and agent_config.tools != ["*"]:
            tool_map = {t.name: t for t in all_tools}
            filtered = []
            for pattern in agent_config.tools:
                if pattern in tool_map:
                    filtered.append(tool_map[pattern])
            all_tools = filtered

        if agent_config.model:
            from langchain.chat_models import init_chat_model

            model = init_chat_model(
                agent_config.model,
                model_provider=config.model.provider,
                temperature=config.model.temperature,
            )
        else:
            model = create_model(config)

        messages = [
            SystemMessage(content=agent_config.system_prompt),
            HumanMessage(content=task),
        ]

        if all_tools:
            model = model.bind_tools(all_tools)
            for _round in range(safe_rounds):
                response = await model.ainvoke(messages)
                messages.append(response)

                if not response.tool_calls:
                    break

                for tc in response.tool_calls:
                    tool_obj = next((t for t in all_tools if t.name == tc["name"]), None)
                    if tool_obj is None:
                        from langchain_core.messages import ToolMessage

                        messages.append(
                            ToolMessage(
                                content=f"Tool not found: {tc['name']}", tool_call_id=tc["id"]
                            )
                        )
                        continue
                    try:
                        result = await tool_obj.ainvoke(tc["args"])
                        from langchain_core.messages import ToolMessage

                        messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
                    except Exception as e:
                        from langchain_core.messages import ToolMessage

                        messages.append(ToolMessage(content=f"Error: {e}", tool_call_id=tc["id"]))
        else:
            response = await model.ainvoke(messages)

        if isinstance(response.content, str):
            return response.content
        return str(response.content)

    except Exception as e:
        logger.error("Delegate task failed for agent '%s': %s", agent_name, e, exc_info=True)
        return f"[error] Delegation to '{agent_name}' failed: {type(e).__name__}: {e}"
