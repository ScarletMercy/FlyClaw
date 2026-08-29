from __future__ import annotations

import logging
from typing import Optional


from src.config import AgentSubconfig

logger = logging.getLogger("flyclaw.agents.registry")


class AgentRegistry:
    def __init__(self):
        self._agents: dict[str, AgentSubconfig] = {}

    def register(self, name: str, config: AgentSubconfig) -> None:
        self._agents[name] = config
        logger.info("Sub-agent registered: %s (tools: %s)", name, config.tools)

    def get(self, name: str) -> Optional[AgentSubconfig]:
        return self._agents.get(name)

    def list_agents(self) -> list[dict]:
        return [
            {
                "name": name,
                "description": cfg.description,
                "tools": cfg.tools,
                "model": cfg.model,
            }
            for name, cfg in self._agents.items()
        ]

    @property
    def count(self) -> int:
        return len(self._agents)

    def has_agent(self, name: str) -> bool:
        return name in self._agents


# ── Module-level singleton — delegates to ServiceContainer ──

from src._container import get_container


def get_agent_registry() -> AgentRegistry:
    return get_container().agent_registry
