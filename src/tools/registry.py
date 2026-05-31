from __future__ import annotations
import logging
from src.agent.tooldef import ToolDef

logger = logging.getLogger("flyclaw.tools.registry")


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: list[ToolDef] = []

    def register(self, tool_def: ToolDef) -> None:
        self._tools.append(tool_def)

    def collect(self) -> list[ToolDef]:
        return list(self._tools)


# ── Module-level singleton — delegates to ServiceContainer ──

from src._container import get_container


def get_tool_registry() -> ToolRegistry:
    return get_container().tool_registry
