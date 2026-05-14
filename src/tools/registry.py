from __future__ import annotations
import logging
from typing import Optional
from src.agent.tooldef import ToolDef

logger = logging.getLogger("myclaw.tools.registry")

class ToolRegistry:
    def __init__(self) -> None:
        self._tools: list[ToolDef] = []
        self._registrations: list[object] = []
    def register(self, tool_def: ToolDef) -> None:
        self._tools.append(tool_def)
    def collect(self) -> list[ToolDef]:
        return list(self._tools)

_registry: Optional[ToolRegistry] = None

def get_tool_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry
