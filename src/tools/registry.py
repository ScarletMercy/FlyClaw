from __future__ import annotations

import logging
from typing import Callable, Optional

from langchain_core.tools import BaseTool

logger = logging.getLogger("myclaw.tools.registry")


class ToolRegistry:
    def __init__(self) -> None:
        self._registrations: list[Callable[[], list[BaseTool]]] = []

    def register(self, func: Callable[[], list[BaseTool]]) -> None:
        self._registrations.append(func)
        logger.debug("Registered tool collector: %s", func.__name__)

    def collect(self) -> list[BaseTool]:
        tools: list[BaseTool] = []
        for func in self._registrations:
            try:
                collected = func()
                tools.extend(collected)
            except Exception as e:
                logger.warning("Tool collector %s failed: %s", func.__name__, e)
        return tools


_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry
