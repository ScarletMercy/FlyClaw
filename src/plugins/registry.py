from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional

from src.agent.tooldef import ToolDef

from .loader import HookResult, PluginRecord, discover_plugins

logger = logging.getLogger("myclaw.plugins.registry")


class PluginRegistry:
    def __init__(self):
        self._records: list[PluginRecord] = []
        self._tools: list[ToolDef] = []
        self._hooks: dict[str, list[Callable]] = {}

    def register_plugin(self, record: PluginRecord) -> None:
        existing = next((r for r in self._records if r.manifest.id == record.manifest.id), None)
        if existing:
            logger.warning("Plugin '%s' already registered, skipping duplicate", record.manifest.id)
            return
        self._records.append(record)
        self._tools.extend(record.tools)
        for hook_name, funcs in record.hooks.items():
            self._hooks.setdefault(hook_name, []).extend(funcs)
        logger.info(
            "Registered plugin '%s': %d tools, %d hooks",
            record.manifest.id,
            len(record.tools),
            sum(len(v) for v in record.hooks.values()),
        )

    def get_all_tools(self) -> list[ToolDef]:
        return list(self._tools)

    def collect_tools(self) -> list[ToolDef]:
        return list(self._tools)

    async def run_hooks(self, hook_name: str, **kwargs) -> list[HookResult]:
        funcs = self._hooks.get(hook_name, [])
        results = []
        for func in funcs:
            try:
                result = func(**kwargs)
                if asyncio.iscoroutine(result):
                    result = await result
                if isinstance(result, dict):
                    result = HookResult(**result)
                elif not isinstance(result, HookResult):
                    result = HookResult()
                results.append(result)
                if result.block:
                    break
            except Exception as e:
                logger.error("Hook %s error in plugin: %s", hook_name, e)
        return results

    @property
    def plugin_count(self) -> int:
        return len(self._records)

    @property
    def tool_count(self) -> int:
        return len(self._tools)

    def list_plugins(self) -> list[dict]:
        return [
            {
                "id": r.manifest.id,
                "name": r.manifest.name,
                "version": r.manifest.version,
                "tools": len(r.tools),
                "hooks": list(r.hooks.keys()),
            }
            for r in self._records
        ]


_registry: Optional[PluginRegistry] = None


def get_plugin_registry() -> PluginRegistry:
    global _registry
    if _registry is None:
        _registry = PluginRegistry()
    return _registry


def init_plugin_registry(extra_dirs: list[str] | None = None) -> PluginRegistry:
    global _registry
    _registry = PluginRegistry()
    records = discover_plugins(extra_dirs)
    for record in records:
        _registry.register_plugin(record)
    return _registry
