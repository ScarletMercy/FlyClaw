"""User-defined hook system for flyclaw event bus."""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable

from src.events.bus import EventBus

logger = logging.getLogger("flyclaw.events.hooks")


class HookSpec:
    """Specification for a user-defined hook."""

    def __init__(self, event: str, handler_path: str, enabled: bool = True):
        self.event = event
        self.handler_path = handler_path
        self.enabled = enabled
        self._handler: Callable | None = None

    def load_handler(self) -> Callable | None:
        """Load the handler from the specified path."""
        if self._handler is not None:
            return self._handler

        path = Path(self.handler_path).expanduser()
        if not path.exists():
            logger.warning("Hook handler not found: %s", self.handler_path)
            return None

        try:
            spec = importlib.util.spec_from_file_location(f"hook_{path.stem}", str(path))
            if spec is None or spec.loader is None:
                logger.warning("Cannot load hook spec: %s", self.handler_path)
                return None

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Handler must be named 'handle'
            handler = getattr(module, "handle", None)
            if handler is None:
                logger.warning("No 'handle' function in %s", self.handler_path)
                return None

            self._handler = handler
            return handler
        except Exception as e:
            logger.error("Failed to load hook handler %s: %s", self.handler_path, e)
            return None


class HookManager:
    """Manages user-defined hooks from configuration."""

    def __init__(self, bus: EventBus):
        self._bus = bus
        self._hooks: list[HookSpec] = []
        self._subscriptions = []

    def load_from_config(self, config: Any) -> int:
        """Load hooks from application config.

        Expected config format:
            hooks:
              hooks:
                - event: "tool.*"
                  handler: "~/.flyclaw/hooks/audit_tool.py"
                  enabled: true

        Or legacy flat list:
            hooks:
              - event: "tool.*"
                handler: "~/.flyclaw/hooks/audit_tool.py"

        Returns:
            Number of hooks loaded
        """
        hooks_config = getattr(config, "hooks", None)
        if hooks_config is None:
            return 0

        # Handle HooksConfig object (new format)
        if hasattr(hooks_config, "hooks"):
            hooks_list = hooks_config.hooks
        elif isinstance(hooks_config, list):
            hooks_list = hooks_config
        else:
            return 0

        if not hooks_list:
            return 0

        count = 0
        for hook_cfg in hooks_list:
            if isinstance(hook_cfg, dict):
                event = hook_cfg.get("event", "")
                handler = hook_cfg.get("handler", "")
                enabled = hook_cfg.get("enabled", True)
            elif hasattr(hook_cfg, "event"):
                event = hook_cfg.event
                handler = hook_cfg.handler
                enabled = hook_cfg.enabled
            else:
                continue

            if not event or not handler:
                logger.warning("Skipping invalid hook config: %s", hook_cfg)
                continue

            spec = HookSpec(event=event, handler_path=handler, enabled=enabled)
            self._hooks.append(spec)

            if enabled:
                handler_fn = spec.load_handler()
                if handler_fn is not None:
                    import asyncio

                    is_async = asyncio.iscoroutinefunction(handler_fn)
                    if is_async:
                        sub = self._bus.subscribe_async(event, handler_fn)
                    else:
                        sub = self._bus.subscribe(event, handler_fn)
                    self._subscriptions.append(sub)
                    logger.info("Hook loaded: %s -> %s", event, handler)
                    count += 1
                else:
                    logger.warning("Failed to load hook handler: %s", handler)

        return count

    def unload_all(self) -> None:
        """Unload all hooks and remove subscriptions."""
        for sub in self._subscriptions:
            self._bus.unsubscribe(sub.event, sub.handler)
        self._subscriptions.clear()
        self._hooks.clear()
        logger.info("All hooks unloaded")

    @property
    def hook_count(self) -> int:
        return len(self._hooks)

    @property
    def active_count(self) -> int:
        return len(self._subscriptions)
