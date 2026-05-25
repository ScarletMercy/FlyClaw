"""flyclaw Event Bus - lightweight pub/sub for system events."""

from __future__ import annotations

from src.events.bus import EventBus
from src.events.hooks import HookManager
from src.events.types import Event, EventCategory, EventContext, Subscription

# Module-level singleton
_bus: EventBus | None = None
_hook_manager: HookManager | None = None


def get_event_bus() -> EventBus:
    """Get or create the global event bus singleton."""
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus


def get_hook_manager() -> HookManager:
    """Get or create the global hook manager singleton."""
    global _hook_manager
    if _hook_manager is None:
        _hook_manager = HookManager(get_event_bus())
    return _hook_manager


def reset_event_bus() -> EventBus:
    """Reset the global event bus (for testing)."""
    global _bus, _hook_manager
    _bus = EventBus()
    _hook_manager = HookManager(_bus)
    return _bus


def subscribe(event: str, handler, priority: int = 0):
    """Subscribe a handler to an event on the global bus."""
    return get_event_bus().subscribe(event, handler, priority)


def subscribe_async(event: str, handler, priority: int = 0):
    """Subscribe an async handler to an event on the global bus."""
    return get_event_bus().subscribe_async(event, handler, priority)


def unsubscribe(event: str, handler) -> bool:
    """Unsubscribe a handler from an event on the global bus."""
    return get_event_bus().unsubscribe(event, handler)


def emit(event: str, **context):
    """Emit an event on the global bus."""
    return get_event_bus().emit(event, **context)


async def emit_async(event: str, **context):
    """Asynchronously emit an event on the global bus."""
    return await get_event_bus().emit_async(event, **context)


__all__ = [
    "Event",
    "EventBus",
    "EventCategory",
    "EventContext",
    "HookManager",
    "HookSpec",
    "Subscription",
    "emit",
    "emit_async",
    "get_event_bus",
    "get_hook_manager",
    "reset_event_bus",
    "subscribe",
    "subscribe_async",
    "unsubscribe",
]
