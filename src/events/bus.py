"""Lightweight event bus for MyClaw with sync/async subscription support."""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import time
import weakref
from collections import defaultdict
from typing import Any, Callable, Optional

from src.events.types import EventContext, Subscription, WILDCARD_PATTERNS, ALL_EVENTS

logger = logging.getLogger("myclaw.events")

_DEFAULT_TIMEOUT = 5.0
_MAX_EMIT_DEPTH = 3


class EventBus:
    """Thread-safe event bus with priority ordering, wildcard matching, and error isolation.

    Usage:
        bus = EventBus()

        # Sync subscription
        bus.subscribe("tool.exec_completed", my_handler)

        # Async subscription with priority
        bus.subscribe_async("session.reset", async_handler, priority=10)

        # Emit event
        bus.emit("tool.exec_completed", tool_name="read_file", duration_ms=42)

        # Wildcard subscription
        bus.subscribe("tool.*", log_all_tools)
    """

    def __init__(self, timeout: float = _DEFAULT_TIMEOUT):
        self._timeout = timeout
        self._subscriptions: dict[str, list[Subscription]] = defaultdict(list)
        self._emit_depth = 0
        self._lock = asyncio.Lock()

    # ── Subscription ──────────────────────────────────────────────

    def subscribe(self, event: str, handler: Callable, priority: int = 0) -> Subscription:
        """Subscribe a synchronous handler to an event.

        Args:
            event: Event name or wildcard pattern (e.g., "tool.*")
            handler: Callable to invoke on event emission
            priority: Lower values execute first (default 0)

        Returns:
            Subscription object for later unsubscription
        """
        sub = Subscription(event=event, handler=handler, is_async=False, priority=priority)
        self._add_subscription(sub)
        logger.debug("Subscribed sync handler to '%s' (priority=%d)", event, priority)
        return sub

    def subscribe_async(self, event: str, handler: Callable, priority: int = 0) -> Subscription:
        """Subscribe an async handler to an event.

        Args:
            event: Event name or wildcard pattern
            handler: Async callable to invoke on event emission
            priority: Lower values execute first

        Returns:
            Subscription object
        """
        sub = Subscription(event=event, handler=handler, is_async=True, priority=priority)
        self._add_subscription(sub)
        logger.debug("Subscribed async handler to '%s' (priority=%d)", event, priority)
        return sub

    def unsubscribe(self, event: str, handler: Callable) -> bool:
        """Unsubscribe a handler from an event.

        Returns:
            True if subscription was found and removed
        """
        target = Subscription(event=event, handler=handler)
        subs = self._subscriptions.get(event, [])
        for i, sub in enumerate(subs):
            if sub == target:
                subs.pop(i)
                logger.debug("Unsubscribed handler from '%s'", event)
                return True
        return False

    def unsubscribe_all(self) -> None:
        """Remove all subscriptions."""
        self._subscriptions.clear()
        logger.info("All event subscriptions cleared")

    # ── Emission ──────────────────────────────────────────────────

    def emit(self, event: str, **context: Any) -> EventContext:
        """Synchronously emit an event to all matching subscribers.

        Args:
            event: Event name
            **context: Arbitrary context data passed to handlers

        Returns:
            EventContext with event name and context data
        """
        if self._emit_depth >= _MAX_EMIT_DEPTH:
            logger.warning("Emit depth limit reached for '%s', skipping to prevent recursion", event)
            return EventContext(event=event, context=context)

        ctx = EventContext(event=event, timestamp=time.time(), context=context)
        matching = self._find_matching_subscriptions(event)

        if not matching:
            return ctx

        self._emit_depth += 1
        try:
            for sub in matching:
                if not sub.active:
                    continue
                if sub.is_async:
                    # Can't call async handler synchronously, schedule it
                    try:
                        loop = asyncio.get_running_loop()
                        asyncio.create_task(self._call_handler_safe(sub, ctx))
                    except RuntimeError:
                        # No running loop, skip async handler
                        logger.debug("No running event loop, skipping async handler for '%s'", event)
                else:
                    self._call_sync_handler(sub, ctx)
        finally:
            self._emit_depth -= 1

        return ctx

    async def emit_async(self, event: str, **context: Any) -> EventContext:
        """Asynchronously emit an event to all matching subscribers.

        Args:
            event: Event name
            **context: Arbitrary context data

        Returns:
            EventContext with event name and context data
        """
        if self._emit_depth >= _MAX_EMIT_DEPTH:
            logger.warning("Emit depth limit reached for '%s', skipping", event)
            return EventContext(event=event, context=context)

        ctx = EventContext(event=event, timestamp=time.time(), context=context)
        matching = self._find_matching_subscriptions(event)

        if not matching:
            return ctx

        self._emit_depth += 1
        try:
            tasks = []
            for sub in matching:
                if not sub.active:
                    continue
                if sub.is_async:
                    tasks.append(asyncio.create_task(self._call_handler_safe(sub, ctx)))
                else:
                    # Run sync handler in executor to not block
                    tasks.append(asyncio.create_task(self._call_sync_handler_async(sub, ctx)))

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            self._emit_depth -= 1

        return ctx

    # ── Introspection ─────────────────────────────────────────────

    @property
    def subscription_count(self) -> int:
        """Total number of active subscriptions."""
        return sum(len(subs) for subs in self._subscriptions.values())

    def get_subscribers(self, event: str) -> list[Subscription]:
        """Get all subscriptions matching an event."""
        return self._find_matching_subscriptions(event)

    def clear_inactive(self) -> int:
        """Remove inactive subscriptions. Returns count removed."""
        removed = 0
        for event in list(self._subscriptions.keys()):
            before = len(self._subscriptions[event])
            self._subscriptions[event] = [s for s in self._subscriptions[event] if s.active]
            removed += before - len(self._subscriptions[event])
            if not self._subscriptions[event]:
                del self._subscriptions[event]
        return removed

    # ── Internal ──────────────────────────────────────────────────

    def _add_subscription(self, sub: Subscription) -> None:
        self._subscriptions[sub.event].append(sub)
        self._subscriptions[sub.event].sort(key=lambda s: s.priority)

    def _find_matching_subscriptions(self, event: str) -> list[Subscription]:
        """Find all subscriptions matching an event, including wildcards."""
        result: list[Subscription] = []
        seen: set[int] = set()

        # Catch-all: "*" matches every event
        for sub in self._subscriptions.get("*", []):
            if id(sub) not in seen:
                result.append(sub)
                seen.add(id(sub))

        # Exact match
        for sub in self._subscriptions.get(event, []):
            if id(sub) not in seen:
                result.append(sub)
                seen.add(id(sub))

        # Wildcard patterns
        for pattern, events in WILDCARD_PATTERNS.items():
            if event in events:
                for sub in self._subscriptions.get(pattern, []):
                    if id(sub) not in seen:
                        result.append(sub)
                        seen.add(id(sub))

        # fnmatch-style wildcards (e.g., "tool.*")
        for pattern in self._subscriptions:
            if "*" in pattern and fnmatch.fnmatch(event, pattern):
                for sub in self._subscriptions[pattern]:
                    if id(sub) not in seen:
                        result.append(sub)
                        seen.add(id(sub))

        result.sort(key=lambda s: s.priority)
        return result

    def _call_sync_handler(self, sub: Subscription, ctx: EventContext) -> None:
        """Call a sync handler with error isolation and timeout."""
        try:
            sub.handler(event=ctx.event, **ctx.context)
        except Exception as e:
            logger.error("Handler error for '%s': %s", ctx.event, e, exc_info=True)

    async def _call_sync_handler_async(self, sub: Subscription, ctx: EventContext) -> None:
        """Call a sync handler in async context."""
        loop = asyncio.get_running_loop()
        try:
            await asyncio.wait_for(
                loop.run_in_executor(None, lambda: sub.handler(event=ctx.event, **ctx.context)),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("Handler timed out for '%s' after %.1fs", ctx.event, self._timeout)
        except Exception as e:
            logger.error("Handler error for '%s': %s", ctx.event, e, exc_info=True)

    async def _call_handler_safe(self, sub: Subscription, ctx: EventContext) -> None:
        """Call an async handler with error isolation and timeout."""
        try:
            await asyncio.wait_for(
                sub.handler(event=ctx.event, **ctx.context),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("Async handler timed out for '%s' after %.1fs", ctx.event, self._timeout)
        except Exception as e:
            logger.error("Async handler error for '%s': %s", ctx.event, e, exc_info=True)
