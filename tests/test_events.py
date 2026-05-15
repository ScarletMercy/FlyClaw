"""Tests for the event bus system."""

import asyncio
import time

import pytest

from src.events.bus import EventBus
from src.events.types import Event, EventContext


class TestEventBusBasic:
    def test_subscribe_and_emit(self):
        bus = EventBus()
        results = []

        def handler(event, **ctx):
            results.append((event, ctx))

        bus.subscribe("test.event", handler)
        bus.emit("test.event", foo="bar")

        assert len(results) == 1
        assert results[0] == ("test.event", {"foo": "bar"})

    def test_multiple_subscribers(self):
        bus = EventBus()
        results = []

        def handler1(event, **ctx):
            results.append(1)

        def handler2(event, **ctx):
            results.append(2)

        bus.subscribe("test.event", handler1)
        bus.subscribe("test.event", handler2)
        bus.emit("test.event")

        assert results == [1, 2]

    def test_unsubscribe(self):
        bus = EventBus()
        results = []

        def handler(event, **ctx):
            results.append(1)

        sub = bus.subscribe("test.event", handler)
        bus.emit("test.event")
        assert len(results) == 1

        bus.unsubscribe("test.event", handler)
        bus.emit("test.event")
        assert len(results) == 1  # No new calls

    def test_unsubscribe_all(self):
        bus = EventBus()
        count = 0

        def handler(event, **ctx):
            nonlocal count
            count += 1

        bus.subscribe("event1", handler)
        bus.subscribe("event2", handler)
        bus.emit("event1")
        assert count == 1

        bus.unsubscribe_all()
        bus.emit("event1")
        bus.emit("event2")
        assert count == 1  # No new calls

    def test_no_subscribers_is_noop(self):
        bus = EventBus()
        ctx = bus.emit("nonexistent.event", data="test")
        assert ctx.event == "nonexistent.event"
        assert ctx.get("data") == "test"


class TestEventBusPriority:
    def test_priority_ordering(self):
        bus = EventBus()
        results = []

        def make_handler(val):
            def handler(event, **ctx):
                results.append(val)
            return handler

        bus.subscribe("test.event", make_handler(3), priority=3)
        bus.subscribe("test.event", make_handler(1), priority=1)
        bus.subscribe("test.event", make_handler(2), priority=2)

        bus.emit("test.event")
        assert results == [1, 2, 3]


class TestEventBusWildcard:
    def test_wildcard_tool_events(self):
        bus = EventBus()
        results = []

        def handler(event, **ctx):
            results.append(event)

        bus.subscribe("tool.*", handler)
        bus.emit("tool.exec_started", tool_name="read_file")
        bus.emit("tool.exec_completed", tool_name="read_file")
        bus.emit("tool.exec_failed", tool_name="exec_command")

        assert len(results) == 3
        assert "tool.exec_started" in results
        assert "tool.exec_completed" in results
        assert "tool.exec_failed" in results

    def test_wildcard_session_events(self):
        bus = EventBus()
        results = []

        def handler(event, **ctx):
            results.append(event)

        bus.subscribe("session.*", handler)
        bus.emit("session.created")
        bus.emit("session.reset")
        bus.emit("session.ended")

        assert len(results) == 3

    def test_exact_match_and_wildcard_both_fire(self):
        bus = EventBus()
        exact_count = 0
        wildcard_count = 0

        def exact_handler(event, **ctx):
            nonlocal exact_count
            exact_count += 1

        def wildcard_handler(event, **ctx):
            nonlocal wildcard_count
            wildcard_count += 1

        bus.subscribe("tool.exec_completed", exact_handler)
        bus.subscribe("tool.*", wildcard_handler)

        bus.emit("tool.exec_completed")

        assert exact_count == 1
        assert wildcard_count == 1


class TestEventBusErrorIsolation:
    def test_handler_error_does_not_stop_others(self):
        bus = EventBus()
        results = []

        def failing_handler(event, **ctx):
            raise ValueError("intentional error")

        def good_handler(event, **ctx):
            results.append("ok")

        bus.subscribe("test.event", failing_handler)
        bus.subscribe("test.event", good_handler)

        bus.emit("test.event")
        assert results == ["ok"]

    def test_all_handlers_fail_gracefully(self):
        bus = EventBus()

        def failing_handler(event, **ctx):
            raise RuntimeError("boom")

        bus.subscribe("test.event", failing_handler)
        # Should not raise
        ctx = bus.emit("test.event")
        assert ctx.event == "test.event"


class TestEventBusAsync:
    @pytest.mark.asyncio
    async def test_async_handler(self):
        bus = EventBus()
        results = []

        async def async_handler(event, **ctx):
            await asyncio.sleep(0.01)
            results.append(ctx.get("value"))

        bus.subscribe_async("test.event", async_handler)
        await bus.emit_async("test.event", value=42)

        assert results == [42]

    @pytest.mark.asyncio
    async def test_mixed_sync_async_handlers(self):
        bus = EventBus()
        results = []

        def sync_handler(event, **ctx):
            results.append("sync")

        async def async_handler(event, **ctx):
            await asyncio.sleep(0.01)
            results.append("async")

        bus.subscribe("test.event", sync_handler)
        bus.subscribe_async("test.event", async_handler)

        await bus.emit_async("test.event")
        assert "sync" in results
        assert "async" in results

    @pytest.mark.asyncio
    async def test_async_handler_timeout(self):
        bus = EventBus(timeout=0.05)

        async def slow_handler(event, **ctx):
            await asyncio.sleep(10)

        bus.subscribe_async("test.event", slow_handler)
        # Should not hang
        await bus.emit_async("test.event")


class TestEventBusIntrospection:
    def test_subscription_count(self):
        bus = EventBus()
        assert bus.subscription_count == 0

        bus.subscribe("e1", lambda **k: None)
        bus.subscribe("e2", lambda **k: None)
        assert bus.subscription_count == 2

    def test_get_subscribers(self):
        bus = EventBus()

        def h1(event, **ctx): pass
        def h2(event, **ctx): pass

        bus.subscribe("test.event", h1)
        bus.subscribe("test.event", h2)

        subs = bus.get_subscribers("test.event")
        assert len(subs) == 2

    def test_clear_inactive(self):
        bus = EventBus()

        def handler(event, **ctx): pass

        sub = bus.subscribe("test.event", handler)
        sub.active = False

        removed = bus.clear_inactive()
        assert removed == 1
        assert bus.subscription_count == 0


class TestEventContext:
    def test_context_get(self):
        ctx = EventContext(event="test", context={"foo": "bar"})
        assert ctx.get("foo") == "bar"
        assert ctx.get("missing", "default") == "default"

    def test_context_contains(self):
        ctx = EventContext(event="test", context={"foo": "bar"})
        assert "foo" in ctx
        assert "missing" not in ctx

    def test_context_getitem(self):
        ctx = EventContext(event="test", context={"foo": "bar"})
        assert ctx["foo"] == "bar"
