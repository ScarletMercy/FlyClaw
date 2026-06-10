"""Event type definitions and context schemas for the flyclaw event bus."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── Wildcard Patterns ──────────────────────────────────────────────

# Maps wildcard patterns (e.g. "tool.*") to the concrete event names they match.
WILDCARD_PATTERNS: dict[str, list[str]] = {
    "app.*": ["app.startup", "app.shutdown", "config.reloaded"],
    "agent.*": [
        "agent_loop.started",
        "agent_loop.completed",
        "agent_loop.resumed",
        "agent.error",
    ],
    "tool.*": [
        "tool.exec_started",
        "tool.exec_completed",
        "tool.exec_failed",
        "tool.approval_pending",
    ],
    "session.*": [
        "session.created",
        "session.reset",
        "session.switched",
        "session.ended",
    ],
    "message.*": ["message.received", "message.replied"],
    "state.*": ["state.saved", "state.loaded", "state.deleted"],
    "command.*": ["command.dispatched", "command.completed", "command.failed"],
    "learning.*": [
        "learning.session_end",
        "learning.memory_extracted",
        "learning.skill_created",
    ],
    "config.*": ["config.reloaded"],
    "kanban.*": [
        "kanban.task.created",
        "kanban.task.claimed",
        "kanban.task.completed",
        "kanban.task.blocked",
        "kanban.task.unblocked",
        "kanban.task.crashed",
        "kanban.task.timed_out",
        "kanban.task.promoted",
        "kanban.task.reclaimed",
        "kanban.task.heartbeat",
    ],
}


# ── Event Context ─────────────────────────────────────────────────


@dataclass
class EventContext:
    """Context data for an event emission."""

    event: str
    timestamp: float = 0.0
    context: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.context.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.context[key]

    def __contains__(self, key: str) -> bool:
        return key in self.context


# ── Subscription ──────────────────────────────────────────────────


@dataclass
class Subscription:
    """A single event subscription."""

    event: str
    handler: Any  # Callable
    is_async: bool = False
    priority: int = 0
    active: bool = True

    def __hash__(self):
        return hash((self.event, id(self.handler)))

    def __eq__(self, other):
        if not isinstance(other, Subscription):
            return False
        return self.event == other.event and self.handler is other.handler
