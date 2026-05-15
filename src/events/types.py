"""Event type definitions and context schemas for the MyClaw event bus."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── Event Categories ──────────────────────────────────────────────

class EventCategory:
    """Namespace for event categories."""
    APP = "app"
    AGENT = "agent"
    TOOL = "tool"
    SESSION = "session"
    MESSAGE = "message"
    STATE = "state"
    COMMAND = "command"
    LEARNING = "learning"
    CONFIG = "config"


# ── Event Names ───────────────────────────────────────────────────

class Event:
    """All event names in the system."""
    # App lifecycle
    APP_STARTUP = "app.startup"
    APP_SHUTDOWN = "app.shutdown"
    CONFIG_RELOADED = "config.reloaded"

    # Agent lifecycle
    AGENT_LOOP_STARTED = "agent_loop.started"
    AGENT_LOOP_COMPLETED = "agent_loop.completed"
    AGENT_LOOP_RESUMED = "agent_loop.resumed"
    AGENT_ERROR = "agent.error"

    # Tool execution
    TOOL_EXEC_STARTED = "tool.exec_started"
    TOOL_EXEC_COMPLETED = "tool.exec_completed"
    TOOL_EXEC_FAILED = "tool.exec_failed"
    TOOL_APPROVAL_PENDING = "tool.approval_pending"

    # Session management
    SESSION_CREATED = "session.created"
    SESSION_RESET = "session.reset"
    SESSION_SWITCHED = "session.switched"
    SESSION_ENDED = "session.ended"

    # Message flow
    MESSAGE_RECEIVED = "message.received"
    MESSAGE_REPLIED = "message.replied"

    # State persistence
    STATE_SAVED = "state.saved"
    STATE_LOADED = "state.loaded"
    STATE_DELETED = "state.deleted"

    # Command dispatch
    COMMAND_DISPATCHED = "command.dispatched"
    COMMAND_COMPLETED = "command.completed"
    COMMAND_FAILED = "command.failed"

    # Learning loop
    LEARNING_SESSION_END = "learning.session_end"
    LEARNING_MEMORY_EXTRACTED = "learning.memory_extracted"
    LEARNING_SKILL_CREATED = "learning.skill_created"


# ── Wildcard Patterns ─────────────────────────────────────────────

# Patterns for wildcard matching (e.g., "tool.*" matches all tool events)
WILDCARD_PATTERNS = {
    f"{EventCategory.APP}.*": [Event.APP_STARTUP, Event.APP_SHUTDOWN, Event.CONFIG_RELOADED],
    f"{EventCategory.AGENT}.*": [Event.AGENT_LOOP_STARTED, Event.AGENT_LOOP_COMPLETED, Event.AGENT_LOOP_RESUMED, Event.AGENT_ERROR],
    f"{EventCategory.TOOL}.*": [Event.TOOL_EXEC_STARTED, Event.TOOL_EXEC_COMPLETED, Event.TOOL_EXEC_FAILED, Event.TOOL_APPROVAL_PENDING],
    f"{EventCategory.SESSION}.*": [Event.SESSION_CREATED, Event.SESSION_RESET, Event.SESSION_SWITCHED, Event.SESSION_ENDED],
    f"{EventCategory.MESSAGE}.*": [Event.MESSAGE_RECEIVED, Event.MESSAGE_REPLIED],
    f"{EventCategory.STATE}.*": [Event.STATE_SAVED, Event.STATE_LOADED, Event.STATE_DELETED],
    f"{EventCategory.COMMAND}.*": [Event.COMMAND_DISPATCHED, Event.COMMAND_COMPLETED, Event.COMMAND_FAILED],
    f"{EventCategory.LEARNING}.*": [Event.LEARNING_SESSION_END, Event.LEARNING_MEMORY_EXTRACTED, Event.LEARNING_SKILL_CREATED],
    f"{EventCategory.CONFIG}.*": [Event.CONFIG_RELOADED],
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
        return self.event == other.event and id(self.handler) == id(other.handler)
