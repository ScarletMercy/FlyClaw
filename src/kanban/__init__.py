"""Kanban-style multi-instance task orchestration.

Adapted from hermes-agent's kanban system. Uses async sub-agent delegation
instead of subprocess spawning for FlyClaw's async-first architecture.
"""

from .types import (
    KanbanTask,
    KanbanRun,
    KanbanComment,
    KanbanEvent,
    KanbanNotifySub,
    TaskLink,
    DispatchResult,
    VALID_STATUSES,
    DEFAULT_CLAIM_TTL_SECONDS,
    DEFAULT_FAILURE_LIMIT,
)
from .store import KanbanStore

__all__ = [
    "KanbanTask",
    "KanbanRun",
    "KanbanComment",
    "KanbanEvent",
    "KanbanNotifySub",
    "TaskLink",
    "DispatchResult",
    "KanbanStore",
    "VALID_STATUSES",
    "DEFAULT_CLAIM_TTL_SECONDS",
    "DEFAULT_FAILURE_LIMIT",
]
