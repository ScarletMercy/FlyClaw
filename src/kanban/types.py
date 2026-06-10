"""Kanban data models — Pydantic types for the task orchestration system.

Maps to the SQLite schema in store.py. State machine:
    triage -> todo -> ready -> running -> done / blocked / archived
                 ^                  |
                 +--- unblock <-----+
"""

from __future__ import annotations

import time
import uuid
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_STATUSES = frozenset({"triage", "todo", "ready", "running", "blocked", "done", "archived"})

DEFAULT_CLAIM_TTL_SECONDS: int = 15 * 60  # 15 minutes
DEFAULT_FAILURE_LIMIT: int = 2


def _short_id(prefix: str = "t") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Core models
# ---------------------------------------------------------------------------


class KanbanTask(BaseModel):
    """A single kanban card / task."""

    id: str = Field(default_factory=lambda: _short_id("t"))
    title: str
    body: Optional[str] = None
    assignee: Optional[str] = None  # maps to AgentSubconfig name
    status: str = "todo"
    priority: int = 0
    created_by: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    # Workspace
    workspace_kind: Literal["scratch", "dir", "worktree"] = "scratch"
    workspace_path: Optional[str] = None

    # Claim / locking
    claim_lock: Optional[str] = None
    claim_expires: Optional[float] = None

    # Multi-tenancy
    tenant: Optional[str] = None
    result: Optional[str] = None
    idempotency_key: Optional[str] = None

    # Failure tracking / circuit breaker
    consecutive_failures: int = 0
    worker_run_id: Optional[str] = None  # RunRegistry run id
    max_runtime_seconds: Optional[int] = None
    last_heartbeat_at: Optional[float] = None
    last_failure_error: Optional[str] = None

    # Skills / retry config
    skills: Optional[list[str]] = None
    max_retries: Optional[int] = None

    # Board isolation
    board: str = "default"


class KanbanRun(BaseModel):
    """Execution record for a single attempt at a task."""

    id: Optional[int] = None
    task_id: str
    profile: Optional[str] = None  # agent name
    status: str = "running"  # running | done | blocked | crashed | timed_out | failed | released
    claim_lock: Optional[str] = None
    claim_expires: Optional[float] = None
    worker_run_id: Optional[str] = None
    started_at: float = Field(default_factory=time.time)
    ended_at: Optional[float] = None
    outcome: Optional[str] = None  # completed | blocked | crashed | timed_out | spawn_failed | gave_up | reclaimed
    summary: Optional[str] = None
    metadata: Optional[dict] = None
    error: Optional[str] = None


class KanbanComment(BaseModel):
    """A comment attached to a task."""

    id: Optional[int] = None
    task_id: str
    author: str
    body: str
    created_at: float = Field(default_factory=time.time)


class KanbanEvent(BaseModel):
    """Audit event for task lifecycle transitions."""

    id: Optional[int] = None
    task_id: str
    run_id: Optional[int] = None
    kind: str  # e.g. "claimed", "completed", "blocked", "crashed"
    payload: Optional[dict] = None
    created_at: float = Field(default_factory=time.time)


class TaskLink(BaseModel):
    """A parent-child dependency edge."""

    parent_id: str
    child_id: str


class KanbanNotifySub(BaseModel):
    """A notification subscription for task events via a channel."""

    task_id: str
    platform: str  # "qq" or "weixin"
    chat_id: str
    thread_id: str = ""
    user_id: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    last_event_id: int = 0


class DispatchResult(BaseModel):
    """Result of a single dispatch_once() tick."""

    reclaimed: int = 0
    promoted: int = 0
    spawned: list[tuple[str, str]] = []  # (task_id, assignee)
    skipped_unassigned: list[str] = []
    crashed: list[str] = []
    auto_blocked: list[str] = []
    timed_out: list[str] = []
