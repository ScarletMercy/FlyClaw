"""Kanban tool surface — 9 async tools for worker and orchestrator agents.

Adapted from hermes-agent's ``tools/kanban_tools.py``. Uses ``ContextVar``
for worker-scoped task ownership enforcement instead of env vars.
"""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from typing import Optional

from src.agent.tooldef import ToolDef

from .types import DEFAULT_CLAIM_TTL_SECONDS, DEFAULT_FAILURE_LIMIT, KanbanTask

logger = logging.getLogger("flyclaw.kanban.tools")


# ---------------------------------------------------------------------------
# Context vars — set by dispatcher when spawning a worker
# ---------------------------------------------------------------------------

_current_kanban_task: ContextVar[Optional[str]] = ContextVar("_current_kanban_task", default=None)
_current_kanban_board: ContextVar[Optional[str]] = ContextVar("_current_kanban_board", default=None)
_current_kanban_agent: ContextVar[Optional[str]] = ContextVar("_current_kanban_agent", default=None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_store():
    """Lazy import to avoid circular dependency at module load.

    Returns None if the kanban store is not initialized (e.g. setup failed).
    """
    from src._container import get_container

    container = get_container()
    if not hasattr(container, "kanban_store") or container.kanban_store is None:
        return None
    return container.kanban_store


_STORE_UNAVAILABLE = "Kanban is not available. The store failed to initialize. Check server logs for details."


def _default_task_id(task_id: Optional[str]) -> Optional[str]:
    """Resolve task_id: explicit arg wins, then context var."""
    return task_id or _current_kanban_task.get()


def _enforce_worker_ownership(tid: str) -> Optional[str]:
    """Workers can only mutate their own task. Returns error message or None."""
    scoped = _current_kanban_task.get()
    if scoped and tid != scoped:
        return (
            f"worker is scoped to task {scoped}; refusing to mutate {tid}. "
            f"Use kanban_comment to hand off information, or kanban_create to "
            f"spawn follow-up work."
        )
    return None


def _board() -> Optional[str]:
    return _current_kanban_board.get()


def _tool_error(msg: str) -> str:
    return f"[error] {msg}"


def _task_summary(task: KanbanTask) -> dict:
    return {
        "id": task.id,
        "title": task.title,
        "status": task.status,
        "assignee": task.assignee,
        "priority": task.priority,
        "created_at": task.created_at,
        "board": task.board,
    }


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


async def kanban_show(task_id: Optional[str] = None) -> str:
    """Read a kanban task's full state including comments, events, and runs.

    Args:
        task_id: Task ID. Defaults to the current kanban task if running as a worker.
    """
    tid = _default_task_id(task_id)
    if not tid:
        return _tool_error("No task_id provided and no current kanban task context")

    store = _get_store()
    if store is None:
        return _tool_error(_STORE_UNAVAILABLE)
    task = await store.get_task(tid)
    if not task:
        return _tool_error(f"Task {tid} not found")

    parents = await store.get_parents(tid)
    children = await store.get_children(tid)
    comments = await store.list_comments(tid)
    events = await store.list_events(tid, limit=20)
    runs = await store.list_runs(tid)

    result = {
        "task": task.model_dump(exclude_none=True),
        "parents": [_task_summary(p) for p in parents],
        "children": [_task_summary(c) for c in children],
        "comments": [{"id": c.id, "author": c.author, "body": c.body[:500], "at": c.created_at} for c in comments],
        "recent_events": [{"id": e.id, "kind": e.kind, "payload": e.payload, "at": e.created_at} for e in events],
        "runs": [
            {
                "id": r.id,
                "status": r.status,
                "outcome": r.outcome,
                "summary": r.summary,
                "error": r.error,
            }
            for r in runs
        ],
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


async def kanban_list(
    assignee: Optional[str] = None,
    status: Optional[str] = None,
    tenant: Optional[str] = None,
    include_archived: bool = False,
    limit: int = 50,
) -> str:
    """List kanban task summaries. Typically used by orchestrator agents.

    Args:
        assignee: Filter by assignee agent name
        status: Filter by status (triage/todo/ready/running/blocked/done/archived)
        tenant: Filter by tenant namespace
        include_archived: Include archived tasks in results
        limit: Maximum number of tasks to return (default 50, max 200)
    """
    store = _get_store()
    if store is None:
        return _tool_error(_STORE_UNAVAILABLE)
    limit = min(limit, 200)
    tasks, total = await store.list_tasks(
        assignee=assignee,
        status=status,
        tenant=tenant,
        board=_board(),
        include_archived=include_archived,
        limit=limit,
    )
    result = {
        "tasks": [_task_summary(t) for t in tasks],
        "total": total,
        "limit": limit,
        "truncated": total > limit,
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


async def kanban_complete(
    task_id: Optional[str] = None,
    summary: Optional[str] = None,
    metadata: Optional[dict] = None,
    result: Optional[str] = None,
    created_cards: Optional[list[str]] = None,
) -> str:
    """Mark your current task as done with a structured handoff.

    Call this when you have finished the task. Provide a summary of what was
    accomplished and any structured facts that downstream tasks might need.

    Args:
        task_id: Task ID (defaults to current worker task)
        summary: Human-readable handoff, 1-3 sentences describing what was done
        metadata: Free-form dict of structured facts for downstream tasks
        result: Short result log line (legacy, prefer summary)
        created_cards: Task IDs you created via kanban_create during this run
    """
    tid = _default_task_id(task_id)
    if not tid:
        return _tool_error("No task_id provided and no current kanban task context")

    err = _enforce_worker_ownership(tid)
    if err:
        return _tool_error(err)

    store = _get_store()
    if store is None:
        return _tool_error(_STORE_UNAVAILABLE)
    ok = await store.complete_task(tid, summary=summary, metadata=metadata, result=result)
    if not ok:
        return _tool_error(f"Failed to complete task {tid} (not in running state?)")

    # Emit event for notifications
    try:
        from src.events import emit_async

        await emit_async(
            "kanban.task.completed",
            task_id=tid,
            kind="completed",
            summary=summary,
        )
    except Exception:
        logger.debug("Failed to emit kanban.task.completed event", exc_info=True)

    msg = f"Task {tid} completed."
    if summary:
        msg += f" Summary: {summary}"
    return msg


async def kanban_block(
    task_id: Optional[str] = None,
    reason: str = "",
) -> str:
    """Transition the task to blocked because you need human input or clarification.

    Args:
        task_id: Task ID (defaults to current worker task)
        reason: What you need answered or resolved, in one or two sentences. Required.
    """
    if not reason:
        return _tool_error("reason is required when blocking a task")

    tid = _default_task_id(task_id)
    if not tid:
        return _tool_error("No task_id provided and no current kanban task context")

    err = _enforce_worker_ownership(tid)
    if err:
        return _tool_error(err)

    store = _get_store()
    if store is None:
        return _tool_error(_STORE_UNAVAILABLE)
    ok = await store.block_task(tid, reason=reason)
    if not ok:
        return _tool_error(f"Failed to block task {tid} (not in running state?)")

    try:
        from src.events import emit_async

        await emit_async("kanban.task.blocked", task_id=tid, kind="blocked", reason=reason)
    except Exception:
        logger.debug("Failed to emit kanban.task.blocked event", exc_info=True)

    return f"Task {tid} blocked. Reason: {reason}"


async def kanban_heartbeat(
    task_id: Optional[str] = None,
    note: Optional[str] = None,
) -> str:
    """Signal that you are still alive during a long operation. Extends the claim TTL.

    Args:
        task_id: Task ID (defaults to current worker task)
        note: Optional progress note describing what you're currently doing
    """
    tid = _default_task_id(task_id)
    if not tid:
        return _tool_error("No task_id provided and no current kanban task context")

    err = _enforce_worker_ownership(tid)
    if err:
        return _tool_error(err)

    store = _get_store()
    if store is None:
        return _tool_error(_STORE_UNAVAILABLE)
    ok = await store.heartbeat_claim(tid, note=note)
    if not ok:
        return _tool_error(f"Heartbeat failed for task {tid} (not running?)")

    msg = f"Heartbeat OK for task {tid}."
    if note:
        msg += f" Note: {note}"
    return msg


async def kanban_comment(
    task_id: str,
    body: str,
) -> str:
    """Append a comment to a task's thread. Comments are visible to all agents.

    Args:
        task_id: Task ID to comment on
        body: Comment body (markdown supported)
    """
    tid = task_id
    if not tid:
        return _tool_error("task_id is required")

    store = _get_store()
    if store is None:
        return _tool_error(_STORE_UNAVAILABLE)
    task = await store.get_task(tid)
    if not task:
        return _tool_error(f"Task {tid} not found")

    author = _current_kanban_agent.get() or "orchestrator"
    comment_id = await store.add_comment(tid, author=author, body=body)
    return f"Comment #{comment_id} added to task {tid}."


async def kanban_create(
    title: str,
    assignee: str,
    body: Optional[str] = None,
    parents: Optional[list[str]] = None,
    tenant: Optional[str] = None,
    priority: int = 0,
    workspace_kind: str = "scratch",
    workspace_path: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    max_runtime_seconds: Optional[int] = None,
    skills: Optional[list[str]] = None,
    max_retries: Optional[int] = None,
) -> str:
    """Create a new kanban task, optionally as a child of existing tasks.

    The child task will wait in 'todo' until all parent tasks are done, then
    automatically promote to 'ready' for the dispatcher to pick up.

    Args:
        title: Task title (required, short summary)
        assignee: Agent name to execute this task (required, must match a sub-agent in config)
        body: Full specification, acceptance criteria, or context
        parents: Parent task IDs — child waits until all parents reach 'done'
        tenant: Namespace for multi-project isolation
        priority: Higher number = dispatched sooner (default 0)
        workspace_kind: Workspace type: scratch (temp), dir (shared), worktree (git)
        workspace_path: Absolute path for dir/worktree workspace kinds
        idempotency_key: Dedup key — if a task with this key exists, return it instead
        max_runtime_seconds: Per-task runtime cap in seconds
        skills: Skill names to force-load into the worker agent
        max_retries: Override circuit breaker failure limit for this task
    """
    store = _get_store()
    if store is None:
        return _tool_error(_STORE_UNAVAILABLE)
    created_by = _current_kanban_agent.get() or "orchestrator"
    board = _board() or "default"

    task = await store.create_task(
        title=title,
        body=body,
        assignee=assignee,
        priority=priority,
        created_by=created_by,
        workspace_kind=workspace_kind,
        workspace_path=workspace_path,
        tenant=tenant,
        idempotency_key=idempotency_key,
        max_runtime_seconds=max_runtime_seconds,
        skills=skills,
        max_retries=max_retries,
        board=board,
        parents=parents,
    )

    try:
        from src.events import emit_async

        await emit_async(
            "kanban.task.created",
            task_id=task.id,
            kind="created",
            title=title,
            assignee=assignee,
        )
    except Exception:
        logger.debug("Failed to emit kanban.task.created event", exc_info=True)

    return json.dumps(
        {
            "id": task.id,
            "status": task.status,
            "title": task.title,
            "assignee": task.assignee,
        },
        ensure_ascii=False,
    )


async def kanban_unblock(task_id: str) -> str:
    """Move a blocked task back to ready. Typically used by orchestrator agents.

    Args:
        task_id: Blocked task ID to return to ready state
    """
    tid = task_id
    if not tid:
        return _tool_error("task_id is required")

    store = _get_store()
    if store is None:
        return _tool_error(_STORE_UNAVAILABLE)
    ok = await store.unblock_task(tid)
    if not ok:
        return _tool_error(f"Failed to unblock task {tid} (not blocked?)")

    try:
        from src.events import emit_async

        await emit_async("kanban.task.unblocked", task_id=tid, kind="unblocked")
    except Exception:
        logger.debug("Failed to emit kanban.task.unblocked event", exc_info=True)

    return f"Task {tid} unblocked and moved to ready."


async def kanban_link(parent_id: str, child_id: str) -> str:
    """Add a parent-child dependency edge. The child task will wait for the parent to complete.

    Args:
        parent_id: Parent task ID (must complete before child)
        child_id: Child task ID (waits for parent)
    """
    if not parent_id or not child_id:
        return _tool_error("Both parent_id and child_id are required")

    store = _get_store()
    if store is None:
        return _tool_error(_STORE_UNAVAILABLE)

    # Verify both tasks exist
    parent = await store.get_task(parent_id)
    if not parent:
        return _tool_error(f"Parent task {parent_id} not found")
    child = await store.get_task(child_id)
    if not child:
        return _tool_error(f"Child task {child_id} not found")

    await store.link_tasks(parent_id, child_id)
    return f"Linked: {parent_id} -> {child_id}"


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def get_tools() -> list[ToolDef]:
    """Return all kanban tools for registration into the agent tool surface."""
    return [
        ToolDef.from_function(kanban_show),
        ToolDef.from_function(kanban_list),
        ToolDef.from_function(kanban_complete),
        ToolDef.from_function(kanban_block),
        ToolDef.from_function(kanban_heartbeat),
        ToolDef.from_function(kanban_comment),
        ToolDef.from_function(kanban_create),
        ToolDef.from_function(kanban_unblock),
        ToolDef.from_function(kanban_link),
    ]
