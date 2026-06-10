"""Async SQLite store for the kanban system.

Adapted from hermes-agent's ``kanban_db.py``. All methods are async and use
aiosqlite with WAL mode + ``BEGIN IMMEDIATE`` write transactions for CAS
safety.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import aiosqlite

from .types import (
    DEFAULT_CLAIM_TTL_SECONDS,
    DEFAULT_FAILURE_LIMIT,
    VALID_STATUSES,
    KanbanComment,
    KanbanEvent,
    KanbanNotifySub,
    KanbanRun,
    KanbanTask,
    TaskLink,
)

logger = logging.getLogger("flyclaw.kanban.store")


# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_CREATE_TASKS_SQL = """
CREATE TABLE IF NOT EXISTS tasks (
    id                   TEXT PRIMARY KEY,
    title                TEXT NOT NULL,
    body                 TEXT,
    assignee             TEXT,
    status               TEXT NOT NULL DEFAULT 'todo',
    priority             INTEGER DEFAULT 0,
    created_by           TEXT,
    created_at           REAL NOT NULL,
    started_at           REAL,
    completed_at         REAL,
    workspace_kind       TEXT NOT NULL DEFAULT 'scratch',
    workspace_path       TEXT,
    claim_lock           TEXT,
    claim_expires        REAL,
    tenant               TEXT,
    result               TEXT,
    idempotency_key      TEXT,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    worker_run_id        TEXT,
    max_runtime_seconds  INTEGER,
    last_heartbeat_at    REAL,
    last_failure_error   TEXT,
    skills               TEXT,
    max_retries          INTEGER,
    board                TEXT NOT NULL DEFAULT 'default'
)
"""

_CREATE_TASK_LINKS_SQL = """
CREATE TABLE IF NOT EXISTS task_links (
    parent_id  TEXT NOT NULL,
    child_id   TEXT NOT NULL,
    PRIMARY KEY (parent_id, child_id)
)
"""

_CREATE_TASK_COMMENTS_SQL = """
CREATE TABLE IF NOT EXISTS task_comments (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id    TEXT NOT NULL,
    author     TEXT NOT NULL,
    body       TEXT NOT NULL,
    created_at REAL NOT NULL
)
"""

_CREATE_TASK_EVENTS_SQL = """
CREATE TABLE IF NOT EXISTS task_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id    TEXT NOT NULL,
    run_id     INTEGER,
    kind       TEXT NOT NULL,
    payload    TEXT,
    created_at REAL NOT NULL
)
"""

_CREATE_TASK_RUNS_SQL = """
CREATE TABLE IF NOT EXISTS task_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id             TEXT NOT NULL,
    profile             TEXT,
    status              TEXT NOT NULL,
    claim_lock          TEXT,
    claim_expires       REAL,
    worker_run_id       TEXT,
    started_at          REAL NOT NULL,
    ended_at            REAL,
    outcome             TEXT,
    summary             TEXT,
    metadata            TEXT,
    error               TEXT
)
"""

_CREATE_NOTIFY_SUBS_SQL = """
CREATE TABLE IF NOT EXISTS kanban_notify_subs (
    task_id       TEXT NOT NULL,
    platform      TEXT NOT NULL,
    chat_id       TEXT NOT NULL,
    thread_id     TEXT NOT NULL DEFAULT '',
    user_id       TEXT,
    created_at    REAL NOT NULL,
    last_event_id INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (task_id, platform, chat_id, thread_id)
)
"""

# Indexes for common queries
_CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)",
    "CREATE INDEX IF NOT EXISTS idx_tasks_assignee ON tasks(assignee)",
    "CREATE INDEX IF NOT EXISTS idx_tasks_board ON tasks(board)",
    "CREATE INDEX IF NOT EXISTS idx_tasks_tenant ON tasks(tenant)",
    "CREATE INDEX IF NOT EXISTS idx_tasks_idempotency ON tasks(idempotency_key)",
    "CREATE INDEX IF NOT EXISTS idx_task_comments_task ON task_comments(task_id)",
    "CREATE INDEX IF NOT EXISTS idx_task_events_task ON task_events(task_id)",
    "CREATE INDEX IF NOT EXISTS idx_task_runs_task ON task_runs(task_id)",
]


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class KanbanStore:
    """Async SQLite store for kanban tasks."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[aiosqlite.Connection] = None
        self._closed = False
        # All connection-mutating operations share ONE aiosqlite connection.
        # aiosqlite serialises individual statements on a background thread, but
        # multi-statement transactions and explicit commits are NOT serialised —
        # two concurrent ``write_txn`` calls hit "cannot start a transaction
        # within a transaction", and a stray ``commit`` can prematurely commit
        # another coroutine's open transaction. This lock serialises every
        # commit-bearing operation so they never interleave on the shared conn.
        self._db_lock = asyncio.Lock()

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        return False

    # ── Connection / transaction ──────────────────────────────────────

    async def initialize(self) -> None:
        """Open connection and create tables."""
        if self._conn is not None:
            return
        self._conn = await aiosqlite.connect(str(self.db_path))
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA busy_timeout=5000")

        for sql in [
            _CREATE_TASKS_SQL,
            _CREATE_TASK_LINKS_SQL,
            _CREATE_TASK_COMMENTS_SQL,
            _CREATE_TASK_EVENTS_SQL,
            _CREATE_TASK_RUNS_SQL,
            _CREATE_NOTIFY_SUBS_SQL,
        ]:
            await self._conn.execute(sql)
        for sql in _CREATE_INDEXES_SQL:
            await self._conn.execute(sql)
        await self._conn.commit()

    async def close(self) -> None:
        # Mark closed FIRST so any straggler worker that calls a store method
        # during shutdown cannot trigger _get_conn() to re-open the connection.
        self._closed = True
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def _get_conn(self) -> aiosqlite.Connection:
        if self._closed:
            raise RuntimeError("KanbanStore is closed")
        if self._conn is None:
            await self.initialize()
        assert self._conn is not None
        return self._conn

    @asynccontextmanager
    async def write_txn(self):
        """``BEGIN IMMEDIATE`` write transaction — serialises writers."""
        conn = await self._get_conn()
        async with self._db_lock:
            await conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise

    # ── Task CRUD ─────────────────────────────────────────────────────

    async def create_task(
        self,
        *,
        title: str,
        body: Optional[str] = None,
        assignee: Optional[str] = None,
        status: str = "todo",
        priority: int = 0,
        created_by: Optional[str] = None,
        workspace_kind: str = "scratch",
        workspace_path: Optional[str] = None,
        tenant: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        max_runtime_seconds: Optional[int] = None,
        skills: Optional[list[str]] = None,
        max_retries: Optional[int] = None,
        board: str = "default",
        parents: Optional[list[str]] = None,
    ) -> KanbanTask:
        """Create a new task. Optionally link to parent tasks."""
        task = KanbanTask(
            title=title,
            body=body,
            assignee=assignee,
            status=status,
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
        )

        async with self.write_txn() as conn:
            # Idempotency: if key exists, return existing task (inside txn)
            if idempotency_key:
                async with conn.execute("SELECT * FROM tasks WHERE idempotency_key = ?", (idempotency_key,)) as cur:
                    row = await cur.fetchone()
                if row:
                    return self._row_to_task(row)

            await conn.execute(
                """INSERT INTO tasks
                   (id, title, body, assignee, status, priority, created_by,
                    created_at, workspace_kind, workspace_path, tenant,
                    idempotency_key, max_runtime_seconds, skills, max_retries, board)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    task.id,
                    task.title,
                    task.body,
                    task.assignee,
                    task.status,
                    task.priority,
                    task.created_by,
                    task.created_at,
                    task.workspace_kind,
                    task.workspace_path,
                    task.tenant,
                    task.idempotency_key,
                    task.max_runtime_seconds,
                    json.dumps(task.skills) if task.skills else None,
                    task.max_retries,
                    task.board,
                ),
            )
            # Link parents
            if parents:
                for pid in parents:
                    await conn.execute(
                        "INSERT OR IGNORE INTO task_links (parent_id, child_id) VALUES (?,?)",
                        (pid, task.id),
                    )

        await self._insert_event(task.id, kind="created")
        return task

    async def get_task(self, task_id: str) -> Optional[KanbanTask]:
        conn = await self._get_conn()
        async with conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)) as cur:
            row = await cur.fetchone()
        return self._row_to_task(row) if row else None

    async def list_tasks(
        self,
        *,
        assignee: Optional[str] = None,
        status: Optional[str] = None,
        tenant: Optional[str] = None,
        board: Optional[str] = None,
        include_archived: bool = False,
        limit: int = 50,
    ) -> tuple[list[KanbanTask], int]:
        """Return (tasks, total_matching)."""
        conn = await self._get_conn()
        conds: list[str] = []
        params: list = []

        if assignee:
            conds.append("assignee = ?")
            params.append(assignee)
        if status:
            conds.append("status = ?")
            params.append(status)
        elif not include_archived:
            conds.append("status != 'archived'")
        if tenant:
            conds.append("tenant = ?")
            params.append(tenant)
        if board:
            conds.append("board = ?")
            params.append(board)

        where = f" WHERE {' AND '.join(conds)}" if conds else ""

        async with conn.execute(f"SELECT COUNT(*) FROM tasks{where}", params) as cur:
            total = (await cur.fetchone())[0]

        async with conn.execute(
            f"SELECT * FROM tasks{where} ORDER BY priority DESC, created_at ASC LIMIT ?",
            params + [limit],
        ) as cur:
            rows = await cur.fetchall()

        return [self._row_to_task(r) for r in rows], total

    async def assign_task(self, task_id: str, assignee: str) -> bool:
        async with self.write_txn() as conn:
            cur = await conn.execute("UPDATE tasks SET assignee = ? WHERE id = ?", (assignee, task_id))
        if cur.rowcount > 0:
            await self._insert_event(task_id, kind="assigned", payload={"assignee": assignee})
        return cur.rowcount > 0

    # ── Claim lifecycle ───────────────────────────────────────────────

    async def claim_task(
        self,
        task_id: str,
        *,
        ttl_seconds: int = DEFAULT_CLAIM_TTL_SECONDS,
    ) -> Optional[KanbanTask]:
        """Atomically transition ``ready -> running``. Returns task on success."""
        claim_token = uuid.uuid4().hex[:12]
        now = time.time()
        expires = now + ttl_seconds

        async with self.write_txn() as conn:
            # Check current status and parent deps inside the txn
            async with conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)) as cur:
                row = await cur.fetchone()
            if not row or row["status"] != "ready":
                return None

            # Verify all parents are done
            async with conn.execute(
                """SELECT COUNT(*) FROM task_links tl
                   JOIN tasks t ON t.id = tl.parent_id
                   WHERE tl.child_id = ? AND t.status != 'done'""",
                (task_id,),
            ) as cur:
                pending = (await cur.fetchone())[0]
            if pending > 0:
                return None

            # CAS: ready -> running
            cas_cur = await conn.execute(
                """UPDATE tasks
                   SET status = 'running', claim_lock = ?, claim_expires = ?,
                       started_at = ?, last_heartbeat_at = ?
                   WHERE id = ? AND status = 'ready'""",
                (claim_token, expires, now, now, task_id),
            )
            if cas_cur.rowcount == 0:
                return None  # Lost the CAS race (defensive — unlikely under BEGIN IMMEDIATE)

            # Create kanban_run record
            await conn.execute(
                """INSERT INTO task_runs (task_id, profile, status, claim_lock,
                   claim_expires, started_at)
                   VALUES (?,?,?,?,?,?)""",
                (task_id, None, "running", claim_token, expires, now),
            )
            run_id_cursor = await conn.execute("SELECT last_insert_rowid()")
            run_id = (await run_id_cursor.fetchone())[0]

            # Update task with current run
            await conn.execute(
                "UPDATE tasks SET worker_run_id = ? WHERE id = ?",
                (str(run_id), task_id),
            )

        task = await self.get_task(task_id)
        if task:
            task.claim_lock = claim_token
            task.claim_expires = expires
            task.worker_run_id = str(run_id)
        await self._insert_event(task_id, kind="claimed", run_id=run_id)
        return task

    async def complete_task(
        self,
        task_id: str,
        *,
        summary: Optional[str] = None,
        metadata: Optional[dict] = None,
        result: Optional[str] = None,
    ) -> bool:
        """Transition ``running -> done``."""
        now = time.time()
        async with self.write_txn() as conn:
            cur = await conn.execute(
                """UPDATE tasks
                   SET status = 'done', completed_at = ?, result = ?,
                       claim_lock = NULL, claim_expires = NULL,
                       consecutive_failures = 0, last_failure_error = NULL
                   WHERE id = ? AND status = 'running'""",
                (now, result or summary, task_id),
            )
            if cur.rowcount == 0:
                return False

            # Update latest run
            await conn.execute(
                """UPDATE task_runs
                   SET status = 'done', outcome = 'completed',
                       ended_at = ?, summary = ?, metadata = ?
                   WHERE task_id = ? AND status = 'running'""",
                (now, summary, json.dumps(metadata) if metadata else None, task_id),
            )

        await self._insert_event(
            task_id,
            kind="completed",
            payload={"summary": summary, "metadata": metadata},
        )
        return True

    async def block_task(
        self,
        task_id: str,
        *,
        reason: str,
    ) -> bool:
        """Transition ``running -> blocked``."""
        now = time.time()
        async with self.write_txn() as conn:
            cur = await conn.execute(
                """UPDATE tasks
                   SET status = 'blocked', last_failure_error = ?,
                       claim_lock = NULL, claim_expires = NULL
                   WHERE id = ? AND status = 'running'""",
                (reason, task_id),
            )
            if cur.rowcount == 0:
                return False

            await conn.execute(
                """UPDATE task_runs
                   SET status = 'blocked', outcome = 'blocked',
                       ended_at = ?, error = ?
                   WHERE task_id = ? AND status = 'running'""",
                (now, reason, task_id),
            )

        await self._insert_event(task_id, kind="blocked", payload={"reason": reason})
        return True

    async def unblock_task(self, task_id: str) -> bool:
        """Transition ``blocked -> ready``."""
        async with self.write_txn() as conn:
            cur = await conn.execute(
                """UPDATE tasks
                   SET status = 'ready', consecutive_failures = 0,
                       last_failure_error = NULL
                   WHERE id = ? AND status = 'blocked'""",
                (task_id,),
            )
        if cur.rowcount > 0:
            await self._insert_event(task_id, kind="unblocked")
        return cur.rowcount > 0

    async def release_task(self, task_id: str, *, reason: str = "") -> bool:
        """Release a running task back to ``ready`` WITHOUT counting as a failure.

        Used on graceful-shutdown cancellation so a task isn't pushed toward the
        circuit-breaker failure limit just because the process was stopped. The
        task's dependencies were already satisfied when it was claimed, so it can
        be re-dispatched immediately as ``ready``.
        """
        now = time.time()
        async with self.write_txn() as conn:
            cur = await conn.execute(
                """UPDATE tasks
                   SET status = 'ready', claim_lock = NULL, claim_expires = NULL,
                       started_at = NULL, worker_run_id = NULL
                   WHERE id = ? AND status = 'running'""",
                (task_id,),
            )
            if cur.rowcount == 0:
                return False
            await conn.execute(
                """UPDATE task_runs
                   SET status = 'released', outcome = 'cancelled', ended_at = ?
                   WHERE task_id = ? AND status = 'running'""",
                (now, task_id),
            )
        await self._insert_event(task_id, kind="released", payload={"reason": reason})
        return True

    async def heartbeat_claim(
        self,
        task_id: str,
        *,
        ttl_seconds: int = DEFAULT_CLAIM_TTL_SECONDS,
        note: Optional[str] = None,
    ) -> bool:
        """Extend the claim TTL and update heartbeat timestamp."""
        now = time.time()
        expires = now + ttl_seconds
        conn = await self._get_conn()
        async with self._db_lock:
            cur = await conn.execute(
                """UPDATE tasks
                   SET claim_expires = ?, last_heartbeat_at = ?
                   WHERE id = ? AND status = 'running'""",
                (expires, now, task_id),
            )
            await conn.commit()
        if cur.rowcount > 0:
            await self._insert_event(task_id, kind="heartbeat", payload={"note": note} if note else None)
        return cur.rowcount > 0

    # ── Dependency graph ──────────────────────────────────────────────

    async def link_tasks(self, parent_id: str, child_id: str) -> None:
        conn = await self._get_conn()
        async with self._db_lock:
            await conn.execute(
                "INSERT OR IGNORE INTO task_links (parent_id, child_id) VALUES (?,?)",
                (parent_id, child_id),
            )
            await conn.commit()
        await self._insert_event(child_id, kind="linked", payload={"parent_id": parent_id})

    async def unlink_tasks(self, parent_id: str, child_id: str) -> bool:
        conn = await self._get_conn()
        async with self._db_lock:
            cur = await conn.execute(
                "DELETE FROM task_links WHERE parent_id = ? AND child_id = ?",
                (parent_id, child_id),
            )
            await conn.commit()
        return cur.rowcount > 0

    async def get_parents(self, task_id: str) -> list[KanbanTask]:
        conn = await self._get_conn()
        async with conn.execute(
            """SELECT t.* FROM tasks t
               JOIN task_links tl ON t.id = tl.parent_id
               WHERE tl.child_id = ?""",
            (task_id,),
        ) as cur:
            rows = await cur.fetchall()
        return [self._row_to_task(r) for r in rows]

    async def get_children(self, task_id: str) -> list[KanbanTask]:
        conn = await self._get_conn()
        async with conn.execute(
            """SELECT t.* FROM tasks t
               JOIN task_links tl ON t.id = tl.child_id
               WHERE tl.parent_id = ?""",
            (task_id,),
        ) as cur:
            rows = await cur.fetchall()
        return [self._row_to_task(r) for r in rows]

    async def recompute_ready(self) -> int:
        """Promote ``todo -> ready`` where all parents are done. Returns count."""
        async with self.write_txn() as conn:
            async with conn.execute(
                """SELECT t.id FROM tasks t
                   WHERE t.status = 'todo'
                   AND NOT EXISTS (
                       SELECT 1 FROM task_links tl
                       JOIN tasks pt ON pt.id = tl.parent_id
                       WHERE tl.child_id = t.id AND pt.status != 'done'
                   )"""
            ) as cur:
                promotable = [r["id"] for r in await cur.fetchall()]

            if not promotable:
                return 0

            placeholders = ",".join("?" * len(promotable))
            await conn.execute(
                f"UPDATE tasks SET status = 'ready' WHERE id IN ({placeholders}) AND status = 'todo'",
                promotable,
            )

        for tid in promotable:
            await self._insert_event(tid, kind="promoted")

        return len(promotable)

    # ── Dispatcher helpers ────────────────────────────────────────────

    async def release_stale_claims(
        self,
        ttl_seconds: int = DEFAULT_CLAIM_TTL_SECONDS,
    ) -> int:
        """Release claims where TTL has expired."""
        now = time.time()
        async with self.write_txn() as conn:
            # Also check heartbeat — if last_heartbeat is stale, reclaim
            # Capture IDs of stale tasks BEFORE updating, so task_runs
            # subquery only matches the tasks we actually reclaimed.
            stale_ids: list[str] = []
            async with conn.execute(
                """SELECT id FROM tasks
                   WHERE status = 'running'
                   AND (
                       (claim_expires IS NOT NULL AND claim_expires < ?)
                       OR (last_heartbeat_at IS NOT NULL AND last_heartbeat_at < ?)
                   )""",
                (now, now - ttl_seconds),
            ) as cur:
                stale_ids = [r["id"] for r in await cur.fetchall()]

            count = len(stale_ids)

            if count > 0:
                placeholders = ",".join("?" * count)
                await conn.execute(
                    f"""UPDATE tasks
                       SET status = 'ready', claim_lock = NULL, claim_expires = NULL,
                           started_at = NULL, worker_run_id = NULL
                       WHERE id IN ({placeholders})""",
                    stale_ids,
                )

                # Mark runs as released — only for the tasks we actually reclaimed
                await conn.execute(
                    f"""UPDATE task_runs
                       SET status = 'released', outcome = 'reclaimed', ended_at = ?
                       WHERE status = 'running'
                       AND task_id IN ({placeholders})""",
                    [now] + stale_ids,
                )

        if count > 0:
            logger.info("Reclaimed %d stale claim(s)", count)
        return count

    async def enforce_max_runtime(self) -> list[str]:
        """Force-release tasks that exceeded their max_runtime_seconds."""
        conn = await self._get_conn()
        now = time.time()
        timed_out: list[str] = []

        async with conn.execute(
            """SELECT id, max_runtime_seconds, started_at FROM tasks
               WHERE status = 'running'
               AND max_runtime_seconds IS NOT NULL
               AND started_at IS NOT NULL"""
        ) as cur:
            rows = await cur.fetchall()

        for row in rows:
            if now - row["started_at"] > row["max_runtime_seconds"]:
                async with self.write_txn() as conn:
                    cur = await conn.execute(
                        """UPDATE tasks
                           SET status = 'ready', claim_lock = NULL,
                               claim_expires = NULL, started_at = NULL,
                               worker_run_id = NULL
                           WHERE id = ? AND status = 'running'""",
                        (row["id"],),
                    )
                    if cur.rowcount == 0:
                        continue  # Task already transitioned by another process
                    await conn.execute(
                        """UPDATE task_runs
                           SET status = 'timed_out', outcome = 'timed_out', ended_at = ?
                           WHERE task_id = ? AND status = 'running'""",
                        (now, row["id"]),
                    )
                timed_out.append(row["id"])
                await self._insert_event(row["id"], kind="timed_out")

        return timed_out

    async def record_task_failure(
        self,
        task_id: str,
        error: str,
        *,
        outcome: str = "crashed",
        failure_limit: int = DEFAULT_FAILURE_LIMIT,
    ) -> bool:
        """Record a worker failure. Returns True if the task was auto-blocked."""
        now = time.time()
        auto_blocked = False

        async with self.write_txn() as conn:
            # Increment consecutive_failures — only if still running
            cur = await conn.execute(
                """UPDATE tasks
                   SET consecutive_failures = consecutive_failures + 1,
                       last_failure_error = ?,
                       status = 'todo', claim_lock = NULL,
                       claim_expires = NULL, started_at = NULL,
                       worker_run_id = NULL
                   WHERE id = ? AND status = 'running'""",
                (error, task_id),
            )
            if cur.rowcount == 0:
                return False  # Task already transitioned by another process

            # Check failure limit
            async with conn.execute(
                "SELECT consecutive_failures, max_retries FROM tasks WHERE id = ?",
                (task_id,),
            ) as cur:
                row = await cur.fetchone()

            if row:
                limit = row["max_retries"] if row["max_retries"] is not None else failure_limit
                if row["consecutive_failures"] >= limit:
                    await conn.execute(
                        """UPDATE tasks SET status = 'blocked'
                           WHERE id = ?""",
                        (task_id,),
                    )
                    auto_blocked = True

            # Update run
            await conn.execute(
                """UPDATE task_runs
                   SET status = ?, outcome = ?, ended_at = ?, error = ?
                   WHERE task_id = ? AND status = 'running'""",
                (outcome, outcome, now, error, task_id),
            )

        kind = "auto_blocked" if auto_blocked else "crashed"
        await self._insert_event(task_id, kind=kind, payload={"error": error})
        return auto_blocked

    async def list_ready_unclaimed(
        self,
        *,
        board: Optional[str] = None,
    ) -> list[KanbanTask]:
        """List ready tasks that have an assignee but no active claim."""
        conn = await self._get_conn()
        conds = [
            "status = 'ready'",
            "assignee IS NOT NULL",
            "claim_lock IS NULL",
        ]
        params: list = []
        if board:
            conds.append("board = ?")
            params.append(board)

        where = " AND ".join(conds)
        async with conn.execute(
            f"SELECT * FROM tasks WHERE {where} ORDER BY priority DESC, created_at ASC",
            params,
        ) as cur:
            rows = await cur.fetchall()
        return [self._row_to_task(r) for r in rows]

    async def count_running(self, *, board: Optional[str] = None) -> int:
        conn = await self._get_conn()
        if board:
            async with conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE status = 'running' AND board = ?",
                (board,),
            ) as cur:
                return (await cur.fetchone())[0]
        async with conn.execute("SELECT COUNT(*) FROM tasks WHERE status = 'running'") as cur:
            return (await cur.fetchone())[0]

    # ── Comments ──────────────────────────────────────────────────────

    async def add_comment(self, task_id: str, author: str, body: str) -> int:
        conn = await self._get_conn()
        async with self._db_lock:
            cur = await conn.execute(
                "INSERT INTO task_comments (task_id, author, body, created_at) VALUES (?,?,?,?)",
                (task_id, author, body, time.time()),
            )
            await conn.commit()
        return cur.lastrowid

    async def list_comments(self, task_id: str, limit: int = 30) -> list[KanbanComment]:
        conn = await self._get_conn()
        async with conn.execute(
            "SELECT * FROM task_comments WHERE task_id = ? ORDER BY created_at DESC LIMIT ?",
            (task_id, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [
            KanbanComment(
                id=r["id"],
                task_id=r["task_id"],
                author=r["author"],
                body=r["body"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    # ── Events / runs ─────────────────────────────────────────────────

    async def _insert_event(
        self,
        task_id: str,
        *,
        kind: str,
        run_id: Optional[int] = None,
        payload: Optional[dict] = None,
    ) -> None:
        try:
            conn = await self._get_conn()
            async with self._db_lock:
                await conn.execute(
                    "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) VALUES (?,?,?,?,?)",
                    (task_id, run_id, kind, json.dumps(payload) if payload else None, time.time()),
                )
                await conn.commit()
        except Exception:
            logger.debug("Failed to insert kanban event", exc_info=True)

    async def list_events(self, task_id: str, limit: int = 50) -> list[KanbanEvent]:
        conn = await self._get_conn()
        async with conn.execute(
            "SELECT * FROM task_events WHERE task_id = ? ORDER BY id DESC LIMIT ?",
            (task_id, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [
            KanbanEvent(
                id=r["id"],
                task_id=r["task_id"],
                run_id=r["run_id"],
                kind=r["kind"],
                payload=json.loads(r["payload"]) if r["payload"] else None,
                created_at=r["created_at"],
            )
            for r in rows
        ]

    async def list_runs(self, task_id: str, limit: int = 10) -> list[KanbanRun]:
        conn = await self._get_conn()
        async with conn.execute(
            "SELECT * FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT ?",
            (task_id, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [
            KanbanRun(
                id=r["id"],
                task_id=r["task_id"],
                profile=r["profile"],
                status=r["status"],
                claim_lock=r["claim_lock"],
                claim_expires=r["claim_expires"],
                worker_run_id=r["worker_run_id"],
                started_at=r["started_at"],
                ended_at=r["ended_at"],
                outcome=r["outcome"],
                summary=r["summary"],
                metadata=json.loads(r["metadata"]) if r["metadata"] else None,
                error=r["error"],
            )
            for r in rows
        ]

    # ── Notifications ─────────────────────────────────────────────────

    async def add_notify_sub(
        self,
        *,
        task_id: str,
        platform: str,
        chat_id: str,
        thread_id: str = "",
        user_id: Optional[str] = None,
    ) -> None:
        conn = await self._get_conn()
        # Use INSERT OR IGNORE to preserve last_event_id on re-subscription
        async with self._db_lock:
            await conn.execute(
                """INSERT OR IGNORE INTO kanban_notify_subs
                   (task_id, platform, chat_id, thread_id, user_id, created_at)
                   VALUES (?,?,?,?,?,?)""",
                (task_id, platform, chat_id, thread_id, user_id, time.time()),
            )
            await conn.commit()

    async def list_notify_subs(self, task_id: str) -> list[dict]:
        conn = await self._get_conn()
        async with conn.execute("SELECT * FROM kanban_notify_subs WHERE task_id = ?", (task_id,)) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def remove_notify_sub(
        self,
        *,
        task_id: str,
        platform: str,
        chat_id: str,
        thread_id: str = "",
    ) -> bool:
        conn = await self._get_conn()
        async with self._db_lock:
            cur = await conn.execute(
                """DELETE FROM kanban_notify_subs
                   WHERE task_id = ? AND platform = ? AND chat_id = ? AND thread_id = ?""",
                (task_id, platform, chat_id, thread_id),
            )
            await conn.commit()
        return cur.rowcount > 0

    # ── Worker context builder ────────────────────────────────────────

    async def build_worker_context(self, task_id: str) -> str:
        """Assemble context text for a worker: parent results, comments, prior runs."""
        parts: list[str] = []

        # Parent task results
        parents = await self.get_parents(task_id)
        if parents:
            parts.append("## Parent Tasks (results)")
            for p in parents:
                summary = p.result or "(no result)"
                parts.append(f"- **{p.title}** [{p.id}] → {summary[:500]}")
            parts.append("")

        # Comments (most recent 30, cap each at 2KB)
        comments = await self.list_comments(task_id, limit=30)
        if comments:
            parts.append("## Comments")
            for c in reversed(comments):  # chronological order
                parts.append(f"[{c.author}] {c.body[:2000]}")
            parts.append("")

        # Prior runs (most recent 10)
        runs = await self.list_runs(task_id, limit=10)
        prior_runs = [r for r in runs if r.outcome is not None]
        if prior_runs:
            parts.append("## Prior Attempts")
            for r in prior_runs:
                outcome = r.outcome or r.status
                err = f" | Error: {r.error[:300]}" if r.error else ""
                summ = f" | Summary: {r.summary[:300]}" if r.summary else ""
                parts.append(f"- Run #{r.id}: {outcome}{err}{summ}")
            parts.append("")

        return "\n".join(parts)

    # ── Board stats ───────────────────────────────────────────────────

    async def board_stats(self, board: str = "default") -> dict:
        conn = await self._get_conn()
        stats: dict[str, int] = {}
        for s in VALID_STATUSES:
            async with conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE status = ? AND board = ?",
                (s, board),
            ) as cur:
                stats[s] = (await cur.fetchone())[0]
        return stats

    # ── Internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _row_to_task(row: aiosqlite.Row) -> KanbanTask:
        skills_raw = row["skills"]
        skills = json.loads(skills_raw) if skills_raw else None
        return KanbanTask(
            id=row["id"],
            title=row["title"],
            body=row["body"],
            assignee=row["assignee"],
            status=row["status"],
            priority=row["priority"],
            created_by=row["created_by"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            workspace_kind=row["workspace_kind"] or "scratch",
            workspace_path=row["workspace_path"],
            claim_lock=row["claim_lock"],
            claim_expires=row["claim_expires"],
            tenant=row["tenant"],
            result=row["result"],
            idempotency_key=row["idempotency_key"],
            consecutive_failures=row["consecutive_failures"],
            worker_run_id=row["worker_run_id"],
            max_runtime_seconds=row["max_runtime_seconds"],
            last_heartbeat_at=row["last_heartbeat_at"],
            last_failure_error=row["last_failure_error"],
            skills=skills,
            max_retries=row["max_retries"],
            board=row["board"] or "default",
        )
