from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import aiosqlite

from .types import TaskCheckpoint, TaskRun

logger = logging.getLogger("flyclaw.task.store")

_DEFAULT_DB = "~/.flyclaw/data/task_runs.db"


class TaskRunStore:
    def __init__(self, db_path: str = _DEFAULT_DB):
        self.db_path = str(Path(db_path).expanduser().resolve())
        self._conn: Optional[aiosqlite.Connection] = None

    async def initialize(self):
        if self._conn is not None:
            return
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS task_runs (
                id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_task_runs_status
            ON task_runs(json_extract(data, '$.status'))
        """)
        await self._conn.commit()

    async def _ensure_initialized(self):
        if self._conn is None:
            await self.initialize()

    async def close(self):
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def save(self, run: TaskRun):
        await self._ensure_initialized()
        run.updated_at = time.time()
        data = run.model_dump_json()
        await self._conn.execute(
            "INSERT OR REPLACE INTO task_runs (id, data, updated_at) VALUES (?, ?, ?)",
            (run.id, data, run.updated_at),
        )
        await self._conn.commit()

    async def get(self, run_id: str) -> Optional[TaskRun]:
        await self._ensure_initialized()
        cursor = await self._conn.execute(
            "SELECT data FROM task_runs WHERE id = ?", (run_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return TaskRun.model_validate_json(row[0])

    async def list_by_status(self, *statuses: str) -> list[TaskRun]:
        await self._ensure_initialized()
        if not statuses:
            return []
        placeholders = ",".join("?" for _ in statuses)
        cursor = await self._conn.execute(
            f"SELECT data FROM task_runs WHERE json_extract(data, '$.status') IN ({placeholders}) ORDER BY updated_at DESC",
            statuses,
        )
        rows = await cursor.fetchall()
        return [TaskRun.model_validate_json(r[0]) for r in rows]

    async def update_status(self, run_id: str, status: str):
        run = await self.get(run_id)
        if run:
            run.status = status
            await self.save(run)

    async def update_checkpoint(self, run_id: str, checkpoint_id: str, **kwargs):
        run = await self.get(run_id)
        if not run:
            return
        for cp in run.checkpoints:
            if cp.id == checkpoint_id:
                for k, v in kwargs.items():
                    setattr(cp, k, v)
                break
        await self.save(run)

    async def delete(self, run_id: str):
        await self._ensure_initialized()
        await self._conn.execute("DELETE FROM task_runs WHERE id = ?", (run_id,))
        await self._conn.commit()


_store: Optional[TaskRunStore] = None


def get_task_store(db_path: str = _DEFAULT_DB) -> TaskRunStore:
    global _store
    if _store is None:
        _store = TaskRunStore(db_path)
    return _store
