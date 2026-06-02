from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

import aiosqlite

from .types import CronJob

logger = logging.getLogger("flyclaw.cron.store")

# Maximum retry attempts for optimistic-lock conflicts in save_job.
# Exposed as a named constant so callers and reviewers can find it easily.
_OPTIMISTIC_LOCK_MAX_RETRIES = 3

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS cron_jobs (
    id TEXT PRIMARY KEY,
    data TEXT NOT NULL,
    version INTEGER DEFAULT 0
)
"""

_CREATE_RUN_LOG_SQL = """
CREATE TABLE IF NOT EXISTS cron_run_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    status TEXT NOT NULL,
    output TEXT,
    error TEXT,
    started_at REAL NOT NULL,
    finished_at REAL,
    created_at REAL DEFAULT (strftime('%s', 'now'))
)
"""

_CREATE_INDEX_JOB_ID_SQL = "CREATE INDEX IF NOT EXISTS idx_run_logs_job_id ON cron_run_logs(job_id)"

_CREATE_INDEX_CREATED_AT_SQL = "CREATE INDEX IF NOT EXISTS idx_run_logs_created_at ON cron_run_logs(created_at)"


class CronStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[aiosqlite.Connection] = None
        # Per-job locks: only serializes retries for the *same* job.
        # Different jobs no longer block each other.
        self._job_locks: dict[str, asyncio.Lock] = {}

    async def __aenter__(self):
        await self._get_conn()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        return False

    async def _get_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            self._conn = await aiosqlite.connect(str(self.db_path))
            await self._conn.execute("PRAGMA journal_mode=WAL")
            await self._conn.execute(_CREATE_TABLE_SQL)
            await self._conn.execute(_CREATE_RUN_LOG_SQL)
            await self._conn.execute(_CREATE_INDEX_JOB_ID_SQL)
            await self._conn.execute(_CREATE_INDEX_CREATED_AT_SQL)
            await self._conn.commit()
        return self._conn

    async def load_jobs(self) -> list[CronJob]:
        conn = await self._get_conn()
        async with conn.execute("SELECT data FROM cron_jobs") as cursor:
            rows = await cursor.fetchall()
        jobs = []
        for (data_str,) in rows:
            try:
                jobs.append(CronJob.model_validate_json(data_str))
            except Exception as e:
                logger.warning("Failed to parse cron job: %s", e)
        return jobs

    def _get_job_lock(self, job_id: str) -> asyncio.Lock:
        """Return a per-job lock, auto-created on first use."""
        # Fast path: lock already exists
        lock = self._job_locks.get(job_id)
        if lock is not None:
            return lock
        # Slow path: create if missing (race is harmless — two locks for the
        # same id just means slightly less serialisation on the very first
        # concurrent access, which the SQL-level optimistic lock already covers).
        lock = asyncio.Lock()
        self._job_locks[job_id] = lock
        return lock

    async def save_job(self, job: CronJob, *, max_retries: int = _OPTIMISTIC_LOCK_MAX_RETRIES) -> None:
        # Per-job lock: only blocks concurrent saves for the *same* job_id.
        # Different jobs run fully in parallel — SQL-level optimistic lock
        # (version column) handles correctness.
        async with self._get_job_lock(job.id):
            conn = await self._get_conn()
            for attempt in range(max_retries + 1):
                new_version = job.version + 1
                old_version = job.version
                # Set version before serialize so JSON data matches the column
                job.version = new_version
                cursor = await conn.execute(
                    "UPDATE cron_jobs SET data = ?, version = ? WHERE id = ? AND version = ?",
                    (job.model_dump_json(), new_version, job.id, old_version),
                )
                await conn.commit()
                if cursor.rowcount > 0:
                    return  # job.version already updated
                # Restore version for retry
                job.version = old_version
                # Version conflict — check if row exists
                async with conn.execute("SELECT version FROM cron_jobs WHERE id = ?", (job.id,)) as cur:
                    row = await cur.fetchone()
                if row is None:
                    # Row doesn't exist — insert
                    job.version = new_version
                    await conn.execute(
                        "INSERT INTO cron_jobs (id, data, version) VALUES (?, ?, ?)",
                        (job.id, job.model_dump_json(), new_version),
                    )
                    await conn.commit()
                    return
                # Row exists but version mismatch — adopt DB column version and retry
                # On RuntimeError below, job.version is left at old_version (restored above)
                if attempt >= max_retries:
                    raise RuntimeError(
                        f"Optimistic lock conflict for cron job {job.id}: failed after {max_retries + 1} attempts"
                    )
                db_version = row[0]
                job.version = db_version
                logger.info(
                    "Retrying save_job for %s (attempt %d/%d, version %d->%d)",
                    job.id,
                    attempt + 2,
                    max_retries + 1,
                    db_version,
                    db_version + 1,
                )

    async def remove_job(self, job_id: str) -> bool:
        # DELETE by PK is atomic in SQLite — no Python-level lock needed.
        conn = await self._get_conn()
        cursor = await conn.execute("DELETE FROM cron_jobs WHERE id = ?", (job_id,))
        await conn.commit()
        self._job_locks.pop(job_id, None)  # clean up per-job lock
        return cursor.rowcount > 0

    async def get_job(self, job_id: str) -> Optional[CronJob]:
        conn = await self._get_conn()
        async with conn.execute("SELECT data FROM cron_jobs WHERE id = ?", (job_id,)) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return CronJob.model_validate_json(row[0])

    async def update_job_state(
        self,
        job_id: str,
        *,
        consecutive_errors: Optional[int] = None,
        last_run_at: Optional[float] = None,
        last_run_status: Optional[str] = None,
        last_error: Optional[str] = None,
        next_run_at: Optional[float] = None,
        running_at: Optional[float] = None,
    ) -> None:
        job = await self.get_job(job_id)
        if job is None:
            return
        updates = {
            "consecutive_errors": consecutive_errors,
            "last_run_at": last_run_at,
            "last_run_status": last_run_status,
            "last_error": last_error,
            "next_run_at": next_run_at,
            "running_at": running_at,
        }
        for key, val in updates.items():
            if val is not None:
                setattr(job, key, val)
        await self.save_job(job)

    async def save_run_log(
        self,
        job_id: str,
        status: str,
        started_at: float,
        finished_at: Optional[float] = None,
        output: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        conn = await self._get_conn()
        await conn.execute(
            "INSERT INTO cron_run_logs (job_id, status, output, error, started_at, finished_at) VALUES (?, ?, ?, ?, ?, ?)",
            (job_id, status, output, error, started_at, finished_at),
        )
        await conn.commit()

    async def get_run_logs(self, job_id: str, limit: int = 50) -> list[dict]:
        conn = await self._get_conn()
        async with conn.execute(
            "SELECT id, job_id, status, output, error, started_at, finished_at FROM cron_run_logs WHERE job_id = ? ORDER BY id DESC LIMIT ?",
            (job_id, limit),
        ) as cursor:
            rows = await cursor.fetchall()
        return [
            {
                "id": r[0],
                "job_id": r[1],
                "status": r[2],
                "output": r[3],
                "error": r[4],
                "started_at": r[5],
                "finished_at": r[6],
            }
            for r in rows
        ]

    async def close(self):
        if self._conn:
            await self._conn.close()
            self._conn = None
