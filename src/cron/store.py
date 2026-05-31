from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import aiosqlite

from .types import CronJob

logger = logging.getLogger("flyclaw.cron.store")


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

    async def save_job(self, job: CronJob) -> None:
        conn = await self._get_conn()
        new_version = job.version + 1
        cursor = await conn.execute(
            "UPDATE cron_jobs SET data = ?, version = ? WHERE id = ? AND version = ?",
            (job.model_dump_json(), new_version, job.id, job.version),
        )
        await conn.commit()
        if cursor.rowcount == 0:
            async with conn.execute("SELECT id FROM cron_jobs WHERE id = ?", (job.id,)) as check_cursor:
                if await check_cursor.fetchone() is None:
                    await conn.execute(
                        "INSERT INTO cron_jobs (id, data, version) VALUES (?, ?, ?)",
                        (job.id, job.model_dump_json(), new_version),
                    )
                    await conn.commit()
                else:
                    logger.warning(
                        "Cron job %s version mismatch (expected %d), forcing update",
                        job.id,
                        job.version,
                    )
                    await conn.execute(
                        "UPDATE cron_jobs SET data = ?, version = ? WHERE id = ?",
                        (job.model_dump_json(), new_version, job.id),
                    )
                    await conn.commit()
        job.version = new_version

    async def remove_job(self, job_id: str) -> bool:
        conn = await self._get_conn()
        cursor = await conn.execute("DELETE FROM cron_jobs WHERE id = ?", (job_id,))
        await conn.commit()
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
