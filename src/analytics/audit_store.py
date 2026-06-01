"""Structured audit log store for tool call tracking.

Provides a SQLite-backed store for recording and querying tool call history.
Used by the Dashboard audit log and analytics features.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import aiosqlite

logger = logging.getLogger("flyclaw.analytics.audit")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    sender_id TEXT NOT NULL DEFAULT '',
    channel TEXT NOT NULL DEFAULT '',
    success INTEGER NOT NULL,
    duration_ms REAL NOT NULL,
    args_preview TEXT NOT NULL DEFAULT '',
    error TEXT,
    timestamp REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tool_calls_tool ON tool_calls(tool_name);
CREATE INDEX IF NOT EXISTS idx_tool_calls_thread ON tool_calls(thread_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_ts ON tool_calls(timestamp);
CREATE INDEX IF NOT EXISTS idx_tool_calls_success ON tool_calls(success);
"""


@dataclass
class AuditEntry:
    id: int
    thread_id: str
    tool_name: str
    sender_id: str
    channel: str
    success: bool
    duration_ms: float
    args_preview: str
    error: Optional[str]
    timestamp: float


class AuditStore:
    """SQLite-backed audit log store (async)."""

    def __init__(self, db_path: str = "~/.flyclaw/data/audit.db"):
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[aiosqlite.Connection] = None

    async def _get_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            self._conn = await aiosqlite.connect(str(self.db_path))
            self._conn.row_factory = aiosqlite.Row
            await self._conn.execute("PRAGMA journal_mode=WAL")
            await self._conn.executescript(_SCHEMA)
            await self._conn.commit()
        return self._conn

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def record_call(
        self,
        thread_id: str,
        tool_name: str,
        sender_id: str = "",
        channel: str = "",
        success: bool = True,
        duration_ms: float = 0.0,
        args_preview: str = "",
        error: Optional[str] = None,
    ) -> int:
        """Record a tool call. Returns the new entry ID."""
        conn = await self._get_conn()
        cursor = await conn.execute(
            """INSERT INTO tool_calls
               (thread_id, tool_name, sender_id, channel, success, duration_ms, args_preview, error, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                thread_id,
                tool_name,
                sender_id,
                channel,
                int(success),
                duration_ms,
                args_preview[:500],
                error,
                time.time(),
            ),
        )
        await conn.commit()
        return cursor.lastrowid

    async def query(
        self,
        tool_name: Optional[str] = None,
        sender_id: Optional[str] = None,
        success: Optional[bool] = None,
        from_ts: Optional[float] = None,
        to_ts: Optional[float] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditEntry]:
        """Query audit entries with filters."""
        conditions = []
        params: list = []

        if tool_name:
            conditions.append("tool_name = ?")
            params.append(tool_name)
        if sender_id:
            conditions.append("sender_id = ?")
            params.append(sender_id)
        if success is not None:
            conditions.append("success = ?")
            params.append(int(success))
        if from_ts:
            conditions.append("timestamp >= ?")
            params.append(from_ts)
        if to_ts:
            conditions.append("timestamp <= ?")
            params.append(to_ts)

        where = " AND ".join(conditions) if conditions else "1=1"
        sql = f"""
            SELECT id, thread_id, tool_name, sender_id, channel, success,
                   duration_ms, args_preview, error, timestamp
            FROM tool_calls
            WHERE {where}
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        conn = await self._get_conn()
        async with conn.execute(sql, params) as cursor:
            rows = await cursor.fetchall()

        return [
            AuditEntry(
                id=r["id"],
                thread_id=r["thread_id"],
                tool_name=r["tool_name"],
                sender_id=r["sender_id"],
                channel=r["channel"],
                success=bool(r["success"]),
                duration_ms=r["duration_ms"],
                args_preview=r["args_preview"],
                error=r["error"],
                timestamp=r["timestamp"],
            )
            for r in rows
        ]

    async def get_stats(self, days: int = 7) -> dict:
        """Get audit statistics for the last N days."""
        from_ts = time.time() - (days * 86400)
        conn = await self._get_conn()

        # Total calls
        async with conn.execute("SELECT COUNT(*) FROM tool_calls WHERE timestamp >= ?", (from_ts,)) as cur:
            row = await cur.fetchone()
            total = row[0]

        # Success count
        async with conn.execute(
            "SELECT COUNT(*) FROM tool_calls WHERE timestamp >= ? AND success = 1", (from_ts,)
        ) as cur:
            row = await cur.fetchone()
            success_count = row[0]

        # Average duration
        async with conn.execute("SELECT AVG(duration_ms) FROM tool_calls WHERE timestamp >= ?", (from_ts,)) as cur:
            row = await cur.fetchone()
            avg_duration = row[0] or 0

        # Top tools
        async with conn.execute(
            """SELECT tool_name,
                      COUNT(*) as count,
                      CAST(SUM(success) AS FLOAT) / COUNT(*) as success_rate,
                      AVG(duration_ms) as avg_duration_ms
               FROM tool_calls
               WHERE timestamp >= ?
               GROUP BY tool_name
               ORDER BY count DESC
               LIMIT 10""",
            (from_ts,),
        ) as cur:
            top_tools = await cur.fetchall()

        # Error count
        async with conn.execute(
            "SELECT COUNT(*) FROM tool_calls WHERE timestamp >= ? AND success = 0", (from_ts,)
        ) as cur:
            row = await cur.fetchone()
            error_count = row[0]

        return {
            "total_calls": total,
            "success_count": success_count,
            "error_count": error_count,
            "success_rate": success_count / total if total > 0 else 0,
            "avg_duration_ms": round(avg_duration, 2),
            "top_tools": [
                {
                    "tool": r[0],
                    "count": r[1],
                    "success_rate": round(r[2], 4) if r[2] is not None else 0,
                    "avg_duration_ms": round(r[3], 2) if r[3] is not None else 0,
                }
                for r in top_tools
            ],
            "period_days": days,
        }

    async def prune(self, older_than_days: int = 90) -> int:
        """Remove entries older than N days. Returns count of removed entries."""
        from_ts = time.time() - (older_than_days * 86400)
        conn = await self._get_conn()
        cursor = await conn.execute("DELETE FROM tool_calls WHERE timestamp < ?", (from_ts,))
        await conn.commit()
        return cursor.rowcount


# Module-level singleton
_store: Optional[AuditStore] = None
_subscriptions: list = []


def get_audit_store(db_path: str = "~/.flyclaw/data/audit.db") -> AuditStore:
    """Get or create the audit store singleton."""
    global _store
    if _store is None:
        _store = AuditStore(db_path)
    return _store


def reset_audit_store(db_path: str = "~/.flyclaw/data/audit.db") -> AuditStore:
    """Reset the audit store singleton (for testing or multi-environment)."""
    global _store, _subscriptions
    _store = AuditStore(db_path)
    _subscriptions = []
    return _store


def subscribe_audit_to_events() -> None:
    """Subscribe the audit store to tool execution events.

    This replaces the direct get_audit_store() calls in the agent loop,
    decoupling audit logging from tool execution.
    """
    global _subscriptions
    store = get_audit_store()

    from src.events import subscribe_async

    async def _on_tool_completed(event, **ctx):
        await store.record_call(
            thread_id=ctx.get("thread_id", ""),
            tool_name=ctx.get("tool_name", ""),
            sender_id=ctx.get("sender_id", ""),
            channel=ctx.get("channel", ""),
            success=True,
            duration_ms=ctx.get("duration_ms", 0.0),
            args_preview=ctx.get("args_preview", ""),
        )

    async def _on_tool_failed(event, **ctx):
        await store.record_call(
            thread_id=ctx.get("thread_id", ""),
            tool_name=ctx.get("tool_name", ""),
            sender_id=ctx.get("sender_id", ""),
            channel=ctx.get("channel", ""),
            success=False,
            duration_ms=ctx.get("duration_ms", 0.0),
            args_preview=ctx.get("args_preview", ""),
            error=ctx.get("error", ""),
        )

    _subscriptions.append(subscribe_async("tool.exec_completed", _on_tool_completed))
    _subscriptions.append(subscribe_async("tool.exec_failed", _on_tool_failed))
