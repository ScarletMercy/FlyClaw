"""Structured audit log store for tool call tracking.

Provides a SQLite-backed store for recording and querying tool call history.
Used by the Dashboard audit log and analytics features.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger("myclaw.analytics.audit")


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
    """SQLite-backed audit log store."""

    def __init__(self, db_path: str = "data/audit.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize the audit table."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
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
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tool_calls_tool ON tool_calls(tool_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tool_calls_thread ON tool_calls(thread_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tool_calls_ts ON tool_calls(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tool_calls_success ON tool_calls(success)")

    def record_call(
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
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(
                """INSERT INTO tool_calls 
                   (thread_id, tool_name, sender_id, channel, success, duration_ms, args_preview, error, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (thread_id, tool_name, sender_id, channel, int(success), duration_ms, args_preview[:500], error, time.time()),
            )
            return cursor.lastrowid

    def query(
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
        params = []

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
        query = f"""
            SELECT id, thread_id, tool_name, sender_id, channel, success, 
                   duration_ms, args_preview, error, timestamp
            FROM tool_calls
            WHERE {where}
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()

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

    def get_stats(self, days: int = 7) -> dict:
        """Get audit statistics for the last N days."""
        from_ts = time.time() - (days * 86400)

        with sqlite3.connect(str(self.db_path)) as conn:
            # Total calls
            total = conn.execute(
                "SELECT COUNT(*) FROM tool_calls WHERE timestamp >= ?", (from_ts,)
            ).fetchone()[0]

            # Success rate
            success_count = conn.execute(
                "SELECT COUNT(*) FROM tool_calls WHERE timestamp >= ? AND success = 1", (from_ts,)
            ).fetchone()[0]

            # Average duration
            avg_duration = conn.execute(
                "SELECT AVG(duration_ms) FROM tool_calls WHERE timestamp >= ?", (from_ts,)
            ).fetchone()[0] or 0

            # Top tools
            top_tools = conn.execute(
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
            ).fetchall()

            # Error count
            error_count = conn.execute(
                "SELECT COUNT(*) FROM tool_calls WHERE timestamp >= ? AND success = 0", (from_ts,)
            ).fetchone()[0]

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

    def prune(self, older_than_days: int = 90) -> int:
        """Remove entries older than N days. Returns count of removed entries."""
        from_ts = time.time() - (older_than_days * 86400)
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute("DELETE FROM tool_calls WHERE timestamp < ?", (from_ts,))
            conn.commit()
            return cursor.rowcount


# Module-level singleton
_store: Optional[AuditStore] = None


def get_audit_store(db_path: str = "data/audit.db") -> AuditStore:
    """Get or create the audit store singleton."""
    global _store
    if _store is None:
        _store = AuditStore(db_path)
    return _store


def reset_audit_store(db_path: str = "data/audit.db") -> AuditStore:
    """Reset the audit store singleton (for testing or multi-environment)."""
    global _store
    _store = AuditStore(db_path)
    return _store
