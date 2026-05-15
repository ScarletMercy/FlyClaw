"""Session index store: SQLite + FTS5 for full-text search over conversation history."""

from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path
from typing import Optional

logger = logging.getLogger("myclaw.session_index")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    thread_id TEXT PRIMARY KEY,
    channel TEXT NOT NULL,
    sender_id TEXT,
    chat_id TEXT,
    chat_type TEXT,
    first_message_at REAL,
    last_message_at REAL,
    message_count INTEGER DEFAULT 0,
    is_active INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL REFERENCES sessions(thread_id),
    message_id TEXT NOT NULL UNIQUE,
    role TEXT NOT NULL,
    content TEXT,
    tool_name TEXT,
    tool_calls TEXT,
    timestamp REAL NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    content=messages,
    content_rowid=id
);

CREATE TRIGGER IF NOT EXISTS messages_fts_insert
    AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_delete
    AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content)
        VALUES('delete', old.id, old.content);
END;

CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);
CREATE INDEX IF NOT EXISTS idx_sessions_channel ON sessions(channel);
CREATE INDEX IF NOT EXISTS idx_sessions_last ON sessions(last_message_at);
CREATE INDEX IF NOT EXISTS idx_sessions_active ON sessions(is_active);
"""

# ── Module-level singleton — delegates to ServiceContainer ──

from src._container import get_container


def get_session_index() -> Optional[SessionIndexStore]:
    return get_container().session_index


def parse_thread_id(thread_id: str) -> dict:
    parts = thread_id.split(":")
    channel = parts[0] if parts else "unknown"
    scope = parts[1] if len(parts) > 1 else ""

    if scope == "user":
        return {"channel": channel, "chat_type": "p2p", "sender_id": parts[2] if len(parts) > 2 else ""}
    if scope == "group":
        return {"channel": channel, "chat_type": "group", "sender_id": ""}
    if scope == "global":
        return {"channel": channel, "chat_type": "p2p", "sender_id": ""}
    if re.match(r"^s\d+$", scope):
        return {"channel": channel, "chat_type": "p2p", "sender_id": parts[2] if len(parts) > 2 else ""}
    return {"channel": channel, "chat_type": "p2p", "sender_id": ""}


def _sanitize_fts5_query(query: str) -> str:
    """Build FTS5 query. Keeps Chinese words intact, joins with OR for recall."""
    query = query.strip()
    if not query:
        return '""'
    if '"' in query:
        return query
    # Split into segments: Chinese char sequences and English words
    tokens = re.findall(r'[一-鿿]+|[a-zA-Z0-9_]+', query)
    if not tokens:
        return '""'
    return " OR ".join(tokens)


class SessionIndexStore:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(db_path)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(_SCHEMA)

    def close(self):
        if self._db:
            self._db.close()
            self._db = None

    def upsert_session(
        self,
        thread_id: str,
        channel: str,
        sender_id: str,
        chat_id: str,
        chat_type: str,
    ) -> None:
        existing = self._db.execute(
            "SELECT thread_id FROM sessions WHERE thread_id = ?", (thread_id,)
        ).fetchone()
        if existing:
            return
        import time
        self._db.execute(
            "INSERT INTO sessions (thread_id, channel, sender_id, chat_id, chat_type, first_message_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (thread_id, channel, sender_id, chat_id, chat_type, time.time()),
        )
        self._db.commit()

    def get_session(self, thread_id: str) -> Optional[dict]:
        row = self._db.execute(
            "SELECT * FROM sessions WHERE thread_id = ?", (thread_id,)
        ).fetchone()
        return dict(row) if row else None

    def add_messages(self, thread_id: str, messages: list[dict]) -> None:
        import time
        for msg in messages:
            self._db.execute(
                "INSERT OR IGNORE INTO messages "
                "(thread_id, message_id, role, content, tool_name, tool_calls, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    thread_id,
                    msg["message_id"],
                    msg["role"],
                    msg.get("content"),
                    msg.get("tool_name"),
                    msg.get("tool_calls"),
                    msg.get("timestamp", time.time()),
                ),
            )
        now = time.time()
        count = self._db.execute(
            "SELECT COUNT(*) FROM messages WHERE thread_id = ?", (thread_id,)
        ).fetchone()[0]
        self._db.execute(
            "UPDATE sessions SET message_count = ?, last_message_at = ? WHERE thread_id = ?",
            (count, now, thread_id),
        )
        self._db.commit()

    def mark_inactive(self, thread_id: str) -> None:
        self._db.execute(
            "UPDATE sessions SET is_active = 0 WHERE thread_id = ?", (thread_id,)
        )
        self._db.commit()

    def search(
        self,
        query: str,
        channel: Optional[str] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[dict]:
        if not query.strip():
            return self._list_recent(limit)

        fts_query = _sanitize_fts5_query(query)
        sql = """
            SELECT
                m.thread_id,
                s.channel,
                s.chat_type,
                s.sender_id,
                s.is_active,
                s.last_message_at,
                s.message_count,
                GROUP_CONCAT(f.content, char(10)) AS snippets
            FROM messages_fts f
            JOIN messages m ON m.id = f.rowid
            JOIN sessions s ON s.thread_id = m.thread_id
            WHERE messages_fts MATCH ?
            GROUP BY m.thread_id
            ORDER BY MAX(m.timestamp) DESC
            LIMIT ? OFFSET ?
        """
        rows = self._db.execute(sql, (fts_query, limit, offset)).fetchall()
        results = []
        for row in rows:
            snippet = row["snippets"] or ""
            snippet_lines = snippet.split("\n")[:2]
            results.append({
                "thread_id": row["thread_id"],
                "channel": row["channel"],
                "chat_type": row["chat_type"],
                "last_message_at": row["last_message_at"],
                "message_count": row["message_count"],
                "is_active": bool(row["is_active"]),
                "snippet": "\n".join(snippet_lines),
            })
        return results

    def _list_recent(self, limit: int) -> list[dict]:
        rows = self._db.execute(
            "SELECT thread_id, channel, chat_type, sender_id, is_active, "
            "last_message_at, message_count FROM sessions "
            "WHERE is_active = 1 ORDER BY last_message_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        results = []
        for row in rows:
            preview_row = self._db.execute(
                "SELECT content FROM messages WHERE thread_id = ? AND role = 'human' "
                "ORDER BY timestamp ASC LIMIT 1",
                (row["thread_id"],),
            ).fetchone()
            preview = (preview_row["content"] or "")[:80] if preview_row else ""
            results.append({
                "thread_id": row["thread_id"],
                "channel": row["channel"],
                "chat_type": row["chat_type"],
                "last_message_at": row["last_message_at"],
                "message_count": row["message_count"],
                "is_active": bool(row["is_active"]),
                "snippet": preview,
            })
        return results
