"""Session index store: SQLite + FTS5 for full-text search over conversation history."""

from __future__ import annotations

import asyncio
import aiosqlite
import logging
import re
import time
from pathlib import Path
from typing import Optional

from src.utils.fts import sanitize_fts5_query, tokenize_batch

logger = logging.getLogger("flyclaw.session_index")

_FTS_TABLE_SQL = """CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    search_text,
    content=messages,
    content_rowid=id
)"""

_SCHEMA = f"""
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
    search_text TEXT,
    tool_name TEXT,
    tool_calls TEXT,
    timestamp REAL NOT NULL
);

{_FTS_TABLE_SQL};

-- 不变量：content 与 search_text 必须同语句写入——单字切分只能发生在应用层，
-- 触发器只会搬运 search_text 列。应用层唯一写入口是 add_messages；绕过它只改
-- content，FTS 将持续索引旧文本（滞后而非损坏）。
CREATE TRIGGER IF NOT EXISTS messages_fts_insert
    AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, search_text) VALUES (new.id, new.search_text);
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_delete
    AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, search_text)
        VALUES('delete', old.id, old.search_text);
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_update
    AFTER UPDATE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, search_text)
        VALUES('delete', old.id, old.search_text);
    INSERT INTO messages_fts(rowid, search_text) VALUES (new.id, new.search_text);
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


class SessionIndexStore:
    def __init__(self, db_path: str):
        self._db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None
        self._lock = asyncio.Lock()

    def _require_db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("SessionIndexStore not initialized. Use await SessionIndexStore.create(path)")
        return self._db

    @classmethod
    async def create(cls, db_path: str) -> SessionIndexStore:
        obj = cls(db_path)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        obj._db = await aiosqlite.connect(db_path)
        obj._db.row_factory = aiosqlite.Row
        await obj._db.execute("PRAGMA journal_mode=WAL")
        await obj._db.execute("PRAGMA busy_timeout=5000")
        await obj._migrate_search_text()
        await obj._db.executescript(_SCHEMA)
        return obj

    async def _migrate_search_text(self) -> None:
        """search_text 一次性迁移：旧 schema 升级、存量切词回填、FTS 重建。

        完成标记最后写入，任何时点被打断，下次启动自动续跑（keyset 按
        search_text IS NULL 续）；以标记而非列存在性判定完成，半迁移状态
        不会被误判。迁移期间拆掉触发器与 FTS 表：对从未入索引的行，update
        触发器会向 FTS 发值不匹配的 'delete'，损坏索引（database disk
        image is malformed）；回填后重建 FTS 并整体 rebuild，触发器由随后
        的 _SCHEMA 创建。全新库无存量可迁，直接打标记。
        """
        db = self._require_db()
        await db.execute("CREATE TABLE IF NOT EXISTS index_meta (key TEXT PRIMARY KEY, value TEXT)")
        cursor = await db.execute("SELECT value FROM index_meta WHERE key = 'fts_search_text'")
        row = await cursor.fetchone()
        if row and row[0] == "1":
            return
        cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages'")
        if not await cursor.fetchone():
            # 全新库：无存量可迁，_SCHEMA 直接建新 schema
            await db.execute("INSERT OR REPLACE INTO index_meta VALUES ('fts_search_text', '1')")
            await db.commit()
            return
        cursor = await db.execute("PRAGMA table_info(messages)")
        cols = {r[1] for r in await cursor.fetchall()}
        if "search_text" not in cols:
            logger.info("Session index: legacy FTS schema detected, migrating to search_text")
            await db.execute("ALTER TABLE messages ADD COLUMN search_text TEXT")
        for trig in ("messages_fts_insert", "messages_fts_delete", "messages_fts_update"):
            await db.execute(f"DROP TRIGGER IF EXISTS {trig}")
        await db.execute("DROP TABLE IF EXISTS messages_fts")
        await db.commit()
        batch_size = 2000
        last_id = 0
        backfilled = 0
        while True:
            cursor = await db.execute(
                "SELECT id, content FROM messages WHERE search_text IS NULL AND id > ? ORDER BY id LIMIT ?",
                (last_id, batch_size),
            )
            rows = await cursor.fetchall()
            if not rows:
                break
            texts = tokenize_batch([r[1] for r in rows])
            await db.executemany(
                "UPDATE messages SET search_text = ? WHERE id = ?",
                [(text, r[0]) for text, r in zip(texts, rows)],
            )
            await db.commit()
            last_id = rows[-1][0]
            backfilled += len(rows)
        if backfilled:
            logger.info("Session index: backfilled search_text for %d messages", backfilled)
        await db.execute(_FTS_TABLE_SQL)
        await db.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")
        await db.execute("INSERT OR REPLACE INTO index_meta VALUES ('fts_search_text', '1')")
        await db.commit()

    async def close(self):
        async with self._lock:
            if self._db:
                await self._db.close()
                self._db = None

    async def upsert_session(
        self,
        thread_id: str,
        channel: str,
        sender_id: str,
        chat_id: str,
        chat_type: str,
    ) -> None:
        async with self._lock:
            db = self._require_db()
            cursor = await db.execute("SELECT thread_id FROM sessions WHERE thread_id = ?", (thread_id,))
            existing = await cursor.fetchone()
            if existing:
                return
            await db.execute(
                "INSERT INTO sessions (thread_id, channel, sender_id, chat_id, chat_type, first_message_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (thread_id, channel, sender_id, chat_id, chat_type, time.time()),
            )
            await db.commit()

    async def get_session(self, thread_id: str) -> Optional[dict]:
        async with self._lock:
            db = self._require_db()
            cursor = await db.execute("SELECT * FROM sessions WHERE thread_id = ?", (thread_id,))
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def add_messages(self, thread_id: str, messages: list[dict]) -> None:
        texts = tokenize_batch([m.get("content") for m in messages])
        async with self._lock:
            db = self._require_db()
            for msg, search_text in zip(messages, texts):
                await db.execute(
                    "INSERT INTO messages "
                    "(thread_id, message_id, role, content, search_text, tool_name, tool_calls, timestamp) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(message_id) DO UPDATE SET "
                    "role=excluded.role, "
                    "content=excluded.content, "
                    "search_text=excluded.search_text, "
                    "tool_name=excluded.tool_name, "
                    "tool_calls=excluded.tool_calls, "
                    "timestamp=excluded.timestamp",
                    (
                        thread_id,
                        msg["message_id"],
                        msg["role"],
                        msg.get("content"),
                        search_text,
                        msg.get("tool_name"),
                        msg.get("tool_calls"),
                        msg.get("timestamp", time.time()),
                    ),
                )
            now = time.time()
            cursor = await db.execute("SELECT COUNT(*) FROM messages WHERE thread_id = ?", (thread_id,))
            count = (await cursor.fetchone())[0]
            await db.execute(
                "UPDATE sessions SET message_count = ?, last_message_at = ? WHERE thread_id = ?",
                (count, now, thread_id),
            )
            await db.commit()

    async def mark_inactive(self, thread_id: str) -> None:
        async with self._lock:
            db = self._require_db()
            await db.execute("UPDATE sessions SET is_active = 0 WHERE thread_id = ?", (thread_id,))
            await db.commit()

    async def search(
        self,
        query: str,
        channel: Optional[str] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[dict]:
        async with self._lock:
            if not query.strip():
                return await self._list_recent(limit)

            fts_query = sanitize_fts5_query(query)
            sql = """
                SELECT
                    m.thread_id,
                    s.channel,
                    s.chat_type,
                    s.sender_id,
                    s.is_active,
                    s.last_message_at,
                    s.message_count,
                    GROUP_CONCAT(m.content, char(10)) AS snippets
                FROM messages_fts f
                JOIN messages m ON m.id = f.rowid
                JOIN sessions s ON s.thread_id = m.thread_id
                WHERE messages_fts MATCH ?
                GROUP BY m.thread_id
                ORDER BY MAX(m.timestamp) DESC
                LIMIT ? OFFSET ?
            """
            db = self._require_db()
            cursor = await db.execute(sql, (fts_query, limit, offset))
            rows = await cursor.fetchall()
            results = []
            for row in rows:
                snippet = row["snippets"] or ""
                snippet_lines = snippet.split("\n")[:2]
                results.append(
                    {
                        "thread_id": row["thread_id"],
                        "channel": row["channel"],
                        "chat_type": row["chat_type"],
                        "last_message_at": row["last_message_at"],
                        "message_count": row["message_count"],
                        "is_active": bool(row["is_active"]),
                        "snippet": "\n".join(snippet_lines),
                    }
                )
            return results

    async def _list_recent(self, limit: int) -> list[dict]:
        # Called only from search() which already holds self._lock
        db = self._require_db()
        cursor = await db.execute(
            "SELECT thread_id, channel, chat_type, sender_id, is_active, "
            "last_message_at, message_count FROM sessions "
            "WHERE is_active = 1 ORDER BY last_message_at DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        results = []
        for row in rows:
            preview_cursor = await db.execute(
                "SELECT content FROM messages WHERE thread_id = ? AND role = 'human' ORDER BY timestamp ASC LIMIT 1",
                (row["thread_id"],),
            )
            preview_row = await preview_cursor.fetchone()
            preview = (preview_row["content"] or "")[:80] if preview_row else ""
            results.append(
                {
                    "thread_id": row["thread_id"],
                    "channel": row["channel"],
                    "chat_type": row["chat_type"],
                    "last_message_at": row["last_message_at"],
                    "message_count": row["message_count"],
                    "is_active": bool(row["is_active"]),
                    "snippet": preview,
                }
            )
        return results

    async def get_indexed_thread_ids(self) -> set[str]:
        async with self._lock:
            db = self._require_db()
            cursor = await db.execute("SELECT DISTINCT thread_id FROM messages")
            rows = await cursor.fetchall()
            return {r[0] for r in rows}

    async def get_thread_messages(self, thread_id: str, limit: int = 100) -> list[dict]:
        async with self._lock:
            db = self._require_db()
            cursor = await db.execute(
                "SELECT message_id, role, content, tool_name, timestamp "
                "FROM messages WHERE thread_id = ? ORDER BY timestamp ASC LIMIT ?",
                (thread_id, limit),
            )
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]
