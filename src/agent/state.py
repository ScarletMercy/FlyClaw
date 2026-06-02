"""Agent state and SQLite-backed session persistence.

Messages are stored in OpenAI chat format (list of dicts).
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

import aiosqlite
from pydantic import BaseModel, Field, field_validator

from src.agent.interrupt import InterruptFlag

logger = logging.getLogger("flyclaw.agent.state")

_VALID_ROLES = {"system", "user", "assistant", "tool"}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    thread_id TEXT PRIMARY KEY,
    messages TEXT NOT NULL,
    metadata TEXT NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_updated ON sessions(updated_at);
"""


class AgentState(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    messages: list[dict[str, Any]] = Field(default_factory=list)
    system_prompt: str = ""
    sender_id: str = ""
    chat_id: str = ""
    chat_type: str = "p2p"
    message_id: str = ""
    user_role: str = ""
    channel: str = ""

    pending_approval: dict[str, Any] | None = None

    @field_validator("messages", mode="after")
    @classmethod
    def _validate_messages(cls, messages: list[dict]) -> list[dict]:
        for msg in messages:
            role = msg.get("role")
            if role not in _VALID_ROLES:
                msg["role"] = "user"
            if msg["role"] == "tool" and not msg.get("tool_call_id"):
                msg["tool_call_id"] = "unknown"
            if msg["role"] == "assistant" and "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    if not tc.get("id"):
                        tc["id"] = "unknown"
                    if not tc.get("function"):
                        tc["function"] = {"name": "unknown", "arguments": "{}"}
        return messages

    def meta_dict(self) -> dict[str, Any]:
        return {
            "system_prompt": self.system_prompt,
            "sender_id": self.sender_id,
            "chat_id": self.chat_id,
            "chat_type": self.chat_type,
            "message_id": self.message_id,
            "user_role": self.user_role,
            "channel": self.channel,
            "pending_approval": self.pending_approval,
        }

    def append_message(self, msg: dict[str, Any]) -> None:
        role = msg.get("role")
        if role not in _VALID_ROLES:
            raise ValueError(f"Invalid message role: {role!r}")
        if role == "tool" and not msg.get("tool_call_id"):
            raise ValueError("Tool messages must have 'tool_call_id'")
        self.messages.append(msg)


_MAX_LOCKS = 4096


class StateStore:
    def __init__(self, db_path: str):
        self._db_path = db_path
        self._conn: Optional[aiosqlite.Connection] = None
        self._init_lock = asyncio.Lock()
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_lock = asyncio.Lock()
        self._interrupt_flags = InterruptFlagStore()

    async def _get_conn(self) -> aiosqlite.Connection:
        """Lazy connection init with double-checked locking."""
        if self._conn is not None:
            return self._conn
        async with self._init_lock:
            if self._conn is not None:
                return self._conn
            if self._db_path != ":memory:":
                Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            conn = await aiosqlite.connect(self._db_path)
            conn.row_factory = aiosqlite.Row
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA busy_timeout=5000")
            await conn.executescript(_SCHEMA)
            await conn.commit()
            self._conn = conn
            return conn

    async def acquire_thread(self, thread_id: str) -> asyncio.Lock:
        async with self._locks_lock:
            if thread_id not in self._locks:
                if len(self._locks) > _MAX_LOCKS:
                    to_remove = [tid for tid, lock in self._locks.items() if not lock.locked()]
                    for tid in to_remove[: len(self._locks) - _MAX_LOCKS // 2]:
                        del self._locks[tid]
                self._locks[thread_id] = asyncio.Lock()
            return self._locks[thread_id]

    async def save(self, thread_id: str, state: AgentState) -> None:
        conn = await self._get_conn()
        await conn.execute(
            """INSERT OR REPLACE INTO sessions (thread_id, messages, metadata, updated_at)
               VALUES (?, ?, ?, ?)""",
            (
                thread_id,
                json.dumps(state.messages, ensure_ascii=False),
                json.dumps(state.meta_dict(), ensure_ascii=False),
                time.time(),
            ),
        )
        await conn.commit()

    async def load(self, thread_id: str) -> AgentState | None:
        conn = await self._get_conn()
        async with conn.execute(
            "SELECT messages, metadata FROM sessions WHERE thread_id = ?",
            (thread_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        messages = json.loads(row[0])
        meta = json.loads(row[1])
        meta["messages"] = messages
        return AgentState.model_validate(meta)

    async def delete(self, thread_id: str) -> bool:
        conn = await self._get_conn()
        cursor = await conn.execute("DELETE FROM sessions WHERE thread_id = ?", (thread_id,))
        await conn.commit()
        return cursor.rowcount > 0

    async def list_threads(self) -> list[str]:
        conn = await self._get_conn()
        async with conn.execute("SELECT thread_id FROM sessions ORDER BY updated_at DESC") as cursor:
            rows = await cursor.fetchall()
        return [r[0] for r in rows]

    async def close(self) -> None:
        if self._conn is not None:
            try:
                await self._conn.close()
            except Exception:
                logger.warning("Error closing StateStore connection", exc_info=True)
            finally:
                self._conn = None

    def get_interrupt_flag(self, thread_id: str) -> InterruptFlag:
        return self._interrupt_flags.get_flag(thread_id)

    def clear_interrupt_flag(self, thread_id: str) -> None:
        self._interrupt_flags.clear_flag(thread_id)


class MemoryStateStore(StateStore):
    def __init__(self):
        self._db_path = ":memory:"
        self._conn: Optional[aiosqlite.Connection] = None
        self._init_lock = asyncio.Lock()
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_lock = asyncio.Lock()
        self._interrupt_flags = InterruptFlagStore()


_MAX_FLAGS = 4096


class InterruptFlagStore:
    """Manages per-thread InterruptFlag instances with LRU cleanup."""

    def __init__(self) -> None:
        self._flags: dict[str, InterruptFlag] = {}
        self._lock = threading.Lock()

    def get_flag(self, thread_id: str) -> InterruptFlag:
        with self._lock:
            if thread_id not in self._flags:
                if len(self._flags) > _MAX_FLAGS:
                    to_remove = [
                        tid for tid, flag in self._flags.items() if not flag.check()[0] and flag.drain_steer() is None
                    ]
                    for tid in to_remove[: len(self._flags) - _MAX_FLAGS // 2]:
                        del self._flags[tid]
                self._flags[thread_id] = InterruptFlag()
            return self._flags[thread_id]

    def clear_flag(self, thread_id: str) -> None:
        with self._lock:
            self._flags.pop(thread_id, None)
