"""Agent state and SQLite-backed session persistence.

Messages are stored in OpenAI chat format (list of dicts).
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
import time
from typing import Any

from pydantic import BaseModel, Field, field_validator

from src.agent.interrupt import InterruptFlag

logger = logging.getLogger("flyclaw.agent.state")

_VALID_ROLES = {"system", "user", "assistant", "tool"}


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
        self._db: sqlite3.Connection | None = None
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_lock = asyncio.Lock()
        self._interrupt_flags = InterruptFlagStore()
        from pathlib import Path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        self._db = sqlite3.connect(self._db_path, timeout=30, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA busy_timeout=5000")
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                thread_id TEXT PRIMARY KEY,
                messages TEXT NOT NULL,
                metadata TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_updated ON sessions(updated_at)")
        self._db.commit()

    async def acquire_thread(self, thread_id: str) -> asyncio.Lock:
        async with self._locks_lock:
            if thread_id not in self._locks:
                if len(self._locks) > _MAX_LOCKS:
                    to_remove = [
                        tid for tid, lock in self._locks.items()
                        if not lock.locked()
                    ]
                    for tid in to_remove[:len(self._locks) - _MAX_LOCKS // 2]:
                        del self._locks[tid]
                self._locks[thread_id] = asyncio.Lock()
            return self._locks[thread_id]

    async def save(self, thread_id: str, state: AgentState) -> None:
        assert self._db is not None
        data = (
            thread_id,
            json.dumps(state.messages, ensure_ascii=False),
            json.dumps(state.meta_dict(), ensure_ascii=False),
            time.time(),
        )

        def _do_save():
            self._db.execute(
                """INSERT OR REPLACE INTO sessions (thread_id, messages, metadata, updated_at)
                   VALUES (?, ?, ?, ?)""",
                data,
            )
            self._db.commit()

        await asyncio.to_thread(_do_save)

    def load(self, thread_id: str) -> AgentState | None:
        assert self._db is not None
        row = self._db.execute(
            "SELECT messages, metadata FROM sessions WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()
        if row is None:
            return None
        messages = json.loads(row[0])
        meta = json.loads(row[1])
        meta["messages"] = messages
        return AgentState.model_validate(meta)

    async def aload(self, thread_id: str) -> AgentState | None:
        return await asyncio.to_thread(self.load, thread_id)

    def load_messages(self, thread_id: str) -> list[dict]:
        assert self._db is not None
        row = self._db.execute(
            "SELECT messages FROM sessions WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()
        if row is None:
            return []
        return json.loads(row[0])

    def delete(self, thread_id: str) -> bool:
        assert self._db is not None
        cursor = self._db.execute("DELETE FROM sessions WHERE thread_id = ?", (thread_id,))
        self._db.commit()
        return cursor.rowcount > 0

    def list_threads(self) -> list[str]:
        assert self._db is not None
        rows = self._db.execute(
            "SELECT thread_id FROM sessions ORDER BY updated_at DESC"
        ).fetchall()
        return [r[0] for r in rows]

    def close(self) -> None:
        if self._db:
            self._db.close()
            self._db = None

    def get_interrupt_flag(self, thread_id: str) -> InterruptFlag:
        return self._interrupt_flags.get_flag(thread_id)

    def clear_interrupt_flag(self, thread_id: str) -> None:
        self._interrupt_flags.clear_flag(thread_id)


class MemoryStateStore(StateStore):
    def __init__(self):
        self._db_path = ":memory:"
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_lock = asyncio.Lock()
        self._interrupt_flags = InterruptFlagStore()
        self._db: sqlite3.Connection | None = None
        self._init_db()


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
                        tid for tid, flag in self._flags.items()
                        if not flag.check()[0] and flag.drain_steer() is None
                    ]
                    for tid in to_remove[:len(self._flags) - _MAX_FLAGS // 2]:
                        del self._flags[tid]
                self._flags[thread_id] = InterruptFlag()
            return self._flags[thread_id]

    def clear_flag(self, thread_id: str) -> None:
        with self._lock:
            self._flags.pop(thread_id, None)
