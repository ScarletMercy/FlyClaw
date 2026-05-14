"""Agent state and SQLite-backed session persistence.

Messages are stored in OpenAI chat format (list of dicts).
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from typing import Any

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger("myclaw.agent.state")

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
                raise ValueError(f"Invalid message role: {role!r}")
            if role == "tool" and not msg.get("tool_call_id"):
                raise ValueError("Tool messages must have 'tool_call_id'")
            if role == "assistant" and "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    if not tc.get("id") or not tc.get("function"):
                        raise ValueError("Tool calls must have 'id' and 'function'")
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

    def copy(self) -> AgentState:
        return AgentState(
            messages=list(self.messages),
            system_prompt=self.system_prompt,
            sender_id=self.sender_id,
            chat_id=self.chat_id,
            chat_type=self.chat_type,
            message_id=self.message_id,
            user_role=self.user_role,
            channel=self.channel,
            pending_approval=self.pending_approval,
        )


class StateStore:
    def __init__(self, db_path: str):
        self._db_path = db_path
        self._db: sqlite3.Connection | None = None
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_lock = asyncio.Lock()
        from pathlib import Path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        self._db = sqlite3.connect(self._db_path)
        self._db.execute("PRAGMA journal_mode=WAL")
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
                self._locks[thread_id] = asyncio.Lock()
            return self._locks[thread_id]

    async def save(self, thread_id: str, state: AgentState) -> None:
        assert self._db is not None
        self._db.execute(
            """INSERT OR REPLACE INTO sessions (thread_id, messages, metadata, updated_at)
               VALUES (?, ?, ?, ?)""",
            (
                thread_id,
                json.dumps(state.messages, ensure_ascii=False),
                json.dumps(state.meta_dict(), ensure_ascii=False),
                time.time(),
            ),
        )
        self._db.commit()

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
        return self.load(thread_id)

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


class MemoryStateStore(StateStore):
    def __init__(self):
        self._db_path = ":memory:"
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_lock = asyncio.Lock()
        self._db: sqlite3.Connection | None = None
        self._init_db()
