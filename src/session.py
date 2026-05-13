from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("myclaw.session")


class SessionTracker:
    def __init__(self, idle_reset_minutes: int = 120):
        self._idle_reset_seconds = idle_reset_minutes * 60
        self._last_activity: dict[str, float] = {}
        self._task: Optional[asyncio.Task] = None

    def touch(self, thread_id: str) -> None:
        self._last_activity[thread_id] = time.monotonic()

    def get_expired_sessions(self) -> list[str]:
        if self._idle_reset_seconds <= 0:
            return []
        now = time.monotonic()
        expired = []
        for tid, last in list(self._last_activity.items()):
            if now - last > self._idle_reset_seconds:
                expired.append(tid)
        return expired

    def remove(self, thread_id: str) -> None:
        self._last_activity.pop(thread_id, None)

    @property
    def active_count(self) -> int:
        return len(self._last_activity)

    def get_sessions(self) -> list[dict]:
        now = time.monotonic()
        return [{"thread_id": tid, "last_active": now - last} for tid, last in self._last_activity.items()]

    async def start_periodic_cleanup(
        self,
        compiled_graph,
        interval_seconds: int = 60,
    ) -> None:
        if self._task is not None and not self._task.done():
            logger.warning("Periodic cleanup already running")
            return

        async def _loop():
            while True:
                await asyncio.sleep(interval_seconds)
                expired = self.get_expired_sessions()
                if not expired:
                    continue
                for tid in expired:
                    try:
                        config = {"configurable": {"thread_id": tid}}
                        state = await compiled_graph.aget_state(config)
                        if state and state.values:
                            await compiled_graph.aupdate_state(config, {"messages": []})
                            logger.info("Session reset (idle): %s", tid)
                            # Mark session as inactive in search index
                            try:
                                from src.session_index.store import get_session_index

                                idx = get_session_index()
                                if idx:
                                    idx.mark_inactive(tid)
                            except Exception:
                                pass
                        self.remove(tid)
                    except Exception as e:
                        logger.debug("Session reset failed for %s: %s", tid, e)
                        self.remove(tid)

        self._task = asyncio.create_task(_loop())
        logger.info(
            "Session tracker started (idle_reset=%dm, check_interval=%ds)",
            self._idle_reset_seconds // 60,
            interval_seconds,
        )

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                self._task = None
            else:
                self._task = None


# ── Multi-session registry ──


@dataclass
class SessionEntry:
    session_id: str       # short display id (s1, s2, ...)
    thread_id: str        # full thread_id for LangGraph
    created_at: float     # unix timestamp
    summary: str = ""     # first user message excerpt


@dataclass
class UserSessions:
    sessions: list[SessionEntry] = field(default_factory=list)
    current_id: Optional[str] = None  # None = using legacy thread_id
    _next_id: int = 1

    def _alloc_id(self) -> str:
        sid = f"s{self._next_id}"
        self._next_id += 1
        return sid


class SessionRegistry:
    """Per-user multi-session management with JSON persistence."""

    def __init__(self):
        self._users: dict[str, UserSessions] = {}
        self._store_path: Optional[str] = None
        self._lock = threading.Lock()

    def init(self, store_path: str) -> None:
        self._store_path = store_path
        Path(store_path).parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        if not self._store_path or not Path(self._store_path).exists():
            return
        try:
            with open(self._store_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for user_key, ud in data.items():
                sessions = [SessionEntry(**s) for s in ud.get("sessions", [])]
                us = UserSessions(
                    sessions=sessions,
                    current_id=ud.get("current_id"),
                    _next_id=ud.get("_next_id", len(sessions) + 1),
                )
                self._users[user_key] = us
            logger.info("Session registry loaded: %d users", len(self._users))
        except Exception as e:
            logger.warning("Failed to load session registry: %s", e)

    def _save(self) -> None:
        if not self._store_path:
            return
        data = {}
        for user_key, us in self._users.items():
            data[user_key] = {
                "sessions": [asdict(s) for s in us.sessions],
                "current_id": us.current_id,
                "_next_id": us._next_id,
            }
        with self._lock:
            try:
                fd, tmp_path = tempfile.mkstemp(
                    dir=os.path.dirname(self._store_path), suffix=".tmp",
                )
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    os.replace(tmp_path, self._store_path)
                except Exception:
                    os.unlink(tmp_path)
                    raise
            except Exception as e:
                logger.warning("Failed to save session registry: %s", e)

    def _get_user(self, user_key: str) -> UserSessions:
        if user_key not in self._users:
            self._users[user_key] = UserSessions()
        return self._users[user_key]

    def get_current(self, user_key: str) -> Optional[str]:
        """Get current thread_id override, or None if using legacy."""
        us = self._users.get(user_key)
        if not us or us.current_id is None:
            return None
        for s in us.sessions:
            if s.session_id == us.current_id:
                return s.thread_id
        return None

    def new_session(self, user_key: str, channel_prefix: str, user_hash: str) -> str:
        """Create a new session for the user. Returns short session_id."""
        us = self._get_user(user_key)
        sid = us._alloc_id()
        thread_id = f"{channel_prefix}:{sid}:{user_hash}"
        us.sessions.append(SessionEntry(
            session_id=sid,
            thread_id=thread_id,
            created_at=time.time(),
            summary="(new)",
        ))
        us.current_id = sid
        self._save()
        logger.info("New session %s for user %s: %s", sid, user_key, thread_id)
        return sid

    def list_sessions(self, user_key: str) -> list[dict]:
        """List all sessions for a user. Returns list of dicts."""
        us = self._users.get(user_key)
        if not us:
            return []
        result = []
        for s in us.sessions:
            result.append({
                "session_id": s.session_id,
                "thread_id": s.thread_id,
                "created_at": s.created_at,
                "summary": s.summary,
                "is_current": s.session_id == us.current_id,
            })
        return result

    def switch_to(self, user_key: str, session_id: str) -> Optional[str]:
        """Switch to a specific session. Returns thread_id or None."""
        us = self._users.get(user_key)
        if not us:
            return None
        if session_id == "default":
            us.current_id = None
            self._save()
            return "default"
        for s in us.sessions:
            if s.session_id == session_id:
                us.current_id = session_id
                self._save()
                return s.thread_id
        return None

    def update_summary(self, user_key: str, thread_id: str, summary: str) -> None:
        """Update the summary for a session matching thread_id."""
        us = self._users.get(user_key)
        if not us:
            return
        for s in us.sessions:
            if s.thread_id == thread_id:
                s.summary = summary
                self._save()
                return


def get_session_summaries(checkpointer_path: str, thread_ids: list[str]) -> dict[str, str]:
    """Read first user message for each thread_id from checkpointer DB.

    Returns dict of thread_id -> summary string.
    """
    if not checkpointer_path or not Path(checkpointer_path).exists():
        return {}
    results = {}
    conn = None
    try:
        conn = sqlite3.connect(checkpointer_path)
        for tid in thread_ids:
            try:
                cnt = conn.execute(
                    "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                    (tid,),
                ).fetchone()[0]
                results[tid] = f"({cnt} messages)"
            except Exception:
                results[tid] = "(unknown)"
    except Exception as e:
        logger.debug("Failed to read session summaries: %s", e)
    finally:
        if conn:
            conn.close()
    return results


def get_threads_for_user(checkpointer_path: str, user_key: str) -> list[dict]:
    """Get all thread_ids from checkpointer that belong to a user.

    Matches threads starting with user_key prefix or containing the user hash.
    """
    if not checkpointer_path or not Path(checkpointer_path).exists():
        return []
    conn = None
    try:
        conn = sqlite3.connect(checkpointer_path)
        # Match legacy thread_id exactly and multi-session threads by pattern
        rows = conn.execute(
            "SELECT thread_id, COUNT(*) as cnt, MAX(rowid) as last_rowid "
            "FROM checkpoints GROUP BY thread_id ORDER BY last_rowid DESC",
        ).fetchall()

        # Extract user hash from user_key (e.g. "qq:user:ABC123" -> "ABC123")
        parts = user_key.split(":")
        user_hash = parts[-1] if parts else ""

        results = []
        for tid, cnt, _ in rows:
            # Legacy: exact match
            # Multi-session: contains the user_hash
            if tid == user_key or (user_hash and user_hash in tid):
                results.append({"thread_id": tid, "checkpoint_count": cnt})
        return results
    except Exception as e:
        logger.debug("Failed to list threads for user: %s", e)
        return []
    finally:
        if conn:
            conn.close()
