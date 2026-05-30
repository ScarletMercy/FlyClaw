from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field


@dataclass
class AcpSession:
    session_id: str
    agent_id: str
    cwd: str
    created_at: float
    last_active: float
    thread_id: str | None = None
    state: dict = field(default_factory=dict)


class AcpSessionManager:
    def __init__(self, max_sessions: int = 100, idle_ttl_seconds: int = 3600) -> None:
        self._sessions: dict[str, AcpSession] = {}
        self._max_sessions = max_sessions
        self._idle_ttl = idle_ttl_seconds

    def create(self, agent_id: str, cwd: str = "") -> str:
        self._evict_idle()
        if len(self._sessions) >= self._max_sessions:
            oldest_id = min(self._sessions, key=lambda k: self._sessions[k].last_active)
            del self._sessions[oldest_id]
        session_id = uuid.uuid4().hex
        now = time.time()
        self._sessions[session_id] = AcpSession(
            session_id=session_id,
            agent_id=agent_id,
            cwd=cwd,
            created_at=now,
            last_active=now,
        )
        return session_id

    def get(self, session_id: str) -> AcpSession | None:
        session = self._sessions.get(session_id)
        if session is not None:
            session.last_active = time.time()
        return session

    def close(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def list_sessions(self) -> list[AcpSession]:
        return list(self._sessions.values())

    def _evict_idle(self) -> None:
        now = time.time()
        expired = [sid for sid, s in self._sessions.items() if now - s.last_active > self._idle_ttl]
        for sid in expired:
            del self._sessions[sid]
