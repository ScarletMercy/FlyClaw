from __future__ import annotations

import secrets
import time
from dataclasses import dataclass


@dataclass
class _Capability:
    token: str
    node_id: str
    created_at: float
    expires_at: float


class CanvasCapabilityManager:
    def __init__(self, ttl_seconds: int = 600):
        self._ttl = ttl_seconds
        self._tokens: dict[str, _Capability] = {}

    def mint(self, node_id: str) -> str:
        token = secrets.token_urlsafe(24)
        now = time.time()
        self._tokens[token] = _Capability(
            token=token,
            node_id=node_id,
            created_at=now,
            expires_at=now + self._ttl,
        )
        return token

    def validate(self, token: str, node_id: str) -> bool:
        cap = self._tokens.get(token)
        if not cap:
            return False
        if cap.node_id != node_id:
            return False
        now = time.time()
        if now > cap.expires_at:
            self._tokens.pop(token, None)
            return False
        cap.expires_at = now + self._ttl
        return True

    def revoke(self, token: str) -> None:
        self._tokens.pop(token, None)
