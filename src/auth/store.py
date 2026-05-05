from __future__ import annotations

import json
import logging
import random
import sqlite3
import string
import time
from pathlib import Path
from typing import Any, Optional

from src.auth.models import Device, PairingCode, User, UserRole

logger = logging.getLogger("myclaw.auth.store")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id    TEXT PRIMARY KEY,
    role       TEXT NOT NULL DEFAULT 'guest',
    display_name TEXT NOT NULL DEFAULT '',
    allowed_tools TEXT NOT NULL DEFAULT '[]',
    denied_tools  TEXT NOT NULL DEFAULT '[]',
    created_at REAL NOT NULL,
    last_seen  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS devices (
    device_id   TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    platform    TEXT NOT NULL DEFAULT '',
    name        TEXT NOT NULL DEFAULT '',
    fingerprint TEXT NOT NULL DEFAULT '',
    trusted     INTEGER NOT NULL DEFAULT 0,
    paired_at   REAL NOT NULL DEFAULT 0,
    last_seen   REAL NOT NULL DEFAULT 0,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS pairing_codes (
    code        TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    device_info TEXT NOT NULL DEFAULT '',
    expires_at  REAL NOT NULL,
    created_at  REAL NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_devices_user ON devices(user_id);
CREATE INDEX IF NOT EXISTS idx_pairing_expires ON pairing_codes(expires_at);
"""


class AuthStore:
    def __init__(self, db_path: str = "data/auth.db"):
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ── Users ──────────────────────────────────────────────

    def get_user(self, user_id: str) -> Optional[User]:
        row = self._conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_user(row)

    def get_or_create_user(
        self,
        user_id: str,
        display_name: str = "",
        default_role: UserRole = UserRole.guest,
    ) -> User:
        user = self.get_user(user_id)
        if user is not None:
            self._touch_user(user_id)
            user.touch()
            return user
        now = time.time()
        self._conn.execute(
            "INSERT INTO users (user_id, role, display_name, allowed_tools, denied_tools, created_at, last_seen) "
            "VALUES (?, ?, ?, '[]', '[]', ?, ?)",
            (user_id, default_role.value, display_name, now, now),
        )
        self._conn.commit()
        logger.info("New user registered: %s (role=%s)", user_id, default_role.value)
        return User(
            user_id=user_id,
            role=default_role,
            display_name=display_name,
            created_at=now,
            last_seen=now,
        )

    def update_user_role(self, user_id: str, role: UserRole) -> bool:
        cur = self._conn.execute(
            "UPDATE users SET role = ? WHERE user_id = ?",
            (role.value, user_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def update_user_tools(
        self,
        user_id: str,
        allowed_tools: Optional[list[str]] = None,
        denied_tools: Optional[list[str]] = None,
    ) -> bool:
        sets: list[str] = []
        params: list[Any] = []
        if allowed_tools is not None:
            sets.append("allowed_tools = ?")
            params.append(json.dumps(allowed_tools))
        if denied_tools is not None:
            sets.append("denied_tools = ?")
            params.append(json.dumps(denied_tools))
        if not sets:
            return False
        params.append(user_id)
        cur = self._conn.execute(f"UPDATE users SET {', '.join(sets)} WHERE user_id = ?", params)
        self._conn.commit()
        return cur.rowcount > 0

    def list_users(self) -> list[User]:
        rows = self._conn.execute("SELECT * FROM users ORDER BY last_seen DESC").fetchall()
        return [self._row_to_user(r) for r in rows]

    def delete_user(self, user_id: str) -> bool:
        self._conn.execute("DELETE FROM devices WHERE user_id = ?", (user_id,))
        cur = self._conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
        self._conn.commit()
        return cur.rowcount > 0

    # ── Devices ─────────────────────────────────────────────

    def register_device(
        self,
        device_id: str,
        user_id: str,
        platform: str = "",
        name: str = "",
        fingerprint: str = "",
        trusted: bool = False,
    ) -> Device:
        now = time.time()
        self._conn.execute(
            "INSERT OR REPLACE INTO devices (device_id, user_id, platform, name, fingerprint, trusted, paired_at, last_seen) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (device_id, user_id, platform, name, fingerprint, int(trusted), now, now),
        )
        self._conn.commit()
        return Device(
            device_id=device_id,
            user_id=user_id,
            platform=platform,
            name=name,
            fingerprint=fingerprint,
            trusted=trusted,
            paired_at=now,
            last_seen=now,
        )

    def get_device(self, device_id: str) -> Optional[Device]:
        row = self._conn.execute("SELECT * FROM devices WHERE device_id = ?", (device_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_device(row)

    def trust_device(self, device_id: str) -> bool:
        cur = self._conn.execute("UPDATE devices SET trusted = 1 WHERE device_id = ?", (device_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def list_user_devices(self, user_id: str) -> list[Device]:
        rows = self._conn.execute(
            "SELECT * FROM devices WHERE user_id = ? ORDER BY last_seen DESC",
            (user_id,),
        ).fetchall()
        return [self._row_to_device(r) for r in rows]

    def delete_device(self, device_id: str) -> bool:
        cur = self._conn.execute("DELETE FROM devices WHERE device_id = ?", (device_id,))
        self._conn.commit()
        return cur.rowcount > 0

    # ── Pairing ─────────────────────────────────────────────

    def create_pairing_code(
        self,
        user_id: str,
        device_info: str = "",
        ttl_seconds: int = 300,
    ) -> PairingCode:
        self._cleanup_expired_codes()
        code = "".join(random.choices(string.digits, k=6))
        now = time.time()
        expires_at = now + ttl_seconds
        self._conn.execute(
            "INSERT OR REPLACE INTO pairing_codes (code, user_id, device_info, expires_at, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (code, user_id, device_info, expires_at, now),
        )
        self._conn.commit()
        logger.info("Pairing code created for user %s", user_id)
        return PairingCode(code=code, user_id=user_id, device_info=device_info, expires_at=expires_at)

    def verify_pairing(
        self,
        code: str,
        device_id: str,
        platform: str = "",
        name: str = "",
    ) -> Optional[User]:
        self._cleanup_expired_codes()
        row = self._conn.execute("SELECT * FROM pairing_codes WHERE code = ?", (code,)).fetchone()
        if row is None:
            logger.warning("Pairing code not found: %s", code)
            return None
        if row["expires_at"] < time.time():
            self._conn.execute("DELETE FROM pairing_codes WHERE code = ?", (code,))
            self._conn.commit()
            logger.warning("Pairing code expired: %s", code)
            return None

        user_id = row["user_id"]
        # Upgrade guest → user on first pairing
        user = self.get_user(user_id)
        if user and user.role == UserRole.guest:
            self.update_user_role(user_id, UserRole.user)
            logger.info("User %s upgraded from guest to user via pairing", user_id)

        # Register device as trusted
        self.register_device(
            device_id=device_id,
            user_id=user_id,
            platform=platform,
            name=name,
            trusted=True,
        )

        # Consume the code
        self._conn.execute("DELETE FROM pairing_codes WHERE code = ?", (code,))
        self._conn.commit()
        logger.info("Device %s paired for user %s", device_id, user_id)

        return self.get_user(user_id)

    def is_trusted_device(self, device_id: str) -> bool:
        row = self._conn.execute("SELECT trusted FROM devices WHERE device_id = ?", (device_id,)).fetchone()
        return row is not None and bool(row["trusted"])

    # ── Internal helpers ────────────────────────────────────

    def _touch_user(self, user_id: str) -> None:
        self._conn.execute(
            "UPDATE users SET last_seen = ? WHERE user_id = ?",
            (time.time(), user_id),
        )
        self._conn.commit()

    def _cleanup_expired_codes(self) -> None:
        self._conn.execute("DELETE FROM pairing_codes WHERE expires_at < ?", (time.time(),))
        self._conn.commit()

    @staticmethod
    def _row_to_user(row: sqlite3.Row) -> User:
        return User(
            user_id=row["user_id"],
            role=UserRole(row["role"]),
            display_name=row["display_name"],
            allowed_tools=json.loads(row["allowed_tools"]),
            denied_tools=json.loads(row["denied_tools"]),
            created_at=row["created_at"],
            last_seen=row["last_seen"],
        )

    @staticmethod
    def _row_to_device(row: sqlite3.Row) -> Device:
        return Device(
            device_id=row["device_id"],
            user_id=row["user_id"],
            platform=row["platform"],
            name=row["name"],
            fingerprint=row["fingerprint"],
            trusted=bool(row["trusted"]),
            paired_at=row["paired_at"],
            last_seen=row["last_seen"],
        )

    def close(self) -> None:
        self._conn.close()
