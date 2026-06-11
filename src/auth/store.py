from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import aiosqlite

from src.auth.models import Device, PairingCode, User, UserRole

logger = logging.getLogger("flyclaw.auth.store")

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
    def __init__(self, db_path: str | None = None):
        if db_path is None:
            from src.instance import data_dir

            db_path = str(data_dir() / "auth.db")
        self._path = Path(db_path).expanduser().resolve()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._init_lock = asyncio.Lock()
        self._conn: Optional[aiosqlite.Connection] = None

    async def _get_conn(self) -> aiosqlite.Connection:
        if self._conn is not None:
            return self._conn
        async with self._init_lock:
            # Double-checked locking: 在锁内再次检查，
            # 防止多个协程同时通过第一次检查后重复创建连接
            if self._conn is not None:
                return self._conn
            conn = await aiosqlite.connect(str(self._path))
            conn.row_factory = aiosqlite.Row
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA busy_timeout=5000")
            await conn.executescript(_SCHEMA)
            await conn.commit()
            self._conn = conn
            return conn

    # ── Users ──────────────────────────────────────────────

    async def get_user(self, user_id: str) -> Optional[User]:
        conn = await self._get_conn()
        async with conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_user(row)

    async def get_or_create_user(
        self,
        user_id: str,
        display_name: str = "",
        default_role: UserRole = UserRole.guest,
    ) -> User:
        async with self._lock:
            user = await self.get_user(user_id)
            if user is not None:
                conn = await self._get_conn()
                await conn.execute(
                    "UPDATE users SET last_seen = ? WHERE user_id = ?",
                    (time.time(), user_id),
                )
                await conn.commit()
                user.touch()
                return user
            now = time.time()
            conn = await self._get_conn()
            await conn.execute(
                "INSERT INTO users (user_id, role, display_name, allowed_tools, denied_tools, created_at, last_seen) "
                "VALUES (?, ?, ?, '[]', '[]', ?, ?)",
                (user_id, default_role.value, display_name, now, now),
            )
            await conn.commit()
            logger.info("New user registered: %s (role=%s)", user_id, default_role.value)
            return User(
                user_id=user_id,
                role=default_role,
                display_name=display_name,
                created_at=now,
                last_seen=now,
            )

    async def update_user_role(self, user_id: str, role: UserRole) -> bool:
        async with self._lock:
            conn = await self._get_conn()
            cur = await conn.execute(
                "UPDATE users SET role = ? WHERE user_id = ?",
                (role.value, user_id),
            )
            await conn.commit()
            return cur.rowcount > 0

    async def update_user_tools(
        self,
        user_id: str,
        allowed_tools: Optional[list[str]] = None,
        denied_tools: Optional[list[str]] = None,
    ) -> bool:
        async with self._lock:
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
            conn = await self._get_conn()
            cur = await conn.execute(f"UPDATE users SET {', '.join(sets)} WHERE user_id = ?", params)
            await conn.commit()
            return cur.rowcount > 0

    async def list_users(self) -> list[User]:
        conn = await self._get_conn()
        async with conn.execute("SELECT * FROM users ORDER BY last_seen DESC") as cursor:
            rows = await cursor.fetchall()
        return [self._row_to_user(r) for r in rows]

    async def delete_user(self, user_id: str) -> bool:
        async with self._lock:
            conn = await self._get_conn()
            await conn.execute("DELETE FROM devices WHERE user_id = ?", (user_id,))
            cur = await conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
            await conn.commit()
            return cur.rowcount > 0

    # ── Devices ─────────────────────────────────────────────

    async def register_device(
        self,
        device_id: str,
        user_id: str,
        platform: str = "",
        name: str = "",
        fingerprint: str = "",
        trusted: bool = False,
    ) -> Device:
        async with self._lock:
            now = time.time()
            conn = await self._get_conn()
            await conn.execute(
                "INSERT OR REPLACE INTO devices (device_id, user_id, platform, name, fingerprint, trusted, paired_at, last_seen) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (device_id, user_id, platform, name, fingerprint, int(trusted), now, now),
            )
            await conn.commit()
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

    async def get_device(self, device_id: str) -> Optional[Device]:
        conn = await self._get_conn()
        async with conn.execute("SELECT * FROM devices WHERE device_id = ?", (device_id,)) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_device(row)

    async def trust_device(self, device_id: str) -> bool:
        async with self._lock:
            conn = await self._get_conn()
            cur = await conn.execute("UPDATE devices SET trusted = 1 WHERE device_id = ?", (device_id,))
            await conn.commit()
            return cur.rowcount > 0

    async def list_user_devices(self, user_id: str) -> list[Device]:
        conn = await self._get_conn()
        async with conn.execute(
            "SELECT * FROM devices WHERE user_id = ? ORDER BY last_seen DESC",
            (user_id,),
        ) as cursor:
            rows = await cursor.fetchall()
        return [self._row_to_device(r) for r in rows]

    async def delete_device(self, device_id: str) -> bool:
        async with self._lock:
            conn = await self._get_conn()
            cur = await conn.execute("DELETE FROM devices WHERE device_id = ?", (device_id,))
            await conn.commit()
            return cur.rowcount > 0

    # ── Pairing ─────────────────────────────────────────────

    async def create_pairing_code(
        self,
        user_id: str,
        device_info: str = "",
        ttl_seconds: int = 300,
    ) -> PairingCode:
        async with self._lock:
            conn = await self._get_conn()
            await conn.execute("DELETE FROM pairing_codes WHERE expires_at < ?", (time.time(),))
            code = uuid.uuid4().hex
            now = time.time()
            expires_at = now + ttl_seconds
            await conn.execute(
                "INSERT OR REPLACE INTO pairing_codes (code, user_id, device_info, expires_at, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (code, user_id, device_info, expires_at, now),
            )
            await conn.commit()
            logger.info("Pairing code created for user %s", user_id)
            return PairingCode(code=code, user_id=user_id, device_info=device_info, expires_at=expires_at)

    async def verify_pairing(
        self,
        code: str,
        device_id: str,
        platform: str = "",
        name: str = "",
    ) -> Optional[User]:
        async with self._lock:
            conn = await self._get_conn()
            await conn.execute("DELETE FROM pairing_codes WHERE expires_at < ?", (time.time(),))
            async with conn.execute("SELECT * FROM pairing_codes WHERE code = ?", (code,)) as cursor:
                row = await cursor.fetchone()
            if row is None:
                logger.warning("Pairing code not found: %s", code)
                return None
            if row["expires_at"] < time.time():
                await conn.execute("DELETE FROM pairing_codes WHERE code = ?", (code,))
                await conn.commit()
                logger.warning("Pairing code expired: %s", code)
                return None

            user_id = row["user_id"]
            # Upgrade guest → user on first pairing
            user = await self.get_user(user_id)
            if user and user.role == UserRole.guest:
                await conn.execute(
                    "UPDATE users SET role = ? WHERE user_id = ?",
                    (UserRole.user.value, user_id),
                )
                logger.info("User %s upgraded from guest to user via pairing", user_id)

            # Register device as trusted
            now = time.time()
            await conn.execute(
                "INSERT OR REPLACE INTO devices (device_id, user_id, platform, name, fingerprint, trusted, paired_at, last_seen) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (device_id, user_id, platform, name, "", 1, now, now),
            )

            # Consume the code
            await conn.execute("DELETE FROM pairing_codes WHERE code = ?", (code,))
            await conn.commit()
            logger.info("Device %s paired for user %s", device_id, user_id)

            return await self.get_user(user_id)

    async def is_trusted_device(self, device_id: str) -> bool:
        conn = await self._get_conn()
        async with conn.execute("SELECT trusted FROM devices WHERE device_id = ?", (device_id,)) as cursor:
            row = await cursor.fetchone()
        return row is not None and bool(row["trusted"])

    # ── Internal helpers ────────────────────────────────────

    @staticmethod
    def _row_to_user(row: aiosqlite.Row) -> User:
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
    def _row_to_device(row: aiosqlite.Row) -> Device:
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

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
