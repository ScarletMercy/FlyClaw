from __future__ import annotations

import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class UserRole(str, Enum):
    owner = "owner"
    admin = "admin"
    user = "user"
    guest = "guest"


ROLE_HIERARCHY: dict[UserRole, int] = {
    UserRole.owner: 100,
    UserRole.admin: 80,
    UserRole.user: 50,
    UserRole.guest: 10,
}

ROLE_PERMISSIONS: dict[UserRole, dict[str, Any]] = {
    UserRole.owner: {
        "tools": "*",
        "admin_api": True,
        "approval_bypass": True,
    },
    UserRole.admin: {
        "tools": "*",
        "admin_api": True,
        "approval_bypass": False,
    },
    UserRole.user: {
        "tools": [
            "exec",
            "web_search",
            "web_fetch",
            "memory_search",
            "memory_add",
            "memory_tools",
            "cron_tools",
            "file_read",
            "file_list",
            "media_tools",
        ],
        "admin_api": False,
        "approval_bypass": False,
    },
    UserRole.guest: {
        "tools": [],
        "admin_api": False,
        "approval_bypass": False,
    },
}


class User(BaseModel):
    user_id: str
    role: UserRole = UserRole.guest
    display_name: str = ""
    allowed_tools: list[str] = Field(default_factory=list)
    denied_tools: list[str] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    last_seen: float = Field(default_factory=time.time)

    @property
    def is_owner(self) -> bool:
        return self.role == UserRole.owner

    @property
    def is_admin(self) -> bool:
        return ROLE_HIERARCHY.get(self.role, 0) >= ROLE_HIERARCHY[UserRole.admin]

    @property
    def permissions(self) -> dict[str, Any]:
        return ROLE_PERMISSIONS.get(self.role, ROLE_PERMISSIONS[UserRole.guest])

    def touch(self) -> None:
        self.last_seen = time.time()


class Device(BaseModel):
    device_id: str
    user_id: str
    platform: str = ""  # "feishu", "qq", "web"
    name: str = ""
    fingerprint: str = ""
    trusted: bool = False
    paired_at: float = Field(default_factory=time.time)
    last_seen: float = Field(default_factory=time.time)


class PairingCode(BaseModel):
    code: str
    user_id: str
    device_info: str = ""
    expires_at: float
    created_at: float = Field(default_factory=time.time)
