from __future__ import annotations

import fnmatch
import logging
from typing import TYPE_CHECKING, Optional

from src.auth.models import ROLE_PERMISSIONS, User, UserRole
from src.auth.store import AuthStore

if TYPE_CHECKING:
    from src.config import AppConfig

logger = logging.getLogger("flyclaw.auth.rbac")


class RBAC:
    def __init__(self, store: AuthStore, config: Optional["AppConfig"] = None):
        self._store = store
        self._config = config

    @property
    def store(self) -> AuthStore:
        return self._store

    def resolve_user(
        self,
        sender_id: str,
        display_name: str = "",
    ) -> User:
        """Resolve or auto-register a user by channel sender_id."""

        default_role = UserRole.guest
        if self._config:
            cfg_default = getattr(self._config.auth, "default_role", "") if hasattr(self._config, "auth") else ""
            if cfg_default:
                try:
                    default_role = UserRole(cfg_default)
                except ValueError:
                    pass

        user = self._store.get_or_create_user(
            user_id=sender_id,
            display_name=display_name,
            default_role=default_role,
        )

        # Upgrade existing user if config default_role is higher than current role
        from src.auth.models import ROLE_HIERARCHY
        if user.role != default_role:
            current_level = ROLE_HIERARCHY.get(user.role, 0)
            default_level = ROLE_HIERARCHY.get(default_role, 0)
            if default_level > current_level:
                logger.info(
                    "Upgrading user %s from %s to %s (config default_role)",
                    user.user_id, user.role.value, default_role.value,
                )
                self._store.update_user_role(user.user_id, default_role)
                user.role = default_role

        return user

    def check_tool_access(self, user: User, tool_name: str) -> bool:
        """Check if a user can access a specific tool.

        Resolution order:
        1. User-level deny list (always blocks)
        2. User-level allow list (overrides role)
        3. Role-level permissions
        """
        # 1. Explicit deny always wins
        for pattern in user.denied_tools:
            if fnmatch.fnmatch(tool_name, pattern):
                logger.debug("Tool %s denied for user %s (explicit deny)", tool_name, user.user_id)
                return False

        # 2. Explicit allow overrides role
        for pattern in user.allowed_tools:
            if fnmatch.fnmatch(tool_name, pattern):
                return True

        # 3. Role-level check
        perms = user.permissions
        role_tools = perms.get("tools", [])
        if role_tools == "*":
            return True
        for pattern in role_tools:
            if fnmatch.fnmatch(tool_name, pattern):
                return True

        return False

    def check_admin_access(self, user: User) -> bool:
        """Check if user has admin API access."""
        return user.permissions.get("admin_api", False) is True

    def check_approval_bypass(self, user: User) -> bool:
        """Check if user can bypass tool approval workflows."""
        return user.permissions.get("approval_bypass", False) is True

    def filter_tools(self, user: User, tools: list) -> list:
        """Filter a tool list based on user permissions."""
        filtered = []
        for tool in tools:
            if self.check_tool_access(user, tool.name):
                filtered.append(tool)
            else:
                logger.debug(
                    "Tool %s filtered out for user %s (role=%s)",
                    tool.name,
                    user.user_id,
                    user.role.value,
                )
        return filtered

    def require_role(self, user: User, minimum_role: UserRole) -> bool:
        """Check if user has at least the minimum required role."""
        from src.auth.models import ROLE_HIERARCHY

        user_level = ROLE_HIERARCHY.get(user.role, 0)
        required_level = ROLE_HIERARCHY.get(minimum_role, 0)
        return user_level >= required_level


# ── Module-level singleton — delegates to ServiceContainer ──

from src._container import get_container


def get_rbac() -> Optional[RBAC]:
    return get_container().rbac
