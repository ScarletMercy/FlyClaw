from __future__ import annotations

import fnmatch
import logging
from typing import Optional

from src.agent.tooldef import ToolDef

from src.auth.models import User

logger = logging.getLogger("flyclaw.tool_policy")


class ToolPolicy:
    def __init__(
        self,
        allowed_patterns: Optional[list[str]] = None,
        denied_patterns: Optional[list[str]] = None,
        owner_only_tools: Optional[list[str]] = None,
    ):
        self.allowed_patterns = allowed_patterns or ["*"]
        self.denied_patterns = denied_patterns or []
        self.owner_only_tools = owner_only_tools or []

    def filter_tools(
        self,
        tools: list[ToolDef],
        sender_id: str = "",
        owner_id: str = "",
        user: Optional[User] = None,
    ) -> list[ToolDef]:
        filtered = []
        for tool in tools:
            name = tool.name

            if self._is_denied(name):
                logger.debug("Tool denied by policy: %s", name)
                continue

            if not self._is_allowed(name):
                logger.debug("Tool not in allow list: %s", name)
                continue

            if self._is_owner_only(name, sender_id, owner_id):
                logger.debug("Tool owner-only, sender not owner: %s (sender=%s)", name, sender_id)
                continue

            filtered.append(tool)

        # Apply RBAC filtering if user is provided
        if user is not None:
            from src.auth.rbac import get_rbac

            rbac = get_rbac()
            if rbac is not None:
                rbac_filtered = []
                for tool in filtered:
                    if rbac.check_tool_access(user, tool.name):
                        rbac_filtered.append(tool)
                    else:
                        logger.debug(
                            "Tool %s filtered by RBAC for user %s (role=%s)",
                            tool.name,
                            user.user_id,
                            user.role.value,
                        )
                filtered = rbac_filtered

        return filtered

    def _is_denied(self, tool_name: str) -> bool:
        for pattern in self.denied_patterns:
            if fnmatch.fnmatch(tool_name, pattern):
                return True
        return False

    def _is_allowed(self, tool_name: str) -> bool:
        if not self.allowed_patterns or self.allowed_patterns == ["*"]:
            return True
        for pattern in self.allowed_patterns:
            if fnmatch.fnmatch(tool_name, pattern):
                return True
        return False

    def _is_owner_only(self, tool_name: str, sender_id: str, owner_id: str) -> bool:
        if not self.owner_only_tools:
            return False
        if not owner_id:
            return False
        if sender_id == owner_id:
            return False
        for pattern in self.owner_only_tools:
            if fnmatch.fnmatch(tool_name, pattern):
                return True
        return False


def apply_tool_policy(
    tools: list[ToolDef],
    sender_id: str = "",
    config=None,
    user: Optional[User] = None,
) -> list[ToolDef]:
    if config is None or not hasattr(config, "tools"):
        return tools

    policy_cfg = getattr(config.tools, "policy", None)
    if policy_cfg is None:
        return tools

    owner_id = getattr(config, "owner_id", "") or ""

    policy = ToolPolicy(
        allowed_patterns=policy_cfg.allow or ["*"],
        denied_patterns=policy_cfg.deny or [],
        owner_only_tools=policy_cfg.owner_only or [],
    )
    return policy.filter_tools(tools, sender_id=sender_id, owner_id=owner_id, user=user)
