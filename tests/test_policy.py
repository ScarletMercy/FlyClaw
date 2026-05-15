"""Tests for tool policy with RBAC integration."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_tool(name):
    t = MagicMock()
    t.name = name
    return t


class TestToolPolicy:
    def test_allow_all_default(self):
        from src.tools.policy import ToolPolicy

        policy = ToolPolicy()
        tools = [_make_tool("exec"), _make_tool("web_search"), _make_tool("custom")]
        filtered = policy.filter_tools(tools)
        assert len(filtered) == 3

    def test_deny_pattern(self):
        from src.tools.policy import ToolPolicy

        policy = ToolPolicy(denied_patterns=["exec*"])
        tools = [_make_tool("exec"), _make_tool("web_search")]
        filtered = policy.filter_tools(tools)
        assert len(filtered) == 1
        assert filtered[0].name == "web_search"

    def test_allow_pattern(self):
        from src.tools.policy import ToolPolicy

        policy = ToolPolicy(allowed_patterns=["web_*"])
        tools = [_make_tool("exec"), _make_tool("web_search"), _make_tool("web_fetch")]
        filtered = policy.filter_tools(tools)
        assert len(filtered) == 2

    def test_owner_only(self):
        from src.tools.policy import ToolPolicy

        policy = ToolPolicy(owner_only_tools=["admin_*"])
        tools = [_make_tool("admin_settings"), _make_tool("exec")]

        # Owner gets all
        filtered = policy.filter_tools(tools, sender_id="owner", owner_id="owner")
        assert len(filtered) == 2

        # Non-owner: admin tool filtered
        filtered = policy.filter_tools(tools, sender_id="guest", owner_id="owner")
        assert len(filtered) == 1
        assert filtered[0].name == "exec"


class TestApplyToolPolicy:
    def test_no_config_returns_all(self):
        from src.tools.policy import apply_tool_policy

        tools = [_make_tool("exec")]
        result = apply_tool_policy(tools, config=None)
        assert len(result) == 1

    def test_config_with_policy(self):
        from src.config import AppConfig
        from src.tools.policy import apply_tool_policy

        config = AppConfig()
        config.tools.policy.deny = ["exec"]
        tools = [_make_tool("exec"), _make_tool("web_search")]
        result = apply_tool_policy(tools, config=config)
        assert len(result) == 1
        assert result[0].name == "web_search"

    def test_rbac_filtering_with_user(self, tmp_path):
        """When user is passed, RBAC filtering is applied."""
        from src.auth.models import User, UserRole
        from src.auth.rbac import RBAC
        from src.auth.store import AuthStore
        from src.config import AppConfig
        from src.tools.policy import apply_tool_policy

        store = AuthStore(db_path=str(tmp_path / "auth.db"))
        rbac = RBAC(store)

        mock_container = MagicMock()
        mock_container.rbac = rbac

        with patch("src.auth.rbac.get_container", return_value=mock_container):
            config = AppConfig()
            guest = User(user_id="g1", role=UserRole.guest)
            tools = [_make_tool("exec"), _make_tool("web_search")]
            result = apply_tool_policy(tools, config=config, user=guest)
            # Guest has no tools allowed by RBAC
            assert len(result) == 0

        store.close()

    def test_rbac_allows_owner_all_tools(self, tmp_path):
        from src.auth.models import User, UserRole
        from src.auth.rbac import RBAC
        from src.auth.store import AuthStore
        from src.config import AppConfig
        from src.tools.policy import apply_tool_policy

        store = AuthStore(db_path=str(tmp_path / "auth.db"))
        rbac = RBAC(store)

        mock_container = MagicMock()
        mock_container.rbac = rbac

        with patch("src.auth.rbac.get_container", return_value=mock_container):
            config = AppConfig()
            owner = User(user_id="owner", role=UserRole.owner)
            tools = [_make_tool("exec"), _make_tool("admin_panel"), _make_tool("web_search")]
            result = apply_tool_policy(tools, config=config, user=owner)
            assert len(result) == 3

        store.close()
