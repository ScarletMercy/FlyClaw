"""Tests for auth RBAC system: models, store, and permission checks."""

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestUserModels:
    def test_user_defaults(self):
        from src.auth.models import User, UserRole

        u = User(user_id="test")
        assert u.role == UserRole.guest
        assert u.display_name == ""
        assert u.allowed_tools == []
        assert u.denied_tools == []
        assert u.is_owner is False
        assert u.is_admin is False

    def test_owner_properties(self):
        from src.auth.models import User, UserRole

        u = User(user_id="owner", role=UserRole.owner)
        assert u.is_owner is True
        assert u.is_admin is True

    def test_admin_properties(self):
        from src.auth.models import User, UserRole

        u = User(user_id="admin", role=UserRole.admin)
        assert u.is_owner is False
        assert u.is_admin is True

    def test_user_touch(self):
        from src.auth.models import User

        u = User(user_id="test", last_seen=0)
        before = time.time()
        u.touch()
        assert u.last_seen >= before

    def test_role_hierarchy(self):
        from src.auth.models import ROLE_HIERARCHY, UserRole

        assert ROLE_HIERARCHY[UserRole.owner] > ROLE_HIERARCHY[UserRole.admin]
        assert ROLE_HIERARCHY[UserRole.admin] > ROLE_HIERARCHY[UserRole.user]
        assert ROLE_HIERARCHY[UserRole.user] > ROLE_HIERARCHY[UserRole.guest]

    def test_role_permissions(self):
        from src.auth.models import ROLE_PERMISSIONS, UserRole

        assert ROLE_PERMISSIONS[UserRole.owner]["tools"] == "*"
        assert ROLE_PERMISSIONS[UserRole.owner]["approval_bypass"] is True
        assert ROLE_PERMISSIONS[UserRole.guest]["tools"] == []
        assert ROLE_PERMISSIONS[UserRole.user]["admin_api"] is False
        assert "exec" in ROLE_PERMISSIONS[UserRole.user]["tools"]


class TestAuthStore:
    def _make_store(self, tmp_path):
        from src.auth.store import AuthStore

        return AuthStore(db_path=str(tmp_path / "auth.db"))

    def test_create_store(self, tmp_path):
        store = self._make_store(tmp_path)
        users = store.list_users()
        assert users == []
        store.close()

    def test_get_or_create_user(self, tmp_path):
        store = self._make_store(tmp_path)
        user = store.get_or_create_user("u1", display_name="Alice")
        assert user.user_id == "u1"
        assert user.display_name == "Alice"
        assert user.role.value == "guest"
        store.close()

    def test_get_or_create_user_idempotent(self, tmp_path):
        store = self._make_store(tmp_path)
        u1 = store.get_or_create_user("u1", display_name="Alice")
        u2 = store.get_or_create_user("u1", display_name="Bob")
        assert u1.user_id == u2.user_id
        # display_name unchanged on re-get (only set on create)
        assert u2.display_name == "Alice"
        store.close()

    def test_update_user_role(self, tmp_path):
        from src.auth.models import UserRole

        store = self._make_store(tmp_path)
        store.get_or_create_user("u1")
        ok = store.update_user_role("u1", UserRole.admin)
        assert ok is True
        user = store.get_user("u1")
        assert user.role == UserRole.admin
        store.close()

    def test_update_user_tools(self, tmp_path):
        store = self._make_store(tmp_path)
        store.get_or_create_user("u1")
        ok = store.update_user_tools("u1", allowed_tools=["exec", "web_search"], denied_tools=["cron_tools"])
        assert ok is True
        user = store.get_user("u1")
        assert "exec" in user.allowed_tools
        assert "cron_tools" in user.denied_tools
        store.close()

    def test_list_users(self, tmp_path):
        store = self._make_store(tmp_path)
        store.get_or_create_user("u1")
        store.get_or_create_user("u2")
        users = store.list_users()
        assert len(users) == 2
        store.close()

    def test_delete_user(self, tmp_path):
        store = self._make_store(tmp_path)
        store.get_or_create_user("u1")
        ok = store.delete_user("u1")
        assert ok is True
        assert store.get_user("u1") is None
        store.close()

    def test_delete_nonexistent_user(self, tmp_path):
        store = self._make_store(tmp_path)
        ok = store.delete_user("ghost")
        assert ok is False
        store.close()

    def test_device_crud(self, tmp_path):
        store = self._make_store(tmp_path)
        store.get_or_create_user("u1")
        dev = store.register_device("d1", "u1", platform="web", name="Chrome")
        assert dev.device_id == "d1"
        assert dev.trusted is False

        ok = store.trust_device("d1")
        assert ok is True
        dev = store.get_device("d1")
        assert dev.trusted is True

        devices = store.list_user_devices("u1")
        assert len(devices) == 1

        ok = store.delete_device("d1")
        assert ok is True
        assert store.get_device("d1") is None
        store.close()

    def test_pairing_flow(self, tmp_path):
        from src.auth.models import UserRole

        store = self._make_store(tmp_path)
        store.get_or_create_user("u1")

        code = store.create_pairing_code("u1", ttl_seconds=300)
        assert len(code.code) == 6
        assert code.code.isdigit()

        user = store.verify_pairing(code.code, "dev1", platform="web")
        assert user is not None
        assert user.role == UserRole.user  # upgraded from guest

        # Device is trusted
        assert store.is_trusted_device("dev1")

        # Code is consumed
        user2 = store.verify_pairing(code.code, "dev2")
        assert user2 is None
        store.close()

    def test_expired_pairing_code(self, tmp_path):
        store = self._make_store(tmp_path)
        store.get_or_create_user("u1")

        code = store.create_pairing_code("u1", ttl_seconds=-1)  # already expired
        user = store.verify_pairing(code.code, "dev1")
        assert user is None
        store.close()


class TestRBAC:
    def _make_rbac(self, tmp_path):
        from src.auth.rbac import RBAC
        from src.auth.store import AuthStore

        store = AuthStore(db_path=str(tmp_path / "auth.db"))
        return RBAC(store, config=None)

    def test_resolve_unknown_user(self, tmp_path):
        from src.auth.models import UserRole

        rbac = self._make_rbac(tmp_path)
        user = rbac.resolve_user("newuser")
        assert user.role == UserRole.guest
        rbac.store.close()

    def test_resolve_user_with_default_role_user(self, tmp_path):
        from src.auth.models import UserRole
        from src.auth.rbac import RBAC
        from src.auth.store import AuthStore
        from src.config import AppConfig

        store = AuthStore(db_path=str(tmp_path / "auth.db"))
        config = AppConfig()
        rbac = RBAC(store, config)
        user = rbac.resolve_user("someuser")
        assert user.role == UserRole.owner
        rbac.store.close()

    def test_check_tool_access_guest(self, tmp_path):
        from src.auth.models import User, UserRole

        rbac = self._make_rbac(tmp_path)
        user = User(user_id="g1", role=UserRole.guest)
        assert rbac.check_tool_access(user, "exec") is False
        assert rbac.check_tool_access(user, "web_search") is False
        rbac.store.close()

    def test_check_tool_access_user(self, tmp_path):
        from src.auth.models import User, UserRole

        rbac = self._make_rbac(tmp_path)
        user = User(user_id="u1", role=UserRole.user)
        assert rbac.check_tool_access(user, "exec") is True
        assert rbac.check_tool_access(user, "web_search") is True
        assert rbac.check_tool_access(user, "some_admin_tool") is False
        rbac.store.close()

    def test_check_tool_access_admin(self, tmp_path):
        from src.auth.models import User, UserRole

        rbac = self._make_rbac(tmp_path)
        user = User(user_id="a1", role=UserRole.admin)
        assert rbac.check_tool_access(user, "exec") is True
        assert rbac.check_tool_access(user, "anything") is True
        rbac.store.close()

    def test_explicit_deny_overrides_role(self, tmp_path):
        from src.auth.models import User, UserRole

        rbac = self._make_rbac(tmp_path)
        user = User(user_id="u1", role=UserRole.admin, denied_tools=["exec"])
        assert rbac.check_tool_access(user, "exec") is False
        rbac.store.close()

    def test_explicit_allow_overrides_role(self, tmp_path):
        from src.auth.models import User, UserRole

        rbac = self._make_rbac(tmp_path)
        user = User(user_id="g1", role=UserRole.guest, allowed_tools=["web_search"])
        assert rbac.check_tool_access(user, "web_search") is True
        # But other tools still denied
        assert rbac.check_tool_access(user, "exec") is False
        rbac.store.close()

    def test_filter_tools(self, tmp_path):
        from unittest.mock import MagicMock

        from src.auth.models import User, UserRole

        rbac = self._make_rbac(tmp_path)

        tools = []
        for name in ["exec", "web_search", "admin_delete", "cron_tools"]:
            t = MagicMock()
            t.name = name
            tools.append(t)

        user = User(user_id="u1", role=UserRole.user)
        filtered = rbac.filter_tools(user, tools)
        names = [t.name for t in filtered]
        assert "exec" in names
        assert "web_search" in names
        assert "admin_delete" not in names
        rbac.store.close()

    def test_check_admin_access(self, tmp_path):
        from src.auth.models import User, UserRole

        rbac = self._make_rbac(tmp_path)
        assert rbac.check_admin_access(User(user_id="a", role=UserRole.admin)) is True
        assert rbac.check_admin_access(User(user_id="u", role=UserRole.user)) is False
        assert rbac.check_admin_access(User(user_id="g", role=UserRole.guest)) is False
        rbac.store.close()

    def test_require_role(self, tmp_path):
        from src.auth.models import User, UserRole

        rbac = self._make_rbac(tmp_path)
        assert rbac.require_role(User(user_id="a", role=UserRole.admin), UserRole.user) is True
        assert rbac.require_role(User(user_id="g", role=UserRole.guest), UserRole.admin) is False
        rbac.store.close()
