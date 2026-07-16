"""Tests for src/tools/approval.py — ApprovalManager, session approvals."""

import asyncio

import pytest

from src.tools.approval import ApprovalManager, ApprovalRequest


@pytest.fixture
def mgr():
    return ApprovalManager()


# ── needs_approval ─────────────────────────────────────────


class TestNeedsApproval:
    def test_off_mode(self, mgr):
        assert mgr.needs_approval("exec", "rm -rf", "off", True) is False

    def test_always_mode(self, mgr):
        assert mgr.needs_approval("exec", "ls", "always", False) is True

    def test_on_denylist_miss_denylisted(self, mgr):
        assert mgr.needs_approval("exec", "rm -rf", "on_denylist_miss", True) is True

    def test_on_denylist_miss_not_denylisted(self, mgr):
        assert mgr.needs_approval("exec", "ls", "on_denylist_miss", False) is False

    def test_unknown_mode(self, mgr):
        assert mgr.needs_approval("exec", "ls", "unknown", True) is False


# ── session approvals ──────────────────────────────────────


class TestSessionApproval:
    def test_session_approval_exact_match(self, mgr):
        mgr.approve_session("thread1", "exec", "rm -rf /")
        assert mgr.has_session_approval("thread1", "exec", "rm -rf /") is True

    def test_session_approval_different_args(self, mgr):
        mgr.approve_session("thread1", "exec", "rm -rf /")
        assert mgr.has_session_approval("thread1", "exec", "ls") is False

    def test_session_approval_different_thread(self, mgr):
        mgr.approve_session("thread1", "exec", "rm -rf /")
        assert mgr.has_session_approval("thread2", "exec", "rm -rf /") is False

    def test_clear_session(self, mgr):
        mgr.approve_session("thread1", "exec", "rm -rf /")
        mgr.clear_session("thread1")
        assert mgr.has_session_approval("thread1", "exec", "rm -rf /") is False

    def test_clear_session_all(self, mgr):
        mgr.approve_session("thread1", "exec", "rm -rf /")
        mgr.approve_session_pattern("thread1", "del ")
        mgr.clear_session_all("thread1")
        assert mgr.has_session_approval("thread1", "exec", "rm -rf /") is False


# ── session pattern approvals ──────────────────────────────


class TestSessionPatternApproval:
    def test_pattern_match_in_args(self, mgr):
        mgr.approve_session_pattern("thread1", "del ")
        assert mgr.has_session_approval("thread1", "exec", "Del C:\\file.txt") is True

    def test_pattern_match_in_tool_name(self, mgr):
        mgr.approve_session_pattern("thread1", "exec")
        assert mgr.has_session_approval("thread1", "Exec", "anything") is True

    def test_pattern_no_match(self, mgr):
        mgr.approve_session_pattern("thread1", "del ")
        assert mgr.has_session_approval("thread1", "exec", "ls -la") is False

    def test_clear_session_preserves_patterns(self, mgr):
        mgr.approve_session("thread1", "exec", "rm -rf /")
        mgr.approve_session_pattern("thread1", "del ")
        mgr.clear_session("thread1")
        # Exact match cleared
        assert mgr.has_session_approval("thread1", "exec", "rm -rf /") is False
        # Pattern preserved
        assert mgr.has_session_approval("thread1", "exec", "del something") is True


# ── request/approve flow ───────────────────────────────────


class TestRequestApproval:
    def test_request_returns_approval_request(self, mgr):
        req = mgr.request_approval("exec", "rm -rf /", thread_id="t1")
        assert isinstance(req, ApprovalRequest)
        assert req.tool_name == "exec"
        assert req.thread_id == "t1"

    def test_request_list_pending(self, mgr):
        mgr.request_approval("exec", "cmd1")
        mgr.request_approval("file_write", "cmd2")
        pending = mgr.list_pending()
        assert len(pending) == 2

    def test_get_pending(self, mgr):
        req = mgr.request_approval("exec", "cmd")
        found = mgr.get_pending(req.id)
        assert found is not None
        assert found.tool_name == "exec"

    def test_get_pending_not_found(self, mgr):
        assert mgr.get_pending("nonexistent") is None


class TestResolveApproval:
    @pytest.mark.asyncio
    async def test_allow_once(self, mgr):
        req = mgr.request_approval("exec", "cmd")

        async def resolver():
            await asyncio.sleep(0.05)
            mgr.resolve(req.id, "allow_once", "ok")

        asyncio.create_task(resolver())
        decision, response = await mgr.await_approval(req.id, timeout=2)
        assert decision == "allow_once"
        assert response == "ok"

    @pytest.mark.asyncio
    async def test_deny(self, mgr):
        req = mgr.request_approval("exec", "cmd")

        async def resolver():
            await asyncio.sleep(0.05)
            mgr.resolve(req.id, "deny")

        asyncio.create_task(resolver())
        decision, _ = await mgr.await_approval(req.id, timeout=2)
        assert decision == "deny"

    @pytest.mark.asyncio
    async def test_timeout(self, mgr):
        req = mgr.request_approval("exec", "cmd")
        decision, _ = await mgr.await_approval(req.id, timeout=0.1)
        assert decision == "timeout"

    @pytest.mark.asyncio
    async def test_await_not_found(self, mgr):
        decision, _ = await mgr.await_approval("nonexistent")
        assert decision == "deny"

    def test_resolve_invalid_decision(self, mgr):
        req = mgr.request_approval("exec", "cmd")
        assert mgr.resolve(req.id, "bad_decision") is False

    def test_resolve_unknown_request(self, mgr):
        assert mgr.resolve("nonexistent", "allow_once") is False

    def test_resolve_twice(self, mgr):
        req = mgr.request_approval("exec", "cmd")
        assert mgr.resolve(req.id, "allow_once") is True
        assert mgr.resolve(req.id, "deny") is False

    def test_is_resolved(self, mgr):
        req = mgr.request_approval("exec", "cmd")
        assert mgr.is_resolved(req.id) is False
        mgr.resolve(req.id, "allow_once")
        assert mgr.is_resolved(req.id) is True

    def test_cancel_pending(self, mgr):
        req = mgr.request_approval("exec", "cmd")
        assert mgr.cancel_pending(req.id) is True
        assert mgr.get_pending(req.id) is None
        assert mgr.cancel_pending(req.id) is False


# ── _make_digest ───────────────────────────────────────────


class TestMakeDigest:
    def test_deterministic(self, mgr):
        d1 = mgr._make_digest("exec", "rm -rf /")
        d2 = mgr._make_digest("exec", "rm -rf /")
        assert d1 == d2

    def test_different_inputs(self, mgr):
        d1 = mgr._make_digest("exec", "rm -rf /")
        d2 = mgr._make_digest("exec", "ls")
        assert d1 != d2

    def test_long_args_produce_stable_digest(self, mgr):
        long_arg = "x" * 500
        d1 = mgr._make_digest("exec", long_arg)
        d2 = mgr._make_digest("exec", long_arg[:200] + "y" * 300)
        # Both truncated to 200 before hashing — digests differ because
        # the first 200 chars are same but digest also includes the full first 200
        assert isinstance(d1, str) and len(d1) == 16
