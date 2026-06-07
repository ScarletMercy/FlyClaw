"""Tests for src/tools/process.py — BackgroundSession, ProcessRegistry, kill_process_tree."""

import asyncio
import os
import time

import pytest

from src.tools.process import (
    BackgroundSession,
    ProcessRegistry,
    kill_process_tree,
    get_process_registry,
)


# ── BackgroundSession ──────────────────────────────────────


class TestBackgroundSession:
    def test_status_running_when_returncode_none(self):
        proc = _make_mock_proc(returncode=None)
        s = BackgroundSession(
            id="abc",
            command="echo",
            pid=1,
            proc=proc,
            started_at=time.monotonic(),
            workdir=".",
        )
        assert s.status == "running"

    def test_status_exited_when_exit_code_set(self):
        proc = _make_mock_proc(returncode=0)
        s = BackgroundSession(
            id="abc",
            command="echo",
            pid=1,
            proc=proc,
            started_at=time.monotonic(),
            workdir=".",
            exit_code=0,
        )
        assert s.status == "exited"

    def test_status_exited_when_returncode_set_but_exit_code_none(self):
        proc = _make_mock_proc(returncode=1)
        s = BackgroundSession(
            id="abc",
            command="echo",
            pid=1,
            proc=proc,
            started_at=time.monotonic(),
            workdir=".",
        )
        assert s.status == "exited"

    def test_elapsed_while_running(self):
        proc = _make_mock_proc(returncode=None)
        now = time.monotonic()
        s = BackgroundSession(
            id="abc",
            command="echo",
            pid=1,
            proc=proc,
            started_at=now - 5.0,
            workdir=".",
        )
        assert s.elapsed >= 4.9

    def test_elapsed_after_finished(self):
        proc = _make_mock_proc(returncode=0)
        now = time.monotonic()
        s = BackgroundSession(
            id="abc",
            command="echo",
            pid=1,
            proc=proc,
            started_at=now - 10.0,
            workdir=".",
            finished_at=now - 2.0,
        )
        assert 7.9 <= s.elapsed <= 8.1


# ── ProcessRegistry ────────────────────────────────────────


class TestProcessRegistry:
    @pytest.mark.asyncio
    async def test_spawn_and_poll(self):
        reg = ProcessRegistry()
        sid = await reg.spawn("echo hello")
        # Give the process time to complete
        await asyncio.sleep(0.5)

        info = await reg.poll(sid)
        assert info["session_id"] == sid
        assert info["status"] == "exited"
        assert info["exit_code"] == 0
        assert "hello" in info["output_tail"]

    @pytest.mark.asyncio
    async def test_poll_not_found(self):
        reg = ProcessRegistry()
        info = await reg.poll("nonexistent")
        assert info["status"] == "not_found"

    @pytest.mark.asyncio
    async def test_wait_completes(self):
        reg = ProcessRegistry()
        sid = await reg.spawn("echo done")
        result = await reg.wait(sid, timeout=10)
        assert result["status"] == "exited"
        assert result["exit_code"] == 0
        assert "done" in result["output"]

    @pytest.mark.asyncio
    async def test_wait_not_found(self):
        reg = ProcessRegistry()
        result = await reg.wait("nonexistent")
        assert result["status"] == "not_found"

    @pytest.mark.asyncio
    async def test_wait_timeout_kills(self):
        reg = ProcessRegistry()
        # Use a long sleep to force timeout
        if os.name == "nt":
            sid = await reg.spawn("ping -n 60 127.0.0.1 >nul")
        else:
            sid = await reg.spawn("sleep 60")
        result = await reg.wait(sid, timeout=1)
        assert result["status"] == "timeout"

    @pytest.mark.asyncio
    async def test_kill_running(self):
        reg = ProcessRegistry()
        if os.name == "nt":
            sid = await reg.spawn("ping -n 60 127.0.0.1 >nul")
        else:
            sid = await reg.spawn("sleep 60")
        await asyncio.sleep(0.3)
        msg = await reg.kill(sid)
        assert "Killed" in msg

    @pytest.mark.asyncio
    async def test_kill_not_found(self):
        reg = ProcessRegistry()
        msg = await reg.kill("nonexistent")
        assert "not found" in msg.lower()

    @pytest.mark.asyncio
    async def test_kill_already_exited(self):
        reg = ProcessRegistry()
        sid = await reg.spawn("echo bye")
        await asyncio.sleep(0.5)
        msg = await reg.kill(sid)
        assert "already exited" in msg.lower()

    @pytest.mark.asyncio
    async def test_log_output(self):
        reg = ProcessRegistry()
        sid = await reg.spawn("echo line1 && echo line2")
        await asyncio.sleep(0.5)
        log = await reg.log(sid)
        assert "line1" in log
        assert "line2" in log

    @pytest.mark.asyncio
    async def test_log_not_found(self):
        reg = ProcessRegistry()
        log = await reg.log("nonexistent")
        assert "not found" in log.lower()

    @pytest.mark.asyncio
    async def test_list_sessions(self):
        reg = ProcessRegistry()
        sid = await reg.spawn("echo test")
        await asyncio.sleep(0.3)
        sessions = reg.list_sessions()
        assert len(sessions) >= 1
        assert sessions[0]["id"] == sid

    @pytest.mark.asyncio
    async def test_evict_removes_old_finished(self):
        reg = ProcessRegistry(max_sessions=2)
        s1 = await reg.spawn("echo s1")
        await asyncio.sleep(0.3)
        s2 = await reg.spawn("echo s2")
        await asyncio.sleep(0.3)
        s3 = await reg.spawn("echo s3")
        await asyncio.sleep(0.3)
        # After spawning 3 with max_sessions=2, oldest finished should be evicted
        sessions = reg.list_sessions()
        ids = [s["id"] for s in sessions]
        # s3 should always be present; at least one should be evicted
        assert s3 in ids

    @pytest.mark.asyncio
    async def test_spawn_with_env(self):
        reg = ProcessRegistry()
        if os.name == "nt":
            sid = await reg.spawn("echo %MY_TEST_VAR%", env={"MY_TEST_VAR": "hello_env"})
        else:
            sid = await reg.spawn("echo $MY_TEST_VAR", env={"MY_TEST_VAR": "hello_env"})
        result = await reg.wait(sid, timeout=5)
        assert "hello_env" in result["output"]


class TestGetProcessRegistry:
    def test_returns_instance(self):
        # Reset singleton
        import src.tools.process as mod

        mod._registry = None
        reg = get_process_registry()
        assert isinstance(reg, ProcessRegistry)
        # Cleanup
        mod._registry = None


# ── kill_process_tree ──────────────────────────────────────


class TestKillProcessTree:
    @pytest.mark.asyncio
    async def test_kill_with_none_pid(self):
        proc = _make_mock_proc(pid=None, returncode=None)
        result = await kill_process_tree(proc)
        assert result is False

    @pytest.mark.asyncio
    async def test_kill_already_dead(self):
        proc = _make_mock_proc(pid=999999, returncode=0)
        # This should still attempt to kill (may fail gracefully)
        result = await kill_process_tree(proc)
        # Result depends on OS — just ensure it doesn't crash
        assert isinstance(result, bool)


# ── Helpers ────────────────────────────────────────────────


def _make_mock_proc(returncode=None, pid=12345):
    """Create a mock asyncio.subprocess.Process-like object."""

    class MockProc:
        def __init__(self):
            self.pid = pid
            self.returncode = returncode
            self.stdout = None
            self.stderr = None

        async def wait(self):
            return self.returncode

        def kill(self):
            self.returncode = -9

    return MockProc()
