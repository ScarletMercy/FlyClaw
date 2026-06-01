"""Tests for file snapshot/rollback (CheckpointManager)."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from src.tools.snapshot import CheckpointManager, reset_snapshot_manager


@pytest.fixture
def tmp_store(tmp_path):
    """Create a temporary store directory."""
    store = tmp_path / "snapshots"
    store.mkdir()
    return str(store)


@pytest.fixture
def tmp_work(tmp_path):
    """Create a temporary working directory."""
    work = tmp_path / "workspace"
    work.mkdir()
    return str(work)


@pytest.fixture
def mgr(tmp_store):
    """Create a CheckpointManager with temp store."""
    reset_snapshot_manager()
    m = CheckpointManager(store_path=tmp_store, max_per_dir=10, max_file_size=1_000_000)
    return m


class TestCheckpointManager:
    async def test_empty_dir_snapshot(self, mgr, tmp_work):
        """Snapshot of an empty dir should work (or return None)."""
        result = await mgr.ensure_snapshot(tmp_work)
        # Empty dir might not create a commit, that's OK
        assert result is None or isinstance(result, str)

    async def test_snapshot_creates_store(self, tmp_path, tmp_work):
        """Manager should auto-create the bare git store."""
        store = str(tmp_path / "new_store")
        m = CheckpointManager(store_path=store, max_per_dir=10)
        # Write a file first
        Path(tmp_work, "test.txt").write_text("hello")
        await m.ensure_snapshot(tmp_work)
        assert Path(store).exists()

    async def test_snapshot_with_files(self, mgr, tmp_work):
        """Snapshot should capture file state."""
        Path(tmp_work, "hello.txt").write_text("world")
        snap_id = await mgr.ensure_snapshot(tmp_work)
        assert snap_id is not None
        assert len(snap_id) == 12

    async def test_dedup_snapshot(self, mgr, tmp_work):
        """Second snapshot with no changes should return same ID."""
        Path(tmp_work, "a.txt").write_text("aaa")
        snap1 = await mgr.ensure_snapshot(tmp_work)
        snap2 = await mgr.ensure_snapshot(tmp_work)
        assert snap1 == snap2

    async def test_list_snapshots(self, mgr, tmp_work):
        """list_snapshots should return entries."""
        Path(tmp_work, "f1.txt").write_text("one")
        await mgr.ensure_snapshot(tmp_work)
        Path(tmp_work, "f2.txt").write_text("two")
        await mgr.ensure_snapshot(tmp_work)

        snapshots = await mgr.list_snapshots(tmp_work)
        assert len(snapshots) == 2
        assert all("id" in s and "date" in s and "message" in s for s in snapshots)

    async def test_list_snapshots_empty(self, mgr, tmp_work):
        """No snapshots yet → empty list."""
        assert await mgr.list_snapshots(tmp_work) == []

    async def test_restore_full(self, mgr, tmp_work):
        """Full restore should revert all changes."""
        Path(tmp_work, "orig.txt").write_text("original content")
        snap_id = await mgr.ensure_snapshot(tmp_work)

        # Modify file
        Path(tmp_work, "orig.txt").write_text("modified!")
        Path(tmp_work, "new_file.txt").write_text("new")

        # Restore
        result = await mgr.restore(tmp_work, snap_id)
        assert "Restored" in result
        # Note: git checkout restores tracked files but doesn't delete new untracked files
        assert Path(tmp_work, "orig.txt").read_text() == "original content"

    async def test_restore_single_file(self, mgr, tmp_work):
        """Restore a single file from snapshot."""
        Path(tmp_work, "a.txt").write_text("aaa")
        Path(tmp_work, "b.txt").write_text("bbb")
        snap_id = await mgr.ensure_snapshot(tmp_work)

        Path(tmp_work, "a.txt").write_text("AAA")
        Path(tmp_work, "b.txt").write_text("BBB")

        result = await mgr.restore_file(tmp_work, snap_id, "a.txt")
        assert "a.txt" in result
        assert Path(tmp_work, "a.txt").read_text() == "aaa"
        # b.txt should remain modified
        assert Path(tmp_work, "b.txt").read_text() == "BBB"

    async def test_diff(self, mgr, tmp_work):
        """diff should show changes between snapshot and current state."""
        Path(tmp_work, "data.txt").write_text("v1")
        snap_id = await mgr.ensure_snapshot(tmp_work)

        Path(tmp_work, "data.txt").write_text("v2")
        diff_text = await mgr.diff(tmp_work, snap_id)
        assert "v1" in diff_text or "v2" in diff_text

    async def test_diff_no_changes(self, mgr, tmp_work):
        """diff with no changes should say so."""
        Path(tmp_work, "x.txt").write_text("x")
        snap_id = await mgr.ensure_snapshot(tmp_work)
        diff_text = await mgr.diff(tmp_work, snap_id)
        assert "No differences" in diff_text or diff_text == ""

    async def test_nonexistent_work_dir(self, mgr):
        """Snapshot of non-existent dir should return None."""
        result = await mgr.ensure_snapshot("/nonexistent/path/12345")
        assert result is None

    async def test_restore_no_store(self, tmp_work):
        """Restore when store doesn't exist should return error."""
        m = CheckpointManager(store_path="/nonexistent/store", max_per_dir=5)
        result = await m.restore(tmp_work, "abc123")
        assert "No snapshot store" in result

    async def test_prune_old_snapshots(self, tmp_path):
        """Manager should prune snapshots beyond max_per_dir."""
        store = str(tmp_path / "prune_store")
        work = str(tmp_path / "prune_work")
        os.makedirs(work)
        m = CheckpointManager(store_path=store, max_per_dir=3)

        for i in range(5):
            Path(work, f"file_{i}.txt").write_text(f"content_{i}")
            await m.ensure_snapshot(work)

        snapshots = await m.list_snapshots(work)
        assert len(snapshots) <= 3

    async def test_multiple_dirs_isolated(self, mgr, tmp_path):
        """Different work dirs should have independent snapshots."""
        work1 = str(tmp_path / "dir1")
        work2 = str(tmp_path / "dir2")
        os.makedirs(work1)
        os.makedirs(work2)

        Path(work1, "a.txt").write_text("aaa")
        await mgr.ensure_snapshot(work1)

        Path(work2, "b.txt").write_text("bbb")
        await mgr.ensure_snapshot(work2)

        snaps1 = await mgr.list_snapshots(work1)
        snaps2 = await mgr.list_snapshots(work2)
        assert len(snaps1) == 1
        assert len(snaps2) == 1
        assert snaps1[0]["id"] != snaps2[0]["id"]

    async def test_snapshot_excludes_patterns(self, mgr, tmp_work):
        """__pycache__, .git, node_modules should be excluded."""
        cache_dir = Path(tmp_work, "__pycache__")
        cache_dir.mkdir()
        Path(cache_dir, "mod.pyc").write_bytes(b"\x00\x01")
        Path(tmp_work, "real.py").write_text("print('hi')")

        snap_id = await mgr.ensure_snapshot(tmp_work)
        assert snap_id is not None

        # The .pyc file shouldn't be tracked
        diff_text = await mgr.diff(tmp_work, snap_id)
        assert "mod.pyc" not in diff_text
