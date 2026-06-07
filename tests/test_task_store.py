"""Tests for src/task/store.py — TaskRunStore CRUD operations."""

import pytest

from src.task.store import TaskRunStore
from src.task.types import TaskCheckpoint, TaskRun


@pytest.fixture
async def store(tmp_path):
    db_path = str(tmp_path / "test_tasks.db")
    s = TaskRunStore(db_path)
    await s.initialize()
    yield s
    await s.close()


def _make_run(**overrides) -> TaskRun:
    defaults = dict(
        goal="Test goal",
        steps=["Step 1", "Step 2"],
        checkpoints=[TaskCheckpoint(at="2026-06-07 12:00:00", prompt="check")],
        status="running",
    )
    defaults.update(overrides)
    return TaskRun(**defaults)


# ── Save / Get ─────────────────────────────────────────────


class TestSaveAndGet:
    @pytest.mark.asyncio
    async def test_save_and_retrieve(self, store):
        run = _make_run()
        await store.save(run)
        loaded = await store.get(run.id)
        assert loaded is not None
        assert loaded.goal == "Test goal"
        assert loaded.steps == ["Step 1", "Step 2"]

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, store):
        result = await store.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_save_updates(self, store):
        run = _make_run()
        await store.save(run)
        run.status = "completed"
        run.current_step = 2
        await store.save(run)
        loaded = await store.get(run.id)
        assert loaded.status == "completed"
        assert loaded.current_step == 2


# ── list_by_status ─────────────────────────────────────────


class TestListByStatus:
    @pytest.mark.asyncio
    async def test_filters_by_status(self, store):
        r1 = _make_run(status="running")
        r2 = _make_run(status="completed")
        await store.save(r1)
        await store.save(r2)

        running = await store.list_by_status("running")
        assert len(running) == 1
        assert running[0].id == r1.id

    @pytest.mark.asyncio
    async def test_multiple_statuses(self, store):
        r1 = _make_run(status="running")
        r2 = _make_run(status="planning")
        await store.save(r1)
        await store.save(r2)

        result = await store.list_by_status("running", "planning")
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_empty_result(self, store):
        result = await store.list_by_status("cancelled")
        assert result == []

    @pytest.mark.asyncio
    async def test_no_statuses(self, store):
        result = await store.list_by_status()
        assert result == []


# ── update_status ──────────────────────────────────────────


class TestUpdateStatus:
    @pytest.mark.asyncio
    async def test_update(self, store):
        run = _make_run(status="running")
        await store.save(run)
        await store.update_status(run.id, "cancelled")
        loaded = await store.get(run.id)
        assert loaded.status == "cancelled"

    @pytest.mark.asyncio
    async def test_update_nonexistent(self, store):
        # Should not crash
        await store.update_status("nonexistent", "cancelled")


# ── update_checkpoint ──────────────────────────────────────


class TestUpdateCheckpoint:
    @pytest.mark.asyncio
    async def test_update_cp(self, store):
        cp = TaskCheckpoint(at="2026-06-07 12:00:00", prompt="check")
        run = _make_run(checkpoints=[cp])
        await store.save(run)
        await store.update_checkpoint(run.id, cp.id, status="scheduled", cron_job_id="cj1")
        loaded = await store.get(run.id)
        assert loaded.checkpoints[0].status == "scheduled"
        assert loaded.checkpoints[0].cron_job_id == "cj1"


# ── delete ─────────────────────────────────────────────────


class TestDelete:
    @pytest.mark.asyncio
    async def test_delete(self, store):
        run = _make_run()
        await store.save(run)
        await store.delete(run.id)
        result = await store.get(run.id)
        assert result is None
