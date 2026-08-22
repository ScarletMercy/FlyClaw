"""Tests for ConsolidationScheduler and _next_occurrence."""

from __future__ import annotations

import asyncio
import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.consolidation_scheduler import (
    ConsolidationScheduler,
    _next_occurrence,
    _TRIGGER_HOUR,
    _TRIGGER_MINUTE,
)


# ─── _next_occurrence pure function ──────────────────────────────────────────


class TestNextOccurrence:
    def test_before_target_time_same_day(self):
        now = datetime.datetime(2026, 6, 11, 2, 30, tzinfo=datetime.timezone.utc)
        result = _next_occurrence(now, 3, 0)
        assert result == datetime.datetime(2026, 6, 11, 3, 0, tzinfo=datetime.timezone.utc)

    def test_after_target_time_next_day(self):
        now = datetime.datetime(2026, 6, 11, 4, 0, tzinfo=datetime.timezone.utc)
        result = _next_occurrence(now, 3, 0)
        assert result == datetime.datetime(2026, 6, 12, 3, 0, tzinfo=datetime.timezone.utc)

    def test_exactly_at_target_time_next_day(self):
        now = datetime.datetime(2026, 6, 11, 3, 0, 0, tzinfo=datetime.timezone.utc)
        result = _next_occurrence(now, 3, 0)
        assert result == datetime.datetime(2026, 6, 12, 3, 0, tzinfo=datetime.timezone.utc)

    def test_seconds_microseconds_zeroed(self):
        now = datetime.datetime(2026, 6, 11, 2, 0, 45, 123456, tzinfo=datetime.timezone.utc)
        result = _next_occurrence(now, 3, 0)
        assert result.second == 0
        assert result.microsecond == 0

    def test_different_hour_minute(self):
        now = datetime.datetime(2026, 6, 11, 10, 0, tzinfo=datetime.timezone.utc)
        result = _next_occurrence(now, 14, 30)
        assert result == datetime.datetime(2026, 6, 11, 14, 30, tzinfo=datetime.timezone.utc)

    def test_cross_month_boundary(self):
        now = datetime.datetime(2026, 6, 30, 23, 0, tzinfo=datetime.timezone.utc)
        result = _next_occurrence(now, 3, 0)
        assert result == datetime.datetime(2026, 7, 1, 3, 0, tzinfo=datetime.timezone.utc)


# ─── ConsolidationScheduler lifecycle ────────────────────────────────────────


class TestSchedulerLifecycle:
    @pytest.mark.asyncio
    async def test_start_creates_task(self):
        scheduler = ConsolidationScheduler()
        container = MagicMock()
        container.config.agents.timezone = "UTC"

        async def _fake_sleep(seconds):
            await asyncio.sleep(9999)

        with patch("asyncio.sleep", side_effect=_fake_sleep):
            await scheduler.start(container)

        assert scheduler._task is not None
        scheduler._task.cancel()
        try:
            await scheduler._task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_start_idempotent(self):
        scheduler = ConsolidationScheduler()
        container = MagicMock()
        container.config.agents.timezone = "UTC"

        async def _fake_sleep(seconds):
            await asyncio.sleep(9999)

        with patch("asyncio.sleep", side_effect=_fake_sleep):
            await scheduler.start(container)
            task1 = scheduler._task
            await scheduler.start(container)
            assert scheduler._task is task1

        scheduler._task.cancel()
        try:
            await scheduler._task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self):
        scheduler = ConsolidationScheduler()
        container = MagicMock()
        container.config.agents.timezone = "UTC"

        async def _fake_sleep(seconds):
            await asyncio.sleep(9999)

        with patch("asyncio.sleep", side_effect=_fake_sleep):
            await scheduler.start(container)

        await scheduler.stop()
        assert scheduler._task is None

    @pytest.mark.asyncio
    async def test_stop_when_not_started(self):
        scheduler = ConsolidationScheduler()
        await scheduler.stop()
        assert scheduler._task is None


# ─── Scheduler _loop dispatch logic ──────────────────────────────────────────


class TestSchedulerLoop:
    @pytest.mark.asyncio
    async def test_non_sunday_calls_only_daily(self):
        scheduler = ConsolidationScheduler()
        container = MagicMock()
        container.config.agents.timezone = "UTC"

        daily_mock = AsyncMock(return_value={"sessions_processed": 0})
        memory_mock = AsyncMock(return_value={"total_memories": 0})

        call_count = 0

        async def _sleep_then_cancel(seconds):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise asyncio.CancelledError()

        with (
            patch("asyncio.sleep", side_effect=_sleep_then_cancel),
            patch("src.services.consolidation_state.run_session_organize", daily_mock),
            patch("src.services.consolidation_state.run_memory_organize", memory_mock),
            patch(
                "src.services.consolidation_scheduler._next_occurrence",
                return_value=datetime.datetime(2026, 6, 15, 3, 0, tzinfo=datetime.timezone.utc),
            ),
        ):
            try:
                await scheduler._loop(container)
            except asyncio.CancelledError:
                pass

        daily_mock.assert_called_once_with(container)
        memory_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_sunday_calls_memory_then_daily(self):
        scheduler = ConsolidationScheduler()
        container = MagicMock()
        container.config.agents.timezone = "UTC"

        daily_mock = AsyncMock(return_value={"sessions_processed": 0})
        memory_mock = AsyncMock(return_value={"total_memories": 0})

        call_count = 0

        async def _sleep_then_cancel(seconds):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise asyncio.CancelledError()

        with (
            patch("asyncio.sleep", side_effect=_sleep_then_cancel),
            patch("src.services.consolidation_state.run_session_organize", daily_mock),
            patch("src.services.consolidation_state.run_memory_organize", memory_mock),
            patch(
                "src.services.consolidation_scheduler._next_occurrence",
                return_value=datetime.datetime(2026, 6, 14, 3, 0, tzinfo=datetime.timezone.utc),
            ),
        ):
            try:
                await scheduler._loop(container)
            except asyncio.CancelledError:
                pass

        assert memory_mock.call_count == 1
        assert daily_mock.call_count == 1

    @pytest.mark.asyncio
    async def test_memory_failure_does_not_block_daily(self):
        scheduler = ConsolidationScheduler()
        container = MagicMock()
        container.config.agents.timezone = "UTC"

        daily_mock = AsyncMock(return_value={"sessions_processed": 0})
        memory_mock = AsyncMock(side_effect=RuntimeError("LLM down"))

        call_count = 0

        async def _sleep_then_cancel(seconds):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise asyncio.CancelledError()

        with (
            patch("asyncio.sleep", side_effect=_sleep_then_cancel),
            patch("src.services.consolidation_state.run_session_organize", daily_mock),
            patch("src.services.consolidation_state.run_memory_organize", memory_mock),
            patch(
                "src.services.consolidation_scheduler._next_occurrence",
                return_value=datetime.datetime(2026, 6, 14, 3, 0, tzinfo=datetime.timezone.utc),
            ),
        ):
            try:
                await scheduler._loop(container)
            except asyncio.CancelledError:
                pass

        memory_mock.assert_called_once()
        daily_mock.assert_called_once_with(container)
