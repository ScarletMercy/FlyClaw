"""Tests for consolidation_scheduler Sunday KV→archive migration trigger.

锁住:周日额外跑 migrate_kv_to_archive,非周日不跑。
沿用 test_consolidation_scheduler.py 的 patch 模式(AsyncMock + context manager)。
"""

from __future__ import annotations

import asyncio
import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.consolidation_scheduler import ConsolidationScheduler


class TestSchedulerMigration:
    @pytest.mark.asyncio
    async def test_sunday_triggers_migration(self):
        scheduler = ConsolidationScheduler()
        container = MagicMock()
        container.config.agents.timezone = "UTC"

        daily_mock = AsyncMock(return_value={"sessions_processed": 0})
        memory_mock = AsyncMock(return_value={"total_memories": 0})
        migrate_mock = AsyncMock(return_value={"migrated": 0})

        call_count = 0

        async def _sleep_then_cancel(seconds):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise asyncio.CancelledError()

        # 2026-06-14 是周日
        with (
            patch("asyncio.sleep", side_effect=_sleep_then_cancel),
            patch("src.services.daily_consolidation.run_daily_consolidation", daily_mock),
            patch("src.services.memory_consolidation.run_memory_consolidation", memory_mock),
            patch("src.services.memory_archive_migration.migrate_kv_to_archive", migrate_mock),
            patch(
                "src.services.consolidation_scheduler._next_occurrence",
                return_value=datetime.datetime(2026, 6, 14, 3, 0, tzinfo=datetime.timezone.utc),
            ),
        ):
            try:
                await scheduler._loop(container)
            except asyncio.CancelledError:
                pass

        daily_mock.assert_called_once_with(container)
        memory_mock.assert_called_once_with(container)
        migrate_mock.assert_called_once_with(container)  # 周日触发迁移

    @pytest.mark.asyncio
    async def test_non_sunday_skips_migration(self):
        scheduler = ConsolidationScheduler()
        container = MagicMock()
        container.config.agents.timezone = "UTC"

        daily_mock = AsyncMock(return_value={"sessions_processed": 0})
        memory_mock = AsyncMock(return_value={"total_memories": 0})
        migrate_mock = AsyncMock(return_value={"migrated": 0})

        call_count = 0

        async def _sleep_then_cancel(seconds):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise asyncio.CancelledError()

        # 2026-06-15 是周一
        with (
            patch("asyncio.sleep", side_effect=_sleep_then_cancel),
            patch("src.services.daily_consolidation.run_daily_consolidation", daily_mock),
            patch("src.services.memory_consolidation.run_memory_consolidation", memory_mock),
            patch("src.services.memory_archive_migration.migrate_kv_to_archive", migrate_mock),
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
        memory_mock.assert_not_called()  # 非周日不跑
        migrate_mock.assert_not_called()  # 非周日不迁移
