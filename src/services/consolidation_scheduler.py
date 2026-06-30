"""Built-in consolidation scheduler (no cron dependency).

Runs as an asyncio background task, waking at 03:00 every day in the
configured timezone.  On Sundays it runs memory consolidation first,
then daily (session) consolidation.  On other days only daily.

Lifecycle:
  - start() called from app.on_startup()
  - stop()  called from app.on_shutdown()
"""

from __future__ import annotations

import asyncio
import datetime
import logging
from typing import Any

from src.utils.tz import get_tz

logger = logging.getLogger("flyclaw.consolidation.scheduler")

_TRIGGER_HOUR = 3
_TRIGGER_MINUTE = 0


class ConsolidationScheduler:
    def __init__(self) -> None:
        self._task: asyncio.Task | None = None

    async def start(self, container: Any) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._loop(container))
        logger.info("Consolidation scheduler started (daily %02d:%02d)", _TRIGGER_HOUR, _TRIGGER_MINUTE)

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
        logger.info("Consolidation scheduler stopped")

    async def _loop(self, container: Any) -> None:
        while True:
            tz_name = getattr(container.config.agents, "timezone", "local")
            tz = get_tz(tz_name)

            now = datetime.datetime.now(tz)
            next_run = _next_occurrence(now, _TRIGGER_HOUR, _TRIGGER_MINUTE)
            delay = (next_run - now).total_seconds()

            logger.info("Next consolidation at %s (in %.0f min)", next_run.isoformat(), delay / 60)
            await asyncio.sleep(delay)

            is_sunday = next_run.weekday() == 6

            if is_sunday:
                logger.info("Sunday consolidation: running memory cleanup first")
                try:
                    from src.services.memory_consolidation import run_memory_consolidation

                    await run_memory_consolidation(container)
                except Exception as e:
                    logger.error("Sunday memory consolidation failed: %s", e, exc_info=True)

            try:
                from src.services.daily_consolidation import run_daily_consolidation

                await run_daily_consolidation(container)
            except Exception as e:
                logger.error("Daily consolidation failed: %s", e, exc_info=True)


def _next_occurrence(now: datetime.datetime, hour: int, minute: int) -> datetime.datetime:
    candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= now:
        candidate += datetime.timedelta(days=1)
    return candidate
