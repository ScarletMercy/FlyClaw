from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

logger = logging.getLogger("myclaw.session")


class SessionTracker:
    def __init__(self, idle_reset_minutes: int = 120):
        self._idle_reset_seconds = idle_reset_minutes * 60
        self._last_activity: dict[str, float] = {}
        self._task: Optional[asyncio.Task] = None

    def touch(self, thread_id: str) -> None:
        self._last_activity[thread_id] = time.monotonic()

    def get_expired_sessions(self) -> list[str]:
        if self._idle_reset_seconds <= 0:
            return []
        now = time.monotonic()
        expired = []
        for tid, last in list(self._last_activity.items()):
            if now - last > self._idle_reset_seconds:
                expired.append(tid)
        return expired

    def remove(self, thread_id: str) -> None:
        self._last_activity.pop(thread_id, None)

    @property
    def active_count(self) -> int:
        return len(self._last_activity)

    def get_sessions(self) -> list[dict]:
        now = time.monotonic()
        return [
            {"thread_id": tid, "last_active": now - last}
            for tid, last in self._last_activity.items()
        ]

    async def start_periodic_cleanup(
        self,
        compiled_graph,
        interval_seconds: int = 60,
    ) -> None:
        if self._task is not None and not self._task.done():
            logger.warning("Periodic cleanup already running")
            return
        async def _loop():
            while True:
                await asyncio.sleep(interval_seconds)
                expired = self.get_expired_sessions()
                if not expired:
                    continue
                for tid in expired:
                    try:
                        config = {"configurable": {"thread_id": tid}}
                        state = await compiled_graph.aget_state(config)
                        if state and state.values:
                            await compiled_graph.aupdate_state(config, {"messages": []})
                            logger.info("Session reset (idle): %s", tid)
                        self.remove(tid)
                    except Exception as e:
                        logger.debug("Session reset failed for %s: %s", tid, e)
                        self.remove(tid)

        self._task = asyncio.create_task(_loop())
        logger.info(
            "Session tracker started (idle_reset=%dm, check_interval=%ds)",
            self._idle_reset_seconds // 60,
            interval_seconds,
        )

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                self._task = None
            else:
                self._task = None
