from __future__ import annotations

import asyncio
import logging
from pathlib import Path

logger = logging.getLogger("myclaw.canvas.live_reload")

_task: asyncio.Task | None = None


async def start_canvas_watcher(root: Path):
    global _task
    try:
        from watchfiles import awatch
    except ImportError:
        logger.debug("watchfiles not installed, canvas live-reload disabled")
        return

    root.mkdir(parents=True, exist_ok=True)

    async def _watch():
        from src.canvas.server import broadcast_reload
        async for _changes in awatch(root):
            await broadcast_reload()

    _task = asyncio.create_task(_watch())
    logger.info("Canvas live-reload watcher started for %s", root)


async def stop_canvas_watcher():
    global _task
    if _task:
        _task.cancel()
        try:
            await _task
        except asyncio.CancelledError:
            pass
