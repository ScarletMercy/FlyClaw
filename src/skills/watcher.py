from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Callable

from .loader import discover_skills
from .types import Skill

logger = logging.getLogger("myclaw.skills.watcher")

_watch_task = None


async def start_skills_watcher(
    directories: list[tuple[str, Path]],
    on_change: Callable[[list[Skill]], None],
):
    global _watch_task
    try:
        from watchfiles import awatch
    except ImportError:
        logger.debug("watchfiles not installed, skill auto-reload disabled")
        return

    dirs = [d for _, d in directories if d.exists()]
    if not dirs:
        return

    async def _watch():
        async for _changes in awatch(*dirs, watch_filter=lambda _, p: "SKILL.md" in str(p)):
            logger.info("Skills changed, reloading...")
            try:
                skills = discover_skills(directories)
                on_change(skills)
            except Exception as e:
                logger.error("Skill reload failed: %s", e)

    _watch_task = asyncio.create_task(_watch())
    logger.info("Skill watcher started for %d directories", len(dirs))


def stop_skills_watcher():
    global _watch_task
    if _watch_task:
        _watch_task.cancel()
        _watch_task = None
