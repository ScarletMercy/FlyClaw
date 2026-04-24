"""Auto-index watcher for memory files. Watches configured directories and
re-indexes changed files into the memory store."""
import asyncio
import logging
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("myclaw.memory.watcher")

_watcher_task: Optional[asyncio.Task] = None

async def start_memory_watcher(
    paths: list[str],
    on_change: Callable[[str, str], None],
) -> None:
    """Start watching memory directories for file changes.

    Args:
        paths: List of directory paths to watch
        on_change: Callback(path, content) called when a file changes
    """
    global _watcher_task
    import watchfiles

    abs_paths = [Path(p).expanduser().resolve() for p in paths if Path(p).expanduser().exists()]
    if not abs_paths:
        logger.warning("No valid memory watch paths")
        return

    async def _watch():
        logger.info("Memory watcher started, watching %d paths", len(abs_paths))
        try:
            async for changes in watchfiles.awatch(*abs_paths, stop_event=asyncio.Event()):
                for change_type, path_str in changes:
                    p = Path(path_str)
                    if p.is_file() and p.suffix in ('.md', '.txt', '.rst', '.py', '.js', '.ts', '.json', '.yaml', '.yml'):
                        try:
                            content = p.read_text(encoding='utf-8')
                            on_change(str(p), content)
                            logger.info("Memory re-indexed: %s (%s)", p.name, change_type.name)
                        except Exception as e:
                            logger.warning("Failed to re-index %s: %s", p, e)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Memory watcher error: %s", e)

    _watcher_task = asyncio.create_task(_watch())

async def stop_memory_watcher() -> None:
    global _watcher_task
    if _watcher_task and not _watcher_task.done():
        _watcher_task.cancel()
        try:
            await _watcher_task
        except asyncio.CancelledError:
            pass
    _watcher_task = None
