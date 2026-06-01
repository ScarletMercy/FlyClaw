"""Tool result persistence — cache large outputs to temp files.

When a tool output exceeds a threshold, it is saved to a temp file
and the message content is truncated with a reference to the file path.
The model can use read_file to retrieve the full content on demand.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path
from typing import Set

logger = logging.getLogger("flyclaw.agent.tool_cache")

_DEFAULT_MAX_CHARS = 8000
_DEFAULT_PREVIEW = 8000

# NOTE: strip_cache_path / _CACHE_PATH_RE moved to compressor.compressor
# (the only caller) to avoid compressor → agent coupling.


def cache_root() -> Path:
    return (Path.home() / ".flyclaw" / "temp").resolve()


def _cache_dir_path(thread_id: str) -> Path:
    """Return the cache directory path for *thread_id* without creating it."""
    safe_id = thread_id.replace(":", "_").replace("/", "_").replace("\\", "_")
    return cache_root() / "tool_cache" / safe_id


def _cache_dir(thread_id: str) -> Path:
    """Return (and create) the cache directory for *thread_id*."""
    base = _cache_dir_path(thread_id)
    base.mkdir(parents=True, exist_ok=True)
    return base


def cache_large_output(
    content: str,
    thread_id: str,
    max_chars: int = _DEFAULT_MAX_CHARS,
    preview: int = _DEFAULT_PREVIEW,
) -> tuple[str, str | None]:
    """Truncate large content and save to a temp file.

    Uses a content-hash based filename so the same content always produces
    the same truncated text — essential for KV cache prefix stability.

    Returns:
        (truncated_content, file_path_or_None)
    """
    if not isinstance(content, str) or len(content) <= max_chars:
        return content, None

    d = _cache_dir(thread_id)
    content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
    filename = f"{content_hash}.txt"
    filepath = d / filename

    try:
        filepath.write_text(content, encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to cache tool output: %s", e)
        return content[:preview] + f"\n... [truncated, {len(content)} chars total]", None

    truncated = (
        content[:preview]
        + f"\n... [content truncated, {len(content)} chars total. "
        + f"Full content saved to: `{filepath}`]"
    )
    return truncated, str(filepath)


def clear_thread_cache(thread_id: str) -> None:
    """Remove all cached files for a thread."""
    d = _cache_dir_path(thread_id)
    if d.exists():
        try:
            for f in d.iterdir():
                f.unlink(missing_ok=True)
            d.rmdir()
        except Exception as e:
            logger.warning("Failed to clear tool cache for %s: %s", thread_id, e)


def clear_orphaned_caches(live_thread_ids: Set[str]) -> int:
    """Remove cache directories whose thread no longer exists.

    Scans the tool_cache root and deletes any subdirectory whose safe_id
    does not correspond to a thread in *live_thread_ids*.

    Args:
        live_thread_ids: Set of thread IDs that are still alive.

    Returns:
        Number of orphaned directories removed.
    """
    root = cache_root() / "tool_cache"
    if not root.exists():
        return 0

    # Build the set of directory names that should be kept
    keep = {_cache_dir_path(tid).name for tid in live_thread_ids}

    removed = 0
    for entry in root.iterdir():
        if entry.is_dir() and entry.name not in keep:
            try:
                shutil.rmtree(entry, ignore_errors=True)
                removed += 1
                logger.debug("Removed orphaned tool cache: %s", entry.name)
            except Exception as e:
                logger.warning("Failed to remove orphaned cache %s: %s", entry.name, e)

    if removed:
        logger.info("Cleared %d orphaned tool cache directories", removed)
    return removed


def clear_all_caches() -> int:
    """Remove the entire tool_cache tree (e.g. at startup or shutdown).

    Returns:
        Number of top-level thread directories removed.
    """
    root = cache_root() / "tool_cache"
    if not root.exists():
        return 0

    count = sum(1 for _ in root.iterdir() if _.is_dir())
    try:
        shutil.rmtree(root, ignore_errors=True)
        logger.info("Cleared all tool caches (%d directories)", count)
    except Exception as e:
        logger.warning("Failed to clear all tool caches: %s", e)
    return count
