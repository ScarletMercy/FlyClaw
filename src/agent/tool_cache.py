"""Tool result persistence — cache large outputs to temp files.

When a tool output exceeds a threshold, it is saved to a temp file
and the message content is truncated with a reference to the file path.
The model can use read_file to retrieve the full content on demand.
"""

from __future__ import annotations

import hashlib
import logging
import tempfile
from pathlib import Path

logger = logging.getLogger("flyclaw.agent.tool_cache")

_DEFAULT_MAX_CHARS = 8000
_DEFAULT_PREVIEW = 8000


def _cache_dir(thread_id: str) -> Path:
    safe_id = thread_id.replace(":", "_").replace("/", "_").replace("\\", "_")
    base = Path(tempfile.gettempdir()) / "flyclaw" / "tool_cache" / safe_id
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
    d = _cache_dir(thread_id)
    if d.exists():
        try:
            for f in d.iterdir():
                f.unlink(missing_ok=True)
            d.rmdir()
        except Exception as e:
            logger.warning("Failed to clear tool cache for %s: %s", thread_id, e)
