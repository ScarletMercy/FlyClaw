"""Media tag delivery: parse <media>path</media> tags from AI replies and send via channel."""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger("flyclaw.media_delivery")

_AUDIO_EXTS = {".ogg", ".opus", ".mp3", ".wav", ".m4a", ".silk", ".amr", ".aac", ".flac"}
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp"}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".svg"}

# Matches <media>path</media>, <qqvoice>path</qqvoice>, etc.
_MEDIA_TAG_RE = re.compile(
    r"<(media|qqmedia|qqvoice|qqaudio|voice|audio|qqvideo|video|qqimg|qqimage|img|image|qqfile|qqdoc|file|document)>"
    r"\s*(.+?)\s*"
    r"</(?:media|qqmedia|qqvoice|qqaudio|voice|audio|qqvideo|video|qqimg|qqimage|img|image|qqfile|qqdoc|file|document)>",
    re.IGNORECASE,
)

# For cleaning after delivery
_MEDIA_CLEAN_RE = re.compile(
    r"</?(?:media|qqmedia|qqvoice|qqaudio|voice|audio|qqvideo|video|qqimg|qqimage|img|image|qqfile|qqdoc|file|document)>",
    re.IGNORECASE,
)


def parse_media_tags(text: str) -> list[tuple[str, str, str]]:
    """Extract (tag_name, path, ext) from media tags."""
    results = []
    seen = set()
    for match in _MEDIA_TAG_RE.finditer(text):
        tag = match.group(1).lower()
        path = match.group(2).strip()
        # Remove surrounding quotes
        if (path.startswith('"') and path.endswith('"')) or (path.startswith("'") and path.endswith("'")):
            path = path[1:-1]
        if path in seen:
            continue
        seen.add(path)
        ext = Path(path).suffix.lower()
        results.append((tag, path, ext))
    return results


def strip_media_tags(text: str) -> str:
    """Remove all media tags from text."""
    return _MEDIA_CLEAN_RE.sub("", text).strip()


def _resolve_type(tag: str, ext: str) -> str:
    """Determine media type from tag name and extension."""
    if tag in ("qqvoice", "qqaudio", "voice", "audio"):
        return "audio"
    if tag in ("qqvideo", "video"):
        return "video"
    if tag in ("qqimg", "qqimage", "img", "image"):
        return "image"
    if tag in ("qqfile", "qqdoc", "file", "document"):
        return "file"
    # Generic <media> — auto-detect by extension
    if ext in _AUDIO_EXTS:
        return "audio"
    if ext in _VIDEO_EXTS:
        return "video"
    if ext in _IMAGE_EXTS:
        return "image"
    return "file"


async def deliver_media(text: str, chat_id: str, channel_prefix: str, channel) -> str:
    """Extract media tags, send files via channel, return cleaned text."""
    media_list = parse_media_tags(text)
    if not media_list:
        return text

    for tag, file_path, ext in media_list:
        p = Path(file_path).resolve()
        # Path traversal protection: reject paths outside workspace
        from src.tools.file_tools import _resolve_path, _BASE_DIR

        try:
            _resolve_path(file_path)
        except ValueError:
            logger.warning("Media path outside workspace rejected: %s", file_path)
            continue
        if not p.exists() or not p.is_file():
            logger.warning("Media file not found: %s", file_path)
            continue

        media_type = _resolve_type(tag, ext)
        try:
            if channel_prefix == "qq":
                if media_type == "audio":
                    data = p.read_bytes()
                    await channel.send_media(chat_id, str(p), media_type, file_bytes=data, file_name=p.name)
                else:
                    await channel.send_media(chat_id, str(p), media_type)
        except Exception as e:
            logger.error("Media delivery failed for %s: %s", file_path, e)

    return strip_media_tags(text)
