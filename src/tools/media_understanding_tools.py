from __future__ import annotations

import base64
import logging
import mimetypes
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger("flyclaw.tools.media_understanding")


def _get_runner():
    from src._container import get_container

    return get_container().media_understanding_runner


def _decode_data_url(data_url: str) -> tuple[bytes, str]:
    if not data_url.startswith("data:"):
        raise ValueError("Invalid data URL: must start with 'data:'")
    if "," not in data_url:
        raise ValueError("Invalid data URL: missing comma separator")
    header, b64 = data_url.split(",", 1)
    if ":" not in header:
        raise ValueError("Invalid data URL: missing MIME type")
    mime_part = header.split(":", 1)[1].split(";")[0]
    return base64.b64decode(b64), mime_part


def _strip_mime_params(content_type: str) -> str:
    return content_type.split(";")[0].strip() if content_type else ""


async def _resolve_media_input(source: str, default_mime: str) -> tuple[bytes, str]:
    """Resolve a media source (data URL, local file path, or HTTP URL) to (bytes, mime_type)."""
    if source.startswith("data:"):
        return _decode_data_url(source)

    # Local file path (no scheme, or Windows drive like D:\...)
    parsed = urlparse(source)
    if not parsed.scheme or (len(parsed.scheme) == 1 and parsed.scheme.isalpha()):
        path = Path(source).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {source}")
        if not path.is_file():
            raise ValueError(f"Not a regular file: {source}")
        data = path.read_bytes()
        mime, _ = mimetypes.guess_type(str(path))
        return data, mime or default_mime

    # Remote URL — use DNS-pinned safe_fetch for SSRF protection
    from src.security.url_safety import safe_fetch

    resp = await safe_fetch(source, timeout=30.0)
    if resp.status_code != 200:
        raise ValueError(f"Failed to download: HTTP {resp.status_code}")
    mime_type = _strip_mime_params(resp.headers.get("content-type", default_mime))
    return resp.content, mime_type


async def describe_media(media_url: str) -> str:
    """Describe/analyze an image or video. Provide a URL, a data: base64 URL, or a local file path.

    Args:
        media_url: URL of the image or video, a data:...;base64,... data URL, or a local file path.
    """
    runner = _get_runner()
    if runner is None:
        return "[error] Media understanding not enabled. Set tools.media_understanding.enabled: true in config."

    try:
        data, mime_type = await _resolve_media_input(media_url, "image/png")

        from src.media_understanding.types import MediaCapability
        from src.media_understanding.runner import MediaUnderstandingRunner

        capability = MediaUnderstandingRunner.guess_capability_from_mime(mime_type) or MediaCapability.IMAGE
        if capability == MediaCapability.AUDIO:
            capability = MediaCapability.IMAGE

        result = await runner.understand(data, capability, mime_type=mime_type)
        if result.error:
            return f"[error] {result.error}"
        label = "video" if capability == MediaCapability.VIDEO else "image"
        return f"[{label} description] ({result.model})\n{result.text}"
    except Exception as e:
        logger.error("describe_media error: %s", e)
        return f"[error] {e}"


def get_tools() -> list:
    from src.agent.tooldef import ToolDef
    from src._container import get_container

    container = get_container()
    mu_config = container.config.tools.media_understanding
    if not mu_config.enabled:
        return []

    model_name = mu_config.name or mu_config.image.name
    if not model_name:
        return []

    return [
        ToolDef.from_function(describe_media),
    ]
