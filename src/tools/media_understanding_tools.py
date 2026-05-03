from __future__ import annotations

import base64
import ipaddress
import logging
import mimetypes
from pathlib import Path
from urllib.parse import urlparse

from langchain_core.tools import tool

logger = logging.getLogger("myclaw.tools.media_understanding")

_runner = None


def _validate_url(url: str) -> None:
    """Validate URL to prevent SSRF attacks."""
    parsed = urlparse(url)
    if parsed.scheme not in ("https", "http"):
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("Missing hostname in URL")
    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        return  # hostname is a domain, not an IP
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
        raise ValueError("URL points to a private or reserved IP address")


def _get_runner():
    global _runner
    if _runner is not None:
        return _runner
    try:
        from src.config import load_config
        from src.media_understanding.runner import MediaUnderstandingRunner

        cfg = load_config()
        if not cfg.tools.media_understanding.enabled:
            return None
        fallback_key = cfg.model.api_key or ""
        _runner = MediaUnderstandingRunner(cfg.tools.media_understanding, fallback_key)
        return _runner
    except Exception as e:
        logger.warning("Failed to init media understanding runner: %s", e)
        return None


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

    # Remote URL
    _validate_url(source)
    import httpx

    async with httpx.AsyncClient(timeout=httpx.Timeout(30)) as client:
        resp = await client.get(source, follow_redirects=True)
        if resp.status_code != 200:
            raise ValueError(f"Failed to download: HTTP {resp.status_code}")
        mime_type = _strip_mime_params(resp.headers.get("content-type", default_mime))
        return resp.content, mime_type


@tool
async def describe_image(image_url: str) -> str:
    """Describe/analyze an image. Provide a URL, a data:image base64 URL, or a local file path.

    Args:
        image_url: URL of the image, a data:image/...;base64,... data URL, or a local file path.
    """
    runner = _get_runner()
    if runner is None:
        return "[error] Media understanding not enabled. Set tools.media_understanding.enabled: true in config."

    try:
        data, mime_type = await _resolve_media_input(image_url, "image/png")

        from src.media_understanding.types import MediaCapability

        result = await runner.understand(data, MediaCapability.IMAGE, mime_type=mime_type)
        if result.error:
            return f"[error] {result.error}"
        return f"[image description] ({result.model})\n{result.text}"
    except Exception as e:
        logger.error("describe_image error: %s", e)
        return f"[error] {e}"


@tool
async def transcribe_audio(audio_url: str) -> str:
    """Transcribe audio/speech to text. Provide a URL, a data:audio base64 URL, or a local file path.

    Args:
        audio_url: URL of the audio file, a data:audio/...;base64,... data URL, or a local file path.
    """
    runner = _get_runner()
    if runner is None:
        return "[error] Media understanding not enabled. Set tools.media_understanding.enabled: true in config."

    try:
        data, mime_type = await _resolve_media_input(audio_url, "audio/wav")

        from src.media_understanding.types import MediaCapability

        result = await runner.understand(data, MediaCapability.AUDIO, mime_type=mime_type)
        if result.error:
            return f"[error] {result.error}"
        return f"[audio transcription] ({result.model})\n{result.text}"
    except Exception as e:
        logger.error("transcribe_audio error: %s", e)
        return f"[error] {e}"


@tool
async def describe_video(video_url: str) -> str:
    """Describe/analyze a video (extracts a frame and describes it).

    Args:
        video_url: URL of the video file, a data:video/...;base64,... data URL, or a local file path.
    """
    runner = _get_runner()
    if runner is None:
        return "[error] Media understanding not enabled. Set tools.media_understanding.enabled: true in config."

    try:
        data, mime_type = await _resolve_media_input(video_url, "video/mp4")

        from src.media_understanding.types import MediaCapability

        result = await runner.understand(data, MediaCapability.VIDEO, mime_type=mime_type)
        if result.error:
            return f"[error] {result.error}"
        return f"[video description] ({result.model})\n{result.text}"
    except Exception as e:
        logger.error("describe_video error: %s", e)
        return f"[error] {e}"
