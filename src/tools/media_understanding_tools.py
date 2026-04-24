from __future__ import annotations

import base64
import ipaddress
import logging
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


@tool
async def describe_image(image_url: str) -> str:
    """Describe/analyze an image. Provide either a URL or a data:image base64 URL.

    Args:
        image_url: URL of the image, or a data:image/...;base64,... data URL.
    """
    runner = _get_runner()
    if runner is None:
        return "[error] Media understanding not enabled. Set tools.media_understanding.enabled: true in config."

    try:
        import httpx

        if image_url.startswith("data:"):
            data, mime_type = _decode_data_url(image_url)
        else:
            _validate_url(image_url)
            async with httpx.AsyncClient(timeout=httpx.Timeout(30)) as client:
                resp = await client.get(image_url, follow_redirects=True)
                if resp.status_code != 200:
                    return f"[error] Failed to download image: HTTP {resp.status_code}"
                data = resp.content
                mime_type = _strip_mime_params(resp.headers.get("content-type", "image/png"))

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
    """Transcribe audio/speech to text. Provide either a URL or a data:audio base64 URL.

    Args:
        audio_url: URL of the audio file, or a data:audio/...;base64,... data URL.
    """
    runner = _get_runner()
    if runner is None:
        return "[error] Media understanding not enabled. Set tools.media_understanding.enabled: true in config."

    try:
        import httpx

        if audio_url.startswith("data:"):
            data, mime_type = _decode_data_url(audio_url)
        else:
            _validate_url(audio_url)
            async with httpx.AsyncClient(timeout=httpx.Timeout(30)) as client:
                resp = await client.get(audio_url, follow_redirects=True)
                if resp.status_code != 200:
                    return f"[error] Failed to download audio: HTTP {resp.status_code}"
                data = resp.content
                mime_type = _strip_mime_params(resp.headers.get("content-type", "audio/wav"))

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
        video_url: URL of the video file, or a data:video/...;base64,... data URL.
    """
    runner = _get_runner()
    if runner is None:
        return "[error] Media understanding not enabled. Set tools.media_understanding.enabled: true in config."

    try:
        import httpx

        if video_url.startswith("data:"):
            data, mime_type = _decode_data_url(video_url)
        else:
            _validate_url(video_url)
            async with httpx.AsyncClient(timeout=httpx.Timeout(60)) as client:
                resp = await client.get(video_url, follow_redirects=True)
                if resp.status_code != 200:
                    return f"[error] Failed to download video: HTTP {resp.status_code}"
                data = resp.content
                mime_type = _strip_mime_params(resp.headers.get("content-type", "video/mp4"))

        from src.media_understanding.types import MediaCapability

        result = await runner.understand(data, MediaCapability.VIDEO, mime_type=mime_type)
        if result.error:
            return f"[error] {result.error}"
        return f"[video description] ({result.model})\n{result.text}"
    except Exception as e:
        logger.error("describe_video error: %s", e)
        return f"[error] {e}"
