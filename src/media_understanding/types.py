from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel


class MediaCapability(str, Enum):
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class MediaResult(BaseModel):
    """Standardized result from media understanding."""

    capability: MediaCapability
    text: str
    provider: str = ""
    model: str = ""
    mime_type: str = ""
    error: Optional[str] = None


def _media_error(capability: MediaCapability, client, mime_type: str, error: str) -> MediaResult:
    """Construct an error MediaResult."""
    return MediaResult(
        capability=capability, text="", provider=client.provider,
        model=client.model, mime_type=mime_type, error=error,
    )


def _media_ok(capability: MediaCapability, text: str, client, mime_type: str, model: str = "") -> MediaResult:
    """Construct a success MediaResult."""
    return MediaResult(
        capability=capability, text=text, provider=client.provider,
        model=model or client.model, mime_type=mime_type,
    )
