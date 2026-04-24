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
