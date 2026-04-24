"""High-level runner that resolves config and dispatches to the right capability."""
from __future__ import annotations

import logging
from typing import Optional

from .types import MediaCapability, MediaResult
from .provider import MediaProviderClient
from .image import understand_image
from .audio import transcribe_audio
from .video import understand_video

logger = logging.getLogger("myclaw.media_understanding")


def _make_capability_client(mu_config, cap_config, fallback_api_key: str = "") -> MediaProviderClient:
    provider = cap_config.provider or mu_config.provider
    name = cap_config.name or mu_config.name
    base_url = cap_config.base_url or mu_config.base_url
    api_key = cap_config.api_key or mu_config.api_key or fallback_api_key
    timeout = mu_config.timeout_seconds
    return MediaProviderClient(provider=provider, name=name, base_url=base_url, api_key=api_key, timeout=timeout)


class MediaUnderstandingRunner:
    """Orchestrates media understanding across all capabilities."""

    def __init__(self, mu_config, fallback_api_key: str = ""):
        self.config = mu_config
        self._fallback_key = fallback_api_key
        self._image_client: Optional[MediaProviderClient] = None
        self._audio_client: Optional[MediaProviderClient] = None
        self._video_client: Optional[MediaProviderClient] = None

    def _get_image_client(self) -> MediaProviderClient:
        if self._image_client is None:
            self._image_client = _make_capability_client(self.config, self.config.image, self._fallback_key)
        return self._image_client

    def _get_audio_client(self) -> MediaProviderClient:
        if self._audio_client is None:
            self._audio_client = _make_capability_client(self.config, self.config.audio, self._fallback_key)
        return self._audio_client

    def _get_video_client(self) -> MediaProviderClient:
        if self._video_client is None:
            self._video_client = _make_capability_client(self.config, self.config.video, self._fallback_key)
        return self._video_client

    async def understand(
        self,
        data: bytes,
        capability: MediaCapability,
        mime_type: str = "",
    ) -> MediaResult:
        if capability == MediaCapability.IMAGE:
            if not self.config.image.enabled:
                return MediaResult(capability=capability, text="", error="Image understanding disabled")
            return await understand_image(
                self._get_image_client(), data,
                mime_type=mime_type or "image/png",
                max_bytes=self.config.max_image_size,
            )
        elif capability == MediaCapability.AUDIO:
            if not self.config.audio.enabled:
                return MediaResult(capability=capability, text="", error="Audio transcription disabled")
            return await transcribe_audio(
                self._get_audio_client(), data,
                mime_type=mime_type or "audio/wav",
                max_bytes=self.config.max_audio_size,
            )
        elif capability == MediaCapability.VIDEO:
            if not self.config.video.enabled:
                return MediaResult(capability=capability, text="", error="Video understanding disabled")
            return await understand_video(
                self._get_video_client(), data,
                mime_type=mime_type or "video/mp4",
                max_bytes=self.config.max_video_size,
            )
        return MediaResult(capability=MediaCapability.IMAGE, text="", error=f"Unknown capability: {capability}")

    @staticmethod
    def guess_capability_from_mime(mime_type: str) -> Optional[MediaCapability]:
        if mime_type.startswith("image/"):
            return MediaCapability.IMAGE
        if mime_type.startswith("audio/"):
            return MediaCapability.AUDIO
        if mime_type.startswith("video/"):
            return MediaCapability.VIDEO
        return None

    @staticmethod
    def guess_capability_from_ext(filename: str) -> Optional[MediaCapability]:
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        image_exts = {"png", "jpg", "jpeg", "gif", "bmp", "webp", "svg", "tiff", "ico"}
        audio_exts = {"wav", "mp3", "ogg", "flac", "aac", "m4a", "wma", "opus"}
        video_exts = {"mp4", "avi", "mkv", "mov", "wmv", "flv", "webm", "m4v"}
        if ext in image_exts:
            return MediaCapability.IMAGE
        if ext in audio_exts:
            return MediaCapability.AUDIO
        if ext in video_exts:
            return MediaCapability.VIDEO
        return None
