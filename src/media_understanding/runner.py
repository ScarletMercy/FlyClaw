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
        self._fallback_clients: Optional[list[MediaProviderClient]] = None

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

    def _get_fallback_clients(self) -> list[MediaProviderClient]:
        if self._fallback_clients is None:
            self._fallback_clients = []
            for fb in self.config.fallbacks:
                self._fallback_clients.append(MediaProviderClient(
                    provider=fb.provider or self.config.provider,
                    name=fb.name,
                    base_url=fb.base_url,
                    api_key=fb.api_key or self.config.api_key or self._fallback_key,
                    timeout=self.config.timeout_seconds,
                ))
        return self._fallback_clients

    def _get_client_for_capability(self, capability: MediaCapability) -> MediaProviderClient:
        if capability == MediaCapability.IMAGE:
            return self._get_image_client()
        elif capability == MediaCapability.AUDIO:
            return self._get_audio_client()
        elif capability == MediaCapability.VIDEO:
            return self._get_video_client()
        return self._get_image_client()

    async def _run_capability(
        self,
        client: MediaProviderClient,
        capability: MediaCapability,
        data: bytes,
        mime_type: str,
        max_bytes: int,
    ) -> MediaResult:
        if capability == MediaCapability.IMAGE:
            return await understand_image(client, data, mime_type=mime_type, max_bytes=max_bytes)
        elif capability == MediaCapability.AUDIO:
            return await transcribe_audio(client, data, mime_type=mime_type, max_bytes=max_bytes)
        elif capability == MediaCapability.VIDEO:
            return await understand_video(client, data, mime_type=mime_type, max_bytes=max_bytes)
        return MediaResult(capability=capability, text="", error=f"Unknown capability: {capability}")

    async def understand(
        self,
        data: bytes,
        capability: MediaCapability,
        mime_type: str = "",
    ) -> MediaResult:
        if capability == MediaCapability.IMAGE and not self.config.image.enabled:
            return MediaResult(capability=capability, text="", error="Image understanding disabled")
        if capability == MediaCapability.AUDIO and not self.config.audio.enabled:
            return MediaResult(capability=capability, text="", error="Audio transcription disabled")
        if capability == MediaCapability.VIDEO and not self.config.video.enabled:
            return MediaResult(capability=capability, text="", error="Video understanding disabled")

        max_bytes = {
            MediaCapability.IMAGE: self.config.max_image_size,
            MediaCapability.AUDIO: self.config.max_audio_size,
            MediaCapability.VIDEO: self.config.max_video_size,
        }.get(capability, 0)

        default_mime = {
            MediaCapability.IMAGE: "image/png",
            MediaCapability.AUDIO: "audio/wav",
            MediaCapability.VIDEO: "video/mp4",
        }.get(capability, "")

        primary = self._get_client_for_capability(capability)
        all_clients = [primary] + self._get_fallback_clients()
        last_result: Optional[MediaResult] = None

        for i, client in enumerate(all_clients):
            try:
                result = await self._run_capability(client, capability, data, mime_type or default_mime, max_bytes)
                if not result.error:
                    if i > 0:
                        logger.info("Fallback to %s succeeded for %s", client.model, capability.value)
                    return result
                last_result = result
                if i < len(all_clients) - 1:
                    logger.warning(
                        "Media understanding failed with %s (%s), trying fallback: %s",
                        client.model, capability.value, result.error,
                    )
            except Exception as e:
                last_result = MediaResult(capability=capability, text="", error=str(e))
                if i < len(all_clients) - 1:
                    logger.warning(
                        "Media understanding raised with %s (%s), trying fallback: %s",
                        client.model, capability.value, e,
                    )

        return last_result or MediaResult(capability=capability, text="", error="No clients available")

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
