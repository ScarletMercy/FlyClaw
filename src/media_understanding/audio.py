from __future__ import annotations

import logging

from .provider import MediaProviderClient
from .types import MediaCapability, MediaResult, _media_error, _media_ok

logger = logging.getLogger("flyclaw.media_understanding.audio")


async def transcribe_audio(
    client: MediaProviderClient,
    audio_data: bytes,
    mime_type: str = "",
    language: str = "",
    prompt: str = "",
    max_bytes: int = 0,
) -> MediaResult:
    try:
        if max_bytes > 0 and len(audio_data) > max_bytes:
            return _media_error(MediaCapability.AUDIO, client, mime_type, f"Audio too large: {len(audio_data)} bytes, limit {max_bytes}")
        result = await client.transcribe_audio(audio_data, mime_type, language, prompt)

        if "error" in result:
            return _media_error(MediaCapability.AUDIO, client, mime_type, result["error"])

        text = result.get("text", "").strip()
        if not text:
            return _media_error(MediaCapability.AUDIO, client, mime_type, "Empty transcription result")

        logger.info("Audio transcribed (%d bytes, %s) -> %d chars", len(audio_data), client.model, len(text))
        return _media_ok(MediaCapability.AUDIO, text, client, mime_type, model=result.get("model", client.model))
    except Exception as e:
        logger.error("Audio transcription failed: %s", e)
        return _media_error(MediaCapability.AUDIO, client, mime_type, str(e))
