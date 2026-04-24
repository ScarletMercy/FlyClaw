from __future__ import annotations

import logging

from .provider import MediaProviderClient
from .types import MediaCapability, MediaResult

logger = logging.getLogger("myclaw.media_understanding.audio")


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
            return MediaResult(
                capability=MediaCapability.AUDIO,
                text="",
                provider=client.provider,
                model=client.model,
                mime_type=mime_type,
                error=f"Audio too large: {len(audio_data)} bytes, limit {max_bytes}",
            )
        result = await client.transcribe_audio(audio_data, mime_type, language, prompt)

        if "error" in result:
            return MediaResult(
                capability=MediaCapability.AUDIO,
                text="",
                provider=client.provider,
                model=client.model,
                mime_type=mime_type,
                error=result["error"],
            )

        text = result.get("text", "").strip()
        if not text:
            return MediaResult(
                capability=MediaCapability.AUDIO,
                text="",
                provider=client.provider,
                model=client.model,
                mime_type=mime_type,
                error="Empty transcription result",
            )

        logger.info("Audio transcribed (%d bytes, %s) -> %d chars", len(audio_data), client.model, len(text))
        return MediaResult(
            capability=MediaCapability.AUDIO,
            text=text,
            provider=client.provider,
            model=result.get("model", client.model),
            mime_type=mime_type,
        )
    except Exception as e:
        logger.error("Audio transcription failed: %s", e)
        return MediaResult(
            capability=MediaCapability.AUDIO,
            text="",
            provider=client.provider,
            model=client.model,
            mime_type=mime_type,
            error=str(e),
        )
