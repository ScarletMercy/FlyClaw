from __future__ import annotations

import io
import logging

from .provider import MediaProviderClient
from .types import MediaCapability, MediaResult

logger = logging.getLogger("myclaw.media_understanding.image")


async def understand_image(
    client: MediaProviderClient,
    image_data: bytes,
    mime_type: str = "image/png",
    prompt: str = "Describe this image in detail. If it contains text, transcribe the text.",
    max_tokens: int = 1024,
    max_bytes: int = 0,
) -> MediaResult:
    try:
        resize_threshold = 4 * 1024 * 1024  # always try to resize if > 4MB
        processed_data, processed_mime = _preprocess_image(image_data, mime_type, max_bytes=resize_threshold)
        if max_bytes > 0 and len(processed_data) > max_bytes:
            return MediaResult(
                capability=MediaCapability.IMAGE,
                text="",
                provider=client.provider,
                model=client.model,
                mime_type=mime_type,
                error=f"Image too large: {len(processed_data)} bytes, limit {max_bytes}",
            )

        result = await client.describe_image(
            processed_data, processed_mime, prompt, max_tokens
        )

        if "error" in result:
            return MediaResult(
                capability=MediaCapability.IMAGE,
                text="",
                provider=client.provider,
                model=client.model,
                mime_type=mime_type,
                error=result["error"],
            )

        text = result.get("text", "").strip()
        if not text:
            return MediaResult(
                capability=MediaCapability.IMAGE,
                text="",
                provider=client.provider,
                model=client.model,
                mime_type=mime_type,
                error="Empty response from vision model",
            )

        logger.info("Image described (%d bytes, %s) -> %d chars", len(image_data), client.model, len(text))
        return MediaResult(
            capability=MediaCapability.IMAGE,
            text=text,
            provider=client.provider,
            model=result.get("model", client.model),
            mime_type=mime_type,
        )
    except Exception as e:
        logger.error("Image understanding failed: %s", e)
        return MediaResult(
            capability=MediaCapability.IMAGE,
            text="",
            provider=client.provider,
            model=client.model,
            mime_type=mime_type,
            error=str(e),
        )


def _preprocess_image(data: bytes, mime_type: str, max_bytes: int = 4 * 1024 * 1024) -> tuple[bytes, str]:
    if len(data) <= max_bytes:
        return data, mime_type

    try:
        from PIL import Image

        img = Image.open(io.BytesIO(data))
        img.thumbnail((1600, 1600), Image.LANCZOS)

        output = io.BytesIO()
        if img.mode == "RGBA":
            img.save(output, format="PNG", optimize=True)
            processed_mime = "image/png"
        else:
            rgb = img.convert("RGB")
            rgb.save(output, format="JPEG", quality=85)
            processed_mime = "image/jpeg"

        processed = output.getvalue()
        if len(processed) < len(data):
            logger.debug("Image resized: %d -> %d bytes", len(data), len(processed))
            return processed, processed_mime
    except ImportError:
        logger.warning("Pillow not installed, cannot resize large image")
    except Exception as e:
        logger.warning("Image resize failed: %s", e)

    return data, mime_type
