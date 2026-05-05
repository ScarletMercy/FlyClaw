"""Video understanding via frame extraction + vision model."""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from .provider import MediaProviderClient
from .types import MediaCapability, MediaResult, _media_error, _media_ok

logger = logging.getLogger("myclaw.media_understanding.video")


async def understand_video(
    client: MediaProviderClient,
    video_data: bytes,
    mime_type: str = "video/mp4",
    prompt: str = "Describe what is happening in this video frame.",
    max_tokens: int = 1024,
    max_bytes: int = 0,
) -> MediaResult:
    try:
        if max_bytes > 0 and len(video_data) > max_bytes:
            return _media_error(MediaCapability.VIDEO, client, mime_type, f"Video too large: {len(video_data)} bytes, limit {max_bytes}")

        frame_data = await _extract_frame(video_data)
        if frame_data is None:
            return _media_error(MediaCapability.VIDEO, client, mime_type, "Failed to extract video frame. Is ffmpeg installed?")

        frames = [frame_data]
        for ts in [2.0, 5.0]:
            extra = await _extract_frame(video_data, timestamp=ts)
            if extra:
                frames.append(extra)

        descriptions = []
        for i, frame in enumerate(frames):
            label = f"Frame at {['1s', '2s', '5s'][i]}" if i < 3 else f"Frame {i + 1}"
            result = await client.describe_video(frame, "image/jpeg", f"{prompt} ({label})", max_tokens)
            if "error" not in result:
                desc = result.get("text", "").strip()
                if desc:
                    descriptions.append(f"[{label}] {desc}")

        if not descriptions:
            return _media_error(MediaCapability.VIDEO, client, mime_type, "Empty response from vision model for all frames")

        text = "\n".join(descriptions)
        logger.info("Video described (%d bytes, %d frames, %s) -> %d chars", len(video_data), len(frames), client.model, len(text))
        return _media_ok(MediaCapability.VIDEO, text, client, mime_type)
    except Exception as e:
        logger.error("Video understanding failed: %s", e)
        return _media_error(MediaCapability.VIDEO, client, mime_type, str(e))


async def _extract_frame(video_data: bytes, timestamp: float = 1.0) -> Optional[bytes]:
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-ss", str(timestamp), "-i", "-",
            "-frames:v", "1", "-f", "image2pipe", "-vcodec", "mjpeg", "-",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(input=video_data), timeout=30)
        if proc.returncode == 0 and stdout:
            return stdout
        return None
    except FileNotFoundError:
        logger.warning("ffmpeg not found, cannot extract video frame")
        return None
    except asyncio.TimeoutError:
        logger.warning("ffmpeg timed out extracting frame")
        try:
            proc.kill()
            await proc.wait()
        except ProcessLookupError:
            pass
        return None
    except Exception as e:
        logger.warning("Frame extraction failed: %s", e)
        return None
