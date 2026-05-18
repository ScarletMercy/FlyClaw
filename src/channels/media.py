from __future__ import annotations

import base64
import logging

import httpx

from typing import Optional

logger = logging.getLogger("myclaw.media")

_MAX_DOWNLOAD_SIZE = 50 * 1024 * 1024


async def download_from_url(url: str, timeout: int = 30) -> Optional[tuple[bytes, str]]:
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout), follow_redirects=True) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                logger.error("Download URL failed: HTTP %d", resp.status_code)
                return None
            content_type = resp.headers.get("content-type", "application/octet-stream")
            content_length = int(resp.headers.get("content-length", 0))
            if content_length > _MAX_DOWNLOAD_SIZE:
                logger.error(
                    "Download URL failed: content-length %d exceeds limit %d", content_length, _MAX_DOWNLOAD_SIZE
                )
                return None
            chunks = []
            total = 0
            async for chunk in resp.aiter_bytes(8192):
                total += len(chunk)
                if total > _MAX_DOWNLOAD_SIZE:
                    logger.error("Download URL failed: streamed size %d exceeds limit %d", total, _MAX_DOWNLOAD_SIZE)
                    return None
                chunks.append(chunk)
            return b"".join(chunks), content_type
    except Exception as e:
        logger.error("Download URL error: %s", e)
        return None


def image_to_base64_url(data: bytes, content_type: str = "image/png") -> str:
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{content_type};base64,{b64}"
