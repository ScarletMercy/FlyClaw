from __future__ import annotations

import asyncio
import base64
import io
import logging
import mimetypes
import os
import tempfile
from pathlib import Path
from typing import Optional

import httpx
import lark_oapi as lark
from lark_oapi.api.im.v1 import (
    CreateFileRequestBody,
    CreateImageRequestBody,
    CreateMessageRequestBody,
    CreateMessageRequest,
)

logger = logging.getLogger("myclaw.media")

_MAX_FILE_SIZE = 30 * 1024 * 1024


async def download_message_resource(
    client: lark.Client,
    message_id: str,
    file_key: str,
) -> Optional[bytes]:
    try:
        resp = await asyncio.to_thread(
            client.im.messageResource.get,
            path_params={"message_id": message_id, "file_key": file_key},
        )
        if not resp.success():
            logger.error("Download resource failed: %s %s", resp.code, resp.msg)
            return None
        data = resp.data
        if hasattr(data, "read"):
            return data.read()
        if isinstance(data, bytes):
            return data
        if hasattr(data, "body"):
            return data.body.read()
        return None
    except Exception as e:
        logger.error("Download resource error: %s", e)
        return None


async def download_image(
    client: lark.Client,
    image_key: str,
) -> Optional[bytes]:
    try:
        resp = await asyncio.to_thread(
            client.im.image.get,
            path_params={"image_key": image_key},
        )
        if not resp.success():
            logger.error("Download image failed: %s %s", resp.code, resp.msg)
            return None
        data = resp.data
        if hasattr(data, "read"):
            return data.read()
        if isinstance(data, bytes):
            return data
        if hasattr(data, "body"):
            return data.body.read()
        return None
    except Exception as e:
        logger.error("Download image error: %s", e)
        return None


async def upload_image(
    client: lark.Client,
    image_data: bytes,
) -> Optional[str]:
    if len(image_data) > _MAX_FILE_SIZE:
        logger.error("Image too large: %d bytes", len(image_data))
        return None
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = f"{tmp_dir}/image.png"
            with open(tmp_path, "wb") as f:
                f.write(image_data)
            with open(tmp_path, "rb") as f:
                resp = await asyncio.to_thread(
                    client.im.image.create,
                    CreateImageRequestBody.builder().image_type("message").image(f).build(),
                )
            if resp.success() and resp.data:
                return resp.data.image_key
            logger.error("Upload image failed: %s %s", resp.code, resp.msg)
            return None
    except Exception as e:
        logger.error("Upload image error: %s", e)
        return None


async def upload_file(
    client: lark.Client,
    file_data: bytes,
    filename: str,
) -> Optional[str]:
    if len(file_data) > _MAX_FILE_SIZE:
        logger.error("File too large: %d bytes", len(file_data))
        return None
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            safe_name = os.path.basename(filename)
            tmp_path = os.path.join(tmp_dir, safe_name)
            with open(tmp_path, "wb") as f:
                f.write(file_data)
            with open(tmp_path, "rb") as f:
                resp = await asyncio.to_thread(
                    client.im.file.create,
                    CreateFileRequestBody.builder()
                    .file_type(_guess_file_type(filename))
                    .file_name(filename)
                    .file(f)
                    .build(),
                )
            if resp.success() and resp.data:
                return resp.data.file_key
            logger.error("Upload file failed: %s %s", resp.code, resp.msg)
            return None
    except Exception as e:
        logger.error("Upload file error: %s", e)
        return None


_MAX_DOWNLOAD_SIZE = 50 * 1024 * 1024  # 50MB


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


def _guess_file_type(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    mime_map = {
        ".pdf": "pdf",
        ".doc": "doc",
        ".docx": "docx",
        ".xls": "xls",
        ".xlsx": "xlsx",
        ".ppt": "ppt",
        ".pptx": "pptx",
        ".mp4": "mp4",
        ".zip": "stream",
        ".txt": "stream",
        ".csv": "stream",
    }
    return mime_map.get(ext, "stream")
