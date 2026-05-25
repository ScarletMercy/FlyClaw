"""QQ Bot tools for flyclaw - guild and channel management."""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Optional

logger = logging.getLogger("flyclaw.qq_tools")

_current_qq_chat_id: ContextVar[str] = ContextVar("_current_qq_chat_id", default="")


def set_current_qq_chat_id(chat_id: str):
    _current_qq_chat_id.set(chat_id)


def _get_qq_channel():
    from src.channels.qq import get_qq_channel
    return get_qq_channel()


def _get_http_and_token():
    ch = _get_qq_channel()
    if ch and ch._http_client and ch._token_manager:
        return ch._http_client, ch._token_manager
    return None, None


async def _qq_get(path: str, description: str = "QQ API"):
    import httpx
    from src.channels.qq import API_BASE
    from src.channels.base import api_request_with_retry

    http_client, token_mgr = _get_http_and_token()
    if not http_client or not token_mgr:
        ch = _get_qq_channel()
        if ch is None:
            logger.error("%s: QQ channel not initialized (get_qq_channel() returned None)", description)
        elif ch._token_manager is None:
            logger.error("%s: QQ channel token_manager is None — start() was never called or failed before auth", description)
        elif ch._http_client is None:
            logger.error("%s: QQ channel http_client is None — start() failed after auth (before httpx client created)", description)
        return None

    token = await token_mgr.get_token()
    headers = {"Authorization": f"QQBot {token}"}

    try:
        resp = await api_request_with_retry(
            lambda: http_client.get(f"{API_BASE}{path}", headers=headers),
            description=description,
        )
        if resp.status_code == 401:
            token_mgr.clear_cache()
            token = await token_mgr.get_token()
            headers["Authorization"] = f"QQBot {token}"
            resp = await api_request_with_retry(
                lambda: http_client.get(f"{API_BASE}{path}", headers=headers),
                description=description,
            )
        if resp.status_code >= 400:
            logger.error("%s failed: %d %s", description, resp.status_code, resp.text[:200])
            return None
        return resp.json()
    except Exception as e:
        logger.error("%s error: %s", description, e)
        return None


async def qq_send_image(image_key: str = "") -> str:
    """Send an image to the current QQ chat. Supports local file paths and URLs.

    Args:
        image_key: Local file path (preferred) or URL of the image.
    """
    chat_id = _current_qq_chat_id.get("")
    if not chat_id:
        return "[error] No current QQ chat context"
    ch = _get_qq_channel()
    if not ch:
        return "[error] QQ channel not initialized"
    ok = await ch.send_image(chat_id, image_key)
    return "Image sent." if ok else "[error] Failed to send image"


async def qq_send_file(file_key: str = "") -> str:
    """Send a file to the current QQ chat. Supports local file paths and URLs.

    Args:
        file_key: Local file path (preferred) or URL of the file.
    """
    chat_id = _current_qq_chat_id.get("")
    if not chat_id:
        return "[error] No current QQ chat context"
    ch = _get_qq_channel()
    if not ch:
        return "[error] QQ channel not initialized"
    ok = await ch.send_file(chat_id, file_key)
    return "File sent." if ok else "[error] Failed to send file"


def get_tools() -> list:
    from src.agent.tooldef import ToolDef
    return [
        ToolDef.from_function(qq_send_image),
        ToolDef.from_function(qq_send_file),
    ]
