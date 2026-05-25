"""WeChat (Weixin) tools for flyclaw - message sending and file transfer."""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Optional

logger = logging.getLogger("flyclaw.weixin_tools")

_current_weixin_chat_id: ContextVar[str] = ContextVar("_current_weixin_chat_id", default="")


def set_current_weixin_chat_id(chat_id: str):
    _current_weixin_chat_id.set(chat_id)


def _get_weixin_channel():
    from src.channels.weixin import get_weixin_channel
    return get_weixin_channel()


async def weixin_send_image(image_key: str = "") -> str:
    """Send an image to the current WeChat chat. Supports local file paths and URLs.

    Args:
        image_key: Local file path (preferred) or URL of the image.
    """
    chat_id = _current_weixin_chat_id.get("")
    if not chat_id:
        return "[error] No current WeChat chat context"
    ch = _get_weixin_channel()
    if not ch:
        return "[error] WeChat channel not initialized"
    ok = await ch.send_image(chat_id, image_key)
    return "Image sent." if ok else "[error] Failed to send image"


async def weixin_send_file(file_key: str = "") -> str:
    """Send a file to the current WeChat chat. Supports local file paths and URLs.

    Args:
        file_key: Local file path (preferred) or URL of the file.
    """
    chat_id = _current_weixin_chat_id.get("")
    if not chat_id:
        return "[error] No current WeChat chat context"
    ch = _get_weixin_channel()
    if not ch:
        return "[error] WeChat channel not initialized"
    ok = await ch.send_file(chat_id, file_key)
    return "File sent." if ok else "[error] Failed to send file"


async def weixin_send_voice(file_path: str = "", caption: Optional[str] = None) -> str:
    """Send a voice/audio file to the current WeChat chat.

    Args:
        file_path: Local file path of the audio file.
        caption: Optional caption for the voice message.
    """
    chat_id = _current_weixin_chat_id.get("")
    if not chat_id:
        return "[error] No current WeChat chat context"
    ch = _get_weixin_channel()
    if not ch:
        return "[error] WeChat channel not initialized"
    ok = await ch.send_voice(chat_id, file_path, caption)
    return "Voice sent." if ok else "[error] Failed to send voice"


def get_tools() -> list:
    from src.agent.tooldef import ToolDef
    return [
        ToolDef.from_function(weixin_send_image),
        ToolDef.from_function(weixin_send_file),
        ToolDef.from_function(weixin_send_voice),
    ]
