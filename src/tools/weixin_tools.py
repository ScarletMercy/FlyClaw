"""WeChat (Weixin) tools for MyClaw - message sending and file transfer."""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Optional

logger = logging.getLogger("myclaw.weixin_tools")

_current_weixin_chat_id: ContextVar[str] = ContextVar("_current_weixin_chat_id", default="")


def set_current_weixin_chat_id(chat_id: str):
    _current_weixin_chat_id.set(chat_id)


def _get_weixin_channel():
    from src.channels.weixin import get_weixin_channel
    return get_weixin_channel()


async def weixin_send_text(chat_id: str = "", text: str = "") -> str:
    """Send a text message to a WeChat user or group via the bot.

    Args:
        chat_id: WeChat user ID or group chat ID. Leave empty to reply in current chat.
        text: Message text content.
    """
    chat_id = chat_id or _current_weixin_chat_id.get("")
    if not chat_id:
        return "[error] No chat_id provided and no current WeChat chat context"
    ch = _get_weixin_channel()
    if not ch:
        return "[error] WeChat channel not initialized"
    result = await ch.send_text(chat_id, text)
    if result is not None:
        return "Message sent."
    return "[error] Failed to send message"


async def weixin_send_image(chat_id: str = "", image_key: str = "") -> str:
    """Send an image to a WeChat user or group. Supports local file paths and URLs.

    Args:
        chat_id: WeChat user ID or group chat ID. Leave empty to reply in current chat.
        image_key: Local file path (preferred) or URL of the image.
    """
    chat_id = chat_id or _current_weixin_chat_id.get("")
    if not chat_id:
        return "[error] No chat_id provided and no current WeChat chat context"
    ch = _get_weixin_channel()
    if not ch:
        return "[error] WeChat channel not initialized"
    ok = await ch.send_image(chat_id, image_key)
    return "Image sent." if ok else "[error] Failed to send image"


async def weixin_send_file(chat_id: str = "", file_key: str = "") -> str:
    """Send a file to a WeChat user or group. Supports local file paths and URLs.

    Args:
        chat_id: WeChat user ID or group chat ID. Leave empty to reply in current chat.
        file_key: Local file path (preferred) or URL of the file.
    """
    chat_id = chat_id or _current_weixin_chat_id.get("")
    if not chat_id:
        return "[error] No chat_id provided and no current WeChat chat context"
    ch = _get_weixin_channel()
    if not ch:
        return "[error] WeChat channel not initialized"
    ok = await ch.send_file(chat_id, file_key)
    return "File sent." if ok else "[error] Failed to send file"


async def weixin_send_voice(chat_id: str = "", file_path: str = "", caption: Optional[str] = None) -> str:
    """Send a voice/audio file to a WeChat user or group.

    Args:
        chat_id: WeChat user ID or group chat ID. Leave empty to reply in current chat.
        file_path: Local file path of the audio file.
        caption: Optional caption for the voice message.
    """
    chat_id = chat_id or _current_weixin_chat_id.get("")
    if not chat_id:
        return "[error] No chat_id provided and no current WeChat chat context"
    ch = _get_weixin_channel()
    if not ch:
        return "[error] WeChat channel not initialized"
    ok = await ch.send_voice(chat_id, file_path, caption)
    return "Voice sent." if ok else "[error] Failed to send voice"


def get_tools() -> list:
    from src.agent.tooldef import ToolDef
    return [
        ToolDef.from_function(weixin_send_text),
        ToolDef.from_function(weixin_send_image),
        ToolDef.from_function(weixin_send_file),
        ToolDef.from_function(weixin_send_voice),
    ]
