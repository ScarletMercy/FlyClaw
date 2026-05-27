"""Unified chat tools for flyclaw - works with all channels."""

from __future__ import annotations

import logging
from contextvars import ContextVar
from pathlib import Path

logger = logging.getLogger("flyclaw.chat_tools")

_current_channel_type: ContextVar[str] = ContextVar("_current_channel_type", default="")
_current_chat_id: ContextVar[str] = ContextVar("_current_chat_id", default="")


def set_current_chat_context(channel_type: str, chat_id: str):
    """Set the current channel type and chat ID for tool execution."""
    _current_channel_type.set(channel_type)
    _current_chat_id.set(chat_id)


def _get_channel():
    """Get the active channel instance based on current context."""
    channel_type = _current_channel_type.get("")
    if channel_type == "qq":
        from src.channels.qq import get_qq_channel
        return get_qq_channel()
    elif channel_type == "weixin":
        from src.channels.weixin import get_weixin_channel
        return get_weixin_channel()
    return None


def _get_context() -> tuple[str, str]:
    """Get current channel type and chat ID."""
    return _current_channel_type.get(""), _current_chat_id.get("")


async def send_image(image_key: str = "") -> str:
    """Send an image to the current chat. Supports local file paths and URLs.

    Args:
        image_key: Local file path (preferred) or URL of the image.
    """
    if not image_key:
        return "[error] image_key 不能为空"
    channel_type, chat_id = _get_context()
    if not channel_type:
        return "[error] No active channel context"
    if not chat_id:
        return "[error] No current chat context"
    ch = _get_channel()
    if not ch:
        return "[error] Channel not initialized"

    # 本地文件前置校验
    resolved_key = image_key
    if not image_key.startswith(("http://", "https://", "data:")):
        try:
            from src.tools.file_tools import _resolve_path
            resolved_key = _resolve_path(image_key)
            p = Path(resolved_key)
        except ValueError as e:
            return f"[error] {e}"
        if not p.exists():
            return f"[error] 文件不存在: {image_key}"
        if not p.is_file():
            return f"[error] 不是文件: {image_key}"
        if p.stat().st_size == 0:
            return f"[error] 文件为空: {image_key}"

    ok = await ch.send_image(chat_id, resolved_key)
    return "Image sent." if ok else "[error] 发送图片失败（可能是格式不支持或上传超时）"


async def send_file(file_key: str = "") -> str:
    """Send a file to the current chat. Supports local file paths and URLs.

    Args:
        file_key: Local file path (preferred) or URL of the file.
    """
    if not file_key:
        return "[error] file_key 不能为空"
    channel_type, chat_id = _get_context()
    if not channel_type:
        return "[error] No active channel context"
    if not chat_id:
        return "[error] No current chat context"
    ch = _get_channel()
    if not ch:
        return "[error] Channel not initialized"

    # 本地文件前置校验
    resolved_key = file_key
    if not file_key.startswith(("http://", "https://")):
        try:
            from src.tools.file_tools import _resolve_path
            resolved_key = _resolve_path(file_key)
            p = Path(resolved_key)
        except ValueError as e:
            return f"[error] {e}"
        if not p.exists():
            return f"[error] 文件不存在: {file_key}"
        if not p.is_file():
            return f"[error] 不是文件: {file_key}"
        if p.stat().st_size == 0:
            return f"[error] 文件为空: {file_key}"

    ok = await ch.send_file(chat_id, resolved_key)
    return "File sent." if ok else "[error] 发送文件失败（可能是格式不支持或上传超时）"


async def send_voice(file_path: str = "") -> str:
    """Send a voice/audio message to the current chat.

    Args:
        file_path: Local file path of the audio file.
    """
    if not file_path:
        return "[error] file_path 不能为空"
    channel_type, chat_id = _get_context()
    if not channel_type:
        return "[error] No active channel context"
    if not chat_id:
        return "[error] No current chat context"
    ch = _get_channel()
    if not ch:
        return "[error] Channel not initialized"

    try:
        from src.tools.file_tools import _resolve_path
        resolved = _resolve_path(file_path)
        p = Path(resolved)
    except ValueError as e:
        return f"[error] {e}"
    if not p.exists():
        return f"[error] 文件不存在: {file_path}"
    if not p.is_file():
        return f"[error] 不是文件: {file_path}"
    if p.stat().st_size == 0:
        return f"[error] 文件为空: {file_path}"

    if hasattr(ch, "send_voice"):
        ok = await ch.send_voice(chat_id, str(p))
    elif hasattr(ch, "send_audio"):
        audio_bytes = p.read_bytes()
        ok = await ch.send_audio(chat_id, audio_bytes, p.name)
    else:
        return "[error] 当前渠道不支持语音消息"

    return "Voice sent." if ok else "[error] 发送语音失败"


def get_tools() -> list:
    from src.agent.tooldef import ToolDef
    return [
        ToolDef.from_function(send_image),
        ToolDef.from_function(send_file),
        ToolDef.from_function(send_voice),
    ]
