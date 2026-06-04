"""Unified chat tools for flyclaw - works with all channels."""

from __future__ import annotations

import logging
from contextvars import ContextVar
from pathlib import Path

from src.media_delivery import _AUDIO_EXTS, _IMAGE_EXTS

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


def _infer_media_type(file_key: str) -> str:
    """Infer media type from file extension (handles URLs with query strings)."""
    from urllib.parse import urlparse

    path = urlparse(file_key).path if file_key.startswith(("http://", "https://")) else file_key
    ext = Path(path).suffix.lower()
    if ext in _IMAGE_EXTS:
        return "image"
    if ext in _AUDIO_EXTS:
        return "audio"
    return "file"


async def send_file(file_key: str = "", force_file: bool = False) -> str:
    """Send a file to the current chat. Supports images, audio, documents, and any other file type.
    If the file is a media file (image or audio), it is sent as native media by default
    (inline display / playable audio); otherwise it is sent as a downloadable file attachment.
    Use force_file=True to always send as a generic file attachment regardless of type.

    Supports local file paths and URLs.

    Args:
        file_key: (Required) The local file path or URL of the file to send.
        force_file: When True, force sending as a generic file attachment
                    instead of native media type (e.g. send an image as a file).
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

    media_type = "file" if force_file else _infer_media_type(file_key)

    # 本地文件前置校验
    resolved_key = file_key
    if not file_key.startswith(("http://", "https://", "data:")):
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

    ok = await ch.send_media(chat_id, resolved_key, media_type)
    return "Media sent." if ok else "[error] 发送失败（可能是格式不支持或上传超时）"


def get_tools() -> list:
    from src.agent.tooldef import ToolDef

    return [
        ToolDef.from_function(send_file),
    ]
