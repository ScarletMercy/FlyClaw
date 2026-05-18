from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Optional

logger = logging.getLogger("myclaw.media_tools")

_current_channel: ContextVar[str] = ContextVar("_current_channel", default="")


def set_current_channel(channel: str):
    _current_channel.set(channel)


async def send_voice(audio_source: str) -> str:
    """Send an audio file as a voice message to the current chat.

    Args:
        audio_source: URL or local file path of the audio file to send.
    """
    from src.tools.cron_tools import _current_chat_id
    from pathlib import Path

    chat_id = _current_chat_id.get("")
    channel = _current_channel.get("")
    if not chat_id:
        return "[error] No active chat"

    # Read audio from local file or URL
    audio = None
    p = Path(audio_source)
    if p.exists() and p.is_file():
        audio = p.read_bytes()
    elif audio_source.startswith(("http://", "https://")):
        import httpx
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(audio_source)
                resp.raise_for_status()
                audio = resp.content
        except Exception as e:
            return f"[error] Failed to download audio: {e}"
    else:
        return f"[error] Not a valid file path or URL: {audio_source}"

    if not audio:
        return "[error] Empty audio data"

    if channel == "qq":
        from src.channels.qq import _qq_channel
        if _qq_channel and await _qq_channel.send_audio(chat_id, audio):
            return f"Voice sent ({len(audio)} bytes)"
    return "[error] Failed to send voice"


def get_tools() -> list:
    from src.agent.tooldef import ToolDef
    return [
        ToolDef.from_function(send_voice),
    ]
