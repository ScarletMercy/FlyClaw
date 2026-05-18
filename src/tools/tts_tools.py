from __future__ import annotations

import logging
import tempfile
from pathlib import Path

logger = logging.getLogger("myclaw.tts_tools")


async def text_to_speech(text: str, voice: str = "zh-CN-YunxiNeural") -> str:
    """将文本转为语音并发送到当前聊天。适用于简短回复、问候、通知等语音场景。

    Args:
        text: 要转为语音的文本内容
        voice: edge-tts 音色名称，默认 zh-CN-YunxiNeural
    """
    from src.tools.cron_tools import _current_chat_id
    from src.tools.media_tools import _current_channel

    chat_id = _current_chat_id.get("")
    channel = _current_channel.get("")
    if not chat_id:
        return "[error] No active chat"

    if not text.strip():
        return "[error] Text is empty"

    import edge_tts

    try:
        communicate = edge_tts.Communicate(text, voice=voice)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name
        await communicate.save(tmp_path)
        audio_bytes = Path(tmp_path).read_bytes()
        Path(tmp_path).unlink(missing_ok=True)
    except Exception as e:
        logger.error("edge-tts synthesis failed: %s", e)
        return f"[error] TTS synthesis failed: {e}"

    if not audio_bytes:
        return "[error] TTS produced empty audio"

    if channel == "qq":
        from src.channels.qq import _qq_channel
        if _qq_channel and await _qq_channel.send_audio(chat_id, audio_bytes):
            return f"Voice sent ({len(audio_bytes)} bytes, voice={voice})"

    return "[error] Failed to send voice to channel"


def get_tools() -> list:
    from src.agent.tooldef import ToolDef
    return [
        ToolDef.from_function(text_to_speech),
    ]
