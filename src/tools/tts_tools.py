from __future__ import annotations

import logging

import aiofiles
import aiofiles.os

from src.tools.result import ToolResult

logger = logging.getLogger("flyclaw.tts_tools")


async def synthesize_speech(text: str, voice: str) -> bytes:
    """edge-tts 合成 mp3 字节；失败抛异常，临时文件总是清理。"""
    import tempfile

    import edge_tts

    communicate = edge_tts.Communicate(text, voice=voice)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        await communicate.save(tmp_path)
        async with aiofiles.open(tmp_path, "rb") as f:
            data = await f.read()
    finally:
        try:
            await aiofiles.os.remove(tmp_path)
        except OSError:
            pass
    return data


async def text_to_speech(text: str, voice: str = "zh-CN-YunxiNeural") -> str | ToolResult:
    """将文本转为语音并发送到当前聊天。适用于简短回复、问候、通知等语音场景。

    Args:
        text: 要转为语音的文本内容
        voice: edge-tts 音色名称，默认 zh-CN-YunxiNeural
    """
    from src.tools.chat_tools import _current_chat_id
    from src.tools.media_tools import _current_channel

    chat_id = _current_chat_id.get("")
    channel = _current_channel.get("")
    if not chat_id:
        return ToolResult.error("No active chat")

    if not text.strip():
        return ToolResult.error("Text is empty")

    try:
        audio_bytes = await synthesize_speech(text, voice)
    except Exception as e:
        logger.error("edge-tts synthesis failed: %s", e)
        return ToolResult.error(f"TTS synthesis failed: {e}")

    if not audio_bytes:
        return ToolResult.error("TTS produced empty audio")

    if channel == "qq":
        from src.channels.qq import _qq_channel

        if _qq_channel and await _qq_channel.send_media(
            chat_id,
            "",
            media_type="audio",
            file_bytes=audio_bytes,
            file_name="speech.mp3",
        ):
            return f"Voice sent ({len(audio_bytes)} bytes, voice={voice})"

    if channel == "weixin":
        from src.channels.weixin import _weixin_channel

        if _weixin_channel:
            try:
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                    tmp_file = f.name
                async with aiofiles.open(tmp_file, "wb") as f:
                    await f.write(audio_bytes)
                try:
                    if await _weixin_channel.send_media(chat_id, tmp_file, media_type="audio"):
                        return f"Voice sent ({len(audio_bytes)} bytes, voice={voice})"
                finally:
                    try:
                        await aiofiles.os.remove(tmp_file)
                    except OSError:
                        pass
            except Exception as e:
                logger.error("WeChat voice send failed: %s", e)

    return ToolResult.error("Failed to send voice to channel")


def get_tools() -> list:
    from src.agent.tooldef import ToolDef

    return [
        ToolDef.from_function(text_to_speech),
    ]
