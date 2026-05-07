from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Optional

from langchain_core.tools import tool

logger = logging.getLogger("myclaw.media_tools")

_current_channel: ContextVar[str] = ContextVar("_current_channel", default="")


def set_current_channel(channel: str):
    _current_channel.set(channel)


@tool
async def send_image_to_chat(chat_id: str, image_url: str) -> str:
    """Send an image to a Feishu chat. Downloads from URL, uploads to Feishu, sends.

    Args:
        chat_id: The Feishu chat ID to send the image to.
        image_url: URL of the image to download and send.
    """
    from src.channels.feishu import get_feishu_client
    from src.channels.media import download_from_url, upload_image

    client = get_feishu_client()
    if client is None:
        return "[error] Feishu client not initialized"

    result = await download_from_url(image_url)
    if result is None:
        return "[error] Failed to download image"

    data, content_type = result
    image_key = await upload_image(client, data)
    if image_key is None:
        return "[error] Failed to upload image to Feishu"

    sent = await _send_image_message(client, chat_id, image_key)
    if sent:
        return f"Image sent to chat {chat_id} (key: {image_key})"
    return "[error] Failed to send image message"


@tool
async def send_file_to_chat(chat_id: str, file_url: str, filename: str) -> str:
    """Send a file to a Feishu chat. Downloads from URL, uploads to Feishu, sends.

    Args:
        chat_id: The Feishu chat ID to send the file to.
        file_url: URL of the file to download and send.
        filename: Name for the file in Feishu.
    """
    from src.channels.feishu import get_feishu_client
    from src.channels.media import download_from_url, upload_file

    client = get_feishu_client()
    if client is None:
        return "[error] Feishu client not initialized"

    result = await download_from_url(file_url)
    if result is None:
        return "[error] Failed to download file"

    data, content_type = result
    file_key = await upload_file(client, data, filename)
    if file_key is None:
        return "[error] Failed to upload file to Feishu"

    sent = await _send_file_message(client, chat_id, file_key)
    if sent:
        return f"File sent to chat {chat_id} ({filename}, key: {file_key})"
    return "[error] Failed to send file message"


async def _send_image_message(client, chat_id: str, image_key: str) -> bool:
    import asyncio
    import json

    from lark_oapi.api.im.v1 import CreateMessageRequestBody, CreateMessageRequest

    try:
        content = json.dumps({"image_key": image_key})
        body = CreateMessageRequestBody.builder().receive_id(chat_id).msg_type("image").content(content).build()
        req = CreateMessageRequest.builder().receive_id_type("chat_id").request_body(body).build()
        resp = await asyncio.to_thread(client.im.v1.message.create, req)
        return resp.success()
    except Exception as e:
        logger.error("Send image message error: %s", e)
        return False


async def _send_file_message(client, chat_id: str, file_key: str) -> bool:
    import asyncio
    import json

    from lark_oapi.api.im.v1 import CreateMessageRequestBody, CreateMessageRequest

    try:
        content = json.dumps({"file_key": file_key})
        body = CreateMessageRequestBody.builder().receive_id(chat_id).msg_type("file").content(content).build()
        req = CreateMessageRequest.builder().receive_id_type("chat_id").request_body(body).build()
        resp = await asyncio.to_thread(client.im.v1.message.create, req)
        return resp.success()
    except Exception as e:
        logger.error("Send file message error: %s", e)
        return False


@tool
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
    elif channel == "feishu":
        from src.channels.feishu import _feishu_channel
        if _feishu_channel and await _feishu_channel.send_audio(chat_id, audio):
            return f"Voice sent ({len(audio)} bytes)"
    return "[error] Failed to send voice"
