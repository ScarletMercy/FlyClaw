"""OpenAI-compatible API client for media understanding.

Supports image description (chat completions with image_url),
audio transcription (audio/transcriptions endpoint),
and any provider that follows the OpenAI API spec.
"""

from __future__ import annotations

import base64
import logging
from typing import Optional

import httpx

logger = logging.getLogger("myclaw.media_understanding.provider")

_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-6",
}


def _resolve_base_url(provider: str, base_url: str) -> str:
    if base_url:
        return base_url.rstrip("/")
    if provider == "anthropic":
        return "https://api.anthropic.com/v1"
    return "https://api.openai.com/v1"


def _default_model(provider: str, name: str, base_url: str = "") -> str:
    if name:
        return name
    if provider in _DEFAULT_MODELS and not base_url:
        return _DEFAULT_MODELS[provider]
    if base_url:
        logger.warning(
            "Custom base_url set but no model name specified. "
            "Set tools.media_understanding.name to the model name your provider expects."
        )
        return ""
    # Unknown provider without base_url or name — cannot guess
    logger.warning(
        "Unknown provider '%s' without base_url or model name. "
        "Set both tools.media_understanding.base_url and tools.media_understanding.name.",
        provider,
    )
    return ""


class MediaProviderClient:
    """HTTP client for OpenAI-compatible media APIs."""

    def __init__(
        self,
        provider: str = "openai",
        name: str = "",
        base_url: str = "",
        api_key: str = "",
        timeout: int = 60,
    ):
        self.provider = provider
        self.model = _default_model(provider, name, base_url)
        self.base_url = _resolve_base_url(provider, base_url)
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            if self.provider == "anthropic":
                h["x-api-key"] = self.api_key
                h["anthropic-version"] = "2023-06-01"
            else:
                h["Authorization"] = f"Bearer {self.api_key}"
        return h

    @staticmethod
    def _image_to_data_url(data: bytes, mime_type: str = "") -> str:
        if not mime_type:
            mime_type = "image/png"
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:{mime_type};base64,{b64}"

    async def describe_image(
        self,
        image_data: bytes,
        mime_type: str = "image/png",
        prompt: str = "Describe this image in detail. If it contains text, transcribe the text.",
        max_tokens: int = 1024,
    ) -> dict:
        if not self.model:
            return {"text": "", "model": "", "error": "No model name configured. Set tools.media_understanding.name."}

        data_url = self._image_to_data_url(image_data, mime_type)

        if self.provider == "anthropic":
            return await self._describe_image_anthropic(data_url, mime_type, prompt, max_tokens)

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "max_tokens": max_tokens,
        }

        async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout)) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        text = ""
        choices = data.get("choices") or []
        if choices:
            text = choices[0].get("message", {}).get("content", "")
        return {"text": text, "model": data.get("model", self.model)}

    async def _describe_image_anthropic(self, data_url: str, mime_type: str, prompt: str, max_tokens: int) -> dict:
        _, after_comma = data_url.split(",", 1)
        media_type = mime_type or data_url.split(":")[1].split(";")[0]

        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": after_comma,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }

        async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout)) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers=self._headers(),
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        text = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")
        return {"text": text, "model": data.get("model", self.model)}

    async def transcribe_audio(
        self,
        audio_data: bytes,
        mime_type: str = "",
        language: str = "",
        prompt: str = "",
    ) -> dict:
        if not self.model:
            return {"text": "", "model": "", "error": "No model name configured. Set tools.media_understanding.name."}

        if not mime_type:
            mime_type = "audio/wav"

        ext_map = {
            "audio/wav": "wav",
            "audio/mpeg": "mp3",
            "audio/mp4": "mp4",
            "audio/ogg": "ogg",
            "audio/flac": "flac",
            "audio/webm": "webm",
            "audio/x-m4a": "m4a",
        }
        ext = ext_map.get(mime_type, "wav")

        form_data = {
            "model": (None, self.model),
            "file": (f"audio.{ext}", audio_data, mime_type),
            "response_format": (None, "json"),
        }
        if language:
            form_data["language"] = (None, language)
        if prompt:
            form_data["prompt"] = (None, prompt)

        headers = {}
        if self.api_key:
            if self.provider == "anthropic":
                headers["x-api-key"] = self.api_key
            else:
                headers["Authorization"] = f"Bearer {self.api_key}"

        async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout)) as client:
            resp = await client.post(
                f"{self.base_url}/audio/transcriptions",
                headers=headers,
                files=form_data,
            )
            if resp.status_code == 404:
                logger.info("audio/transcriptions not supported, falling back to chat completions")
                return await self._transcribe_audio_via_chat(client, audio_data, mime_type)
            if resp.status_code == 413:
                return {"text": "", "model": self.model, "error": "Audio file too large for transcription API."}
            resp.raise_for_status()
            data = resp.json()

        text = data.get("text", "")
        return {"text": text, "model": data.get("model", self.model)}

    async def _transcribe_audio_via_chat(self, client, audio_data: bytes, mime_type: str) -> dict:
        """Fallback: send audio as base64 to chat completions for transcription."""
        # Audio too large for base64 chat payload (>5MB raw ≈ >7MB base64)
        if len(audio_data) > 5 * 1024 * 1024:
            return {"text": "", "model": self.model, "error": "Audio too large for chat-based transcription (>5MB). Use a smaller file or a provider with /audio/transcriptions support."}

        b64 = base64.b64encode(audio_data).decode("ascii")
        fmt = mime_type.split("/")[-1] if "/" in mime_type else "wav"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please transcribe the following audio. Output only the transcribed text, nothing else."},
                    {"type": "input_audio", "input_audio": {"data": b64, "format": fmt}},
                ],
            }
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 4096,
        }

        try:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error("Audio chat transcription failed: %s %s", e.response.status_code, e.response.text[:200])
            return {"text": "", "model": self.model, "error": f"Audio transcription via chat failed: {e.response.status_code}. The model may not support audio input."}

        data = resp.json()
        text = ""
        for choice in data.get("choices") or []:
            msg = choice.get("message") or {}
            msg_content = msg.get("content")
            if msg_content is None:
                continue
            if isinstance(msg_content, str):
                text += msg_content
            elif isinstance(msg_content, list):
                for block in msg_content:
                    if isinstance(block, str):
                        text += block
                    elif isinstance(block, dict) and block.get("type") == "text":
                        text += block.get("text", "")

        if not text.strip():
            return {"text": "", "model": self.model, "error": "Audio transcription returned empty result. The model may not support audio input."}

        return {"text": text.strip(), "model": data.get("model", self.model)}

    async def describe_video_native(
        self,
        video_data: bytes,
        mime_type: str = "video/mp4",
        prompt: str = "Describe this video in detail. What is happening?",
        max_tokens: int = 2048,
    ) -> dict:
        """Send video directly to the model (no frame extraction).

        Uses OpenAI-compatible video_url content type.
        Falls back gracefully — caller should catch errors and use frame extraction.
        """
        if not self.model:
            return {"text": "", "model": "", "error": "No model name configured."}

        data_url = self._image_to_data_url(video_data, mime_type)

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "video_url", "video_url": {"url": data_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": max_tokens,
        }

        async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout)) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        text = ""
        choices = data.get("choices") or []
        if choices:
            text = choices[0].get("message", {}).get("content", "")
        return {"text": text, "model": data.get("model", self.model)}
