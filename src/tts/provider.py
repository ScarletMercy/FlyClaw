"""TTS providers: OpenAI, ElevenLabs, Azure."""

from __future__ import annotations

import html as _html
import logging
from typing import Optional

import httpx

from src.config import ModelConfig, TtsConfig

logger = logging.getLogger("myclaw.tts")


class OpenAITtsProvider:
    """OpenAI-compatible TTS provider (works with OpenAI, Groq, Together, local, etc.)."""

    async def synthesize(self, text: str, config: TtsConfig, model_config: ModelConfig) -> bytes:
        text = text[: config.max_chars]
        if not text.strip():
            return b""
        api_key = config.api_key or model_config.api_key or ""
        base_url = (config.base_url or model_config.base_url or "https://api.openai.com").rstrip("/")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": config.model, "input": text, "voice": config.voice, "response_format": "mp3"}
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{base_url}/v1/audio/speech", json=payload, headers=headers)
            resp.raise_for_status()
            return resp.content


class ElevenLabsTtsProvider:
    """ElevenLabs TTS provider."""

    async def synthesize(self, text: str, config: TtsConfig, model_config: ModelConfig) -> bytes:
        text = text[: config.max_chars]
        if not text.strip():
            return b""
        api_key = config.api_key  # ElevenLabs requires its own API key
        if not api_key:
            raise ValueError("ElevenLabs provider requires tts.api_key to be set")
        model = config.model or "eleven_multilingual_v2"
        voice = config.voice or "Rachel"
        headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
        payload = {"text": text, "model_id": model, "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}}
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice}",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            return resp.content


class AzureTtsProvider:
    """Azure Cognitive Services TTS provider."""

    async def synthesize(self, text: str, config: TtsConfig, model_config: ModelConfig) -> bytes:
        text = text[: config.max_chars]
        if not text.strip():
            return b""
        # Azure needs api_key and region (base_url format: "eastus.api.cognitive.microsoft.com")
        api_key = config.api_key
        region = config.base_url or "eastus"
        if not api_key:
            raise ValueError("Azure TTS provider requires tts.api_key to be set")
        voice = config.voice or "en-US-JennyNeural"
        headers = {"Ocp-Apim-Subscription-Key": api_key, "Content-Type": "application/ssml+xml"}
        ssml = f"<speak version='1.0' xml:lang='en-US'><voice name='{_html.escape(voice)}'>{text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')}</voice></speak>"
        url = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, content=ssml.encode("utf-8"), headers=headers)
            resp.raise_for_status()
            return resp.content


class TtsProvider:
    """TTS provider dispatcher. Routes to the configured provider."""

    _PROVIDERS = {
        "openai": OpenAITtsProvider,
        "elevenlabs": ElevenLabsTtsProvider,
        "azure": AzureTtsProvider,
    }

    def __init__(self, config: TtsConfig, model_config: ModelConfig):
        self.config = config
        self.model_config = model_config
        provider_cls = self._PROVIDERS.get(config.provider, OpenAITtsProvider)
        self._inner = provider_cls()

    async def synthesize(self, text: str) -> bytes:
        return await self._inner.synthesize(text, self.config, self.model_config)
