"""Chat model client using the openai library directly."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI
import httpx

logger = logging.getLogger("myclaw.agent.client")


@dataclass
class ChatResponse:
    content: str
    tool_calls: list[Any] = field(default_factory=list)


class ChatClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.0,
    ):
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0),
            max_retries=2,
        )
        self.model = model
        self.temperature = temperature
        self.base_url = base_url

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        **extra: Any,
    ) -> ChatResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        kwargs.update(extra)

        resp = await self._client.chat.completions.create(**kwargs)
        if not resp.choices:
            raw = resp.model_dump()
            error_info = raw.get("error", {})
            if isinstance(error_info, dict):
                msg = error_info.get("message", "")
            elif isinstance(error_info, str):
                msg = error_info
            else:
                msg = raw.get("message", "")
            if msg:
                raise ValueError(f"Model {self.model} API error: {msg}")
            raise ValueError(f"Model {self.model} returned no choices (empty response)")
        choice = resp.choices[0].message
        if choice is None:
            raise ValueError(f"Model {self.model} returned empty message")
        return ChatResponse(
            content=choice.content or "",
            tool_calls=choice.tool_calls or [],
        )

    async def chat_simple(self, messages: list[dict], **extra: Any) -> str:
        resp = await self.chat(messages, tools=None, **extra)
        return resp.content

    def __repr__(self) -> str:
        return f"ChatClient(model={self.model!r}, base_url={self.base_url!r})"


class FallbackChain:
    def __init__(self, primary: ChatClient, fallbacks: list[ChatClient] | None = None):
        self._all = [primary] + (fallbacks or [])
        self._active_idx = 0
        self._cooldowns: dict[int, float] = {}
        self._model_meta: list[dict] = []

    @property
    def active(self) -> ChatClient:
        return self._all[self._active_idx]

    def switch_to(self, idx: int) -> None:
        if 0 <= idx < len(self._all):
            self._active_idx = idx
            self._cooldowns.pop(id(self._all[idx]), None)

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        **extra: Any,
    ) -> ChatResponse:
        now = time.time()
        errors: list[tuple[int, Exception]] = []
        order = [self._active_idx] + [
            i for i in range(len(self._all)) if i != self._active_idx
        ]
        for i in order:
            client = self._all[i]
            if now < self._cooldowns.get(id(client), 0):
                continue
            try:
                return await client.chat(messages, tools=tools, **extra)
            except Exception as e:
                errors.append((i, e))
                err_lower = str(e).lower()
                cooldown = 0
                # Don't fallback on empty response - it's a message format issue
                if "no choices" in err_lower or "empty message" in err_lower or "api error" in err_lower:
                    logger.error("Model %d returned empty/API error - NOT falling back (message format issue)", i)
                    raise e
                if "rate" in err_lower or "429" in err_lower:
                    cooldown = 30
                elif "overload" in err_lower or "503" in err_lower or "529" in err_lower:
                    cooldown = 60
                elif "auth" in err_lower or "401" in err_lower or "403" in err_lower:
                    cooldown = 300
                elif "billing" in err_lower or "quota" in err_lower:
                    cooldown = 3600
                if cooldown:
                    self._cooldowns[id(client)] = now + cooldown
                logger.warning("Model %d failed, cooldown %ds: %s", i, cooldown, e)
        if errors:
            raise errors[-1][1]
        raise RuntimeError("No models available")

    async def chat_simple(self, messages: list[dict], **extra: Any) -> str:
        resp = await self.chat(messages, tools=None, **extra)
        return resp.content

    def __repr__(self) -> str:
        active = self._all[self._active_idx]
        return f"FallbackChain(active={active!r}, total={len(self._all)})"


class ModelRef:
    def __init__(self, model: ChatClient | FallbackChain):
        self.model = model
        self.ctx_window: dict | None = None


def create_client(
    provider: str,
    name: str,
    temperature: float,
    base_url: str | None = None,
    api_key: str | None = None,
) -> ChatClient:
    return ChatClient(
        base_url=base_url or "",
        api_key=api_key or "",
        model=name,
        temperature=temperature,
    )


def create_chain(config) -> FallbackChain:
    primary = create_client(
        config.model.provider,
        config.model.name,
        config.model.temperature,
        base_url=config.model.base_url,
        api_key=config.model.api_key,
    )
    fallbacks = []
    for fb in config.model.fallbacks or []:
        fallbacks.append(create_client(
            fb.provider,
            fb.name,
            getattr(fb, "temperature", config.model.temperature),
            base_url=fb.base_url or config.model.base_url,
            api_key=fb.api_key or config.model.api_key,
        ))
    return FallbackChain(primary, fallbacks)
