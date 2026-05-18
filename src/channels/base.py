from __future__ import annotations

import asyncio
import collections
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import httpx

logger = logging.getLogger("myclaw.channels.base")

_MAX_API_RETRIES = 3


class Channel(ABC):
    """Abstract base class for messaging channels."""

    def __init__(self) -> None:
        self._processed_messages: collections.deque[str] = collections.deque(maxlen=5000)
        self._processed_messages_lock = asyncio.Lock()

    @abstractmethod
    def set_message_callback(self, callback: Callable) -> None:
        """Set the callback function for incoming messages."""
        pass

    @abstractmethod
    async def start(self) -> None:
        """Start the channel and begin listening for messages."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the channel and cleanup resources."""
        pass

    @abstractmethod
    async def send_text(
        self,
        chat_id: str,
        text: str,
        reply_to: Optional[str] = None,
    ) -> Any:
        """Send a text message to a chat."""
        pass

    @abstractmethod
    async def send_image(self, chat_id: str, image_key: str) -> bool:
        """Send an image message to a chat."""
        pass

    @abstractmethod
    async def send_file(self, chat_id: str, file_key: str) -> bool:
        """Send a file message to a chat."""
        pass

    @abstractmethod
    async def send_card(
        self,
        chat_id: str,
        card_content: str,
        reply_to: Optional[str] = None,
    ) -> Any:
        """Send an interactive card to a chat."""
        pass

    # ── Shared helpers ──────────────────────────────────────

    async def check_dedup(self, message_id: str) -> bool:
        """Check and record message ID. Returns True if duplicate (should skip)."""
        async with self._processed_messages_lock:
            if message_id in self._processed_messages:
                return True
            self._processed_messages.append(message_id)
            return False

    @staticmethod
    def chunk_text(text: str, limit: int = 4000) -> list[str]:
        """Split text into chunks respecting newline boundaries."""
        if len(text) <= limit:
            return [text]
        chunks = []
        while text:
            if len(text) <= limit:
                chunks.append(text)
                break
            cut = text.rfind("\n", 0, limit)
            if cut < limit // 2:
                cut = limit
            chunks.append(text[:cut])
            text = text[cut:].lstrip("\n")
        return chunks


async def api_request_with_retry(
    request_fn: Callable,
    *,
    description: str = "API request",
    max_retries: int = _MAX_API_RETRIES,
    retry_on_server_error: bool = True,
) -> Any:
    """Execute an HTTP request with retry for 429, 5xx, and network errors."""
    import random as _random

    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = await request_fn()
        except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as e:
            last_exc = e
            if attempt >= max_retries:
                logger.warning("%s: network error persisted after %d retries: %s", description, max_retries, e)
                raise
            wait = min(1.0 * (2 ** attempt), 8.0) + _random.uniform(0, 0.5)
            logger.warning("%s: network error, retry %d/%d in %.1fs: %s", description, attempt + 1, max_retries, wait, e)
            await asyncio.sleep(wait)
            continue

        if getattr(resp, "status_code", 0) == 429:
            if attempt >= max_retries:
                logger.warning("%s: 429 rate limit persisted after %d retries", description, max_retries)
                return resp
            retry_after = getattr(resp, "headers", {}).get("Retry-After")
            if retry_after:
                try:
                    wait = float(retry_after)
                except ValueError:
                    wait = 1.0
            else:
                wait = min(1.0 * (2**attempt), 8.0)
                wait += _random.uniform(0, wait * 0.1)
            logger.warning("%s: 429 rate limited, retry %d/%d in %.1fs", description, attempt + 1, max_retries, wait)
            await asyncio.sleep(wait)
            continue

        if retry_on_server_error and getattr(resp, "status_code", 0) >= 500:
            if attempt >= max_retries:
                logger.warning("%s: server error %d persisted after %d retries", description, resp.status_code, max_retries)
                return resp
            wait = min(1.0 * (2 ** attempt), 8.0) + _random.uniform(0, 0.5)
            logger.warning("%s: server error %d, retry %d/%d in %.1fs", description, resp.status_code, attempt + 1, max_retries, wait)
            await asyncio.sleep(wait)
            continue

        return resp
