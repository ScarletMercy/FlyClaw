from __future__ import annotations

import asyncio
import logging
from typing import Optional

import lark_oapi as lark
from lark_oapi.api.im.v1 import (
    CreateMessageReactionRequest,
    CreateMessageReactionRequestBody,
    DeleteMessageReactionRequest,
)

logger = logging.getLogger("myclaw.typing")

_TYPING_EMOJI = "Typing"
_MAX_DURATION = 60
_BACKOFF_CODES = {99991400, 99991403, 429}
_MAX_CONSECUTIVE_FAILURES = 2
_consecutive_failures_lock = asyncio.Lock()
_consecutive_failures = 0


class TypingIndicator:
    def __init__(self, client: lark.Client, enabled: bool = True):
        self._client = client
        self._enabled = enabled
        self._active: dict[str, str] = {}
        self._timers: dict[str, asyncio.Task] = {}

    async def start(self, message_id: str) -> None:
        if not self._enabled:
            return
        global _consecutive_failures
        async with _consecutive_failures_lock:
            if _consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                return
        try:
            body = CreateMessageReactionRequestBody.builder().reaction_type({"emoji_type": _TYPING_EMOJI}).build()
            req = CreateMessageReactionRequest.builder().message_id(message_id).request_body(body).build()
            resp = await asyncio.to_thread(
                self._client.im.v1.message_reaction.create,
                req,
            )
            if resp.success() and resp.data:
                reaction_id = resp.data.reaction_id
                self._active[message_id] = reaction_id
                async with _consecutive_failures_lock:
                    _consecutive_failures = 0
                self._timers[message_id] = asyncio.create_task(self._auto_stop(message_id, _MAX_DURATION))
            else:
                code = getattr(resp, "code", 0)
                if code in _BACKOFF_CODES:
                    async with _consecutive_failures_lock:
                        _consecutive_failures += 1
                    logger.warning(
                        "Typing indicator rate-limited, backing off (failures=%d)",
                        _consecutive_failures,
                    )
                else:
                    logger.debug("Typing indicator failed: %s %s", resp.code, resp.msg)
        except Exception as e:
            logger.debug("Typing indicator error: %s", e)

    async def stop(self, message_id: str) -> None:
        reaction_id = self._active.pop(message_id, None)
        timer = self._timers.pop(message_id, None)
        if timer:
            timer.cancel()
        if not reaction_id:
            return
        try:
            req = DeleteMessageReactionRequest.builder().message_id(message_id).reaction_id(reaction_id).build()
            await asyncio.to_thread(
                self._client.im.v1.message_reaction.delete,
                req,
            )
        except Exception:
            pass

    async def stop_all(self) -> None:
        message_ids = list(self._active.keys())
        for mid in message_ids:
            await self.stop(mid)

    async def _auto_stop(self, message_id: str, delay: int) -> None:
        try:
            await asyncio.sleep(delay)
            await self.stop(message_id)
        except asyncio.CancelledError:
            pass
