from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time as _time
from typing import Any, Callable, Optional

import httpx
import lark_oapi as lark
from lark_oapi.api.im.v1 import (
    CreateMessageRequest,
    CreateMessageRequestBody,
    CreateMessageResponseBody,
    ReplyMessageRequest,
)
from lark_oapi.ws.client import Client as WsClient
from lark_oapi.event.dispatcher_handler import EventDispatcherHandlerBuilder
from lark_oapi.api.im.v1.model.p2_im_message_receive_v1 import P2ImMessageReceiveV1

from .base import Channel, api_request_with_retry

logger = logging.getLogger("myclaw.feishu")

_feishu_client: Optional[lark.Client] = None
_feishu_channel: Optional["FeishuChannel"] = None
_bot_info: Optional[dict] = None
_token_cache: dict[str, dict] = {}
_token_cache_lock = asyncio.Lock()
_mu_runner = None  # Fallback only; prefer _feishu_channel._mu_runner


def set_media_understanding_runner(runner):
    """Set the media understanding runner for auto-processing media attachments."""
    if _feishu_channel:
        _feishu_channel._mu_runner = runner
    else:
        global _mu_runner
        _mu_runner = runner


def _resolve_api_base(domain: str) -> str:
    if domain == "lark":
        return "https://open.larksuite.com/open-apis"
    return "https://open.feishu.cn/open-apis"


async def feishu_api_request(request_fn, *, description: str = "Feishu API") -> httpx.Response:
    """Public wrapper for Feishu API requests with 429 retry logic."""
    return await api_request_with_retry(request_fn, description=description)


async def _get_tenant_token(app_id: str, app_secret: str, domain: str) -> str:
    cache_key = f"{domain}|{app_id}"
    async with _token_cache_lock:
        cached = _token_cache.get(cache_key)
        if cached and cached["expires_at"] > _time.time() + 60:
            return cached["token"]

        api_base = _resolve_api_base(domain)
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
            try:
                resp = await api_request_with_retry(
                    lambda: client.post(
                        f"{api_base}/auth/v3/tenant_access_token/internal",
                        json={"app_id": app_id, "app_secret": app_secret},
                        timeout=10,
                    ),
                    description="Tenant token refresh",
                )
            except Exception as e:
                raise RuntimeError(f"Token refresh request failed: {e}") from e
        try:
            data = resp.json()
        except Exception as e:
            raise RuntimeError(f"Token response parse error: {e}") from e
        if data.get("code") != 0 or not data.get("tenant_access_token"):
            raise RuntimeError(f"Token error: {data.get('msg', 'unknown')}")
        token = data["tenant_access_token"]
        expire = data.get("expire", 7200)
        _token_cache[cache_key] = {"token": token, "expires_at": _time.time() + expire}
        return token


def _format_for_card_md(text: str) -> str:
    """Convert standard markdown to Feishu card-compatible markdown.

    Feishu card markdown has limited support for formatting:
    - Lists (- item) are not rendered properly, convert to bullet points with newlines
    - Multiple newlines get collapsed, use double newline for paragraph breaks
    """
    import html
    import re

    # Escape HTML entities to prevent XSS in card content
    text = html.escape(text, quote=False)

    # Convert markdown list items "- item" to "• item" with explicit newlines
    text = re.sub(r"\n-\s*", "\n• ", text)
    text = re.sub(r"^-\s*", "• ", text)
    return text


def _merge_streaming_text(previous: str, incoming: str) -> str:
    if not incoming:
        return previous
    if not previous or incoming == previous:
        return incoming
    if incoming.startswith(previous):
        return incoming
    if previous.startswith(incoming):
        return previous
    if incoming in previous:
        return previous
    if previous in incoming:
        return incoming
    max_overlap = min(len(previous), len(incoming))
    for overlap in range(max_overlap, 0, -1):
        if previous[-overlap:] == incoming[:overlap]:
            return previous + incoming[overlap:]
    return previous + incoming


class StreamingCardSession:
    def __init__(self, app_id: str, app_secret: str, domain: str):
        self._app_id = app_id
        self._app_secret = app_secret
        self._domain = domain
        self._card_id: Optional[str] = None
        self._message_id: Optional[str] = None
        self._sequence: int = 0
        self._current_text: str = ""
        self._pending_text: Optional[str] = None
        self._closed: bool = False
        self._last_update: float = float("-inf")
        self._throttle_ms: int = 100
        self._token: Optional[str] = None
        self._token_expires: float = 0
        self._http_client: Optional[httpx.AsyncClient] = None

    async def start(
        self,
        lark_client: lark.Client,
        chat_id: str,
        reply_to: Optional[str] = None,
    ) -> Optional[str]:
        api_base = _resolve_api_base(self._domain)
        if not self._http_client:
            self._http_client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
        token = await _get_tenant_token(self._app_id, self._app_secret, self._domain)
        if token:
            self._token = token
            self._token_expires = _time.monotonic() + 1800

        card_json = {
            "schema": "2.0",
            "config": {
                "streaming_mode": True,
                "summary": {"content": "[Generating...]"},
                "streaming_config": {
                    "print_frequency_ms": {"default": 50},
                    "print_step": {"default": 1},
                },
            },
            "body": {"elements": [{"tag": "markdown", "content": "⏳ Thinking...", "element_id": "content"}]},
        }

        resp = await api_request_with_retry(
            lambda: self._http_client.post(
                f"{api_base}/cardkit/v1/cards",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json={"type": "card_json", "data": json.dumps(card_json)},
                timeout=10,
            ),
            description="Create streaming card",
        )
        data = resp.json()
        if data.get("code") != 0 or not data.get("data", {}).get("card_id"):
            logger.error("Create streaming card failed: %s", data.get("msg"))
            if self._http_client:
                await self._http_client.aclose()
                self._http_client = None
            return None
        self._card_id = data["data"]["card_id"]
        self._sequence = 1

        card_content = json.dumps({"type": "card", "data": {"card_id": self._card_id}})
        try:
            if reply_to:
                body = CreateMessageRequestBody.builder().msg_type("interactive").content(card_content).build()
                req = ReplyMessageRequest.builder().message_id(reply_to).request_body(body).build()
                resp_data = await asyncio.to_thread(
                    lark_client.im.v1.message.reply,
                    req,
                )
            else:
                body = (
                    CreateMessageRequestBody.builder()
                    .receive_id(chat_id)
                    .msg_type("interactive")
                    .content(card_content)
                    .build()
                )
                req = CreateMessageRequest.builder().receive_id_type("chat_id").request_body(body).build()
                resp_data = await asyncio.to_thread(
                    lark_client.im.v1.message.create,
                    req,
                )
            if hasattr(resp_data, "data") and resp_data.data:
                self._message_id = resp_data.data.message_id
        except Exception as e:
            logger.error("Send streaming card failed: %s", e)
            if self._http_client:
                await self._http_client.aclose()
                self._http_client = None
            return None

        logger.debug("Streaming card started: %s", self._card_id)
        return self._card_id

    async def update(self, text: str, flush: bool = False) -> None:
        if not self._card_id or self._closed:
            return

        merged_input = _merge_streaming_text(self._pending_text or self._current_text, text)
        if not merged_input or merged_input == self._current_text:
            return

        now = _time.monotonic() * 1000
        if not flush and now - self._last_update < self._throttle_ms:
            self._pending_text = merged_input
            return

        self._pending_text = None
        self._last_update = now
        await self._do_update(merged_input)

    async def _ensure_token(self) -> Optional[str]:
        now = _time.monotonic()
        if self._token and now < self._token_expires:
            return self._token
        self._token = await _get_tenant_token(self._app_id, self._app_secret, self._domain)
        if self._token:
            self._token_expires = now + 1800  # token valid for ~2h, refresh at 30m
        return self._token

    async def _do_update(self, merged_input: str) -> None:
        if not self._card_id:
            return
        merged_text = _merge_streaming_text(self._current_text, merged_input)
        if not merged_text or merged_text == self._current_text:
            return

        self._current_text = merged_text
        self._sequence += 1

        api_base = _resolve_api_base(self._domain)
        try:
            token = await self._ensure_token()
            if not token:
                return
            await api_request_with_retry(
                lambda: self._http_client.put(
                    f"{api_base}/cardkit/v1/cards/{self._card_id}/elements/content/content",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "content": _format_for_card_md(merged_text),
                        "sequence": self._sequence,
                        "uuid": f"s_{self._card_id}_{self._sequence}",
                    },
                    timeout=10,
                ),
                description="Streaming card update",
            )
        except Exception as e:
            logger.debug("Streaming card update failed: %s", e)

    async def close(self, final_text: Optional[str] = None) -> None:
        if not self._card_id or self._closed:
            return
        self._closed = True

        pending = _merge_streaming_text(self._current_text, self._pending_text or "")
        text = _merge_streaming_text(pending, final_text) if final_text else pending

        api_base = _resolve_api_base(self._domain)
        try:
            token = await self._ensure_token()
            if text and text != self._current_text:
                self._sequence += 1
                try:
                    await api_request_with_retry(
                        lambda: self._http_client.put(
                            f"{api_base}/cardkit/v1/cards/{self._card_id}/elements/content/content",
                            headers={
                                "Authorization": f"Bearer {token}",
                                "Content-Type": "application/json",
                            },
                            json={
                                "content": _format_for_card_md(text),
                                "sequence": self._sequence,
                                "uuid": f"s_{self._card_id}_{self._sequence}",
                            },
                            timeout=10,
                        ),
                        description="Streaming card final update",
                    )
                except Exception as e:
                    logger.debug("Streaming card final update failed: %s", e)
                self._current_text = text

            self._sequence += 1
            summary = text.replace("\n", " ").strip()[:50]
            try:
                await api_request_with_retry(
                    lambda: self._http_client.patch(
                        f"{api_base}/cardkit/v1/cards/{self._card_id}/settings",
                        headers={
                            "Authorization": f"Bearer {token}",
                            "Content-Type": "application/json; charset=utf-8",
                        },
                        json={
                            "settings": json.dumps(
                                {
                                    "config": {
                                        "streaming_mode": False,
                                        "summary": {"content": summary},
                                    }
                                }
                            ),
                            "sequence": self._sequence,
                            "uuid": f"c_{self._card_id}_{self._sequence}",
                        },
                        timeout=10,
                    ),
                    description="Streaming card close",
                )
            except Exception as e:
                logger.debug("Streaming card close failed: %s", e)
        finally:
            if self._http_client:
                await self._http_client.aclose()
                self._http_client = None

        logger.debug("Streaming card closed: %s", self._card_id)
        self._card_id = None


def get_feishu_client() -> Optional[lark.Client]:
    return _feishu_client


def _build_lark_client(app_id: str, app_secret: str, domain: str) -> lark.Client:
    domain_map = {"feishu": lark.FEISHU_DOMAIN, "lark": lark.LARK_DOMAIN}
    d = domain_map.get(domain, lark.FEISHU_DOMAIN)
    return lark.Client.builder().app_id(app_id).app_secret(app_secret).domain(d).build()


async def get_bot_info(app_id: str, app_secret: str, domain: str) -> Optional[dict]:
    global _bot_info
    if _feishu_channel and _feishu_channel._bot_info:
        return _feishu_channel._bot_info
    if _bot_info:
        return _bot_info
    try:
        api_base = _resolve_api_base(domain)
        token = await _get_tenant_token(app_id, app_secret, domain)
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
            resp = await api_request_with_retry(
                lambda: client.get(
                    f"{api_base}/bot/v3/info/",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=10,
                ),
                description="Get bot info",
            )
        data = resp.json()
        if data.get("code") == 0 and data.get("bot"):
            bot = data["bot"]
            info = {"open_id": bot.get("open_id", ""), "name": bot.get("app_name", "")}
            if _feishu_channel:
                _feishu_channel._bot_info = info
            else:
                _bot_info = info
            logger.info("Bot identity: %s (%s)", info["name"], info["open_id"])
            return info
    except Exception as e:
        logger.error("Failed to get bot info: %s", e)
    return None


def _parse_text_content(content: str) -> str:
    try:
        data = json.loads(content)
        return data.get("text", "")
    except (json.JSONDecodeError, TypeError):
        return str(content)


def _parse_post_content(content: str) -> str:
    try:
        data = json.loads(content)
        parts = []
        for locale, body in data.items():
            if isinstance(body, dict) and "content" in body:
                for line in body["content"]:
                    for elem in line:
                        tag = elem.get("tag", "")
                        if tag == "text":
                            parts.append(elem.get("text", ""))
                        elif tag == "a":
                            parts.append(f"[{elem.get('text', elem.get('href', ''))}]({elem.get('href', '')})")
                        elif tag == "at":
                            parts.append(f"@{elem.get('user_name', elem.get('user_id', ''))}")
        return "".join(parts)
    except (json.JSONDecodeError, TypeError):
        return str(content)


def _parse_message_content(msg_type: str, content: str) -> str:
    if msg_type == "text":
        return _parse_text_content(content)
    elif msg_type == "post":
        return _parse_post_content(content)
    elif msg_type == "interactive":
        try:
            data = json.loads(content)
            elements = []
            if "header" in data:
                elements.append(data["header"].get("title", {}).get("tag", ""))
            for elem in data.get("elements", []):
                if "text" in elem:
                    elements.append(elem["text"].get("content", elem["text"].get("tag", "")))
                elif "markdown" in elem:
                    elements.append(elem["markdown"].get("content", ""))
            return "\n".join(elements)
        except (json.JSONDecodeError, KeyError, TypeError):
            return str(content)
    return _parse_media_content(msg_type, content)


def _parse_image_content(content: str) -> str:
    try:
        data = json.loads(content)
        image_key = data.get("image_key", "")
        return f"[image: {image_key}]" if image_key else "[image]"
    except (json.JSONDecodeError, TypeError):
        return "[image]"


def _parse_file_content(content: str) -> str:
    try:
        data = json.loads(content)
        filename = data.get("file_name", data.get("file_key", ""))
        return f"[file: {filename}]" if filename else "[file]"
    except (json.JSONDecodeError, TypeError):
        return "[file]"


def _parse_media_content(msg_type: str, content: str) -> str:
    if msg_type == "image":
        return _parse_image_content(content)
    elif msg_type == "file":
        return _parse_file_content(content)
    elif msg_type in ("audio", "video", "sticker"):
        return f"[{msg_type}]"
    return f"[{msg_type}] {content[:200]}"


async def _process_media_message(
    msg_type: str,
    content: str,
    message_id: str,
    feishu_client,
    mu_runner,
) -> str:
    """Download and understand a media message. Returns description text or empty string.

    Feishu message types:
      image  -> content has {"image_key": "img_v3_xxx"}
      audio  -> content has {"file_key": "file_v3_xxx"}
      video  -> content has {"file_key": "file_v3_xxx"}
      file   -> content has {"file_key": "file_v3_xxx", "file_name": "doc.pdf"}
    """
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return ""

    try:
        if msg_type == "image":
            image_key = data.get("image_key", "")
            if not image_key or not mu_runner:
                return ""
            from src.channels.media import download_image
            from src.media_understanding.types import MediaCapability

            image_bytes = await download_image(feishu_client, image_key)
            if not image_bytes:
                return ""
            result = await mu_runner.understand(image_bytes, MediaCapability.IMAGE, mime_type="image/png")
            if result.text:
                return f"\n[📎 图片理解]\n{result.text}"
            if result.error:
                return f"\n[⚠️ 图片理解失败: {result.error}]"
            return ""

        elif msg_type in ("audio", "video", "file"):
            file_key = data.get("file_key", "")
            if not file_key or not mu_runner:
                return ""
            from src.channels.media import download_message_resource
            from src.media_understanding.types import MediaCapability
            from src.media_understanding.runner import MediaUnderstandingRunner

            media_bytes = await download_message_resource(feishu_client, message_id, file_key)
            if not media_bytes:
                return ""

            cap = None
            mime = ""
            if msg_type == "audio":
                cap = MediaCapability.AUDIO
                mime = "audio/mp3"
            elif msg_type == "video":
                cap = MediaCapability.VIDEO
                mime = "video/mp4"
            elif msg_type == "file":
                filename = data.get("file_name", "")
                if filename:
                    cap = MediaUnderstandingRunner.guess_capability_from_ext(filename)
                if cap is None:
                    return ""

            if cap is None:
                return ""

            result = await mu_runner.understand(media_bytes, cap, mime_type=mime)
            label = {
                MediaCapability.IMAGE: "图片理解",
                MediaCapability.AUDIO: "语音转录",
                MediaCapability.VIDEO: "视频理解",
            }.get(cap, "媒体理解")
            if result.text:
                return f"\n[📎 {label}]\n{result.text}"
            if result.error:
                return f"\n[⚠️ {label}失败: {result.error}]"
            return ""

    except Exception as e:
        logger.warning("Media processing failed (type=%s): %s", msg_type, e)

    return ""


def _extract_mentions(event_data: dict, bot_open_id: str) -> tuple[str, list[dict]]:
    sender = event_data.get("sender", {})
    sender_id = sender.get("sender_id", {}).get("open_id", "unknown")
    sender_type = sender.get("sender_type", "")

    message = event_data.get("message", {})
    msg_type = message.get("message_type", "text")
    content = message.get("content", "{}")
    chat_id = message.get("chat_id", "")
    chat_type = message.get("chat_type", "p2p")
    message_id = message.get("message_id", "")

    mentions = []
    raw_mentions = event_data.get("message", {}).get("mentions", []) or []
    bot_mentioned = False
    for m in raw_mentions:
        mentions.append(
            {
                "key": m.get("key", ""),
                "id": m.get("id", {}).get("open_id", ""),
                "name": m.get("name", ""),
            }
        )
        if m.get("id", {}).get("open_id") == bot_open_id:
            bot_mentioned = True

    text = _parse_message_content(msg_type, content)

    if chat_type == "group" and bot_mentioned:
        text = re.sub(r"@_user_\d+\s*", "", text).strip()

    return text, {
        "sender_id": sender_id,
        "sender_type": sender_type,
        "chat_id": chat_id,
        "chat_type": chat_type,
        "message_id": message_id,
        "bot_mentioned": bot_mentioned,
        "mentions": mentions,
    }


class FeishuChannel(Channel):
    def __init__(self, config):
        super().__init__()
        global _feishu_client, _feishu_channel
        self.config = config
        _feishu_client = _build_lark_client(config.app_id, config.app_secret, config.domain)
        self.client = _feishu_client
        _feishu_channel = self
        self._running = False
        self._chat_queues: dict[str, asyncio.Queue] = {}
        self._streaming_sessions: dict[str, StreamingCardSession] = {}
        self._on_message_callback: Optional[Callable] = None
        self._bot_info: Optional[dict] = None
        self._mu_runner = None
        self._ws_client: Optional[WsClient] = None
        self._ws_handler = None
        self._ws_domain_url = None
        self._main_loop = None

    def set_message_callback(self, callback: Callable):
        self._on_message_callback = callback

    async def start(self):
        if not self.config.enabled:
            logger.info("Feishu channel disabled")
            return

        # Wrap async callback for sync SDK
        self._main_loop = asyncio.get_running_loop()

        def _sync_on_message(data):
            try:
                future = asyncio.run_coroutine_threadsafe(self._on_message(data), self._main_loop)

                def _done(fut):
                    try:
                        fut.result()
                    except Exception as exc:
                        logger.error("Message handler exception: %s", exc, exc_info=True)

                future.add_done_callback(_done)
            except Exception as e:
                logger.error("Message callback error: %s", e, exc_info=True)

        bot = await get_bot_info(self.config.app_id, self.config.app_secret, self.config.domain)
        if not bot:
            logger.error("Cannot start Feishu: bot identity check failed")
            return

        handler = EventDispatcherHandlerBuilder("", "").register_p2_im_message_receive_v1(_sync_on_message).build()

        domain_url = lark.FEISHU_DOMAIN if self.config.domain != "lark" else lark.LARK_DOMAIN
        self._ws_handler = handler
        self._ws_domain_url = domain_url

        def _create_ws_client():
            return WsClient(
                self.config.app_id,
                self.config.app_secret,
                event_handler=self._ws_handler,
                domain=self._ws_domain_url,
                log_level=lark.LogLevel.WARNING,
            )

        self._ws_client = _create_ws_client()
        # Log unhandled event types, suppress noise
        _lark_logger = logging.getLogger("Lark")
        _lark_logger.handlers.clear()
        _lark_logger.propagate = False

        # Rate-limit connection error logs to prevent log spam during network outages
        _conn_error_state = {"count": 0, "last_log": 0.0}

        class _EventFilter(logging.Filter):
            _SUPPRESSED = ("connect failed", "receive message loop exit", "processor not found")

            def filter(self, record):
                msg = record.getMessage()
                if any(s in msg for s in self._SUPPRESSED):
                    now = _time.monotonic()
                    _conn_error_state["count"] += 1
                    # Log first occurrence, then at most once per 60 seconds
                    if now - _conn_error_state["last_log"] >= 60:
                        _conn_error_state["last_log"] = now
                        suppressed = _conn_error_state["count"] - 1
                        if suppressed > 0:
                            record.msg = f"{msg} ({suppressed} similar errors suppressed)"
                            record.args = ()
                        _conn_error_state["count"] = 0
                        return True
                    return False
                return True

        _fh = logging.StreamHandler()
        _fh.setFormatter(logging.Formatter("[lark] %(message)s"))
        _fh.setLevel(logging.ERROR)
        _fh.addFilter(_EventFilter())
        _lark_logger.addHandler(_fh)
        _lark_logger.setLevel(logging.WARNING)
        self._running = True

        _WS_BACKOFF = [1, 2, 5, 10, 30, 60, 120]

        def _start_ws():
            backoff_idx = 0
            while self._running:
                try:
                    self._ws_client = _create_ws_client()
                    self._ws_client.start()
                    backoff_idx = 0
                except Exception as e:
                    if not self._running:
                        break
                    delay = _WS_BACKOFF[min(backoff_idx, len(_WS_BACKOFF) - 1)]
                    logger.warning("Feishu WS disconnected, reconnect in %ds: %s", delay, e)
                    import time
                    time.sleep(delay)
                    backoff_idx += 1
            self._running = False

        import threading

        ws_thread = threading.Thread(target=_start_ws, daemon=True)
        ws_thread.start()

        logger.info("Feishu WebSocket channel started (bot: %s)", bot["name"])

    async def _on_message(self, event: Any):
        try:
            if not self._on_message_callback:
                return
            if not event.event or not event.event.message:
                return
        except Exception as e:
            logger.error("Message parsing error: %s", e, exc_info=True)
            return

        msg = event.event.message
        sender = event.event.sender
        message_id = msg.message_id
        if not message_id:
            return

        if await self.check_dedup(message_id):
            return

        bot = await get_bot_info(self.config.app_id, self.config.app_secret, self.config.domain)
        bot_open_id = bot["open_id"] if bot else ""

        sender_id = sender.sender_id.open_id if sender and sender.sender_id else "unknown"
        chat_id = msg.chat_id
        chat_type = msg.chat_type or "p2p"
        msg_type = msg.message_type or "text"
        content = msg.content or "{}"

        mentions = msg.mentions or []
        mention_list = []
        bot_mentioned = False
        for m in mentions:
            mid = m.id.open_id if m.id else ""
            mention_list.append({"key": m.key or "", "id": mid, "name": m.name or ""})
            if mid == bot_open_id:
                bot_mentioned = True

        text = _parse_message_content(msg_type, content)

        # Auto-process media attachments (image/audio/video/file)
        if self._mu_runner and msg_type in ("image", "audio", "video", "file"):
            media_desc = await _process_media_message(msg_type, content, message_id, self.client, self._mu_runner)
            if media_desc:
                text = media_desc
            if not text.strip():
                text = f"[{msg_type} message received, understanding returned empty]"

        if chat_type == "group" and bot_mentioned:
            text = re.sub(r"@_user_\d+\s*", "", text).strip()
        if not text.strip():
            return

        meta = {
            "sender_id": sender_id,
            "chat_id": chat_id,
            "chat_type": chat_type,
            "message_id": message_id,
            "bot_mentioned": bot_mentioned,
            "mentions": mention_list,
        }

        if chat_type == "group":
            if self.config.group_policy == "disabled":
                return
            if self.config.group_policy == "allowlist":
                if chat_id not in (self.config.group_allow_from or []):
                    return
            if self.config.require_mention and not bot_mentioned:
                return
        elif chat_type == "p2p":
            if self.config.dm_policy == "allowlist":
                if sender_id not in (self.config.allow_from or []):
                    return

        logger.info(
            "Feishu message: chat=%s sender=%s type=%s text=%.100s (len=%d)",
            chat_id,
            sender_id,
            chat_type,
            text,
            len(text),
        )

        await self._on_message_callback(
            text=text,
            sender_id=sender_id,
            chat_id=chat_id,
            chat_type=chat_type,
            message_id=message_id,
            reply_fn=lambda t: self.send_text(chat_id, t, message_id),
            stream_fn=self._create_stream_sender(chat_id, message_id),
        )

    async def send_text(
        self,
        chat_id: str,
        text: str,
        reply_to: Optional[str] = None,
    ):
        chunks = self.chunk_text(text, 4000)
        for chunk in chunks:
            await self._do_send(chat_id, chunk, reply_to=reply_to)
            if len(chunks) > 1:
                await asyncio.sleep(0.1)

    async def _do_send(
        self,
        chat_id: str,
        text: str,
        msg_type: str = "text",
        reply_to: Optional[str] = None,
    ):
        from src.tools.feishu_tools import _lark_call_with_retry

        if msg_type == "text":
            content = json.dumps({"text": text}, ensure_ascii=False)
        else:
            content = text

        try:
            if reply_to:
                body = CreateMessageRequestBody.builder().msg_type(msg_type).content(content).build()
                req = ReplyMessageRequest.builder().message_id(reply_to).request_body(body).build()
                resp = await _lark_call_with_retry(
                    lambda req=req: self.client.im.v1.message.reply(req),
                    description="Feishu reply message",
                )
            else:
                body = (
                    CreateMessageRequestBody.builder().receive_id(chat_id).msg_type(msg_type).content(content).build()
                )
                req = CreateMessageRequest.builder().receive_id_type("chat_id").request_body(body).build()
                resp = await _lark_call_with_retry(
                    lambda req=req: self.client.im.v1.message.create(req),
                    description="Feishu send message",
                )
            if not resp.success():
                logger.error("Send message failed: %s %s", resp.code, resp.msg)
                return None
            return resp.data
        except Exception as e:
            logger.error("Send message error: %s", e)
            return None

    async def send_reply(
        self,
        chat_id: str,
        text: str,
        message_id: str,
    ):
        from src.tools.feishu_tools import _lark_call_with_retry

        try:
            content = json.dumps({"text": text}, ensure_ascii=False)
            body = CreateMessageRequestBody.builder().msg_type("text").content(content).build()
            req = ReplyMessageRequest.builder().message_id(message_id).request_body(body).build()
            resp = await _lark_call_with_retry(
                lambda req=req: self.client.im.v1.message.reply(req),
                description="Feishu reply message",
            )
            if resp.success():
                return resp.data
            logger.warning("Reply failed (%s %s), falling back to send_text", resp.code, resp.msg)
            await self.send_text(chat_id, text)
            return None
        except Exception as e:
            logger.error("Reply error: %s", e)
            await self.send_text(chat_id, text)
            return None

    def _create_stream_sender(self, chat_id: str, message_id: str):
        session_key = f"{chat_id}:{message_id}"
        session_started = False
        fallback_accumulated: list[str] = []
        lock = asyncio.Lock()

        async def stream_fn(delta: str, done: bool = False, flush: bool = False):
            nonlocal session_started

            if not self.config.streaming:
                fallback_accumulated.append(delta)
                if done:
                    full = "".join(fallback_accumulated)
                    if full.strip():
                        await self.send_text(chat_id, full)
                    fallback_accumulated.clear()
                return

            async with lock:
                # Clean up stale streaming sessions (TTL: 10 minutes)
                now = _time.monotonic()
                stale_keys = [k for k, v in self._streaming_sessions.items() if now - v._last_update > 600]
                for k in stale_keys:
                    try:
                        await self._streaming_sessions[k].close()
                    except Exception:
                        pass
                    del self._streaming_sessions[k]

                session = self._streaming_sessions.get(session_key)

                if not session and not session_started and not done:
                    session_started = True
                    session = StreamingCardSession(
                        self.config.app_id,
                        self.config.app_secret,
                        self.config.domain,
                    )
                    card_msg_id = await session.start(self.client, chat_id, reply_to=message_id)
                    if card_msg_id is None:
                        session = None
                    else:
                        self._streaming_sessions[session_key] = session

                if session and delta:
                    await session.update(delta, flush=flush)

                if done:
                    if session:
                        final_text = "".join(fallback_accumulated) + delta
                        await session.close(final_text if final_text.strip() else None)
                        self._streaming_sessions.pop(session_key, None)
                    elif fallback_accumulated:
                        full = "".join(fallback_accumulated)
                        if full.strip():
                            await self.send_text(chat_id, full)
                        fallback_accumulated.clear()

        return stream_fn

    async def send_image(self, chat_id: str, image_key: str) -> bool:
        content = json.dumps({"image_key": image_key})
        result = await self._do_send(chat_id, content, msg_type="image")
        return result is not None

    async def send_file(self, chat_id: str, file_key: str) -> bool:
        content = json.dumps({"file_key": file_key})
        result = await self._do_send(chat_id, content, msg_type="file")
        return result is not None

    async def send_audio(self, chat_id: str, audio_bytes: bytes, file_name: str = "speech.mp3") -> bool:
        """Upload audio bytes and send as audio message to chat.

        Uses native HTTP calls to Feishu IM file API to avoid lark SDK blocking issues.
        """
        api_base = _resolve_api_base(self.config.domain)
        token = await _get_tenant_token(self.config.app_id, self.config.app_secret, self.config.domain)
        if not token:
            logger.error("send_audio: failed to get tenant token")
            return False

        headers = {"Authorization": f"Bearer {token}"}

        # Step 1: Upload file via IM file API (native HTTP, no lark SDK)
        import io

        form_data = {
            "file_type": (None, "stream"),
            "file_name": (None, file_name),
            "file": (file_name, io.BytesIO(audio_bytes), "application/octet-stream"),
        }

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as hc:
                upload_resp = await feishu_api_request(
                    lambda: hc.post(f"{api_base}/im/v1/files", headers=headers, files=form_data),
                    description="Audio file upload",
                )
        except Exception as e:
            logger.error("send_audio: upload failed: %s", e)
            return False

        if upload_resp is None or upload_resp.status_code >= 400:
            logger.error("send_audio: upload HTTP %s", upload_resp.status_code if upload_resp else "no response")
            return False

        data = upload_resp.json()
        file_key = data.get("data", {}).get("file_key", "")
        if not file_key:
            logger.error("send_audio: no file_key in response: %s", data.get("msg", ""))
            return False

        # Step 2: Send as file message (stream uploads can only be sent as "file" type, not "audio")
        content = json.dumps({"file_key": file_key})
        result = await self._do_send(chat_id, content, msg_type="file")
        return result is not None

    async def send_card(
        self,
        chat_id: str,
        card_content: str,
        reply_to: Optional[str] = None,
    ) -> Any:
        """Send an interactive card to a chat."""
        return await self._do_send(chat_id, card_content, msg_type="interactive", reply_to=reply_to)

    async def send_approval_card(
        self,
        chat_id: str,
        request_id: str,
        command_preview: str,
        denylisted: bool = False,
    ) -> Optional[str]:
        api_base = _resolve_api_base(self.config.domain)
        token = await _get_tenant_token(self.config.app_id, self.config.app_secret, self.config.domain)

        warning = "⚠️ DANGEROUS COMMAND" if denylisted else "Command requires approval"
        card_json = {
            "schema": "2.0",
            "config": {"streaming_mode": False},
            "header": {
                "title": {"tag": "plain_text", "content": "🔐 Approval Required"},
                "template": "red" if denylisted else "orange",
            },
            "body": {
                "elements": [
                    {"tag": "markdown", "content": f"**{warning}**\n```\n{command_preview}\n```"},
                    {
                        "tag": "action",
                        "actions": [
                            {
                                "tag": "button",
                                "text": {"tag": "plain_text", "content": "✅ Allow Once"},
                                "type": "primary",
                                "value": json.dumps({"request_id": request_id, "decision": "allow_once"}),
                                "url": "",
                            },
                            {
                                "tag": "button",
                                "text": {"tag": "plain_text", "content": "✅ Allow Always"},
                                "type": "primary",
                                "value": json.dumps({"request_id": request_id, "decision": "allow_always"}),
                                "url": "",
                            },
                            {
                                "tag": "button",
                                "text": {"tag": "plain_text", "content": "❌ Deny"},
                                "type": "danger",
                                "value": json.dumps({"request_id": request_id, "decision": "deny"}),
                                "url": "",
                            },
                        ],
                    },
                ]
            },
        }

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
                resp = await api_request_with_retry(
                    lambda: client.post(
                        f"{api_base}/cardkit/v1/cards",
                        headers={
                            "Authorization": f"Bearer {token}",
                            "Content-Type": "application/json",
                        },
                        json={"type": "card_json", "data": json.dumps(card_json)},
                        timeout=10,
                    ),
                    description="Create approval card",
                )
            data = resp.json()
            if data.get("code") != 0 or not data.get("data", {}).get("card_id"):
                logger.error("Create approval card failed: %s", data.get("msg"))
                return None

            card_id = data["data"]["card_id"]
            card_content = json.dumps({"type": "card", "data": {"card_id": card_id}})
            await self._do_send(chat_id, card_content, msg_type="interactive")
            return card_id
        except Exception as e:
            logger.error("Send approval card failed: %s", e)
            return None

    async def stop(self):
        self._running = False
        self._ws_client = None
        logger.info("Feishu channel stopped")
