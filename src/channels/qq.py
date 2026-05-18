"""QQ Bot channel implementation for MyClaw.

Uses the official QQ Bot API (WebSocket + HTTP) for real-time messaging.
Supports C2C (direct), group, and guild channel messages.
Supports markdown, typing indicator, audio/TTS, local file upload, 429 retry.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time as _time
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

from .base import Channel, api_request_with_retry

logger = logging.getLogger("myclaw.qq")

# Module-level singleton for tool access
_qq_channel: Optional[QQChannel] = None  # forward ref; set in QQChannel.__init__


def get_qq_channel() -> Optional[QQChannel]:
    """Get the global QQChannel instance (for tool access)."""
    return _qq_channel


# --- QQ Bot API constants ---
API_BASE = "https://api.sgroup.qq.com"
TOKEN_URL = "https://bots.qq.com/app/getAppAccessToken"
USER_AGENT = "MyClawQQBot/0.1.0"

# WebSocket opcodes
OP_DISPATCH = 0
OP_HEARTBEAT = 1
OP_IDENTIFY = 2
OP_RESUME = 6
OP_RECONNECT = 7
OP_INVALID_SESSION = 9
OP_HELLO = 10
OP_HEARTBEAT_ACK = 11

# Intent bits
INTENT_PUBLIC_GUILD_MESSAGES = 1 << 30
INTENT_DIRECT_MESSAGE = 1 << 12
INTENT_GROUP_AND_C2C = 1 << 25
FULL_INTENTS = INTENT_PUBLIC_GUILD_MESSAGES | INTENT_DIRECT_MESSAGE | INTENT_GROUP_AND_C2C

# Reconnect config
_RECONNECT_DELAYS = [1, 2, 5, 10, 30, 60]
_MAX_RECONNECT_ATTEMPTS = 100
_MSG_SEQ_MAX = 65536


# ---------------------------------------------------------------------------
# Token Manager
# ---------------------------------------------------------------------------


class _QQTokenManager:
    """Manages QQ Bot access token with auto-refresh."""

    def __init__(self, app_id: str, client_secret: str):
        self._app_id = app_id
        self._client_secret = client_secret
        self._token: Optional[str] = None
        self._expires_at: float = 0.0
        self._lock = asyncio.Lock()
        self._refresh_task: Optional[asyncio.Task] = None
        self._running = False

    async def get_token(self) -> str:
        async with self._lock:
            if self._token and _time.time() < self._expires_at - 300:
                return self._token
            return await self._fetch_token()

    async def _fetch_token(self) -> str:
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
            resp = await api_request_with_retry(
                lambda: client.post(
                    TOKEN_URL,
                    json={"appId": self._app_id, "clientSecret": self._client_secret},
                ),
                description="QQ token fetch",
            )
        try:
            data = resp.json()
        except Exception as e:
            raise RuntimeError(f"QQ token response parse error: {e}") from e
        token = data.get("access_token")
        if not token:
            raise RuntimeError(f"QQ token error: {data}")
        expires_in = int(data.get("expires_in", 7200))
        self._token = token
        self._expires_at = _time.time() + expires_in
        logger.debug("QQ token refreshed, expires in %ds", expires_in)
        return token

    def clear_cache(self):
        self._token = None
        self._expires_at = 0.0

    def start_background_refresh(self):
        self._running = True
        self._refresh_task = asyncio.create_task(self._refresh_loop())

    async def _refresh_loop(self):
        while self._running:
            try:
                await asyncio.sleep(115 * 60)
                await self.get_token()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Token refresh failed: %s", e)
                await asyncio.sleep(60)

    def stop_background_refresh(self):
        self._running = False
        if self._refresh_task and not self._refresh_task.done():
            self._refresh_task.cancel()


# ---------------------------------------------------------------------------
# WebSocket Client
# ---------------------------------------------------------------------------


class _QQWebSocketClient:
    """Manages the QQ Bot WebSocket gateway connection."""

    def __init__(
        self,
        app_id: str,
        token_manager: _QQTokenManager,
        on_dispatch: Callable,
        log: logging.Logger,
    ):
        self._app_id = app_id
        self._token_mgr = token_manager
        self._on_dispatch = on_dispatch
        self._log = log
        self._ws = None
        self._session_id: Optional[str] = None
        self._last_seq: Optional[int] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._recv_task: Optional[asyncio.Task] = None
        self._running = False
        self._reconnect_attempts = 0
        self._reconnecting = False  # Guard against concurrent reconnect loops

    async def connect(self):
        self._running = True
        token = await self._token_mgr.get_token()

        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
            resp = await api_request_with_retry(
                lambda: client.get(
                    f"{API_BASE}/gateway",
                    headers={"Authorization": f"QQBot {token}"},
                ),
                description="QQ gateway fetch",
            )
        data = resp.json()
        ws_url = data.get("url")
        if not ws_url:
            raise RuntimeError(f"Failed to get QQ gateway URL: {data}")

        import websockets

        # SSL: try certifi CA bundle first, fall back to no-verify on Windows
        ssl_ctx = None
        try:
            import certifi
            import ssl
            ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            pass

        try:
            connect_kwargs = {
                "user_agent_header": USER_AGENT,
                "ping_interval": None,
            }
            if ssl_ctx:
                connect_kwargs["ssl"] = ssl_ctx
            self._ws = await websockets.connect(ws_url, **connect_kwargs)
        except Exception as e:
            is_ssl_error = "ssl" in str(e).lower() or "certificate" in str(e).lower()
            if is_ssl_error:
                self._log.warning("SSL cert verification failed, retrying without verify")
                try:
                    import ssl
                    no_verify = ssl.create_default_context()
                    no_verify.check_hostname = False
                    no_verify.verify_mode = 0  # ssl.CERT_NONE
                    self._ws = await websockets.connect(
                        ws_url,
                        user_agent_header=USER_AGENT,
                        ping_interval=None,
                        ssl=no_verify,
                    )
                except Exception as e2:
                    self._log.error("WebSocket connect failed (no-verify): %s", e2)
                    await self._schedule_reconnect()
                    return
            else:
                self._log.error("WebSocket connect failed: %s", e)
                await self._schedule_reconnect()
                return

        try:
            raw = await asyncio.wait_for(self._ws.recv(), timeout=30)
            hello = json.loads(raw)
        except Exception as e:
            self._log.error("Failed to receive HELLO: %s", e)
            await self._cleanup()
            await self._schedule_reconnect()
            return

        if hello.get("op") != OP_HELLO:
            self._log.error("Expected HELLO, got op=%s", hello.get("op"))
            await self._cleanup()
            await self._schedule_reconnect()
            return

        heartbeat_interval = hello.get("d", {}).get("heartbeat_interval", 41250)

        if self._session_id and self._last_seq is not None:
            await self._ws.send(
                json.dumps(
                    {
                        "op": OP_RESUME,
                        "d": {
                            "token": f"QQBot {token}",
                            "session_id": self._session_id,
                            "seq": self._last_seq,
                        },
                    }
                )
            )
            self._log.info("Resuming QQ session %s", self._session_id)
        else:
            await self._ws.send(
                json.dumps(
                    {
                        "op": OP_IDENTIFY,
                        "d": {
                            "token": f"QQBot {token}",
                            "intents": FULL_INTENTS,
                            "shard": [0, 1],
                        },
                    }
                )
            )
            self._log.info("Identified with QQ gateway")

        # Cancel stale tasks from previous connection before creating new ones
        for old_task in (self._heartbeat_task, self._recv_task):
            if old_task and not old_task.done():
                old_task.cancel()
        self._heartbeat_task = asyncio.create_task(self._heartbeat(heartbeat_interval))
        self._recv_task = asyncio.create_task(self._recv_loop())
        self._reconnect_attempts = 0

    async def _heartbeat(self, interval_ms: int):
        interval = interval_ms / 1000.0
        while self._running:
            try:
                await asyncio.sleep(interval)
                if self._ws and self._running:
                    payload = {"op": OP_HEARTBEAT, "d": self._last_seq}
                    await self._ws.send(json.dumps(payload))
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._log.debug("Heartbeat error: %s", e)
                break

    async def _recv_loop(self):
        import websockets

        try:
            async for raw in self._ws:
                if not self._running:
                    break
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                op = msg.get("op")
                t = msg.get("t")
                d = msg.get("d")
                s = msg.get("s")

                if s is not None:
                    self._last_seq = s

                if op == OP_DISPATCH:
                    if t == "READY":
                        self._session_id = d.get("session_id") if d else None
                        self._log.info("QQ gateway READY, session=%s", self._session_id)
                    try:
                        await self._on_dispatch(t, d if d else {})
                    except Exception as e:
                        self._log.error("Dispatch handler error for %s: %s", t, e)

                elif op == OP_HEARTBEAT_ACK:
                    pass

                elif op == OP_RECONNECT:
                    self._log.warning("QQ gateway requested reconnect")
                    await self._cleanup()
                    await self._schedule_reconnect()

                elif op == OP_INVALID_SESSION:
                    can_resume = d if isinstance(d, bool) else False
                    if not can_resume:
                        self._session_id = None
                        self._last_seq = None
                        self._token_mgr.clear_cache()
                    await self._cleanup()
                    await self._schedule_reconnect(3)

        except websockets.ConnectionClosed as e:
            self._log.warning("WebSocket closed: code=%s reason=%s", e.code, e.reason)
            if self._running:
                await self._cleanup()
                await self._schedule_reconnect()
        except Exception as e:
            self._log.error("Recv loop error: %s", e)
            if self._running:
                await self._cleanup()
                await self._schedule_reconnect()

    async def _cleanup(self):
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        self._heartbeat_task = None

        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def _schedule_reconnect(self, custom_delay: Optional[float] = None):
        if not self._running:
            return
        if self._reconnecting:
            return  # Already reconnecting
        self._reconnecting = True
        try:
            while self._running:
                self._reconnect_attempts += 1
                if self._reconnect_attempts > _MAX_RECONNECT_ATTEMPTS:
                    self._log.error("Max reconnect attempts reached, giving up")
                    return

                idx = min(self._reconnect_attempts - 1, len(_RECONNECT_DELAYS) - 1)
                delay = custom_delay if custom_delay is not None else _RECONNECT_DELAYS[idx]
                custom_delay = None
                self._log.info("Reconnecting in %.1fs (attempt %d)", delay, self._reconnect_attempts)
                await asyncio.sleep(delay)

                try:
                    await self.connect()
                    return
                except Exception as e:
                    self._log.error("Reconnect failed: %s", e)
        finally:
            self._reconnecting = False

    async def stop(self):
        self._running = False
        await self._cleanup()
        if self._recv_task and not self._recv_task.done():
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
        self._recv_task = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_chat_id(chat_id: str) -> tuple[str, str]:
    """Parse chat_id like 'c2c:openid' -> ('c2c', 'openid')."""
    if ":" in chat_id:
        prefix, id_part = chat_id.split(":", 1)
        return prefix, id_part
    return "c2c", chat_id


# ---------------------------------------------------------------------------
# QQChannel
# ---------------------------------------------------------------------------


class QQChannel(Channel):
    """QQ Bot channel implementation."""

    def __init__(self, config):
        super().__init__()
        global _qq_channel
        self.config = config
        self._running = False
        self._on_message_callback: Optional[Callable] = None
        self._token_manager: Optional[_QQTokenManager] = None
        self._ws_client: Optional[_QQWebSocketClient] = None
        self._http_client: Optional[httpx.AsyncClient] = None
        self._seq_counter = 0
        self._typing_unsupported = False
        # Typing indicator state: {chat_id -> Task}
        self._typing_tasks: dict[str, asyncio.Task] = {}
        self._mu_runner = None  # Media understanding runner for audio transcription
        _qq_channel = self

    def set_message_callback(self, callback: Callable):
        self._on_message_callback = callback

    def set_media_understanding_runner(self, runner):
        self._mu_runner = runner

    async def start(self):
        if not self.config.enabled:
            logger.info("QQ channel disabled")
            return

        self._token_manager = _QQTokenManager(self.config.app_id, self.config.client_secret)
        for _attempt in range(3):
            try:
                await self._token_manager.get_token()
                break
            except Exception as e:
                if _attempt >= 2:
                    logger.error("QQ Bot auth failed after 3 attempts: %s", e)
                    return
                wait = 2 ** _attempt
                logger.warning("QQ Bot auth attempt %d failed, retry in %ds: %s", _attempt + 1, wait, e)
                await asyncio.sleep(wait)

        self._token_manager.start_background_refresh()
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            transport=httpx.AsyncHTTPTransport(retries=1),
        )

        self._ws_client = _QQWebSocketClient(
            self.config.app_id,
            self._token_manager,
            self._on_dispatch,
            logger,
        )
        self._running = True

        async def _start_ws():
            try:
                await self._ws_client.connect()
            except Exception as e:
                logger.error("QQ WebSocket connect failed: %s", e)

        asyncio.create_task(_start_ws())
        logger.info("QQ Bot channel starting...")

    async def stop(self):
        self._running = False
        # Cancel all typing tasks
        for task in self._typing_tasks.values():
            if not task.done():
                task.cancel()
        self._typing_tasks.clear()
        if self._ws_client:
            await self._ws_client.stop()
        if self._token_manager:
            self._token_manager.stop_background_refresh()
        if self._http_client:
            await self._http_client.aclose()
        logger.info("QQ channel stopped")

    # --- Dispatch handler ---

    async def _on_dispatch(self, event_type: str, data: dict):
        logger.debug("QQ dispatch: %s", event_type)
        if not self._on_message_callback:
            return

        if event_type == "C2C_MESSAGE_CREATE":
            logger.debug("QQ C2C raw data: %s", {k: v for k, v in data.items() if k not in ("author",)})
            author = data.get("author", {})
            sender_id = author.get("user_openid", "unknown")
            content = data.get("content", "")
            message_id = data.get("id", "")
            chat_type = "p2p"
            chat_id = f"c2c:{sender_id}"

        elif event_type == "GROUP_AT_MESSAGE_CREATE":
            author = data.get("author", {})
            sender_id = author.get("member_openid", "unknown")
            content = data.get("content", "")
            message_id = data.get("id", "")
            group_openid = data.get("group_openid", "")
            chat_type = "group"
            chat_id = f"group:{group_openid}"

        elif event_type == "AT_MESSAGE_CREATE":
            author = data.get("author", {})
            sender_id = author.get("id", author.get("user_openid", "unknown"))
            content = data.get("content", "")
            message_id = data.get("id", "")
            channel_id = data.get("channel_id", "")
            chat_type = "group"  # guild channel maps to group
            chat_id = f"channel:{channel_id}"

        elif event_type == "DIRECT_MESSAGE_CREATE":
            author = data.get("author", {})
            sender_id = author.get("id", author.get("user_openid", "unknown"))
            content = data.get("content", "")
            message_id = data.get("id", "")
            guild_id = data.get("guild_id", "")
            chat_type = "p2p"  # guild DM maps to p2p
            chat_id = f"dm:{guild_id}"

        else:
            return

        if not message_id:
            return

        # Message dedup
        if await self.check_dedup(message_id):
            return

        # Policy checks
        if chat_type == "group":
            if self.config.group_policy == "disabled":
                logger.debug("QQ group message dropped: group_policy=disabled")
                return
            if self.config.group_policy == "allowlist":
                group_oid = chat_id.split(":", 1)[1] if ":" in chat_id else ""
                if group_oid not in (self.config.group_allow_from or []):
                    logger.debug("QQ group message dropped: group %s not in allowlist", group_oid)
                    return
        elif chat_type == "p2p":
            if self.config.dm_policy == "allowlist":
                if sender_id not in (self.config.allow_from or []):
                    logger.debug("QQ DM dropped: sender %s not in allowlist", sender_id)
                    return

        text = content.strip()

        # Handle file messages (docx, pdf, etc.) — QQ sends these with attachment info
        attachment = data.get("attachment", {})
        if attachment and not text:
            fname = attachment.get("filename", "") or attachment.get("file_name", "")
            if fname:
                text = f"[收到文件: {fname}]"
            else:
                text = "[收到文件]"

        # Process attachments
        attachments = data.get("attachments", [])
        image_urls = []
        video_urls = []
        audio_urls = []
        file_urls = []
        has_media = False
        if attachments:
            for att in attachments:
                ct = att.get("content_type", "")
                url = att.get("url", "")
                if ct.startswith("image") and url:
                    image_urls.append(url)
                elif ct.startswith("image"):
                    text += "\n[image]"
                    has_media = True
                elif ct.startswith("video") and url:
                    video_urls.append(url)
                    has_media = True
                elif ct.startswith("video"):
                    text += "\n[video]"
                    has_media = True
                elif ct.startswith("audio") and url:
                    audio_urls.append(url)
                    has_media = True
                elif ct.startswith("audio"):
                    text += "\n[audio]"
                    has_media = True
                elif ct == "file" and url:
                    fname = att.get("filename", "")
                    size = att.get("size", 0)
                    file_urls.append({"url": url, "filename": fname, "size": size})
                    has_media = True

        # Check file_info / media fields (QQ may use these for video/file messages)
        file_info = data.get("file_info", "")
        media_info = data.get("media", "")
        if file_info or media_info:
            has_media = True
            if not text:
                text = "[收到文件/媒体消息]"

        # Build media context text
        if video_urls and not text.strip():
            text = "请描述这个视频"
            for url in video_urls:
                text += f"\n[video_url: {url}]"
        elif video_urls:
            for url in video_urls:
                text += f"\n[video_url: {url}]"
        elif has_media and not text.strip():
            text = "[收到媒体消息]"

        # If only images were sent (no text), build a describe request
        if not text.strip() and image_urls:
            text = "请描述这张图片"
            if len(image_urls) > 1:
                text = f"请描述这 {len(image_urls)} 张图片"
            for url in image_urls:
                text += f"\n[image_url: {url}]"
        elif image_urls:
            for url in image_urls:
                text += f"\n[image_url: {url}]"

        # Audio transcription
        if audio_urls and self._mu_runner:
            for aurl in audio_urls:
                try:
                    async with httpx.AsyncClient(timeout=30) as dl:
                        resp = await dl.get(aurl)
                        resp.raise_for_status()
                        audio_data = resp.content
                    from src.media_understanding.types import MediaCapability
                    result = await self._mu_runner.understand(audio_data, MediaCapability.AUDIO)
                    if result.text:
                        text += f"\n[语音转文字]: {result.text}"
                    else:
                        text += "\n[语音消息（转写失败）]"
                except Exception as e:
                    logger.warning("QQ audio transcription failed: %s", e)
                    text += "\n[语音消息（转写失败）]"
        elif audio_urls:
            for aurl in audio_urls:
                text += f"\n[audio_url: {aurl}]"

        # File attachments
        if file_urls:
            for finfo in file_urls:
                fname = finfo["filename"]
                furl = finfo["url"]
                text += f"\n[收到文件: {fname} (url: {furl})]"
            if not text.strip():
                text = f"[收到文件: {file_urls[0]['filename']}]"

        logger.info(
            "QQ message: chat=%s sender=%s type=%s text=%.100s",
            chat_id,
            sender_id,
            chat_type,
            text,
        )

        # Inject chat_id for QQ send tools (so they can default to current chat)
        try:
            from src.tools.qq_tools import set_current_qq_chat_id
            set_current_qq_chat_id(chat_id)
        except ImportError:
            pass  # qq_tools not available, tools will handle missing chat_id

        # Start typing indicator for C2C
        typing_task = None
        if chat_type == "p2p":
            typing_task = asyncio.create_task(self._typing_loop(chat_id, message_id))
            self._typing_tasks[chat_id] = typing_task

        try:
            await self._on_message_callback(
                text=text,
                sender_id=sender_id,
                chat_id=chat_id,
                chat_type=chat_type,
                message_id=message_id,
                reply_fn=lambda t: self.send_text(chat_id, t, message_id),
                stream_fn=self._create_stream_sender(chat_id, message_id),
            )
        finally:
            # Stop typing
            if typing_task and not typing_task.done():
                typing_task.cancel()

    # --- Typing indicator ---

    async def _send_typing(
        self,
        chat_id: str,
        msg_id: str | None = None,
    ) -> bool:
        """Send typing indicator for C2C chat via input_notify API."""
        if self._typing_unsupported:
            return False
        kind, target = _parse_chat_id(chat_id)
        if kind != "c2c":
            return False
        if not self._http_client or not self._token_manager:
            return False
        try:
            self._seq_counter = (self._seq_counter + 1) % _MSG_SEQ_MAX
            body: dict = {
                "msg_type": 6,
                "input_notify": {
                    "input_type": 1,
                    "input_second": 60,
                },
                "msg_seq": self._seq_counter,
            }
            if msg_id:
                body["msg_id"] = msg_id
            result = await self._api_post(
                f"/v2/users/{target}/messages",
                body,
                description="QQ typing notify",
            )
            if result is None:
                return False
            return True
        except Exception as e:
            logger.debug("QQ typing error: %s", e)
            return False

    async def _typing_loop(self, chat_id: str, msg_id: str | None = None):
        """Send typing indicator every 50 seconds (QQ timeout is 60s)."""
        try:
            while True:
                await self._send_typing(chat_id, msg_id)
                await asyncio.sleep(50)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug("Typing loop error: %s", e)
        finally:
            self._typing_tasks.pop(chat_id, None)

    # --- Core HTTP request ---

    async def _api_post(
        self,
        path: str,
        body: dict,
        *,
        description: str = "QQ API",
    ) -> Optional[dict]:
        """Send authenticated POST to QQ API with 401 retry and 429 retry."""
        if not self._http_client or not self._token_manager:
            return None

        token = await self._token_manager.get_token()
        headers = {"Authorization": f"QQBot {token}", "Content-Type": "application/json"}
        url = f"{API_BASE}{path}"

        try:
            resp = await api_request_with_retry(
                lambda: self._http_client.post(url, headers=headers, json=body),
                description=description,
            )
            if resp.status_code == 401:
                self._token_manager.clear_cache()
                token = await self._token_manager.get_token()
                headers["Authorization"] = f"QQBot {token}"
                resp = await api_request_with_retry(
                    lambda: self._http_client.post(url, headers=headers, json=body),
                    description=description,
                )
            if resp.status_code >= 400:
                logger.error("%s failed: status=%d body=%s", description, resp.status_code, resp.text[:200])
                return None
            return resp.json()
        except Exception as e:
            logger.error("%s error: %s", description, e)
            return None

    # --- Message sending ---

    async def send_text(
        self,
        chat_id: str,
        text: str,
        reply_to: Optional[str] = None,
    ) -> Any:
        chunks = self.chunk_text(text, 2000)
        results = []
        for chunk in chunks:
            result = await self._send_message(chat_id, chunk, reply_to)
            results.append(result)
            if len(chunks) > 1:
                await asyncio.sleep(0.1)
        return results[-1] if results else None

    async def _send_message(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        msg_type: Optional[int] = None,
        markdown: Optional[dict] = None,
        media: Optional[dict] = None,
    ) -> Any:
        """Send a message. Determines msg_type automatically if not specified."""
        kind, target = _parse_chat_id(chat_id)
        self._seq_counter = (self._seq_counter + 1) % _MSG_SEQ_MAX

        # Determine msg_type
        if msg_type is not None:
            pass  # explicitly set
        elif media:
            msg_type = 7
        elif markdown and self.config.markdown_support:
            msg_type = 2
        else:
            msg_type = 0

        if kind == "c2c":
            body: dict = {
                "content": content,
                "msg_type": msg_type,
                "msg_seq": self._seq_counter,
            }
            if reply_to:
                body["msg_id"] = reply_to
            if media:
                body["media"] = media
            if msg_type == 2 and markdown:
                body["markdown"] = markdown
            return await self._api_post(f"/v2/users/{target}/messages", body, description="QQ send C2C")

        elif kind == "group":
            body = {
                "content": content,
                "msg_type": msg_type,
                "msg_seq": self._seq_counter,
            }
            if reply_to:
                body["msg_id"] = reply_to
            if media:
                body["media"] = media
            if msg_type == 2 and markdown:
                body["markdown"] = markdown
            return await self._api_post(f"/v2/groups/{target}/messages", body, description="QQ send group")

        elif kind == "channel":
            body = {
                "content": content,
                "msg_type": msg_type,
                "msg_seq": self._seq_counter,
            }
            if reply_to:
                body["msg_id"] = reply_to
            if media:
                body["media"] = media
            if msg_type == 2 and markdown:
                body["markdown"] = markdown
            return await self._api_post(f"/channels/{target}/messages", body, description="QQ send channel")

        elif kind == "dm":
            body = {
                "content": content,
                "msg_type": msg_type,
                "msg_seq": self._seq_counter,
            }
            if reply_to:
                body["msg_id"] = reply_to
            if media:
                body["media"] = media
            if msg_type == 2 and markdown:
                body["markdown"] = markdown
            return await self._api_post(f"/dms/{target}/messages", body, description="QQ send DM")

        else:
            logger.error("Unknown chat_id format: %s", chat_id)
            return None

    # --- Media upload ---

    async def _upload_media(
        self,
        chat_id: str,
        file_type: int,
        url: Optional[str] = None,
        file_data: Optional[str] = None,
        file_name: Optional[str] = None,
    ) -> Optional[str]:
        """Upload media and return file_info. file_type: 1=image, 2=video, 3=audio, 4=file."""
        kind, target = _parse_chat_id(chat_id)
        if kind == "c2c":
            path = f"/v2/users/{target}/files"
        elif kind == "group":
            path = f"/v2/groups/{target}/files"
        else:
            logger.warning("QQ media upload not supported for chat type: %s", kind)
            return None

        body: dict = {"file_type": file_type}
        if url:
            body["url"] = url
        if file_data:
            body["file_data"] = file_data
        if file_name:
            body["file_name"] = file_name

        result = await self._api_post(path, body, description=f"QQ upload media type={file_type}")
        if result:
            return result.get("file_info") or result.get("attachment", {}).get("file_info")
        return None

    async def _upload_local_file(
        self,
        chat_id: str,
        file_path: str,
        file_type: int,
    ) -> Optional[str]:
        """Upload a local file as base64. file_type: 1=image, 2=video, 3=audio, 4=file."""
        p = Path(file_path)
        if not p.exists():
            logger.error("File not found: %s", file_path)
            return None
        size = p.stat().st_size
        max_size = 25 * 1024 * 1024  # 25 MB
        if size > max_size:
            logger.error("File too large for base64 upload (%d MB, max %d MB): %s", size // (1024 * 1024), max_size // (1024 * 1024), file_path)
            return None
        data = p.read_bytes()
        b64 = base64.b64encode(data).decode()
        return await self._upload_media(
            chat_id,
            file_type,
            file_data=b64,
            file_name=p.name,
        )

    # --- Public send methods ---

    async def send_image(self, chat_id: str, image_key: str) -> bool:
        """Send an image. image_key can be a URL or local file path."""
        kind, _ = _parse_chat_id(chat_id)

        # Guild channels: use markdown image syntax for URLs
        if kind in ("channel", "dm"):
            if image_key.startswith(("http://", "https://")):
                return await self._send_message(chat_id, f"![]({image_key})") is not None
            return False

        # Try local file upload first
        if not image_key.startswith(("http://", "https://", "data:")):
            p = Path(image_key)
            if p.exists():
                file_info = await self._upload_local_file(chat_id, image_key, file_type=1)
                if file_info:
                    result = await self._send_message(chat_id, "", msg_type=7, media={"file_info": file_info})
                    return result is not None

        # URL-based upload
        file_info = await self._upload_media(chat_id, file_type=1, url=image_key)
        if file_info:
            result = await self._send_message(chat_id, "", msg_type=7, media={"file_info": file_info})
            return result is not None

        # Fallback: send as text with image URL
        return await self._send_message(chat_id, image_key) is not None

    async def send_file(self, chat_id: str, file_key: str) -> bool:
        """Send a file. file_key can be a URL or local file path."""
        kind, _ = _parse_chat_id(chat_id)
        if kind in ("channel", "dm"):
            logger.warning("QQ file send not supported for chat type: %s", kind)
            return False

        # Local file
        if not file_key.startswith(("http://", "https://")):
            p = Path(file_key)
            if p.exists():
                file_info = await self._upload_local_file(chat_id, file_key, file_type=4)
                if file_info:
                    result = await self._send_message(chat_id, "", msg_type=7, media={"file_info": file_info})
                    return result is not None

        # URL-based
        file_info = await self._upload_media(chat_id, file_type=4, url=file_key)
        if file_info:
            result = await self._send_message(chat_id, "", msg_type=7, media={"file_info": file_info})
            return result is not None
        return False

    async def send_audio(self, chat_id: str, audio_bytes: bytes, file_name: str = "speech.mp3") -> bool:
        """Send audio bytes as a voice message (C2C and group only)."""
        kind, _ = _parse_chat_id(chat_id)
        if kind not in ("c2c", "group"):
            logger.warning("QQ audio send not supported for chat type: %s", kind)
            return False

        b64 = base64.b64encode(audio_bytes).decode()
        file_info = await self._upload_media(
            chat_id,
            file_type=3,
            file_data=b64,
            file_name=file_name,
        )
        if file_info:
            result = await self._send_message(chat_id, "", msg_type=7, media={"file_info": file_info})
            return result is not None
        return False

    async def send_card(
        self,
        chat_id: str,
        card_content: str,
        reply_to: Optional[str] = None,
    ) -> Any:
        """Send card content. Uses markdown if supported, otherwise plain text."""
        text = card_content
        try:
            data = json.loads(card_content)
            if isinstance(data, dict):
                elements = []
                for elem in data.get("elements", []):
                    if "content" in elem:
                        elements.append(elem["content"])
                    elif "text" in elem:
                        elements.append(elem["text"].get("content", str(elem["text"])))
                if elements:
                    text = "\n".join(elements)
        except (json.JSONDecodeError, TypeError):
            pass

        if self.config.markdown_support:
            kind, _ = _parse_chat_id(chat_id)
            if kind in ("c2c", "group"):
                return await self._send_message(
                    chat_id,
                    text,
                    reply_to=reply_to,
                    msg_type=2,
                    markdown={"content": text},
                )

        return await self.send_text(chat_id, text, reply_to)

    # --- Stream sender ---

    def _create_stream_sender(self, chat_id: str, message_id: str):
        accumulated: list[str] = []

        async def stream_fn(delta: str, done: bool = False, flush: bool = False):
            if delta:
                accumulated.append(delta)
            if done:
                full = "".join(accumulated)
                if full.strip():
                    await self.send_text(chat_id, full, message_id)
                accumulated.clear()

        return stream_fn

    # --- Helpers ---
