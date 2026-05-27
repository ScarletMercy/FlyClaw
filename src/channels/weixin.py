"""Weixin (WeChat) channel implementation for flyclaw.

Connects to personal WeChat accounts via Tencent's iLink Bot API.
Uses long-poll for inbound messages and AES-128-ECB encrypted CDN for media.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import mimetypes
import os
import re
import secrets
import struct
import tempfile
import textwrap
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import quote, urlparse

from .base import Channel

logger = logging.getLogger("flyclaw.weixin")

WEIXIN_COPY_LINE_WIDTH = 120

try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None
    AIOHTTP_AVAILABLE = False

try:
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    CRYPTO_AVAILABLE = True
except ImportError:
    default_backend = None
    Cipher = None
    algorithms = None
    modes = None
    CRYPTO_AVAILABLE = False

ILINK_BASE_URL = "https://ilinkai.weixin.qq.com"
WEIXIN_CDN_BASE_URL = "https://novac2c.cdn.weixin.qq.com/c2c"
ILINK_APP_ID = "bot"
CHANNEL_VERSION = "2.2.0"
ILINK_APP_CLIENT_VERSION = (2 << 16) | (2 << 8) | 0

EP_GET_UPDATES = "ilink/bot/getupdates"
EP_SEND_MESSAGE = "ilink/bot/sendmessage"
EP_SEND_TYPING = "ilink/bot/sendtyping"
EP_GET_CONFIG = "ilink/bot/getconfig"
EP_GET_UPLOAD_URL = "ilink/bot/getuploadurl"
EP_GET_BOT_QR = "ilink/bot/get_bot_qrcode"
EP_GET_QR_STATUS = "ilink/bot/get_qrcode_status"

LONG_POLL_TIMEOUT_MS = 35_000
API_TIMEOUT_MS = 15_000
CONFIG_TIMEOUT_MS = 10_000
QR_TIMEOUT_MS = 35_000

MAX_CONSECUTIVE_FAILURES = 3
RETRY_DELAY_SECONDS = 2
BACKOFF_DELAY_SECONDS = 30
SESSION_EXPIRED_ERRCODE = -14
RATE_LIMIT_ERRCODE = -2
MESSAGE_DEDUP_TTL_SECONDS = 300

MEDIA_IMAGE = 1
MEDIA_VIDEO = 2
MEDIA_FILE = 3
MEDIA_VOICE = 4

ITEM_TEXT = 1
ITEM_IMAGE = 2
ITEM_VOICE = 3
ITEM_FILE = 4
ITEM_VIDEO = 5

MSG_TYPE_USER = 1
MSG_TYPE_BOT = 2
MSG_STATE_FINISH = 2

TYPING_START = 1
TYPING_STOP = 2

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_TABLE_RULE_RE = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$")
_FENCE_RE = re.compile(r"^```([^\n`]*)\s*$")

_WEIXIN_CDN_ALLOWLIST: frozenset[str] = frozenset(
    {
        "novac2c.cdn.weixin.qq.com",
        "ilinkai.weixin.qq.com",
        "wx.qlogo.cn",
        "thirdwx.qlogo.cn",
        "res.wx.qq.com",
        "mmbiz.qpic.cn",
        "mmbiz.qlogo.cn",
    }
)

_weixin_channel: Optional[WeixinChannel] = None

_FLYCLAW_DATA_DIR = Path.home() / ".flyclaw" / "data"


def get_weixin_channel() -> Optional[WeixinChannel]:
    return _weixin_channel


def check_weixin_requirements() -> bool:
    return AIOHTTP_AVAILABLE and CRYPTO_AVAILABLE


def _safe_id(value: Optional[str], keep: int = 8) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "?"
    if len(raw) <= keep:
        return raw
    return raw[:keep]


def _json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _pkcs7_pad(data: bytes, block_size: int = 16) -> bytes:
    pad_len = block_size - (len(data) % block_size)
    return data + bytes([pad_len] * pad_len)


def _aes128_ecb_encrypt(plaintext: bytes, key: bytes) -> bytes:
    cipher = Cipher(algorithms.AES(key), modes.ECB(), backend=default_backend())
    encryptor = cipher.encryptor()
    return encryptor.update(_pkcs7_pad(plaintext)) + encryptor.finalize()


def _aes128_ecb_decrypt(ciphertext: bytes, key: bytes) -> bytes:
    cipher = Cipher(algorithms.AES(key), modes.ECB(), backend=default_backend())
    decryptor = cipher.decryptor()
    padded = decryptor.update(ciphertext) + decryptor.finalize()
    if not padded:
        return padded
    pad_len = padded[-1]
    if 1 <= pad_len <= 16 and padded.endswith(bytes([pad_len]) * pad_len):
        return padded[:-pad_len]
    return padded


def _aes_padded_size(size: int) -> int:
    return ((size + 1 + 15) // 16) * 16


def _random_wechat_uin() -> str:
    value = struct.unpack(">I", secrets.token_bytes(4))[0]
    return base64.b64encode(str(value).encode("utf-8")).decode("ascii")


def _base_info() -> dict[str, Any]:
    return {"channel_version": CHANNEL_VERSION}


def _headers(token: Optional[str], body: str) -> dict[str, str]:
    hdrs = {
        "Content-Type": "application/json",
        "AuthorizationType": "ilink_bot_token",
        "Content-Length": str(len(body.encode("utf-8"))),
        "X-WECHAT-UIN": _random_wechat_uin(),
        "iLink-App-Id": ILINK_APP_ID,
        "iLink-App-ClientVersion": str(ILINK_APP_CLIENT_VERSION),
    }
    if token:
        hdrs["Authorization"] = f"Bearer {token}"
    return hdrs


def _account_dir() -> Path:
    path = Path.home() / ".flyclaw" / "weixin" / "accounts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _account_file(account_id: str) -> Path:
    if "/" in account_id or "\\" in account_id or ".." in account_id:
        raise ValueError(f"Invalid account_id: {account_id!r}")
    result = _account_dir() / f"{account_id}.json"
    if not result.resolve().is_relative_to(_account_dir().resolve()):
        raise ValueError(f"Account ID escapes directory: {account_id!r}")
    return result


def _atomic_json_write(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def save_weixin_account(
    *,
    account_id: str,
    token: str,
    base_url: str,
    user_id: str = "",
) -> None:
    payload = {
        "token": token,
        "base_url": base_url,
        "user_id": user_id,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path = _account_file(account_id)
    _atomic_json_write(path, payload)
    try:
        path.chmod(0o600)
    except OSError:
        pass


def load_weixin_account(account_id: str) -> Optional[dict[str, Any]]:
    path = _account_file(account_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _parse_aes_key(aes_key_b64: str) -> bytes:
    decoded = base64.b64decode(aes_key_b64)
    if len(decoded) == 16:
        return decoded
    if len(decoded) == 32:
        text = decoded.decode("ascii", errors="ignore")
        if text and all(ch in "0123456789abcdefABCDEF" for ch in text):
            return bytes.fromhex(text)
    raise ValueError(f"unexpected aes_key format ({len(decoded)} decoded bytes)")


def _guess_chat_type(message: dict[str, Any], account_id: str) -> tuple[str, str]:
    room_id = str(message.get("room_id") or message.get("chat_room_id") or "").strip()
    to_user_id = str(message.get("to_user_id") or "").strip()
    is_group = bool(room_id) or (to_user_id and account_id and to_user_id != account_id and message.get("msg_type") == 1)
    if is_group:
        effective_id = room_id or to_user_id
        if not effective_id:
            return "group", ""
        return "group", effective_id
    return "p2p", str(message.get("from_user_id") or "")


def _cdn_download_url(cdn_base_url: str, encrypted_query_param: str) -> str:
    return f"{cdn_base_url.rstrip('/')}/download?encrypted_query_param={quote(encrypted_query_param, safe='')}"


def _cdn_upload_url(cdn_base_url: str, upload_param: str, filekey: str) -> str:
    return (
        f"{cdn_base_url.rstrip('/')}/upload"
        f"?encrypted_query_param={quote(upload_param, safe='')}"
        f"&filekey={quote(filekey, safe='')}"
    )


def _assert_weixin_cdn_url(url: str) -> None:
    try:
        parsed = urlparse(url)
        scheme = parsed.scheme.lower()
        host = parsed.hostname or ""
    except Exception as exc:
        raise ValueError(f"Unparseable media URL: {url!r}") from exc
    if scheme not in {"http", "https"}:
        raise ValueError(f"Media URL has disallowed scheme {scheme!r}")
    if host not in _WEIXIN_CDN_ALLOWLIST:
        raise ValueError(f"Media URL host {host!r} is not in the WeChat CDN allowlist.")


def _validate_outbound_url(url: str) -> None:
    import ipaddress as _ipaddress

    parsed = urlparse(url)
    if parsed.scheme.lower() not in ("http", "https"):
        raise ValueError(f"Disallowed URL scheme: {parsed.scheme!r}")
    host = parsed.hostname
    if not host:
        raise ValueError("URL has no hostname")
    try:
        addr = _ipaddress.ip_address(host)
    except ValueError:
        return
    if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
        raise ValueError(f"Private/reserved IP not allowed: {host}")


def _media_reference(item: dict[str, Any], key: str) -> dict[str, Any]:
    return (item.get(key) or {}).get("media") or {}


def _mime_from_filename(filename: str) -> str:
    return mimetypes.guess_type(filename)[0] or "application/octet-stream"


def _cache_media(data: bytes, filename: str) -> str:
    cache_dir = _FLYCLAW_DATA_DIR / "weixin_media"
    cache_dir.mkdir(parents=True, exist_ok=True)
    prefix = hashlib.md5(data).hexdigest()[:12]
    path = cache_dir / f"{prefix}_{filename}"
    if not path.exists():
        path.write_bytes(data)
    return str(path)


def _is_stale_session_ret(
    ret: Optional[int],
    errcode: Optional[int],
    errmsg: Optional[str],
) -> bool:
    if ret != RATE_LIMIT_ERRCODE and errcode != RATE_LIMIT_ERRCODE:
        return False
    return (errmsg or "").lower() == "unknown error"


def _extract_text(item_list: list[dict[str, Any]]) -> str:
    for item in item_list:
        if item.get("type") == ITEM_TEXT:
            text = str((item.get("text_item") or {}).get("text") or "")
            ref = item.get("ref_msg") or {}
            ref_item = ref.get("message_item") or {}
            ref_type = ref_item.get("type")
            if ref_type in {ITEM_IMAGE, ITEM_VIDEO, ITEM_FILE, ITEM_VOICE}:
                title = ref.get("title") or ""
                prefix = f"[引用媒体: {title}]\n" if title else "[引用媒体]\n"
                return f"{prefix}{text}".strip()
            if ref_item:
                parts: list[str] = []
                if ref.get("title"):
                    parts.append(str(ref["title"]))
                ref_text = _extract_text([ref_item])
                if ref_text:
                    parts.append(ref_text)
                if parts:
                    return f"[引用: {' | '.join(parts)}]\n{text}".strip()
            return text
    for item in item_list:
        if item.get("type") == ITEM_VOICE:
            voice_text = str((item.get("voice_item") or {}).get("text") or "")
            if voice_text:
                return voice_text
    return ""


def _sync_buf_path(account_id: str) -> Path:
    return _account_dir() / f"{account_id}.sync.json"


def _load_sync_buf(account_id: str) -> str:
    path = _sync_buf_path(account_id)
    if not path.exists():
        return ""
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("get_updates_buf", "")
    except Exception:
        return ""


def _save_sync_buf(account_id: str, sync_buf: str) -> None:
    path = _sync_buf_path(account_id)
    _atomic_json_write(path, {"get_updates_buf": sync_buf})


def _make_ssl_connector() -> Optional[aiohttp.TCPConnector]:
    try:
        import ssl
        import certifi
    except ImportError:
        return None
    if not AIOHTTP_AVAILABLE:
        return None
    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    return aiohttp.TCPConnector(ssl=ssl_ctx)


async def _api_post(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    endpoint: str,
    payload: dict[str, Any],
    token: Optional[str],
    timeout_ms: int,
) -> dict[str, Any]:
    body = _json_dumps({**payload, "base_info": _base_info()})
    url = f"{base_url.rstrip('/')}/{endpoint}"
    timeout = aiohttp.ClientTimeout(total=timeout_ms / 1000)
    async with session.post(url, data=body, headers=_headers(token, body), timeout=timeout) as response:
        raw = await response.text()
        if not response.ok:
            raise RuntimeError(f"iLink POST {endpoint} HTTP {response.status}: {raw[:200]}")
        return json.loads(raw)


async def _api_get(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    endpoint: str,
    timeout_ms: int,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/{endpoint}"
    hdrs = {
        "iLink-App-Id": ILINK_APP_ID,
        "iLink-App-ClientVersion": str(ILINK_APP_CLIENT_VERSION),
    }
    timeout = aiohttp.ClientTimeout(total=timeout_ms / 1000)
    async with session.get(url, headers=hdrs, timeout=timeout) as response:
        raw = await response.text()
        if not response.ok:
            raise RuntimeError(f"iLink GET {endpoint} HTTP {response.status}: {raw[:200]}")
        return json.loads(raw)


async def _get_updates(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    token: str,
    sync_buf: str,
    timeout_ms: int,
) -> dict[str, Any]:
    try:
        return await _api_post(
            session,
            base_url=base_url,
            endpoint=EP_GET_UPDATES,
            payload={"get_updates_buf": sync_buf},
            token=token,
            timeout_ms=timeout_ms,
        )
    except asyncio.TimeoutError:
        return {"ret": 0, "msgs": [], "get_updates_buf": sync_buf}


async def _send_message_api(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    token: str,
    to: str,
    text: str,
    context_token: Optional[str],
    client_id: str,
) -> dict[str, Any]:
    if not text or not text.strip():
        raise ValueError("_send_message_api: text must not be empty")
    message: dict[str, Any] = {
        "from_user_id": "",
        "to_user_id": to,
        "client_id": client_id,
        "message_type": MSG_TYPE_BOT,
        "message_state": MSG_STATE_FINISH,
        "item_list": [{"type": ITEM_TEXT, "text_item": {"text": text}}],
    }
    if context_token:
        message["context_token"] = context_token
    return await _api_post(
        session,
        base_url=base_url,
        endpoint=EP_SEND_MESSAGE,
        payload={"msg": message},
        token=token,
        timeout_ms=API_TIMEOUT_MS,
    )


async def _send_typing_api(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    token: str,
    to_user_id: str,
    typing_ticket: str,
    status: int,
) -> None:
    await _api_post(
        session,
        base_url=base_url,
        endpoint=EP_SEND_TYPING,
        payload={
            "ilink_user_id": to_user_id,
            "typing_ticket": typing_ticket,
            "status": status,
        },
        token=token,
        timeout_ms=CONFIG_TIMEOUT_MS,
    )


async def _get_config_api(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    token: str,
    user_id: str,
    context_token: Optional[str],
) -> dict[str, Any]:
    payload: dict[str, Any] = {"ilink_user_id": user_id}
    if context_token:
        payload["context_token"] = context_token
    return await _api_post(
        session,
        base_url=base_url,
        endpoint=EP_GET_CONFIG,
        payload=payload,
        token=token,
        timeout_ms=CONFIG_TIMEOUT_MS,
    )


async def _get_upload_url_api(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    token: str,
    to_user_id: str,
    media_type: int,
    filekey: str,
    rawsize: int,
    rawfilemd5: str,
    filesize: int,
    aeskey_hex: str,
) -> dict[str, Any]:
    return await _api_post(
        session,
        base_url=base_url,
        endpoint=EP_GET_UPLOAD_URL,
        payload={
            "filekey": filekey,
            "media_type": media_type,
            "to_user_id": to_user_id,
            "rawsize": rawsize,
            "rawfilemd5": rawfilemd5,
            "filesize": filesize,
            "no_need_thumb": True,
            "aeskey": aeskey_hex,
        },
        token=token,
        timeout_ms=API_TIMEOUT_MS,
    )


async def _upload_ciphertext(
    session: aiohttp.ClientSession,
    *,
    ciphertext: bytes,
    upload_url: str,
) -> str:
    async def _do_upload() -> str:
        async with session.post(upload_url, data=ciphertext, headers={"Content-Type": "application/octet-stream"}) as response:
            if response.status == 200:
                encrypted_param = response.headers.get("x-encrypted-param")
                if encrypted_param:
                    await response.read()
                    return encrypted_param
                raw = await response.text()
                raise RuntimeError(f"CDN upload missing x-encrypted-param header: {raw[:200]}")
            raw = await response.text()
            raise RuntimeError(f"CDN upload HTTP {response.status}: {raw[:200]}")

    return await asyncio.wait_for(_do_upload(), timeout=120)


async def _download_bytes(
    session: aiohttp.ClientSession,
    *,
    url: str,
    timeout_seconds: float = 60.0,
) -> bytes:
    async def _do_download() -> bytes:
        async with session.get(url) as response:
            response.raise_for_status()
            return await response.read()

    return await asyncio.wait_for(_do_download(), timeout=timeout_seconds)


async def _download_and_decrypt_media(
    session: aiohttp.ClientSession,
    *,
    cdn_base_url: str,
    encrypted_query_param: Optional[str],
    aes_key_b64: Optional[str],
    full_url: Optional[str],
    timeout_seconds: float,
) -> bytes:
    if encrypted_query_param:
        raw = await _download_bytes(
            session,
            url=_cdn_download_url(cdn_base_url, encrypted_query_param),
            timeout_seconds=timeout_seconds,
        )
    elif full_url:
        _assert_weixin_cdn_url(full_url)
        raw = await _download_bytes(session, url=full_url, timeout_seconds=timeout_seconds)
    else:
        raise RuntimeError("media item had neither encrypt_query_param nor full_url")
    if aes_key_b64:
        raw = _aes128_ecb_decrypt(raw, _parse_aes_key(aes_key_b64))
    return raw


# ── Markdown formatting ──────────────────────────────────────────────────


def _split_table_row(line: str) -> list[str]:
    row = line.strip()
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|"):
        row = row[:-1]
    return [cell.strip() for cell in row.split("|")]


def _rewrite_headers_for_weixin(line: str) -> str:
    match = _HEADER_RE.match(line)
    if not match:
        return line.rstrip()
    level = len(match.group(1))
    title = match.group(2).strip()
    if level == 1:
        return f"【{title}】"
    return f"**{title}**"


def _rewrite_table_block_for_weixin(lines: list[str]) -> str:
    if len(lines) < 2:
        return "\n".join(lines)
    headers = _split_table_row(lines[0])
    body_rows = [_split_table_row(line) for line in lines[2:] if line.strip()]
    if not headers or not body_rows:
        return "\n".join(lines)
    formatted_rows: list[str] = []
    for row in body_rows:
        pairs = []
        for idx, header in enumerate(headers):
            if idx >= len(row):
                break
            label = header or f"Column {idx + 1}"
            value = row[idx].strip()
            if value:
                pairs.append((label, value))
        if not pairs:
            continue
        if len(pairs) == 1:
            label, value = pairs[0]
            formatted_rows.append(f"- {label}: {value}")
            continue
        if len(pairs) == 2:
            label, value = pairs[0]
            other_label, other_value = pairs[1]
            formatted_rows.append(f"- {label}: {value}")
            formatted_rows.append(f"  {other_label}: {other_value}")
            continue
        summary = " | ".join(f"{label}: {value}" for label, value in pairs)
        formatted_rows.append(f"- {summary}")
    return "\n".join(formatted_rows) if formatted_rows else "\n".join(lines)


def _normalize_markdown_blocks(content: str) -> str:
    lines = content.splitlines()
    result: list[str] = []
    in_code_block = False
    blank_run = 0
    for raw_line in lines:
        line = raw_line.rstrip()
        if _FENCE_RE.match(line.strip()):
            in_code_block = not in_code_block
            result.append(line)
            blank_run = 0
            continue
        if in_code_block:
            result.append(line)
            continue
        if not line.strip():
            blank_run += 1
            if blank_run <= 1:
                result.append("")
            continue
        blank_run = 0
        result.append(line)
    return "\n".join(result).strip()


def _wrap_copy_friendly_lines(content: str) -> str:
    if not content:
        return content
    wrapped: list[str] = []
    in_code_block = False
    for raw_line in content.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if _FENCE_RE.match(stripped):
            in_code_block = not in_code_block
            wrapped.append(line)
            continue
        if (
            in_code_block
            or len(line) <= WEIXIN_COPY_LINE_WIDTH
            or not stripped
            or stripped.startswith("|")
            or _TABLE_RULE_RE.match(stripped)
        ):
            wrapped.append(line)
            continue
        wrapped_lines = textwrap.wrap(
            line,
            width=WEIXIN_COPY_LINE_WIDTH,
            break_long_words=False,
            break_on_hyphens=False,
            replace_whitespace=False,
            drop_whitespace=True,
        )
        wrapped.extend(wrapped_lines or [line])
    return "\n".join(wrapped).strip()


def _split_markdown_blocks(content: str) -> list[str]:
    if not content:
        return []
    blocks: list[str] = []
    lines = content.splitlines()
    current: list[str] = []
    in_code_block = False
    for raw_line in lines:
        line = raw_line.rstrip()
        if _FENCE_RE.match(line.strip()):
            if not in_code_block and current:
                blocks.append("\n".join(current).strip())
                current = []
            current.append(line)
            in_code_block = not in_code_block
            if not in_code_block:
                blocks.append("\n".join(current).strip())
                current = []
            continue
        if in_code_block:
            current.append(line)
            continue
        if not line.strip():
            if current:
                blocks.append("\n".join(current).strip())
                current = []
            continue
        current.append(line)
    if current:
        blocks.append("\n".join(current).strip())
    return [block for block in blocks if block]


def _split_delivery_units(content: str) -> list[str]:
    units: list[str] = []
    for block in _split_markdown_blocks(content):
        if _FENCE_RE.match(block.splitlines()[0].strip()):
            units.append(block)
            continue
        current: list[str] = []
        for raw_line in block.splitlines():
            line = raw_line.rstrip()
            if not line.strip():
                if current:
                    units.append("\n".join(current).strip())
                    current = []
                continue
            is_continuation = bool(current) and raw_line.startswith((" ", "\t"))
            if is_continuation:
                current.append(line)
                continue
            if current:
                units.append("\n".join(current).strip())
            current = [line]
        if current:
            units.append("\n".join(current).strip())
    return [unit for unit in units if unit]


def _looks_like_chatty_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if len(stripped) > 48:
        return False
    if line.startswith((" ", "\t")):
        return False
    if stripped.startswith((">", "-", "*", "【", "#", "|")):
        return False
    if _TABLE_RULE_RE.match(stripped):
        return False
    if re.match(r"^\*\*[^*]+\*\*$", stripped):
        return False
    if re.match(r"^\d+\.\s", stripped):
        return False
    return True


def _looks_like_heading_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _HEADER_RE.match(stripped):
        return True
    return len(stripped) <= 24 and stripped.endswith((":", "："))


def _should_split_short_chat_block(block: str) -> bool:
    lines = [line for line in block.splitlines() if line.strip()]
    if not 2 <= len(lines) <= 6:
        return False
    if _looks_like_heading_line(lines[0]):
        return False
    return all(_looks_like_chatty_line(line) for line in lines)


def _pack_markdown_blocks(content: str, max_length: int) -> list[str]:
    if len(content) <= max_length:
        return [content]
    packed: list[str] = []
    current = ""
    for block in _split_markdown_blocks(content):
        candidate = block if not current else f"{current}\n\n{block}"
        if len(candidate) <= max_length:
            current = candidate
            continue
        if current:
            packed.append(current)
            current = ""
        if len(block) <= max_length:
            current = block
            continue
        chunked = Channel.chunk_text(block, max_length)
        packed.extend(chunked)
    if current:
        packed.append(current)
    return packed


def _split_text_for_delivery(content: str, max_length: int, split_per_line: bool = False) -> list[str]:
    if not content:
        return []
    if split_per_line:
        if len(content) <= max_length and "\n" not in content:
            return [content]
        chunks: list[str] = []
        for unit in _split_delivery_units(content):
            if len(unit) <= max_length:
                chunks.append(unit)
                continue
            chunks.extend(_pack_markdown_blocks(unit, max_length))
        return [c for c in chunks if c] or [content]
    if len(content) <= max_length:
        return (
            [u for u in _split_delivery_units(content) if u]
            if _should_split_short_chat_block(content)
            else [content]
        )
    return _pack_markdown_blocks(content, max_length) or [content]


def format_message(content: Optional[str]) -> str:
    if content is None:
        return ""
    return _wrap_copy_friendly_lines(_normalize_markdown_blocks(content))


# ── ContextTokenStore ─────────────────────────────────────────────────────


class ContextTokenStore:
    def __init__(self):
        self._root = _account_dir()
        self._cache: dict[str, str] = {}

    def _path(self, account_id: str) -> Path:
        return self._root / f"{account_id}.context-tokens.json"

    def _key(self, account_id: str, user_id: str) -> str:
        return f"{account_id}:{user_id}"

    def restore(self, account_id: str) -> None:
        path = self._path(account_id)
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("weixin: failed to restore context tokens for %s: %s", _safe_id(account_id), exc)
            return
        restored = 0
        for user_id, token in data.items():
            if isinstance(token, str) and token:
                self._cache[self._key(account_id, user_id)] = token
                restored += 1
        if restored:
            logger.info("weixin: restored %d context token(s) for %s", restored, _safe_id(account_id))

    def get(self, account_id: str, user_id: str) -> Optional[str]:
        return self._cache.get(self._key(account_id, user_id))

    def set(self, account_id: str, user_id: str, token: str) -> None:
        self._cache[self._key(account_id, user_id)] = token
        self._persist(account_id)

    def _persist(self, account_id: str) -> None:
        prefix = f"{account_id}:"
        payload = {
            key[len(prefix):]: value
            for key, value in self._cache.items()
            if key.startswith(prefix)
        }
        try:
            _atomic_json_write(self._path(account_id), payload)
        except Exception as exc:
            logger.warning("weixin: failed to persist context tokens for %s: %s", _safe_id(account_id), exc)


class TypingTicketCache:
    def __init__(self, ttl_seconds: float = 600.0):
        self._ttl_seconds = ttl_seconds
        self._cache: dict[str, tuple[str, float]] = {}

    def get(self, user_id: str) -> Optional[str]:
        entry = self._cache.get(user_id)
        if not entry:
            return None
        if time.time() - entry[1] >= self._ttl_seconds:
            self._cache.pop(user_id, None)
            return None
        return entry[0]

    def set(self, user_id: str, ticket: str) -> None:
        self._cache[user_id] = (ticket, time.time())


# ── WeixinChannel ─────────────────────────────────────────────────────────


class WeixinChannel(Channel):
    MAX_MESSAGE_LENGTH = 2000

    def __init__(self, config):
        super().__init__()
        global _weixin_channel
        self.config = config
        self._running = False
        self._on_message_callback: Optional[Callable] = None
        self._token_store = ContextTokenStore()
        self._typing_cache = TypingTicketCache()
        self._poll_session: Optional[aiohttp.ClientSession] = None
        self._send_session: Optional[aiohttp.ClientSession] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._content_dedup: dict[str, float] = {}
        self._inflight: set[asyncio.Task] = set()
        self._dedup_last_cleanup: float = 0.0

        self._account_id = str(getattr(config, "account_id", "") or "").strip()
        self._token = str(getattr(config, "token", "") or "").strip()
        self._base_url = str(getattr(config, "base_url", ILINK_BASE_URL) or ILINK_BASE_URL).strip().rstrip("/")
        self._cdn_base_url = str(getattr(config, "cdn_base_url", WEIXIN_CDN_BASE_URL) or WEIXIN_CDN_BASE_URL).strip().rstrip("/")
        self._dm_policy = str(getattr(config, "dm_policy", "open") or "open").strip().lower()
        self._group_policy = str(getattr(config, "group_policy", "disabled") or "disabled").strip().lower()
        self._allow_from: list[str] = list(getattr(config, "allowed_users", []) or [])
        self._group_allow_from: list[str] = list(getattr(config, "group_allowed_users", []) or [])
        self._split_multiline_messages = bool(getattr(config, "split_multiline_messages", False))
        self._send_chunk_delay_seconds = 1.5
        self._send_chunk_retries = 4
        self._send_chunk_retry_delay_seconds = 1.0

        if self._account_id and not self._token:
            persisted = load_weixin_account(self._account_id)
            if persisted:
                self._token = str(persisted.get("token") or "").strip()
                self._base_url = str(persisted.get("base_url") or self._base_url).strip().rstrip("/")

        _weixin_channel = self

    def set_message_callback(self, callback: Callable) -> None:
        self._on_message_callback = callback

    async def start(self) -> None:
        if not self.config.enabled:
            logger.info("Weixin channel disabled")
            return
        if not check_weixin_requirements():
            logger.warning("Weixin startup failed: aiohttp and cryptography are required")
            return
        if not self._token:
            logger.warning("Weixin startup failed: token is required")
            return
        if not self._account_id:
            logger.warning("Weixin startup failed: account_id is required")
            return

        self._poll_session = aiohttp.ClientSession(trust_env=True, connector=_make_ssl_connector())
        _no_timeout = aiohttp.ClientTimeout(total=None, connect=None, sock_connect=None, sock_read=None)
        self._send_session = aiohttp.ClientSession(trust_env=True, connector=_make_ssl_connector(), timeout=_no_timeout)
        self._token_store.restore(self._account_id)
        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop(), name="weixin-poll")
        logger.info("Weixin channel starting... account=%s", _safe_id(self._account_id))

    async def stop(self) -> None:
        self._running = False
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        self._poll_task = None
        for task in list(self._inflight):
            task.cancel()
        for task in list(self._inflight):
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._inflight.clear()
        if self._poll_session and not self._poll_session.closed:
            await self._poll_session.close()
        self._poll_session = None
        if self._send_session and not self._send_session.closed:
            await self._send_session.close()
        self._send_session = None
        cache_dir = _FLYCLAW_DATA_DIR / "weixin_media"
        if cache_dir.exists():
            now = time.time()
            for f in cache_dir.iterdir():
                if f.is_file() and (now - f.stat().st_mtime) > 7 * 86400:
                    try:
                        f.unlink()
                    except OSError:
                        pass
        logger.info("Weixin channel stopped")

    async def send_text(self, chat_id: str, text: str, reply_to: Optional[str] = None) -> Any:
        if not self._send_session or not self._token:
            return None
        context_token = self._token_store.get(self._account_id, chat_id)
        chunks = [c for c in _split_text_for_delivery(format_message(text), self.MAX_MESSAGE_LENGTH, self._split_multiline_messages) if c and c.strip()]
        last_id = None
        for idx, chunk in enumerate(chunks):
            client_id = f"flyclaw-weixin-{uuid.uuid4().hex}"
            await self._send_text_chunk(chat_id=chat_id, chunk=chunk, context_token=context_token, client_id=client_id)
            last_id = client_id
            if idx < len(chunks) - 1 and self._send_chunk_delay_seconds > 0:
                await asyncio.sleep(self._send_chunk_delay_seconds)
        return last_id

    async def send_image(self, chat_id: str, image_key: str) -> bool:
        if not image_key:
            return False
        try:
            result = await self._send_file_internal(chat_id, image_key, "")
            return result is not None
        except Exception as exc:
            logger.error("weixin send_image failed to=%s: %s", _safe_id(chat_id), exc)
            return False

    async def send_file(self, chat_id: str, file_key: str) -> bool:
        if not file_key:
            return False
        try:
            result = await self._send_file_internal(chat_id, file_key, "")
            return result is not None
        except Exception as exc:
            logger.error("weixin send_file failed to=%s: %s", _safe_id(chat_id), exc)
            return False

    async def send_card(self, chat_id: str, card_content: str, reply_to: Optional[str] = None) -> Any:
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
        return await self.send_text(chat_id, text, reply_to)

    async def send_voice(self, chat_id: str, audio_path: str, caption: Optional[str] = None) -> bool:
        fallback_caption = caption or "[voice message]"
        try:
            result = await self._send_file_internal(chat_id, audio_path, fallback_caption, force_file_attachment=False)
            return result is not None
        except Exception as exc:
            logger.error("weixin send_voice failed to=%s: %s", _safe_id(chat_id), exc)
            return False

    async def send_typing(self, chat_id: str) -> None:
        if not self._send_session or not self._token:
            return
        typing_ticket = self._typing_cache.get(chat_id)
        if not typing_ticket:
            return
        try:
            await _send_typing_api(
                self._send_session,
                base_url=self._base_url,
                token=self._token,
                to_user_id=chat_id,
                typing_ticket=typing_ticket,
                status=TYPING_START,
            )
        except Exception as exc:
            logger.debug("weixin typing start failed for %s: %s", _safe_id(chat_id), exc)

    async def stop_typing(self, chat_id: str) -> None:
        if not self._send_session or not self._token:
            return
        typing_ticket = self._typing_cache.get(chat_id)
        if not typing_ticket:
            return
        try:
            await _send_typing_api(
                self._send_session,
                base_url=self._base_url,
                token=self._token,
                to_user_id=chat_id,
                typing_ticket=typing_ticket,
                status=TYPING_STOP,
            )
        except Exception as exc:
            logger.debug("weixin typing stop failed for %s: %s", _safe_id(chat_id), exc)

    def _spawn_task(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._inflight.add(task)
        task.add_done_callback(self._inflight.discard)
        return task

    # ── Poll loop ────────────────────────────────────────────────────────

    async def _poll_loop(self) -> None:
        assert self._poll_session is not None
        sync_buf = _load_sync_buf(self._account_id)
        timeout_ms = LONG_POLL_TIMEOUT_MS
        consecutive_failures = 0
        session_expired_count = 0

        while self._running:
            try:
                response = await _get_updates(
                    self._poll_session,
                    base_url=self._base_url,
                    token=self._token,
                    sync_buf=sync_buf,
                    timeout_ms=timeout_ms,
                )
                suggested_timeout = response.get("longpolling_timeout_ms")
                if isinstance(suggested_timeout, int) and suggested_timeout > 0:
                    timeout_ms = suggested_timeout

                ret = response.get("ret", 0)
                errcode = response.get("errcode", 0)
                if ret not in {0, None} or errcode not in {0, None}:
                    if (
                        ret == SESSION_EXPIRED_ERRCODE
                        or errcode == SESSION_EXPIRED_ERRCODE
                        or _is_stale_session_ret(ret, errcode, response.get("errmsg"))
                    ):
                        session_expired_count += 1
                        if session_expired_count > 3:
                            logger.error(
                                "weixin: Session expired %d times, stopping poll loop. "
                                "Please re-login via setup.",
                                session_expired_count,
                            )
                            break
                        logger.error(
                            "weixin: Session expired (%d/3); pausing for 10 minutes",
                            session_expired_count,
                        )
                        await asyncio.sleep(600)
                        consecutive_failures = 0
                        continue
                    consecutive_failures += 1
                    logger.warning(
                        "weixin: getUpdates failed ret=%s errcode=%s errmsg=%s (%d/%d)",
                        ret, errcode, response.get("errmsg", ""),
                        consecutive_failures, MAX_CONSECUTIVE_FAILURES,
                    )
                    await asyncio.sleep(
                        BACKOFF_DELAY_SECONDS if consecutive_failures >= MAX_CONSECUTIVE_FAILURES else RETRY_DELAY_SECONDS
                    )
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        consecutive_failures = 0
                    continue

                consecutive_failures = 0
                session_expired_count = 0
                new_sync_buf = str(response.get("get_updates_buf") or "")
                if new_sync_buf:
                    sync_buf = new_sync_buf
                    _save_sync_buf(self._account_id, sync_buf)

                for message in response.get("msgs") or []:
                    self._spawn_task(self._process_message_safe(message))
            except asyncio.CancelledError:
                break
            except Exception as exc:
                consecutive_failures += 1
                logger.error("weixin: poll error (%d/%d): %s", consecutive_failures, MAX_CONSECUTIVE_FAILURES, exc)
                await asyncio.sleep(
                    BACKOFF_DELAY_SECONDS if consecutive_failures >= MAX_CONSECUTIVE_FAILURES else RETRY_DELAY_SECONDS
                )
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    consecutive_failures = 0

    async def _process_message_safe(self, message: dict[str, Any]) -> None:
        try:
            await self._process_message(message)
        except Exception as exc:
            logger.error("weixin: unhandled inbound error from=%s: %s", _safe_id(message.get("from_user_id")), exc, exc_info=True)

    async def _process_message(self, message: dict[str, Any]) -> None:
        assert self._poll_session is not None
        sender_id = str(message.get("from_user_id") or "").strip()
        if not sender_id or sender_id == self._account_id:
            return

        message_id = str(message.get("message_id") or "").strip()
        if message_id:
            is_dup = await self.check_dedup(message_id)
            if is_dup:
                return

        item_list = message.get("item_list") or []
        text = _extract_text(item_list)
        if text:
            now = time.time()
            content_key = f"content:{sender_id}:{hashlib.md5(text.encode()).hexdigest()}"
            if content_key in self._content_dedup and (now - self._content_dedup[content_key]) < MESSAGE_DEDUP_TTL_SECONDS:
                return
            self._content_dedup[content_key] = now
            now_mono = time.monotonic()
            if len(self._content_dedup) > 1000 or now_mono - self._dedup_last_cleanup > 60:
                cutoff = now - MESSAGE_DEDUP_TTL_SECONDS
                self._content_dedup = {
                    k: v for k, v in self._content_dedup.items()
                    if v >= cutoff
                }
                self._dedup_last_cleanup = now_mono

        chat_type, effective_chat_id = _guess_chat_type(message, self._account_id)
        if not effective_chat_id:
            logger.warning(
                "weixin: skipping message with undetermined chat_id from=%s",
                _safe_id(sender_id),
            )
            return
        if chat_type == "group":
            if self._group_policy == "disabled":
                return
            if self._group_policy == "allowlist" and effective_chat_id not in self._group_allow_from:
                return
        else:
            if self._dm_policy == "disabled":
                return
            if self._dm_policy == "allowlist" and sender_id not in self._allow_from:
                return

        context_token = str(message.get("context_token") or "").strip()
        if context_token:
            self._token_store.set(self._account_id, sender_id, context_token)
        self._spawn_task(self._maybe_fetch_typing_ticket(sender_id, context_token or None))

        media_paths: list[str] = []
        media_mimes: list[str] = []
        for item in item_list:
            await self._collect_media(item, media_paths, media_mimes)
            ref_message = item.get("ref_msg") or {}
            ref_item = ref_message.get("message_item")
            if isinstance(ref_item, dict):
                await self._collect_media(ref_item, media_paths, media_mimes)

        if not text and not media_paths:
            return

        if not self._on_message_callback:
            return

        chat_type_str = "group" if chat_type == "group" else "p2p"
        reply_fn = lambda t: self.send_text(effective_chat_id, t, message_id)
        stream_fn = self._create_stream_sender(effective_chat_id, message_id)

        # Inject chat_id for WeChat send tools (so they can default to current chat)
        try:
            from src.tools.weixin_tools import set_current_weixin_chat_id
            set_current_weixin_chat_id(effective_chat_id)
        except ImportError:
            pass  # weixin_tools not available, tools will handle missing chat_id

        logger.info("weixin: inbound from=%s type=%s media=%d", _safe_id(sender_id), chat_type_str, len(media_paths))
        await self._on_message_callback(
            text=text,
            sender_id=sender_id,
            chat_id=effective_chat_id,
            chat_type=chat_type_str,
            message_id=message_id or "",
            reply_fn=reply_fn,
            stream_fn=stream_fn,
        )

    async def _maybe_fetch_typing_ticket(self, user_id: str, context_token: Optional[str]) -> None:
        if not self._poll_session or not self._token:
            return
        if self._typing_cache.get(user_id):
            return
        try:
            response = await _get_config_api(
                self._poll_session,
                base_url=self._base_url,
                token=self._token,
                user_id=user_id,
                context_token=context_token,
            )
            typing_ticket = str(response.get("typing_ticket") or "")
            if typing_ticket:
                self._typing_cache.set(user_id, typing_ticket)
        except Exception as exc:
            logger.debug("weixin: getConfig failed for %s: %s", _safe_id(user_id), exc)

    async def _collect_media(self, item: dict[str, Any], media_paths: list[str], media_mimes: list[str]) -> None:
        item_type = item.get("type")
        if item_type == ITEM_IMAGE:
            path = await self._download_image(item)
            if path:
                media_paths.append(path)
                media_mimes.append("image/jpeg")
        elif item_type == ITEM_VIDEO:
            path = await self._download_video(item)
            if path:
                media_paths.append(path)
                media_mimes.append("video/mp4")
        elif item_type == ITEM_FILE:
            path, mime = await self._download_file(item)
            if path:
                media_paths.append(path)
                media_mimes.append(mime)
        elif item_type == ITEM_VOICE:
            path = await self._download_voice(item)
            if path:
                media_paths.append(path)
                media_mimes.append("audio/silk")

    async def _download_image(self, item: dict[str, Any]) -> Optional[str]:
        media = _media_reference(item, "image_item")
        try:
            data = await _download_and_decrypt_media(
                self._poll_session,
                cdn_base_url=self._cdn_base_url,
                encrypted_query_param=media.get("encrypt_query_param"),
                aes_key_b64=(
                    (item.get("image_item") or {}).get("aeskey")
                    and base64.b64encode(bytes.fromhex(str((item.get("image_item") or {}).get("aeskey")))).decode("ascii")
                    or media.get("aes_key")
                ),
                full_url=media.get("full_url"),
                timeout_seconds=30.0,
            )
            return _cache_media(data, ".jpg")
        except Exception as exc:
            logger.warning("weixin: image download failed: %s", exc)
            return None

    async def _download_video(self, item: dict[str, Any]) -> Optional[str]:
        media = _media_reference(item, "video_item")
        try:
            data = await _download_and_decrypt_media(
                self._poll_session,
                cdn_base_url=self._cdn_base_url,
                encrypted_query_param=media.get("encrypt_query_param"),
                aes_key_b64=media.get("aes_key"),
                full_url=media.get("full_url"),
                timeout_seconds=120.0,
            )
            return _cache_media(data, "video.mp4")
        except Exception as exc:
            logger.warning("weixin: video download failed: %s", exc)
            return None

    async def _download_file(self, item: dict[str, Any]) -> tuple[Optional[str], str]:
        file_item = item.get("file_item") or {}
        media = file_item.get("media") or {}
        filename = str(file_item.get("file_name") or "document.bin")
        mime = _mime_from_filename(filename)
        try:
            data = await _download_and_decrypt_media(
                self._poll_session,
                cdn_base_url=self._cdn_base_url,
                encrypted_query_param=media.get("encrypt_query_param"),
                aes_key_b64=media.get("aes_key"),
                full_url=media.get("full_url"),
                timeout_seconds=60.0,
            )
            return _cache_media(data, filename), mime
        except Exception as exc:
            logger.warning("weixin: file download failed: %s", exc)
            return None, mime

    async def _download_voice(self, item: dict[str, Any]) -> Optional[str]:
        voice_item = item.get("voice_item") or {}
        media = voice_item.get("media") or {}
        if voice_item.get("text"):
            return None
        try:
            data = await _download_and_decrypt_media(
                self._poll_session,
                cdn_base_url=self._cdn_base_url,
                encrypted_query_param=media.get("encrypt_query_param"),
                aes_key_b64=media.get("aes_key"),
                full_url=media.get("full_url"),
                timeout_seconds=60.0,
            )
            return _cache_media(data, ".silk")
        except Exception as exc:
            logger.warning("weixin: voice download failed: %s", exc)
            return None

    # ── Send helpers ─────────────────────────────────────────────────────

    async def _send_text_chunk(
        self,
        *,
        chat_id: str,
        chunk: str,
        context_token: Optional[str],
        client_id: str,
    ) -> None:
        last_error: Optional[Exception] = None
        retried_without_token = False
        for attempt in range(self._send_chunk_retries + 1):
            try:
                resp = await _send_message_api(
                    self._send_session,
                    base_url=self._base_url,
                    token=self._token,
                    to=chat_id,
                    text=chunk,
                    context_token=context_token,
                    client_id=client_id,
                )
                if resp and isinstance(resp, dict):
                    ret = resp.get("ret")
                    errcode = resp.get("errcode")
                    if (ret is not None and ret not in {0}) or (errcode is not None and errcode not in {0}):
                        is_session_expired = (
                            ret == SESSION_EXPIRED_ERRCODE
                            or errcode == SESSION_EXPIRED_ERRCODE
                            or _is_stale_session_ret(ret, errcode, resp.get("errmsg"))
                        )
                        if is_session_expired and not retried_without_token and context_token:
                            retried_without_token = True
                            context_token = None
                            logger.warning("weixin: session expired for %s; retrying without context_token", _safe_id(chat_id))
                            continue
                        is_rate_limited = ret == RATE_LIMIT_ERRCODE or errcode == RATE_LIMIT_ERRCODE
                        if is_rate_limited:
                            last_error = RuntimeError(f"iLink rate limited: ret={ret} errcode={errcode}")
                            if attempt >= self._send_chunk_retries:
                                break
                            wait = self._send_chunk_retry_delay_seconds * 3
                            await asyncio.sleep(wait)
                            continue
                        errmsg = resp.get("errmsg") or resp.get("msg") or "unknown error"
                        raise RuntimeError(f"iLink sendmessage error: ret={ret} errcode={errcode} errmsg={errmsg}")
                return
            except Exception as exc:
                last_error = exc
                if attempt >= self._send_chunk_retries:
                    break
                wait = self._send_chunk_retry_delay_seconds * (attempt + 1)
                logger.warning("weixin: send chunk failed to=%s attempt=%d/%d: %s", _safe_id(chat_id), attempt + 1, self._send_chunk_retries + 1, exc)
                if wait > 0:
                    await asyncio.sleep(wait)
        if last_error:
            raise last_error

    async def _send_file_internal(
        self,
        chat_id: str,
        path: str,
        caption: str,
        force_file_attachment: bool = False,
    ) -> Optional[str]:
        assert self._send_session is not None and self._token is not None
        if path.startswith(("http://", "https://")):
            data = await self._download_remote_media(path)
            cleanup_path = data
            file_path = Path(data)
        else:
            cleanup_path = None
            file_path = Path(path)

        try:
            plaintext = file_path.read_bytes()
            media_type, item_builder = self._outbound_media_builder(str(file_path), force_file_attachment=force_file_attachment)
            filekey = secrets.token_hex(16)
            aes_key = secrets.token_bytes(16)
            rawsize = len(plaintext)
            rawfilemd5 = hashlib.md5(plaintext).hexdigest()
            upload_response = await _get_upload_url_api(
                self._send_session,
                base_url=self._base_url,
                token=self._token,
                to_user_id=chat_id,
                media_type=media_type,
                filekey=filekey,
                rawsize=rawsize,
                rawfilemd5=rawfilemd5,
                filesize=_aes_padded_size(rawsize),
                aeskey_hex=aes_key.hex(),
            )
            upload_param = str(upload_response.get("upload_param") or "")
            upload_full_url = str(upload_response.get("upload_full_url") or "")
            ciphertext = _aes128_ecb_encrypt(plaintext, aes_key)

            if upload_full_url:
                upload_url = upload_full_url
            elif upload_param:
                upload_url = _cdn_upload_url(self._cdn_base_url, upload_param, filekey)
            else:
                raise RuntimeError(f"getUploadUrl returned neither upload_param nor upload_full_url")

            encrypted_query_param = await _upload_ciphertext(self._send_session, ciphertext=ciphertext, upload_url=upload_url)
            context_token = self._token_store.get(self._account_id, chat_id)
            aes_key_for_api = base64.b64encode(aes_key.hex().encode("ascii")).decode("ascii")
            item_kwargs = {
                "encrypt_query_param": encrypted_query_param,
                "aes_key_for_api": aes_key_for_api,
                "ciphertext_size": len(ciphertext),
                "plaintext_size": rawsize,
                "filename": file_path.name,
                "rawfilemd5": rawfilemd5,
            }
            if media_type == MEDIA_VOICE and str(file_path).endswith(".silk"):
                item_kwargs["encode_type"] = 6
                item_kwargs["sample_rate"] = 24000
                item_kwargs["bits_per_sample"] = 16
            media_item = item_builder(**item_kwargs)

            if caption:
                await _send_message_api(
                    self._send_session,
                    base_url=self._base_url,
                    token=self._token,
                    to=chat_id,
                    text=format_message(caption),
                    context_token=context_token,
                    client_id=f"flyclaw-weixin-{uuid.uuid4().hex}",
                )

            last_message_id = f"flyclaw-weixin-{uuid.uuid4().hex}"
            await _api_post(
                self._send_session,
                base_url=self._base_url,
                endpoint=EP_SEND_MESSAGE,
                payload={
                    "msg": {
                        "from_user_id": "",
                        "to_user_id": chat_id,
                        "client_id": last_message_id,
                        "message_type": MSG_TYPE_BOT,
                        "message_state": MSG_STATE_FINISH,
                        "item_list": [media_item],
                        **({"context_token": context_token} if context_token else {}),
                    }
                },
                token=self._token,
                timeout_ms=API_TIMEOUT_MS,
            )
            return last_message_id
        finally:
            if cleanup_path and Path(cleanup_path).exists():
                try:
                    Path(cleanup_path).unlink()
                except OSError:
                    pass

    def _outbound_media_builder(self, path: str, force_file_attachment: bool = False):
        mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
        if mime.startswith("image/"):
            return MEDIA_IMAGE, lambda **kw: {
                "type": ITEM_IMAGE,
                "image_item": {
                    "media": {
                        "encrypt_query_param": kw["encrypt_query_param"],
                        "aes_key": kw["aes_key_for_api"],
                        "encrypt_type": 1,
                    },
                    "mid_size": kw["ciphertext_size"],
                },
            }
        if mime.startswith("video/"):
            return MEDIA_VIDEO, lambda **kw: {
                "type": ITEM_VIDEO,
                "video_item": {
                    "media": {
                        "encrypt_query_param": kw["encrypt_query_param"],
                        "aes_key": kw["aes_key_for_api"],
                        "encrypt_type": 1,
                    },
                    "video_size": kw["ciphertext_size"],
                    "play_length": kw.get("play_length", 0),
                    "video_md5": kw.get("rawfilemd5", ""),
                },
            }
        if path.endswith(".silk") and not force_file_attachment:
            return MEDIA_VOICE, lambda **kw: {
                "type": ITEM_VOICE,
                "voice_item": {
                    "media": {
                        "encrypt_query_param": kw["encrypt_query_param"],
                        "aes_key": kw["aes_key_for_api"],
                        "encrypt_type": 1,
                    },
                    "encode_type": kw.get("encode_type"),
                    "bits_per_sample": kw.get("bits_per_sample"),
                    "sample_rate": kw.get("sample_rate"),
                    "playtime": kw.get("playtime", 0),
                },
            }
        return MEDIA_FILE, lambda **kw: {
            "type": ITEM_FILE,
            "file_item": {
                "media": {
                    "encrypt_query_param": kw["encrypt_query_param"],
                    "aes_key": kw["aes_key_for_api"],
                    "encrypt_type": 1,
                },
                "file_name": kw["filename"],
                "len": str(kw["plaintext_size"]),
            },
        }

    async def _download_remote_media(self, url: str) -> str:
        _validate_outbound_url(url)
        assert self._send_session is not None

        async def _do_fetch() -> bytes:
            async with self._send_session.get(url) as response:
                response.raise_for_status()
                return await response.read()

        data = await asyncio.wait_for(_do_fetch(), timeout=30)
        suffix = Path(url.split("?", 1)[0]).suffix or ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(data)
            return handle.name

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
