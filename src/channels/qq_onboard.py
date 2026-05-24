"""QQ Bot scan-to-configure module.

Calls q.qq.com create_bind_task / poll_bind_result APIs to generate a
connect URL and poll for scan completion.  On success returns the bot's
app_id, client_secret (decrypted locally via AES-256-GCM), and the
scanner's user_openid.

Reference: https://bot.q.qq.com/wiki/develop/api-v2/
"""

from __future__ import annotations

import base64
import logging
import os
import time
from enum import IntEnum
from typing import Optional, Tuple
from urllib.parse import quote

import httpx

logger = logging.getLogger("myclaw.qq_onboard")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PORTAL_HOST = os.getenv("QQ_PORTAL_HOST", "q.qq.com")

ONBOARD_CREATE_PATH = "/lite/create_bind_task"
ONBOARD_POLL_PATH = "/lite/poll_bind_result"
ONBOARD_POLL_INTERVAL = 2.0
ONBOARD_API_TIMEOUT = 10.0

QR_URL_TEMPLATE = (
    "https://q.qq.com/qqbot/openclaw/connect.html"
    "?task_id={task_id}&_wv=2&source=myclaw"
)

_MAX_REFRESHES = 3


# ---------------------------------------------------------------------------
# AES-256-GCM crypto
# ---------------------------------------------------------------------------


def _generate_bind_key() -> str:
    """Generate a 256-bit random AES key and return it as base64."""
    return base64.b64encode(os.urandom(32)).decode()


def _decrypt_secret(encrypted_base64: str, key_base64: str) -> str:
    """Decrypt a base64-encoded AES-256-GCM ciphertext.

    Ciphertext layout (after base64-decoding)::

        IV (12 bytes) || ciphertext (N bytes) || AuthTag (16 bytes)
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    key = base64.b64decode(key_base64)
    raw = base64.b64decode(encrypted_base64)
    iv = raw[:12]
    ciphertext_with_tag = raw[12:]
    aesgcm = AESGCM(key)
    plaintext = aesgcm.decrypt(iv, ciphertext_with_tag, None)
    return plaintext.decode("utf-8")


# ---------------------------------------------------------------------------
# Bind status
# ---------------------------------------------------------------------------


class BindStatus(IntEnum):
    NONE = 0
    PENDING = 1
    COMPLETED = 2
    EXPIRED = 3


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _get_api_headers() -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "MyClawQQBot/0.1.0",
    }


def _create_bind_task(timeout: float = ONBOARD_API_TIMEOUT) -> Tuple[str, str]:
    """Create a bind task and return (task_id, aes_key_base64)."""
    url = f"https://{PORTAL_HOST}{ONBOARD_CREATE_PATH}"
    key = _generate_bind_key()

    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        resp = client.post(url, json={"key": key}, headers=_get_api_headers())
        resp.raise_for_status()
        data = resp.json()

    if data.get("retcode") != 0:
        raise RuntimeError(data.get("msg", "create_bind_task failed"))

    task_id = data.get("data", {}).get("task_id")
    if not task_id:
        raise RuntimeError("create_bind_task: missing task_id in response")

    logger.debug("create_bind_task ok: task_id=%s", task_id)
    return task_id, key


def _poll_bind_result(
    task_id: str,
    timeout: float = ONBOARD_API_TIMEOUT,
) -> Tuple[BindStatus, str, str, str]:
    """Poll the bind result for task_id.

    Returns (status, bot_appid, bot_encrypt_secret, user_openid).
    """
    url = f"https://{PORTAL_HOST}{ONBOARD_POLL_PATH}"

    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        resp = client.post(url, json={"task_id": task_id}, headers=_get_api_headers())
        resp.raise_for_status()
        data = resp.json()

    if data.get("retcode") != 0:
        raise RuntimeError(data.get("msg", "poll_bind_result failed"))

    d = data.get("data", {})
    return (
        BindStatus(d.get("status", 0)),
        str(d.get("bot_appid", "")),
        d.get("bot_encrypt_secret", ""),
        d.get("user_openid", ""),
    )


def build_connect_url(task_id: str) -> str:
    """Build the connect URL for a given task_id."""
    return QR_URL_TEMPLATE.format(task_id=quote(task_id))


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------


def qr_register(timeout_seconds: int = 600) -> Optional[dict]:
    """Run the QQBot scan-to-configure registration flow.

    Creates a bind task, prints the connect URL, polls for scan
    completion, and decrypts the returned credentials.

    :returns:
        ``{"app_id": ..., "client_secret": ..., "user_openid": ...}`` on
        success, or ``None`` on failure / expiry.
    """
    deadline = time.monotonic() + timeout_seconds

    for refresh_count in range(_MAX_REFRESHES + 1):
        try:
            task_id, aes_key = _create_bind_task()
        except Exception as exc:
            logger.warning("Failed to create bind task: %s", exc)
            return None

        url = build_connect_url(task_id)

        print()
        print(f"  请打开以下链接完成配置：")
        print(f"  {url}")
        print()

        while time.monotonic() < deadline:
            try:
                status, app_id, encrypted_secret, user_openid = _poll_bind_result(task_id)
            except Exception as exc:
                logger.debug("Poll error (will retry): %s", exc)
                time.sleep(ONBOARD_POLL_INTERVAL)
                continue

            if status == BindStatus.COMPLETED:
                try:
                    client_secret = _decrypt_secret(encrypted_secret, aes_key)
                except Exception as exc:
                    logger.error("Failed to decrypt client_secret: %s", exc)
                    return None
                print()
                print(f"  扫码完成！(App ID: {app_id})")
                if user_openid:
                    print(f"  扫码者 OpenID: {user_openid}")
                return {
                    "app_id": app_id,
                    "client_secret": client_secret,
                    "user_openid": user_openid,
                }

            if status == BindStatus.EXPIRED:
                if refresh_count >= _MAX_REFRESHES:
                    logger.warning("QR expired %d times, giving up", _MAX_REFRESHES)
                    return None
                print(f"\n  链接已过期，正在刷新... ({refresh_count + 1}/{_MAX_REFRESHES})")
                break

            time.sleep(ONBOARD_POLL_INTERVAL)
        else:
            logger.warning("Poll timed out after %ds", timeout_seconds)
            return None

    return None
