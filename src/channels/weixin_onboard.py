"""WeChat QR scan-to-login onboarding for MyClaw setup wizard."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Optional

logger = logging.getLogger("myclaw.weixin_onboard")

ILINK_BASE_URL = "https://ilinkai.weixin.qq.com"
EP_GET_BOT_QR = "ilink/bot/get_bot_qrcode"
EP_GET_QR_STATUS = "ilink/bot/get_qrcode_status"
QR_TIMEOUT_MS = 35_000

ILINK_APP_ID = "bot"
CHANNEL_VERSION = "2.2.0"
ILINK_APP_CLIENT_VERSION = (2 << 16) | (2 << 8) | 0


def _headers() -> dict[str, str]:
    return {
        "iLink-App-Id": ILINK_APP_ID,
        "iLink-App-ClientVersion": str(ILINK_APP_CLIENT_VERSION),
    }


async def _api_get(
    session,
    *,
    base_url: str,
    endpoint: str,
    timeout_ms: int,
) -> dict:
    import aiohttp

    url = f"{base_url.rstrip('/')}/{endpoint}"
    timeout = aiohttp.ClientTimeout(total=timeout_ms / 1000)
    async with session.get(url, headers=_headers(), timeout=timeout) as response:
        raw = await response.text()
        if not response.ok:
            raise RuntimeError(f"iLink GET {endpoint} HTTP {response.status}: {raw[:200]}")
        return json.loads(raw)


def qr_login(timeout_seconds: int = 480) -> Optional[dict[str, str]]:
    """Run the interactive iLink QR login flow (synchronous wrapper).

    Returns a credential dict on success, or None if login fails or times out.
    """
    try:
        import aiohttp
    except ImportError:
        print("  缺少 aiohttp 库，请运行: pip install aiohttp")
        return None

    result: Optional[dict[str, str]] = None

    async def _run():
        nonlocal result
        try:
            from cryptography.hazmat.backends import default_backend
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        except ImportError:
            print("  缺少 cryptography 库，请运行: pip install cryptography")
            return

        connector = None
        try:
            import ssl
            import certifi

            ssl_ctx = ssl.create_default_context(cafile=certifi.where())
            connector = aiohttp.TCPConnector(ssl=ssl_ctx)
        except ImportError:
            pass

        async with aiohttp.ClientSession(trust_env=True, connector=connector) as session:
            try:
                qr_resp = await _api_get(
                    session,
                    base_url=ILINK_BASE_URL,
                    endpoint=f"{EP_GET_BOT_QR}?bot_type=3",
                    timeout_ms=QR_TIMEOUT_MS,
                )
            except Exception as exc:
                logger.error("weixin: failed to fetch QR code: %s", exc)
                return

            qrcode_value = str(qr_resp.get("qrcode") or "")
            qrcode_url = str(qr_resp.get("qrcode_img_content") or "")
            if not qrcode_value:
                logger.error("weixin: QR response missing qrcode")
                return

            qr_scan_data = qrcode_url if qrcode_url else qrcode_value

            print("\n请使用微信扫描以下二维码：")
            if qrcode_url:
                print(qrcode_url)
            try:
                import qrcode

                qr = qrcode.QRCode()
                qr.add_data(qr_scan_data)
                qr.make(fit=True)
                qr.print_ascii(invert=True)
            except Exception as exc:
                print(f"（终端二维码渲染失败: {exc}，请直接打开上面的二维码链接）")

            deadline = time.monotonic() + timeout_seconds
            current_base_url = ILINK_BASE_URL
            refresh_count = 0

            while time.monotonic() < deadline:
                try:
                    status_resp = await _api_get(
                        session,
                        base_url=current_base_url,
                        endpoint=f"{EP_GET_QR_STATUS}?qrcode={qrcode_value}",
                        timeout_ms=QR_TIMEOUT_MS,
                    )
                except asyncio.TimeoutError:
                    await asyncio.sleep(1)
                    continue
                except Exception as exc:
                    logger.warning("weixin: QR poll error: %s", exc)
                    await asyncio.sleep(1)
                    continue

                status = str(status_resp.get("status") or "wait")
                if status == "wait":
                    print(".", end="", flush=True)
                elif status == "scaned":
                    print("\n已扫码，请在微信里确认...")
                elif status == "scaned_but_redirect":
                    redirect_host = str(status_resp.get("redirect_host") or "")
                    if redirect_host:
                        current_base_url = f"https://{redirect_host}"
                elif status == "expired":
                    refresh_count += 1
                    if refresh_count > 3:
                        print("\n二维码多次过期，请重新执行登录。")
                        return
                    print(f"\n二维码已过期，正在刷新... ({refresh_count}/3)")
                    try:
                        qr_resp = await _api_get(
                            session,
                            base_url=ILINK_BASE_URL,
                            endpoint=f"{EP_GET_BOT_QR}?bot_type=3",
                            timeout_ms=QR_TIMEOUT_MS,
                        )
                        qrcode_value = str(qr_resp.get("qrcode") or "")
                        qrcode_url = str(qr_resp.get("qrcode_img_content") or "")
                        qr_scan_data = qrcode_url if qrcode_url else qrcode_value
                        if qrcode_url:
                            print(qrcode_url)
                        try:
                            import qrcode as _qrcode

                            qr = _qrcode.QRCode()
                            qr.add_data(qr_scan_data)
                            qr.make(fit=True)
                            qr.print_ascii(invert=True)
                        except Exception:
                            pass
                    except Exception as exc:
                        logger.error("weixin: QR refresh failed: %s", exc)
                        return
                elif status == "confirmed":
                    account_id = str(status_resp.get("ilink_bot_id") or "")
                    token = str(status_resp.get("bot_token") or "")
                    base_url = str(status_resp.get("baseurl") or ILINK_BASE_URL)
                    user_id = str(status_resp.get("ilink_user_id") or "")
                    if not account_id or not token:
                        logger.error("weixin: QR confirmed but credential payload was incomplete")
                        return

                    from src.channels.weixin import save_weixin_account

                    save_weixin_account(
                        account_id=account_id,
                        token=token,
                        base_url=base_url,
                        user_id=user_id,
                    )
                    print(f"\n微信连接成功，account_id={account_id}")
                    result = {
                        "account_id": account_id,
                        "token": token,
                        "base_url": base_url,
                        "user_id": user_id,
                    }
                    return
                await asyncio.sleep(1)

            print("\n微信登录超时。")

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor() as pool:
        try:
            pool.submit(lambda: asyncio.run(_run())).result(timeout=timeout_seconds + 30)
        except concurrent.futures.TimeoutError:
            logger.error("weixin: onboard timed out")

    return result
