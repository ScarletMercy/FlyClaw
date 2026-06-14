"""Tests for Dashboard cookie-session auth (C2 fix).

Crypto helpers (_sign_session / _verify_session) are pure functions → strict TDD.
Route behaviour (redirect / login / cookie / no-token-in-html) → integration tests
against a dashboard router mounted on a throwaway FastAPI app.
"""

import time

import pytest


# ---------------------------------------------------------------------------
# Test helper
# ---------------------------------------------------------------------------


def _make_dashboard_client(auth_token: str):
    """Mount the dashboard router on a throwaway FastAPI app, set _app_ref, return (client, module)."""
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient
    from unittest.mock import MagicMock

    from src.config import AppConfig
    from src.dashboard import routes as dash

    cfg = AppConfig()
    cfg.gateway.auth_token = auth_token
    app = FastAPI()
    mock_app = MagicMock()
    mock_app.config = cfg
    dash.register_dashboard(app, mock_app)
    client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
    return client, dash


# ---------------------------------------------------------------------------
# _sign_session / _verify_session —— 纯函数
# ---------------------------------------------------------------------------


def test_session_roundtrip_valid():
    from src.dashboard import routes as dash

    val = dash._sign_session("k", now=1000)
    assert dash._verify_session("k", val, now=1000) is True


def test_session_expired_is_rejected():
    from src.dashboard import routes as dash

    val = dash._sign_session("k", now=0)  # expiry = 0 + 7*86400
    assert dash._verify_session("k", val, now=7 * 86400 + 100) is False


def test_session_tamper_detected():
    from src.dashboard import routes as dash

    val = dash._sign_session("k", now=0)
    payload, _ = val.split(".")
    forged = f"{payload}.{'0' * 64}"  # 篡改签名
    assert dash._verify_session("k", forged, now=0) is False


def test_session_wrong_key_rejected():
    from src.dashboard import routes as dash

    val = dash._sign_session("k", now=0)
    assert dash._verify_session("other-key", val, now=0) is False


def test_session_garbage_rejected():
    from src.dashboard import routes as dash

    assert dash._verify_session("k", "garbage", now=0) is False
    assert dash._verify_session("k", "", now=0) is False
    assert dash._verify_session("k", "no-dot-here", now=0) is False


# ---------------------------------------------------------------------------
# Route behaviour —— 集成测试
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dashboard_redirects_to_login_without_cookie():
    client, _ = _make_dashboard_client("secret")
    resp = await client.get("/dashboard", follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/dashboard/login"
    await client.aclose()


@pytest.mark.asyncio
async def test_dashboard_serves_with_valid_cookie_and_no_token_leak():
    client, dash = _make_dashboard_client("secret")
    cookie_val = dash._sign_session("secret", now=int(time.time()))
    resp = await client.get("/dashboard", cookies={"fc_auth": cookie_val}, follow_redirects=False)
    assert resp.status_code == 200
    assert "secret" not in resp.text  # token 绝不能出现在 HTML
    assert "AUTH_TOKEN" not in resp.text  # 旧的内嵌点也该彻底没了
    assert "auth_token" not in resp.text  # 模板占位符不应残留
    await client.aclose()


@pytest.mark.asyncio
async def test_login_sets_cookie_on_correct_token():
    client, _ = _make_dashboard_client("secret")
    resp = await client.post("/dashboard/login", data={"token": "secret"}, follow_redirects=False)
    assert resp.status_code == 303
    assert "fc_auth" in resp.headers.get("set-cookie", "").lower()
    await client.aclose()


@pytest.mark.asyncio
async def test_login_rejects_wrong_token():
    client, _ = _make_dashboard_client("secret")
    resp = await client.post("/dashboard/login", data={"token": "wrong"}, follow_redirects=False)
    assert resp.status_code == 401
    await client.aclose()


@pytest.mark.asyncio
async def test_dashboard_open_when_auth_disabled():
    """auth_token 为空时（auth 未启用），dashboard 直接放行，不要求登录。"""
    client, _ = _make_dashboard_client("")
    resp = await client.get("/dashboard", follow_redirects=False)
    assert resp.status_code == 200
    await client.aclose()


# ---------------------------------------------------------------------------
# 登录后的 cookie 必须能访问 /api/dashboard/*（验证 cookie path 覆盖 API 端点）
# + 向后兼容（Bearer header / ?token= 仍工作）
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cookie_from_login_reaches_dashboard_api():
    """登录拿到的 cookie 必须能访问受保护的 /api/dashboard/* —— 主流程闭环。"""
    client, _ = _make_dashboard_client("secret")
    resp = await client.post("/dashboard/login", data={"token": "secret"}, follow_redirects=False)
    assert resp.status_code == 303
    # 同一 client（cookie 进 jar）访问受保护 API
    resp2 = await client.get("/api/dashboard/status")
    assert resp2.status_code == 200
    await client.aclose()


@pytest.mark.asyncio
async def test_api_rejects_without_any_credential():
    """无 cookie / 无 Bearer / 无 ?token= 访问 API → 401。"""
    client, _ = _make_dashboard_client("secret")
    assert (await client.get("/api/dashboard/status")).status_code == 401
    await client.aclose()


@pytest.mark.asyncio
async def test_bearer_header_still_works():
    """向后兼容：Bearer header 仍能访问（API 客户端）。"""
    client, _ = _make_dashboard_client("secret")
    resp = await client.get("/api/dashboard/status", headers={"Authorization": "Bearer secret"})
    assert resp.status_code == 200
    await client.aclose()


@pytest.mark.asyncio
async def test_query_token_still_works():
    """向后兼容：?token= 查询参数仍能访问（SSE / EventSource 客户端）。"""
    client, _ = _make_dashboard_client("secret")
    resp = await client.get("/api/dashboard/status", params={"token": "secret"})
    assert resp.status_code == 200
    await client.aclose()


@pytest.mark.asyncio
async def test_wrong_bearer_rejected_without_cookie_fallback():
    """给了错误 Bearer 必须直接 401，不能静默回退到 cookie 校验。"""
    client, _ = _make_dashboard_client("secret")
    resp = await client.get("/api/dashboard/status", headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401
    await client.aclose()
