"""Tests for the gateway HTTP API."""

import os, sys, pytest


def _make_gateway(tmp_path):
    from src.config import AppConfig
    from src.gateway import create_gateway
    from httpx import ASGITransport, AsyncClient
    from unittest.mock import AsyncMock, MagicMock
    from src._container import set_container

    config = AppConfig()
    config.gateway.auth_token = ""
    loop = AsyncMock()
    loop.run = AsyncMock(return_value=MagicMock(messages=[{"role": "assistant", "content": "test"}]))
    app = create_gateway(config, loop)

    mock_container = MagicMock()
    mock_container.rbac = None
    mock_container.dispatcher = None
    mock_container.session_index = None
    mock_container.cron_service = None
    mock_container.memory_searcher = None
    set_container(mock_container)

    transport = ASGITransport(app=app)
    client = AsyncClient(transport=transport, base_url="http://test")
    return client, config


@pytest.mark.asyncio
async def test_healthz(tmp_path):
    client, _ = _make_gateway(tmp_path)
    resp = await client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    await client.aclose()


@pytest.mark.asyncio
async def test_readyz(tmp_path):
    client, _ = _make_gateway(tmp_path)
    assert (await client.get("/readyz")).status_code == 200
    await client.aclose()


@pytest.mark.asyncio
async def test_chat_completions_invalid_json(tmp_path):
    client, _ = _make_gateway(tmp_path)
    resp = await client.post("/v1/chat/completions", content="not json", headers={"Content-Type": "application/json"})
    assert resp.status_code == 400
    await client.aclose()


@pytest.mark.asyncio
async def test_chat_completions_wrong_content_type(tmp_path):
    client, _ = _make_gateway(tmp_path)
    resp = await client.post("/v1/chat/completions", content="data", headers={"Content-Type": "text/plain"})
    assert resp.status_code == 415
    await client.aclose()


@pytest.mark.asyncio
async def test_auth_endpoints_disabled_when_auth_off(tmp_path):
    client, config = _make_gateway(tmp_path)
    config.auth.enabled = False
    assert (await client.post("/api/pair", json={"code": "123456", "device_id": "d1"})).status_code == 400
    await client.aclose()


@pytest.mark.asyncio
async def test_list_users_without_rbac(tmp_path):
    client, _ = _make_gateway(tmp_path)
    resp = await client.get("/api/users")
    assert resp.status_code == 500
    assert "error" in resp.json()
    await client.aclose()


@pytest.mark.asyncio
async def test_pending_approvals(tmp_path):
    client, _ = _make_gateway(tmp_path)
    resp = await client.get("/api/approval/pending")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
    await client.aclose()
