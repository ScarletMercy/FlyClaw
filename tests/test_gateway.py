"""Tests for the gateway HTTP API."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_gateway(tmp_path):
    """Create a FastAPI test client with a minimal config."""
    from src.config import AppConfig
    from src.gateway import create_gateway
    from httpx import ASGITransport, AsyncClient

    config = AppConfig()
    config.gateway.auth_token = ""  # no auth for tests

    # Create a minimal compiled graph mock
    from unittest.mock import AsyncMock, MagicMock

    graph = AsyncMock()
    graph.ainvoke = AsyncMock(return_value={"messages": []})

    async def _empty_stream(*a, **kw):
        return
        yield  # makes this an async generator

    graph.astream_events = _empty_stream

    app = create_gateway(config, graph)

    transport = ASGITransport(app=app)
    client = AsyncClient(transport=transport, base_url="http://test")
    return client, config


@pytest.mark.asyncio
async def test_healthz(tmp_path):
    client, _ = _make_gateway(tmp_path)
    resp = await client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data
    await client.aclose()


@pytest.mark.asyncio
async def test_readyz(tmp_path):
    client, _ = _make_gateway(tmp_path)
    resp = await client.get("/readyz")
    assert resp.status_code == 200
    await client.aclose()


@pytest.mark.asyncio
async def test_chat_completions_invalid_json(tmp_path):
    client, _ = _make_gateway(tmp_path)
    resp = await client.post(
        "/v1/chat/completions",
        content="not json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 400
    await client.aclose()


@pytest.mark.asyncio
async def test_chat_completions_wrong_content_type(tmp_path):
    client, _ = _make_gateway(tmp_path)
    resp = await client.post(
        "/v1/chat/completions",
        content="data",
        headers={"Content-Type": "text/plain"},
    )
    assert resp.status_code == 415
    await client.aclose()


@pytest.mark.asyncio
async def test_auth_endpoints_disabled_when_auth_off(tmp_path):
    """Auth endpoints return 400 when auth is disabled."""
    client, config = _make_gateway(tmp_path)
    config.auth.enabled = False

    resp = await client.post("/api/pair", json={"code": "123456", "device_id": "d1"})
    assert resp.status_code == 400
    await client.aclose()


@pytest.mark.asyncio
async def test_list_users_without_rbac(tmp_path):
    """List users returns 500 when RBAC not initialized."""
    client, _ = _make_gateway(tmp_path)
    # Don't set up RBAC singleton — require_rbac dependency returns 500
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
