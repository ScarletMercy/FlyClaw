"""Tests for gateway HTTP routes — healthz, auth, config API, chat completions."""

import asyncio
import hashlib
import hmac
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.config import AppConfig


def _make_gateway_app(auth_token="test-token"):
    from src.config import AppConfig
    from src.gateway import create_gateway
    from src._container import set_container

    config = AppConfig()
    config.gateway.auth_token = auth_token

    agent_loop = MagicMock()
    cron_service = None

    app = create_gateway(config, agent_loop, cron_service)

    mock_container = MagicMock()
    mock_container.rbac = None
    mock_container.dispatcher = None
    mock_container.session_index = None
    mock_container.cron_service = None
    mock_container.memory_searcher = None
    set_container(mock_container)

    return app, config


@pytest.fixture
def gateway_app():
    app, config = _make_gateway_app()
    yield app


@pytest.fixture
async def client(gateway_app):
    transport = ASGITransport(app=gateway_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestHealthz:
    @pytest.mark.asyncio
    async def test_healthz_returns_ok(self, client):
        resp = await client.get("/healthz")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


class TestChatCompletionsAuth:
    @pytest.mark.asyncio
    async def test_no_auth_returns_401(self, client):
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_wrong_auth_returns_401(self, client):
        resp = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer wrong-token"},
            json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_valid_auth_passes_auth_check(self, client):
        with patch("src.gateway._get_app") as mock_app:
            mock_app_obj = MagicMock()
            mock_app_obj.agent_loop = MagicMock()
            mock_app_obj.agent_loop.run = AsyncMock(
                return_value=MagicMock(messages=[{"role": "assistant", "content": "test reply"}])
            )
            mock_app_obj.config = AppConfig()
            mock_app_obj.state_store = MagicMock()
            mock_app_obj.state_store.load = AsyncMock(return_value=None)
            mock_app_obj.state_store.save = AsyncMock()
            mock_app.return_value = mock_app_obj

            resp = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer test-token"},
                json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert resp.status_code in (200, 500)


class TestConfigAPI:
    @pytest.mark.asyncio
    async def test_get_config_returns_json(self, client):
        import src.gateway as gw_mod

        mock_app = MagicMock()
        mock_app.config = AppConfig()
        gw_mod._app_ref = mock_app
        try:
            resp = await client.get(
                "/api/config",
                headers={"Authorization": "Bearer test-token"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "model" in data or "gateway" in data
        finally:
            gw_mod._app_ref = None
