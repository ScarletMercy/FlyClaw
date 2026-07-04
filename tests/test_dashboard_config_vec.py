"""Tests for dashboard /api/dashboard/config vector fields.

锁住:dashboard_config 返回 memory_store.vector_enabled / vector_model。
vector_enabled=False 时 vector_model 恒为空串(无论实际配置值)。
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.config import AppConfig
from src.dashboard import routes as dash


def _make_client_and_app():
    cfg = AppConfig()
    cfg.gateway.auth_token = ""  # auth disabled → 直接放行
    app = FastAPI()
    mock_app = MagicMock()
    mock_app.config = cfg
    dash.register_dashboard(app, mock_app)
    client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
    return client, cfg


class TestDashboardConfigVector:
    @pytest.mark.asyncio
    async def test_config_returns_vector_fields_when_enabled(self):
        client, cfg = _make_client_and_app()
        cfg.memory_store.enabled = True
        cfg.memory_store.vector_enabled = True
        cfg.memory_store.vector_model = "bge-m3"

        resp = await client.get("/api/dashboard/config")
        assert resp.status_code == 200
        ms = resp.json()["memory_store"]
        assert ms["vector_enabled"] is True
        assert ms["vector_model"] == "bge-m3"
        assert ms["archive_enabled"] is True  # = memory_store.enabled
        await client.aclose()

    @pytest.mark.asyncio
    async def test_config_vector_disabled_returns_empty_model(self):
        client, cfg = _make_client_and_app()
        cfg.memory_store.vector_enabled = False
        cfg.memory_store.vector_model = "text-embedding-3-small"  # 有值但不该返回

        resp = await client.get("/api/dashboard/config")
        assert resp.status_code == 200
        ms = resp.json()["memory_store"]
        assert ms["vector_enabled"] is False
        assert ms["vector_model"] == ""  # disabled 时空串
        await client.aclose()
