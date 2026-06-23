"""Dashboard model list/switch API 测试。

锁住修复:dashboard model API 用 agent_loop._client(不再 model_ref 永远 None → 500)。
FallbackChain 场景:list 列所有模型、switch 真切 _active_idx。
"""

import pytest
from unittest.mock import MagicMock


def _make_client_and_app(auth_token: str = ""):
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient
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


@pytest.mark.asyncio
async def test_list_models_fallback_chain_lists_all():
    from src.agent.client import FallbackChain

    client, dash = _make_client_and_app("")  # auth disabled → 直接放行
    primary = MagicMock(model="primary-m")
    fb = MagicMock(model="fb-m")
    chain = FallbackChain(primary, [fb], multimodal_flags=[False, True])
    dash._app_ref.agent_loop._client = chain

    resp = await client.get("/api/dashboard/models")
    assert resp.status_code == 200
    data = resp.json()
    assert [m["name"] for m in data["available"]] == ["primary-m", "fb-m"]
    assert data["current"]["name"] == "primary-m"  # active_idx=0
    await client.aclose()


@pytest.mark.asyncio
async def test_switch_model_fallback_chain_switches_active():
    from src.agent.client import FallbackChain

    client, dash = _make_client_and_app("")
    primary = MagicMock(model="primary-m")
    fb = MagicMock(model="fb-m")
    chain = FallbackChain(primary, [fb], multimodal_flags=[False, True])
    dash._app_ref.agent_loop._client = chain

    resp = await client.post("/api/dashboard/models/switch", json={"provider": "x", "name": "fb-m"})
    assert resp.status_code == 200
    assert resp.json()["current"] == "fb-m"
    assert chain._active_idx == 1  # 真切换
    await client.aclose()


@pytest.mark.asyncio
async def test_switch_model_no_agent_loop_returns_500():
    # agent_loop 未初始化 → 500(清晰错误,不再 model_ref 的隐性 None)
    client, dash = _make_client_and_app("")
    dash._app_ref.agent_loop = None

    resp = await client.post("/api/dashboard/models/switch", json={"provider": "x", "name": "y"})
    assert resp.status_code == 500
    await client.aclose()


@pytest.mark.asyncio
async def test_switch_model_single_client_uses_swap_client_and_multimodal():
    # finding #1/#2 非链分支: 单 ChatClient(非 FallbackChain)切到多模态 fb
    # → create_client 传 multimodal + swap_client(非裸赋 _client,同步 compressor)
    from unittest.mock import patch
    from src.config import ModelFallback

    client, dash = _make_client_and_app("")
    dash._app_ref.agent_loop._client = MagicMock()  # 非 FallbackChain → 走非链分支
    dash._app_ref.config.model.fallbacks = [ModelFallback(provider="x", name="fb-mm", multimodal=True)]

    new_model = MagicMock()
    with patch("src.agent.client.create_client", return_value=new_model) as mock_create:
        resp = await client.post("/api/dashboard/models/switch", json={"provider": "x", "name": "fb-mm"})

    assert resp.status_code == 200
    assert resp.json()["current"] == "x/fb-mm"
    # multimodal 从 fb.multimodal=True 透传给 create_client(修 #2)
    assert mock_create.call_args.kwargs["multimodal"] is True
    # 走 swap_client(非裸赋 _client)——同步 compressor._client,修 #1
    dash._app_ref.agent_loop.swap_client.assert_called_once_with(new_model)
    await client.aclose()
