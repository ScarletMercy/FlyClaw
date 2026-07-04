"""Tests for src.setup._verify_embedding_api_key_async via mocked httpx."""

from __future__ import annotations

import httpx
import pytest

from src.setup import _verify_embedding_api_key_async


@pytest.fixture
def mock_httpx(monkeypatch):
    """monkeypatch httpx.AsyncClient，返回 (queue, FakeResponse)。

    queue 里放 FakeResponse 或 Exception（post 时抛）。
    _verify_embedding_api_key_async 内部 `import httpx` 拿到同一个模块对象，
    所以 monkeypatch httpx.AsyncClient 即可生效。
    """
    queue: list = []

    class FakeAsyncClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None, headers=None):
            if not queue:
                raise RuntimeError("no response queued")
            item = queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

    class FakeResponse:
        def __init__(self, status_code=200, json_data=None, text=""):
            self.status_code = status_code
            self._json = json_data or {}
            self.text = text

        def json(self):
            return self._json

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    return queue, FakeResponse


class TestVerifyEmbeddingApiKeyAsync:
    @pytest.mark.asyncio
    async def test_success_with_embedding_returns_dimension(self, mock_httpx):
        """200 + embedding → (True, 维度字符串)。"""
        queue, FakeResp = mock_httpx
        embedding = [0.1] * 1536
        queue.append(FakeResp(200, json_data={"data": [{"index": 0, "embedding": embedding}]}))
        ok, msg = await _verify_embedding_api_key_async("https://api.example.com", "sk-test", "text-embedding-3-small")
        assert ok is True
        assert msg == "1536"

    @pytest.mark.asyncio
    async def test_success_empty_data_returns_empty_dim(self, mock_httpx):
        """200 + 无 embedding（data 空）→ (True, "")。"""
        queue, FakeResp = mock_httpx
        queue.append(FakeResp(200, json_data={"data": []}))
        ok, msg = await _verify_embedding_api_key_async("https://api.example.com", "sk-test", "text-embedding-3-small")
        assert ok is True
        assert msg == ""

    @pytest.mark.asyncio
    async def test_http_error_returns_false_with_status(self, mock_httpx):
        """401 → (False, "HTTP 401...")。"""
        queue, FakeResp = mock_httpx
        queue.append(FakeResp(401, text="unauthorized"))
        ok, msg = await _verify_embedding_api_key_async("https://api.example.com", "sk-test", "text-embedding-3-small")
        assert ok is False
        assert msg.startswith("HTTP 401")
        assert "unauthorized" in msg

    @pytest.mark.asyncio
    async def test_network_error_returns_false_with_message(self, mock_httpx):
        """网络异常 → (False, "network down")。"""
        queue, _ = mock_httpx
        queue.append(ConnectionError("network down"))
        ok, msg = await _verify_embedding_api_key_async("https://api.example.com", "sk-test", "text-embedding-3-small")
        assert ok is False
        assert msg == "network down"
