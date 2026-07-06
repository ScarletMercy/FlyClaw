"""Tests for src.setup._verify_embedding_api_key via mocked httpx.Client.

_verify_embedding_api_key 不再走 asyncio.run+AsyncOpenAI（后者在 Windows 上对
部分 TLS 接口挂死），改用同步 httpx.Client 直发 /embeddings。这里 mock httpx.Client。
"""

from __future__ import annotations

import httpx
import pytest

from src.setup import _verify_embedding_api_key


@pytest.fixture
def mock_httpx(monkeypatch):
    """monkeypatch httpx.Client，返回 (queue, posted_url, FakeResp)。

    queue 放 FakeResp 或 Exception（post 时抛）；posted_url 抓 POST 的 URL。
    data 元素用 dict（真实 /embeddings 返回 {"data":[{"index":0,"embedding":[...]}]}）。
    """
    queue: list = []
    posted_url: list = []

    class FakeResp:
        def __init__(self, status_code=200, data=None, text=""):
            self.status_code = status_code
            self._data = data
            self.text = text

        def json(self):
            return {"data": self._data or []}

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, json=None, headers=None):
            posted_url.append(url)
            if not queue:
                raise RuntimeError("no response queued")
            item = queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

    import httpx as _httpx

    monkeypatch.setattr(_httpx, "Client", FakeClient)
    return queue, posted_url, FakeResp


def _emb(dim):
    return {"index": 0, "embedding": [0.1] * dim}


class TestVerifyEmbeddingApiKey:
    def test_success_with_embedding_returns_dimension(self, mock_httpx):
        """成功 + embedding → (True, 维度字符串)。"""
        queue, _, FakeResp = mock_httpx
        queue.append(FakeResp(data=[_emb(1536)]))
        ok, msg = _verify_embedding_api_key("https://api.example.com/v1", "sk-test", "text-embedding-3-small")
        assert ok is True
        assert msg == "1536"

    def test_success_empty_data_returns_empty_dim(self, mock_httpx):
        """成功 + 无 embedding（data 空）→ (True, "")。"""
        queue, _, FakeResp = mock_httpx
        queue.append(FakeResp(data=[]))
        ok, msg = _verify_embedding_api_key("https://api.example.com/v1", "sk-test", "text-embedding-3-small")
        assert ok is True
        assert msg == ""

    def test_posts_to_base_url_embeddings(self, mock_httpx):
        """URL = base_url + /embeddings（base_url 惯例已含 /v1，不再追加）。"""
        queue, posted_url, FakeResp = mock_httpx
        queue.append(FakeResp(data=[_emb(8)]))
        _verify_embedding_api_key("https://api.example.com/v1", "sk-test", "m")
        assert posted_url == ["https://api.example.com/v1/embeddings"]

    def test_http_error_returns_false_with_status(self, mock_httpx):
        """HTTP 401 → (False, 含状态码与错误信息)。"""
        queue, _, FakeResp = mock_httpx
        queue.append(FakeResp(status_code=401, text='{"error":{"message":"unauthorized"}}'))
        ok, msg = _verify_embedding_api_key("https://api.example.com/v1", "sk-test", "text-embedding-3-small")
        assert ok is False
        assert "HTTP 401" in msg
        assert "unauthorized" in msg

    def test_network_error_returns_false_with_message(self, mock_httpx):
        """网络异常 → (False, 异常信息)。"""
        queue, _, _ = mock_httpx
        queue.append(httpx.ConnectError("network down"))
        ok, msg = _verify_embedding_api_key("https://api.example.com/v1", "sk-test", "text-embedding-3-small")
        assert ok is False
        assert "network down" in msg
