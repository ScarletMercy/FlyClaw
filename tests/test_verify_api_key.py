"""Tests for src.setup._verify_api_key via mocked httpx.Client.

镜像 test_verify_embedding_api_key.py 的队列式 mock，但针对同步 httpx 路径
（_verify_api_key 不再走 asyncio.run+AsyncOpenAI——后者在 Windows 上对本接口
会挂死，改用同步 httpx.Client 直发 /chat/completions）。
"""

from __future__ import annotations

import httpx
import pytest

from src.setup import _verify_api_key


@pytest.fixture
def mock_httpx(monkeypatch):
    """monkeypatch httpx.Client，返回 (queue, captured, posted_url, FakeResp)。

    queue 放 FakeResp 或 Exception（post 时抛）；captured 抓构造 kwargs；
    posted_url 抓实际 POST 的 URL。_verify_api_key 内部 `import httpx` 后用
    httpx.Client，所以 monkeypatch httpx.Client 即可。
    """
    queue: list = []
    captured: dict = {}
    posted_url: list = []

    class FakeResp:
        def __init__(self, status_code, text=""):
            self.status_code = status_code
            self.text = text

    class FakeClient:
        def __init__(self, *a, **k):
            captured.update(k)

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
    return queue, captured, posted_url, FakeResp


class TestVerifyApiKey:
    def test_success_returns_true(self, mock_httpx):
        """200 → (True, "")。"""
        queue, _, _, FakeResp = mock_httpx
        queue.append(FakeResp(200, '{"id":"x"}'))
        ok, msg = _verify_api_key("openai", "gpt-4o-mini", "https://api.example.com/v1", "sk-test")
        assert ok is True
        assert msg == ""

    def test_uses_30s_timeout(self, mock_httpx):
        """同步验证仍带 30s 超时（httpx 层）。"""
        queue, captured, _, FakeResp = mock_httpx
        queue.append(FakeResp(200))
        _verify_api_key("openai", "gpt-4o-mini", "https://api.example.com/v1", "sk-test")
        assert captured.get("timeout") == 30.0

    def test_posts_to_base_url_chat_completions(self, mock_httpx):
        """URL = base_url + /chat/completions（base_url 惯例已含 /v1）。"""
        queue, _, posted_url, FakeResp = mock_httpx
        queue.append(FakeResp(200))
        _verify_api_key("openai", "gpt-4o-mini", "https://api.example.com/v1", "sk-test")
        assert posted_url == ["https://api.example.com/v1/chat/completions"]

    def test_base_url_empty_uses_openai_default(self, mock_httpx):
        """base_url 为空时回退 OpenAI 默认（不产生相对 URL）。"""
        queue, _, posted_url, FakeResp = mock_httpx
        queue.append(FakeResp(200))
        _verify_api_key("openai", "gpt-4o", "", "sk-test")
        assert posted_url == ["https://api.openai.com/v1/chat/completions"]

    def test_http_error_returns_false_with_status(self, mock_httpx):
        """非 200 → (False, "HTTP <code>: ...")。"""
        queue, _, _, FakeResp = mock_httpx
        queue.append(FakeResp(401, '{"error":{"message":"unauthorized"}}'))
        ok, msg = _verify_api_key("openai", "gpt-4o-mini", "https://api.example.com/v1", "sk-test")
        assert ok is False
        assert "HTTP 401" in msg
        assert "unauthorized" in msg

    def test_network_error_returns_false_with_message(self, mock_httpx):
        """网络异常 → (False, 异常信息)。"""
        queue, _, _, _ = mock_httpx
        queue.append(httpx.ConnectError("network down"))
        ok, msg = _verify_api_key("openai", "gpt-4o-mini", "https://api.example.com/v1", "sk-test")
        assert ok is False
        assert "network down" in msg
