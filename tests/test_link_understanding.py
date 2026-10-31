"""Tests for src/link_understanding.py — URL detection and deduplication."""

import pytest

from src.link_understanding import URL_PATTERN, detect_and_preview_links


# ── URL_PATTERN ────────────────────────────────────────────


class TestUrlPattern:
    def test_http_url(self):
        text = "Check https://example.com for details"
        urls = URL_PATTERN.findall(text)
        assert urls == ["https://example.com"]

    def test_http_url(self):
        text = "Visit http://example.com/page"
        urls = URL_PATTERN.findall(text)
        assert urls == ["http://example.com/page"]

    def test_multiple_urls(self):
        text = "See https://a.com and https://b.com/page?q=1"
        urls = URL_PATTERN.findall(text)
        assert len(urls) == 2

    def test_url_with_path(self):
        text = "Go to https://example.com/path/to/page"
        urls = URL_PATTERN.findall(text)
        assert urls[0] == "https://example.com/path/to/page"

    def test_no_url(self):
        text = "Just plain text without any URL"
        urls = URL_PATTERN.findall(text)
        assert urls == []

    def test_url_in_parentheses(self):
        text = "Check (https://example.com) for info"
        urls = URL_PATTERN.findall(text)
        assert len(urls) == 1

    def test_url_in_brackets(self):
        text = "See [https://example.com] for info"
        urls = URL_PATTERN.findall(text)
        assert len(urls) == 1


# ── detect_and_preview_links ───────────────────────────────


class TestDetectAndPreviewLinks:
    @pytest.mark.asyncio
    async def test_no_urls(self):
        result = await detect_and_preview_links("Just text, no links")
        assert result == ""

    @pytest.mark.asyncio
    async def test_with_urls_but_fetch_fails(self):
        # URL detection should work even if preview fails
        # We test that it doesn't crash
        result = await detect_and_preview_links("Check https://invalid.example.com/page")
        # Result is either empty (fetch failed) or has preview text
        assert isinstance(result, str)


# ── _fetch_preview 失败信封消费 ────────────────────────────────


class TestFetchPreviewErrorEnvelope:
    """失败信封(status=error)不产预览；成功信封照常产预览。"""

    @pytest.mark.asyncio
    async def test_toolresult_error_yields_no_preview(self, monkeypatch):
        from src.tools.result import ToolResult

        async def fake_fetch(url):
            return ToolResult.error("[error] Error fetching URL: boom")

        monkeypatch.setattr("src.tools.web_tools.web_fetch", fake_fetch)

        from src.link_understanding import _fetch_preview

        assert await _fetch_preview("https://x.example.com") is None

    @pytest.mark.asyncio
    async def test_success_result_still_produces_preview(self, monkeypatch):
        from src.tools.result import ToolResult

        async def fake_fetch(url):
            return ToolResult("Page Title Here\nsome body text")

        monkeypatch.setattr("src.tools.web_tools.web_fetch", fake_fetch)

        from src.link_understanding import _fetch_preview

        preview = await _fetch_preview("https://x.example.com")
        assert preview is not None
        assert "Page Title Here" in preview
