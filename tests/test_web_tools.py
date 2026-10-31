"""Tests for src/tools/web_tools.py — URL helpers, HTML conversion, content type detection."""

import pytest

from src.tools.web_tools import (
    _is_binary_content_type,
    _strip_tags,
    _unescape_unicode,
    _html_to_markdown,
    MAX_MARKDOWN_LENGTH,
    FETCH_TIMEOUT,
    BING_TIMEOUT,
)


# ── _is_binary_content_type ────────────────────────────────


class TestIsBinaryContentType:
    def test_pdf(self):
        assert _is_binary_content_type("application/pdf") is True

    def test_zip(self):
        assert _is_binary_content_type("application/zip") is True

    def test_image_png(self):
        assert _is_binary_content_type("image/png") is True

    def test_image_jpeg(self):
        assert _is_binary_content_type("image/jpeg") is True

    def test_video_mp4(self):
        assert _is_binary_content_type("video/mp4") is True

    def test_audio_mp3(self):
        assert _is_binary_content_type("audio/mpeg") is True

    def test_html(self):
        assert _is_binary_content_type("text/html") is False

    def test_plain_text(self):
        assert _is_binary_content_type("text/plain") is False

    def test_json(self):
        assert _is_binary_content_type("application/json") is False

    def test_case_insensitive(self):
        assert _is_binary_content_type("Application/PDF") is True
        assert _is_binary_content_type("IMAGE/PNG") is True

    def test_content_type_with_charset(self):
        assert _is_binary_content_type("text/html; charset=utf-8") is False


# ── _strip_tags ────────────────────────────────────────────


class TestStripTags:
    def test_no_tags(self):
        assert _strip_tags("hello world") == "hello world"

    def test_simple_tag(self):
        assert _strip_tags("<b>bold</b>") == "bold"

    def test_nested_tags(self):
        result = _strip_tags("<div><p>Hello</p></div>")
        assert "Hello" in result

    def test_attributes(self):
        result = _strip_tags('<a href="http://example.com">Link</a>')
        assert "Link" in result

    def test_whitespace_collapse(self):
        result = _strip_tags("  hello   world  ")
        assert result == "hello world"


# ── _unescape_unicode ──────────────────────────────────────


class TestUnescapeUnicode:
    def test_basic(self):
        assert _unescape_unicode("\\u4f60\\u597d") == "你好"

    def test_no_escapes(self):
        assert _unescape_unicode("hello") == "hello"

    def test_mixed(self):
        result = _unescape_unicode("hello \\u4e16\\u754c")
        assert result == "hello 世界"

    def test_empty(self):
        assert _unescape_unicode("") == ""


# ── _html_to_markdown ──────────────────────────────────────


class TestHtmlToMarkdown:
    def test_simple_html(self):
        result = _html_to_markdown("<p>Hello <b>world</b></p>")
        assert "Hello" in result
        assert "world" in result

    def test_script_tags_removed(self):
        result = _html_to_markdown("<script>alert('xss')</script><p>Content</p>")
        assert "alert" not in result
        assert "Content" in result

    def test_style_tags_removed(self):
        result = _html_to_markdown("<style>body{color:red}</style><p>Text</p>")
        assert "color" not in result
        assert "Text" in result

    def test_plain_text_passthrough(self):
        result = _html_to_markdown("just text")
        assert "just text" in result

    def test_empty_input(self):
        result = _html_to_markdown("")
        assert result == ""


# ── Constants ──────────────────────────────────────────────


class TestConstants:
    def test_max_markdown_length(self):
        assert MAX_MARKDOWN_LENGTH == 100_000

    def test_fetch_timeout(self):
        assert FETCH_TIMEOUT == 30.0

    def test_bing_timeout(self):
        assert BING_TIMEOUT == 15.0


# ── web_fetch 失败信封 ────────────────────────────────


class TestWebFetchErrorEnvelope:
    """失败路径返回 ToolResult.error；content 保留 [error] 前缀措辞。"""

    @pytest.mark.asyncio
    async def test_invalid_url(self):
        from src.tools.web_tools import web_fetch

        result = await web_fetch("not a url")
        assert result.is_error is True
        assert result.content == "[error] Invalid URL: not a url"

    @pytest.mark.asyncio
    async def test_url_exceeds_max_length(self):
        from src.tools.web_tools import MAX_URL_LENGTH, web_fetch

        result = await web_fetch("http://" + "a" * (MAX_URL_LENGTH + 10))
        assert result.is_error is True
        assert "exceeds maximum length" in result.content

    @pytest.mark.asyncio
    async def test_timeout_returns_toolresult_error(self, monkeypatch):
        """safe_fetch 抛超时 → ToolResult.error（safe_fetch 为函数内 import，patch 源模块）。"""
        import httpx

        async def fake_safe_fetch(url, **kw):
            raise httpx.TimeoutException("timed out")

        monkeypatch.setattr("src.security.url_safety.safe_fetch", fake_safe_fetch)

        from src.tools.web_tools import web_fetch

        result = await web_fetch("https://example.com/slow")
        assert result.is_error is True
        assert "timed out" in result.content
