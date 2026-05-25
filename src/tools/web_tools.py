"""Web search and fetch tools.

web_fetch: direct HTTP fetch with HTML-to-markdown conversion (no external API).
web_search: Tavily API search (requires API key). Falls back to Bing scraping when no key is set.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from urllib.parse import urlparse

import httpx

from src.agent.tooldef import ToolDef

logger = logging.getLogger("flyclaw.web_tools")

_cached_api_key: str | None = None

FETCH_TIMEOUT = 30.0
BING_TIMEOUT = 15.0
MAX_MARKDOWN_LENGTH = 100_000
MAX_URL_LENGTH = 2000

_BING_HEADERS = {
    "User-Agent": "flyclaw/1.0 (compatible; web-search)",
    "Accept": "text/html,application/xhtml+xml,*/*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

# Binary content types that we cannot meaningfully extract text from
_BINARY_CONTENT_TYPES = [
    "application/pdf",
    "application/zip",
    "application/x-tar",
    "application/gzip",
    "application/x-bzip2",
    "image/",
    "video/",
    "audio/",
]


def _get_api_key() -> str:
    global _cached_api_key
    if _cached_api_key is not None:
        return _cached_api_key
    try:
        from src.config import load_config
        cfg = load_config()
        _cached_api_key = cfg.tools.web_search.api_key or ""
    except Exception as e:
        logger.warning("Failed to load web search config: %s", e)
        _cached_api_key = ""
    return _cached_api_key


def _is_binary_content_type(content_type: str) -> bool:
    ct = content_type.lower()
    return any(bt in ct for bt in _BINARY_CONTENT_TYPES)


def _strip_tags(html: str) -> str:
    text = re.sub(r"<[^>]+>", "", html)
    return re.sub(r"\s+", " ", text).strip()


def _unescape_unicode(s: str) -> str:
    return re.sub(
        r"\\u([0-9a-fA-F]{4})",
        lambda m: chr(int(m.group(1), 16)),
        s,
    )


def _html_to_markdown(html: str) -> str:
    try:
        from markdownify import markdownify as md
        return md(html)
    except ImportError:
        # Fallback: strip tags manually
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


async def _bing_search(query: str, max_results: int = 5) -> str:
    from urllib.parse import quote_plus

    query = _unescape_unicode(query)
    url = f"https://cn.bing.com/search?q={quote_plus(query)}"
    logger.info("bing_search request URL: %s", url)
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=BING_TIMEOUT,
            headers=_BING_HEADERS,
        ) as client:
            response = await client.get(url)
        logger.info("bing_search actual URL: %s", response.url)

        if response.status_code != 200:
            return f"[bing_search error] HTTP {response.status_code}"

        html = response.text

        algo_blocks = re.findall(
            r'<li\s+class="b_algo"[^>]*>(.*?)</li>', html, re.DOTALL
        )

        results: list[str] = []
        for block in algo_blocks[:max_results]:
            h2_a = re.search(
                r'<h2[^>]*>.*?<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
                block,
                re.DOTALL,
            )
            if not h2_a:
                continue
            link = h2_a.group(1)
            title = _strip_tags(h2_a.group(2))

            cap_p = re.search(
                r'<div\s+class="b_caption">\s*<p[^>]*>(.*?)</p>',
                block,
                re.DOTALL,
            )
            snippet = _strip_tags(cap_p.group(1)) if cap_p else ""

            entry_parts: list[str] = []
            if title:
                entry_parts.append(f"**{title}**")
            if link:
                entry_parts.append(link)
            if snippet:
                entry_parts.append(snippet[:400])
            if entry_parts:
                results.append("\n".join(entry_parts))

        return "\n\n".join(results) if results else "No results found."
    except httpx.TimeoutException:
        return f"[bing_search error] Request timed out after {BING_TIMEOUT}s"
    except Exception as e:
        logger.error("bing_search error for query='%s': %s", query, e)
        return f"[bing_search error] {e}"


async def web_fetch(url: str) -> str:
    """Fetch content from a URL and convert it to markdown text.

    Uses direct HTTP request — no external API key needed.
    HTML pages are converted to markdown. Other content types are returned as-is.

    Args:
        url: The URL to fetch.
    """
    start = time.time()

    if len(url) > MAX_URL_LENGTH:
        return f"URL exceeds maximum length of {MAX_URL_LENGTH} characters"

    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return f"Invalid URL: {url}"
    except Exception:
        return f"Invalid URL: {url}"

    # SSRF protection
    from src.security.url_safety import is_safe_url
    safe, reason = is_safe_url(url)
    if not safe:
        logger.warning("Blocked URL fetch (SSRF): %s — %s", url, reason)
        return f"Blocked: {reason}"

    # Prefer HTTPS
    if parsed.scheme == "http":
        url = url.replace("http://", "https://", 1)

    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=FETCH_TIMEOUT,
            max_redirects=10,
            headers={
                "Accept": "text/markdown, text/html, text/plain, */*",
                "User-Agent": "flyclaw/1.0",
            },
        ) as client:
            response = await client.get(url)

        content_type = response.headers.get("content-type", "")
        raw_bytes = len(response.content)

        if _is_binary_content_type(content_type):
            return (
                f"Binary content ({content_type}, {raw_bytes} bytes). "
                f"Cannot extract text."
            )

        if "text/html" in content_type:
            markdown_content = _html_to_markdown(response.text)
        else:
            markdown_content = response.text

        if len(markdown_content) > MAX_MARKDOWN_LENGTH:
            markdown_content = (
                markdown_content[:MAX_MARKDOWN_LENGTH]
                + "\n\n[Content truncated due to length...]"
            )

        elapsed_ms = int((time.time() - start) * 1000)
        return (
            f"Content from {url} ({raw_bytes} bytes, {elapsed_ms}ms):\n\n"
            f"{markdown_content}\n\n---"
        )

    except httpx.TimeoutException:
        return f"Request timed out after {FETCH_TIMEOUT}s"
    except httpx.HTTPError as e:
        return f"HTTP error: {e}"
    except Exception as e:
        return f"Error fetching URL: {e}"


async def web_search(query: str, max_results: int = 5) -> str:
    """Search the web using Tavily. Returns titles, URLs, and content snippets.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return. Default 5.
    """
    api_key = _get_api_key()
    logger.info("web_search raw query: %r", query)
    query = _unescape_unicode(query)
    logger.info("web_search after unescape: %r", query)
    if not api_key:
        logger.info("No Tavily API key configured, falling back to Bing search")
        return await _bing_search(query, max_results)

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)
        response = await asyncio.to_thread(
            client.search,
            query=query,
            max_results=max_results,
            include_answer=False,
        )

        results = []
        for r in response.get("results", []):
            title = r.get("title", "")
            url = r.get("url", "")
            content = r.get("content", "")
            entry_parts = []
            if title:
                entry_parts.append(f"**{title}**")
            if url:
                entry_parts.append(url)
            if content:
                entry_parts.append(content[:400])
            if entry_parts:
                results.append("\n".join(entry_parts))

        return "\n\n".join(results) if results else "No results found."
    except Exception as e:
        logger.warning("Tavily failed for query='%s', falling back to Bing: %s", query, e)
        return await _bing_search(query, max_results)


def get_tools() -> list[ToolDef]:
    return [ToolDef.from_function(web_fetch), ToolDef.from_function(web_search)]
