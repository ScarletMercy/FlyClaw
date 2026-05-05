"""Web search and fetch tools using Tavily API."""

from __future__ import annotations

import asyncio
import logging

from langchain_core.tools import tool

logger = logging.getLogger("myclaw.web_tools")

_cached_api_key: str | None = None


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


@tool
async def web_search(query: str, max_results: int = 5) -> str:
    """Search the web using Tavily. Returns titles, URLs, and content snippets.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return. Default 5.
    """
    api_key = _get_api_key()
    if not api_key:
        return "[error] Tavily API key not configured. Set tools.web_search.api_key or TAVILY_API_KEY in config."

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
        logger.error("web_search error for query='%s': %s", query, e)
        return f"[web_search error] {e}"


@tool
async def web_fetch(url: str, max_chars: int = 8000) -> str:
    """Fetch and extract the main content from a web page using Tavily.

    Args:
        url: The URL of the web page to fetch.
        max_chars: Maximum number of characters to return. Content will be truncated if longer. Default 8000.
    """
    api_key = _get_api_key()
    if not api_key:
        return "[error] Tavily API key not configured. Set tools.web_search.api_key or TAVILY_API_KEY in config."

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)
        response = await asyncio.to_thread(client.extract, urls=[url])

        results = response.get("results", [])
        if not results:
            return f"[web_fetch] No content extracted from {url}"

        raw_content = results[0].get("raw_content", "")
        if not raw_content:
            return f"[web_fetch] No raw content found for {url}"

        if len(raw_content) > max_chars:
            raw_content = raw_content[:max_chars] + "..."

        return raw_content
    except Exception as e:
        logger.error("web_fetch error for url='%s': %s", url, e)
        return f"[web_fetch error] {e}"
