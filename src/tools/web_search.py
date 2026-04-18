from __future__ import annotations

import asyncio
import logging

from langchain_core.tools import tool

logger = logging.getLogger("myclaw.web_search")

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
