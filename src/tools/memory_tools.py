"""Memory search tool for the agent.

Uses lazy singleton pattern (like media_understanding_tools.py).
"""
from __future__ import annotations

import logging
from typing import Optional

from langchain_core.tools import tool

logger = logging.getLogger("myclaw.tools.memory")

_searcher = None


def set_memory_searcher(searcher):
    global _searcher
    _searcher = searcher


def _get_searcher():
    return _searcher


@tool
async def memory_search(query: str, max_results: int = 6) -> str:
    """Search the memory/knowledge base for relevant information.

    Use this when you need to recall previously indexed documents, notes,
    or knowledge that may help answer the user's question.

    Args:
        query: The search query describing what information you need.
        max_results: Maximum number of results to return (default 6).
    """
    searcher = _get_searcher()
    if not searcher:
        return "Memory search is not available (not configured or not initialized)."

    try:
        results = await searcher.search(query, max_results=max_results)
        if not results:
            return f"No results found for: {query}"

        lines = []
        for i, r in enumerate(results):
            source = r.get("path", "unknown")
            score = r.get("score", 0)
            content = r.get("content", "")
            lines.append(f"[{i + 1}] (score={score}, source={source})")
            lines.append(content[:300])
            if len(content) > 300:
                lines.append("...")
            lines.append("")

        return "\n".join(lines)
    except Exception as e:
        logger.error("Memory search failed: %s", e)
        return f"Memory search error: {e}"
