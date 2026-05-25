"""Agent tool for searching conversation history."""

from __future__ import annotations

import logging
from datetime import datetime

logger = logging.getLogger("flyclaw.session_search_tools")


def _format_results(results: list[dict]) -> str:
    lines = []
    for r in results:
        ts = (
            datetime.fromtimestamp(r["last_message_at"]).strftime("%m-%d %H:%M")
            if r.get("last_message_at")
            else "?"
        )
        active = "" if r.get("is_active", True) else " [expired]"
        snippet = (r.get("snippet") or "")[:120]
        reason = f" ← {r['reason']}" if r.get("reason") else ""
        lines.append(
            f"[{r.get('channel', '?')}] {ts} ({r.get('message_count', 0)}msgs) {snippet}{active}{reason}"
        )
    return "\n".join(lines)


async def session_search(query: str, limit: int = 3) -> str:
    """Search historical conversation records. Supports keywords and semantic search.

    Examples:
      session_search("Docker deploy") -- search for Docker conversations
      session_search("yesterday script") -- search history
      session_search("")  -- browse recent sessions
    """
    from src.session_index.store import get_session_index

    store = get_session_index()
    if not store:
        return "Search not enabled"

    if not query.strip():
        results = store.search("", limit=limit)
        if not results:
            return "No sessions"
        return _format_results(results)

    # Try LLM semantic search first
    results = await _try_llm_search(store, query, limit)
    if results is not None:
        return _format_results(results) if results else "No results found"

    # Fallback: FTS5 keyword search
    results = store.search(query, limit=limit)
    if not results:
        return "No results found"
    return _format_results(results)


async def _try_llm_search(store, query: str, limit: int) -> list[dict] | None:
    """Try LLM semantic search. Returns results or None on failure (caller should fallback)."""
    try:
        from src.config import load_config

        config = load_config()
        sc = config.session_search

        model_name = sc.search_model
        if not model_name:
            return None  # No model configured, fallback to FTS5

        base_url = sc.search_model_base_url or config.model.base_url
        api_key = sc.search_model_api_key or config.model.api_key

        if not base_url or not api_key:
            return None

        from src.session_index.search import llm_search

        return await llm_search(
            store=store,
            query=query,
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            max_candidates=20,
            max_results=limit,
        )
    except Exception as e:
        logger.warning("LLM search failed, falling back to FTS5: %s", e)
        return None


def get_tools() -> list:
    from src.agent.tooldef import ToolDef
    return [
        ToolDef.from_function(session_search),
    ]
