"""Agent tool for searching conversation history."""

from __future__ import annotations

from datetime import datetime


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
        lines.append(
            f"[{r.get('channel', '?')}] {ts} ({r.get('message_count', 0)}msgs) {snippet}{active}"
        )
    return "\n".join(lines)


async def session_search(query: str = "", limit: int = 3) -> str:
    """Search historical conversation records by keyword (FTS5).

    Args:
        query: Search keyword. Empty string returns recent sessions.
        limit: Max number of results to return. Default 3.
    """
    from src.session_index.store import get_session_index

    store = get_session_index()
    if not store:
        return "Search not enabled"

    results = store.search(query, limit=limit)
    if not results:
        return "No sessions" if not query.strip() else "No results found"
    return _format_results(results)


def get_tools() -> list:
    from src.agent.tooldef import ToolDef
    return [
        ToolDef.from_function(session_search),
    ]
