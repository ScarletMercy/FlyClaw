"""AI-powered tools: memory search and sub-agent status."""

from __future__ import annotations

import logging

from src._container import get_container

logger = logging.getLogger("myclaw.tools.ai")

# ── Memory search ──────────────────────────────────────────


async def memory_search(query: str, max_results: int = 6) -> str:
    """Search the memory/knowledge base for relevant information.

    Use this when you need to recall previously indexed documents, notes,
    or knowledge that may help answer the user's question.

    Args:
        query: The search query describing what information you need.
        max_results: Maximum number of results to return (default 6).
    """
    searcher = get_container().memory_searcher
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


# ── Sub-agent status ───────────────────────────────────────


async def subagent_status() -> str:
    """List recent sub-agent runs and their status. Shows agent name, depth, duration, and outcome."""
    from src.agents.run_registry import get_run_registry

    registry = get_run_registry()
    runs = await registry.list_runs(limit=20)
    active = await registry.active_count()

    if not runs:
        return f"No sub-agent runs recorded. Active: {active}"

    lines = [f"Sub-agent runs ({active} active):"]
    for r in runs:
        status = r["status"]
        duration = ""
        if r.get("completed_at") and r.get("started_at"):
            dur = r["completed_at"] - r["started_at"]
            duration = f" ({dur:.1f}s)"
        task = r.get("task", "")[:60]
        line = f"- [{r['id']}] {r['agent_name']} depth={r['depth']} {status}{duration}"
        if task:
            line += f" — {task}"
        lines.append(line)
    return "\n".join(lines)


def get_tools() -> list:
    from src.agent.tooldef import ToolDef
    return [
        ToolDef.from_function(memory_search),
        ToolDef.from_function(subagent_status),
    ]
