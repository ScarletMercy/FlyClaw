"""AI-powered tools: sub-agent status."""

from __future__ import annotations

import logging

logger = logging.getLogger("myclaw.tools.ai")


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
        ToolDef.from_function(subagent_status),
    ]
