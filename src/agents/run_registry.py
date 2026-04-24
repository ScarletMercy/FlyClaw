"""Sub-agent run registry with depth tracking via contextvars.

Tracks active sub-agent runs, enforces depth limits, and provides
status querying. Pattern follows the original openclaw
src/agents/subagent-registry.ts and src/agents/subagent-depth.ts.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextvars import ContextVar
from typing import Literal, Optional, TypedDict

logger = logging.getLogger("myclaw.agents.registry")


class SubagentRun(TypedDict):
    id: str
    agent_name: str
    task: str
    parent_id: Optional[str]
    depth: int
    status: Literal["pending", "running", "completed", "error"]
    started_at: float
    completed_at: Optional[float]
    result: Optional[str]


_current_depth: ContextVar[int] = ContextVar("_current_depth", default=0)

_registry: Optional[RunRegistry] = None


class RunRegistry:
    """In-memory registry for tracking sub-agent runs."""

    def __init__(self, max_history: int = 200):
        self._runs: dict[str, SubagentRun] = {}
        self._lock = asyncio.Lock()
        self._max_history = max_history

    async def start_run(
        self,
        agent_name: str,
        task: str,
        parent_id: Optional[str] = None,
        depth: int = 1,
    ) -> str:
        run_id = str(uuid.uuid4())[:8]
        run: SubagentRun = {
            "id": run_id,
            "agent_name": agent_name,
            "task": task[:200],
            "parent_id": parent_id,
            "depth": depth,
            "status": "running",
            "started_at": time.time(),
            "completed_at": None,
            "result": None,
        }
        async with self._lock:
            self._runs[run_id] = run
            await self._evict()
        logger.info("Sub-agent run started: %s [%s] depth=%d", agent_name, run_id, depth)
        return run_id

    async def complete_run(self, run_id: str, result: str = "") -> None:
        async with self._lock:
            run = self._runs.get(run_id)
            if run:
                run["status"] = "completed"
                run["completed_at"] = time.time()
                run["result"] = result[:500] if result else None

    async def fail_run(self, run_id: str, error: str = "") -> None:
        async with self._lock:
            run = self._runs.get(run_id)
            if run:
                run["status"] = "error"
                run["completed_at"] = time.time()
                run["result"] = error[:500] if error else None

    async def list_runs(self, agent_name: Optional[str] = None, limit: int = 20) -> list[dict]:
        async with self._lock:
            runs = list(self._runs.values())
            if agent_name:
                runs = [r for r in runs if r["agent_name"] == agent_name]
            # Sort by started_at descending (most recent first)
            runs.sort(key=lambda r: r["started_at"], reverse=True)
            return runs[:limit]

    async def active_count(self) -> int:
        async with self._lock:
            return sum(1 for r in self._runs.values() if r["status"] == "running")

    async def _evict(self):
        """Remove oldest completed runs beyond max_history."""
        if len(self._runs) <= self._max_history:
            return
        completed = sorted(
            [(rid, r) for rid, r in self._runs.items() if r["status"] in ("completed", "error")],
            key=lambda x: x[1].get("completed_at") or 0,
        )
        to_remove = len(self._runs) - self._max_history
        for rid, _ in completed[:to_remove]:
            del self._runs[rid]


def get_run_registry() -> RunRegistry:
    global _registry
    if _registry is None:
        _registry = RunRegistry()
    return _registry


def init_run_registry() -> RunRegistry:
    global _registry
    _registry = RunRegistry()
    return _registry


def get_current_depth() -> int:
    return _current_depth.get(0)


def set_current_depth(depth: int) -> None:
    _current_depth.set(depth)
