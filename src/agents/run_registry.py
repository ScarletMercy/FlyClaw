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

logger = logging.getLogger("flyclaw.agents.registry")


class SubagentRun(TypedDict):
    id: str
    agent_name: str
    task: str
    parent_id: Optional[str]
    depth: int
    status: Literal["pending", "running", "completed", "error", "timeout", "interrupted"]
    started_at: float
    completed_at: Optional[float]
    result: Optional[str]
    interrupt_requested: bool
    last_activity_at: Optional[float]


_current_depth: ContextVar[int] = ContextVar("_current_depth", default=0)


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
            "interrupt_requested": False,
            "last_activity_at": time.time(),
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

    async def timeout_run(self, run_id: str, error: str = "") -> None:
        async with self._lock:
            run = self._runs.get(run_id)
            if run:
                run["status"] = "timeout"
                run["completed_at"] = time.time()
                run["result"] = error[:500] if error else None

    def touch(self, run_id: str) -> None:
        run = self._runs.get(run_id)
        if run and run["status"] == "running":
            run["last_activity_at"] = time.time()

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

    async def get_active_tree(self) -> list[dict]:
        """Return currently running sub-agents with elapsed time."""
        async with self._lock:
            now = time.time()
            active = []
            for r in self._runs.values():
                if r["status"] in ("running", "pending"):
                    active.append({
                        "id": r["id"],
                        "agent_name": r["agent_name"],
                        "task": r["task"][:100],
                        "depth": r["depth"],
                        "status": r["status"],
                        "started_at": r["started_at"],
                        "elapsed": round(now - r["started_at"], 1),
                        "last_activity_at": r.get("last_activity_at"),
                        "idle_seconds": round(now - (r.get("last_activity_at") or r["started_at"]), 1),
                        "interrupt_requested": r.get("interrupt_requested", False),
                    })
            return sorted(active, key=lambda x: x["started_at"])

    async def request_interrupt(self, run_id: str) -> bool:
        """Request interruption of a running sub-agent. Returns True if found."""
        async with self._lock:
            run = self._runs.get(run_id)
            if run and run["status"] in ("running", "pending"):
                run["interrupt_requested"] = True
                return True
            return False

    def is_interrupt_requested(self, run_id: str) -> bool:
        """Non-async check if interrupt was requested (called from sync code)."""
        run = self._runs.get(run_id)
        if run:
            return run.get("interrupt_requested", False)
        return False

    async def _evict(self) -> None:
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


# ── Module-level singleton — delegates to ServiceContainer ──

from src._container import get_container


def get_run_registry() -> RunRegistry:
    return get_container().run_registry


def get_current_depth() -> int:
    return _current_depth.get(0)


def set_current_depth(depth: int) -> None:
    _current_depth.set(depth)
