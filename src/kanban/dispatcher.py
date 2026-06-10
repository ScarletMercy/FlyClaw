"""Kanban dispatcher — periodic tick that claims tasks and spawns async workers.

Adapted from hermes-agent's ``dispatch_once()`` and ``_default_spawn()``.
Instead of spawning subprocesses, uses FlyClaw's ``delegate_task`` to run
sub-agents asynchronously.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from .store import KanbanStore
from .tools import _current_kanban_agent, _current_kanban_board, _current_kanban_task
from .types import (
    DEFAULT_CLAIM_TTL_SECONDS,
    DEFAULT_FAILURE_LIMIT,
    DispatchResult,
    KanbanTask,
)

logger = logging.getLogger("flyclaw.kanban.dispatcher")

# Track in-flight worker tasks so shutdown can cancel them.
_active_workers: set[asyncio.Task] = set()

# Kanban tools that must always be available to workers, even if the
# agent's AgentSubconfig.tools filters down to a restricted set.
_KANBAN_ESSENTIAL_TOOLS = frozenset(
    [
        "kanban_complete",
        "kanban_block",
        "kanban_heartbeat",
        "kanban_show",
    ]
)


def _register_worker(task: asyncio.Task) -> None:
    """Track a worker task and auto-remove when done."""
    _active_workers.add(task)
    task.add_done_callback(_active_workers.discard)


async def cancel_active_workers(timeout: float = 5.0) -> None:
    """Cancel and await all in-flight worker tasks. Called during shutdown."""
    if not _active_workers:
        return
    workers = list(_active_workers)
    for t in workers:
        t.cancel()
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*workers, return_exceptions=True),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "cancel_active_workers: timed out after %.1fs, %d workers still pending",
            timeout,
            len(workers),
        )
        results = []
    for r in results:
        if isinstance(r, Exception) and not isinstance(r, asyncio.CancelledError):
            logger.debug("Worker cleanup error: %s", r)
    _active_workers.clear()


# ---------------------------------------------------------------------------
# Worker spawning
# ---------------------------------------------------------------------------


async def _spawn_worker(task: KanbanTask, store: KanbanStore, *, failure_limit: int = DEFAULT_FAILURE_LIMIT) -> None:
    """Spawn an async sub-agent worker for a claimed task.

    This runs inside a ``asyncio.create_task`` so multiple workers can execute
    concurrently. The kanban tools (kanban_complete, kanban_block, etc.) are
    force-injected into the worker's tool surface regardless of the agent's
    tool whitelist, so the worker can always signal completion or request help.

    Note: We build the AgentLoop directly instead of using _run_single so we
    can force-include kanban essential tools and handle kanban-specific lifecycle
    (auto-complete, circuit breaker, CancelledError). The depth guard in
    delegate_task is intentionally bypassed — the dispatcher runs outside any
    agent loop, so _current_depth is 0 (default). We pass depth=1 for
    RunRegistry bookkeeping only.
    """
    from src.agents.delegate import (
        _register_builtin_tools,
        _filter_tools,
        _ESSENTIAL_TOOLS,
        _resolve_child_timeout,
        _HEARTBEAT_INTERVAL,
        _HEARTBEAT_STALE_IDLE_SECONDS,
    )
    from src.agents.run_registry import get_run_registry
    from src.agents.registry import get_agent_registry
    from src.config import load_config
    from src.agent.loop import AgentLoop
    from src.agent.state import AgentState, MemoryStateStore
    from src.agent.client import create_chain, create_client

    config = load_config()
    registry = get_agent_registry()
    agent_config = registry.get(task.assignee)
    if agent_config is None:
        available = [a["name"] for a in registry.list_agents()]
        raise ValueError(f"Unknown agent: '{task.assignee}'. Available: {', '.join(available)}")

    # Build worker context (parent results, comments, prior runs)
    context = await store.build_worker_context(task.id)

    # Full prompt for the worker
    task_prompt = (
        f"work kanban task {task.id}\n\n"
        f"Title: {task.title}\n"
        f"Body: {task.body or ''}\n\n"
        f"--- Worker Context ---\n{context}"
    )

    # Resolve timeout
    timeout = task.max_runtime_seconds or config.delegation.child_timeout_seconds

    run_registry = get_run_registry()

    # Set kanban task context so tools enforce ownership
    task_token = _current_kanban_task.set(task.id)
    board_token = _current_kanban_board.set(task.board)
    agent_token = _current_kanban_agent.set(task.assignee or "unknown")

    start = time.monotonic()
    run_id = None
    agent_loop = None
    monitor_task: asyncio.Task | None = None

    try:
        run_id = await run_registry.start_run(task.assignee, task_prompt, depth=1)

        # Collect ALL tools from the parent loop, then filter + force-include kanban tools
        all_tools = _register_builtin_tools(config)
        all_tools = _filter_tools(all_tools, config)

        # Apply agent-level filtering (same as _run_single does)
        if agent_config.tools and agent_config.tools != ["*"]:
            tool_map = {t.name: t for t in all_tools}
            filtered = [tool_map[p] for p in agent_config.tools if p in tool_map]
            filtered_names = {t.name for t in filtered}
            for name in _ESSENTIAL_TOOLS:
                if name not in filtered_names and name in tool_map:
                    filtered.append(tool_map[name])
            # Force-include kanban essentials so workers can always signal completion
            for name in _KANBAN_ESSENTIAL_TOOLS:
                if name not in filtered_names and name in tool_map:
                    filtered.append(tool_map[name])
            all_tools = filtered

        # Create client for the worker
        if agent_config.model:
            client = create_client(
                config.model.provider,
                agent_config.model,
                config.model.temperature,
                base_url=config.model.base_url,
                api_key=config.model.api_key,
            )
        else:
            client = create_chain(config)

        # Skills prompt from container
        skills_prompt = ""
        try:
            from src._container import get_container

            container = get_container()
            if container.agent_loop:
                skills_prompt = container.agent_loop._skills_prompt
        except Exception:
            pass

        state_store = MemoryStateStore()
        agent_loop = AgentLoop(
            client=client,
            tools=all_tools,
            state_store=state_store,
            config=config,
            skills_prompt=skills_prompt,
            context_window_tokens=config.model.context_window,
        )
        agent_loop._auto_deny_approval = True

        # Build system prompt
        system_prompt = agent_config.system_prompt
        if context:
            system_prompt = f"{system_prompt}\n\nContext:\n{context}" if system_prompt else f"Context:\n{context}"

        if task.max_runtime_seconds:
            # Explicit per-task cap: honour it directly (floor only). Bypass the
            # delegation ceiling so the wait_for timeout doesn't disagree with
            # store.enforce_max_runtime (which uses the raw value) and silently
            # kill long-running tasks the operator explicitly allowed.
            floor = float(getattr(config.delegation, "child_timeout_floor", 30))
            effective_timeout = max(floor, float(task.max_runtime_seconds))
        else:
            effective_timeout = _resolve_child_timeout(timeout, config)

        state = AgentState(
            messages=[{"role": "user", "content": task_prompt}],
            system_prompt=system_prompt,
            sender_id="",
            chat_id="",
            channel="",
        )

        max_rounds = config.delegation.max_iterations
        child_thread_id = f"kanban:{task.id}"

        # Monitor: heartbeat + interrupt detection (mirrors _run_single's _monitor)
        if run_id:

            async def _monitor(_run_id: str, _loop: AgentLoop, _tid: str):
                last_touch = time.monotonic()
                while True:
                    await asyncio.sleep(1.5)
                    now = time.monotonic()
                    # Heartbeat: touch run activity timestamp periodically
                    if now - last_touch >= _HEARTBEAT_INTERVAL:
                        run_registry.touch(_run_id)
                        last_touch = now
                    # Check interrupt
                    if run_registry.is_interrupt_requested(_run_id):
                        flag = _loop._store.get_interrupt_flag(_tid)
                        flag.interrupt("Interrupted by parent")
                        return

            monitor_task = asyncio.create_task(
                _monitor(run_id, agent_loop, child_thread_id),
                name=f"kanban-monitor-{task.id}",
            )

        # Stale detection (mirrors _run_single's _stale_detector)
        stale_task: asyncio.Task | None = None
        if run_id:

            async def _stale_detector(_run_id: str):
                while True:
                    await asyncio.sleep(_HEARTBEAT_INTERVAL)
                    run = run_registry._runs.get(_run_id)
                    if not run or run["status"] != "running":
                        return
                    last = run.get("last_activity_at") or run["started_at"]
                    idle = time.time() - last
                    if idle > _HEARTBEAT_STALE_IDLE_SECONDS:
                        logger.warning(
                            "Kanban worker run '%s' appears stale: no activity for %.0fs",
                            _run_id,
                            idle,
                        )

            stale_task = asyncio.create_task(
                _stale_detector(run_id),
                name=f"kanban-stale-{task.id}",
            )

        # Pre-start interrupt check (mirrors _run_single)
        if run_id and run_registry.is_interrupt_requested(run_id):
            await run_registry.complete_run(run_id, result="[interrupted]")
            return

        # Set _current_thread_id for exec tool approval tracking
        from src.tools.exec import _current_thread_id

        _tid_token = _current_thread_id.set(child_thread_id)
        try:
            result_state = await asyncio.wait_for(
                agent_loop.run(state, child_thread_id, max_rounds=max_rounds),
                timeout=effective_timeout,
            )
        finally:
            _current_thread_id.reset(_tid_token)

        # Extract result text (three-tier fallback, same as _run_single)
        result_text = ""
        # Tier 1: last assistant message with content
        for msg in reversed(result_state.messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                result_text = msg["content"]
                break
        # Tier 2: last non-error tool result from last assistant's tool calls
        if not result_text:
            for msg in reversed(result_state.messages):
                if msg.get("role") == "assistant":
                    for tc in msg.get("tool_calls", []):
                        tc_id = tc.get("id", "")
                        if tc_id:
                            for m in result_state.messages:
                                if m.get("role") == "tool" and m.get("tool_call_id") == tc_id:
                                    content = m.get("content", "")
                                    if content and not content.startswith("[error]"):
                                        result_text = content[:2000]
                                        break
                            if result_text:
                                break
                    break
        # Tier 3: concatenate raw tool results
        if not result_text:
            tool_parts = []
            for msg in result_state.messages:
                if msg.get("role") == "tool" and msg.get("content"):
                    tool_parts.append(msg["content"][:500])
            if tool_parts:
                result_text = "[No final summary] Tool results:\n" + "\n---\n".join(tool_parts[-3:])

        duration = round(time.monotonic() - start, 2)
        await run_registry.complete_run(run_id, result=result_text[:500])

        # Auto-complete: if the worker finished but didn't call kanban_complete
        t = await store.get_task(task.id)
        if t and t.status == "running":
            ok = await store.complete_task(
                task.id,
                summary=result_text[:500] if result_text else "Worker completed without explicit kanban_complete",
            )
            if ok:
                try:
                    from src.events import emit_async

                    await emit_async(
                        "kanban.task.completed",
                        task_id=task.id,
                        kind="completed",
                        summary="auto-completed",
                    )
                except Exception:
                    logger.debug("Failed to emit kanban auto-complete event", exc_info=True)

    except asyncio.CancelledError:
        # Graceful shutdown: return task to ready WITHOUT counting as a failure,
        # so process restarts don't push it toward the circuit-breaker limit.
        try:
            await store.release_task(task.id, reason="Worker cancelled (shutdown)")
        except Exception:
            logger.debug("Failed to release task %s on cancellation", task.id, exc_info=True)
        if run_id:
            try:
                await run_registry.fail_run(run_id, error="Cancelled")
            except Exception:
                pass
        raise

    except asyncio.TimeoutError:
        duration = round(time.monotonic() - start, 2)
        if run_id:
            try:
                await run_registry.timeout_run(run_id, error=f"Timeout after {timeout}s")
            except Exception:
                pass
        logger.warning("Kanban worker timed out for task %s (%.0fs)", task.id, timeout)
        await _handle_timeout(store, task.id, timeout, failure_limit=failure_limit)

    except Exception as exc:
        duration = round(time.monotonic() - start, 2)
        if run_id:
            try:
                await run_registry.fail_run(run_id, error=str(exc))
            except Exception:
                pass
        await _handle_error(store, task.id, str(exc), failure_limit=failure_limit)

    finally:
        for t in (monitor_task, stale_task):
            if t:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
        _current_kanban_task.reset(task_token)
        _current_kanban_board.reset(board_token)
        _current_kanban_agent.reset(agent_token)


async def _handle_timeout(
    store: KanbanStore, task_id: str, timeout: int | float, *, failure_limit: int = DEFAULT_FAILURE_LIMIT
) -> None:
    auto_blocked = await store.record_task_failure(
        task_id, f"Timeout after {timeout}s", outcome="timed_out", failure_limit=failure_limit
    )
    if auto_blocked:
        logger.info("Task %s auto-blocked after timeout", task_id)
    try:
        from src.events import emit_async

        await emit_async("kanban.task.timed_out", task_id=task_id, kind="timed_out")
    except Exception:
        logger.debug("Failed to emit kanban timeout event", exc_info=True)


async def _handle_error(
    store: KanbanStore, task_id: str, error: str, *, failure_limit: int = DEFAULT_FAILURE_LIMIT
) -> None:
    logger.error("Kanban worker failed for task %s: %s", task_id, error)
    auto_blocked = await store.record_task_failure(task_id, error, outcome="crashed", failure_limit=failure_limit)
    if auto_blocked:
        logger.info("Task %s auto-blocked after crash", task_id)
    try:
        from src.events import emit_async

        await emit_async("kanban.task.crashed", task_id=task_id, kind="crashed", error=error)
    except Exception:
        logger.debug("Failed to emit kanban crash event", exc_info=True)


# ---------------------------------------------------------------------------
# Dispatch tick
# ---------------------------------------------------------------------------


async def dispatch_once(
    store: KanbanStore,
    *,
    dry_run: bool = False,
    max_spawn: Optional[int] = None,
    failure_limit: int = DEFAULT_FAILURE_LIMIT,
    ttl_seconds: int = DEFAULT_CLAIM_TTL_SECONDS,
    board: Optional[str] = None,
) -> DispatchResult:
    """Run one dispatcher tick.

    Steps:
    1. Release stale claims (TTL expired / heartbeat gone)
    2. Enforce max runtime limits
    3. Promote todo -> ready where all parents done
    4. For each ready + assigned + unclaimed task: claim + spawn async worker
    """
    result = DispatchResult()

    # 1. Release stale claims
    result.reclaimed = await store.release_stale_claims(ttl_seconds)

    # 2. Enforce max runtime
    result.timed_out = await store.enforce_max_runtime()

    # 3. Promote todo -> ready
    result.promoted = await store.recompute_ready()

    # 4. Spawn workers for ready+assigned+unclaimed tasks
    ready_tasks = await store.list_ready_unclaimed(board=board)
    running_count = await store.count_running(board=board)
    spawned = 0

    for task in ready_tasks:
        if max_spawn is not None and running_count + spawned >= max_spawn:
            break

        if not task.assignee:
            result.skipped_unassigned.append(task.id)
            continue

        if dry_run:
            result.spawned.append((task.id, task.assignee))
            continue

        # Atomic claim
        claimed = await store.claim_task(task.id, ttl_seconds=ttl_seconds)
        if not claimed:
            continue  # Already claimed by a concurrent tick

        try:
            # Create tracked background task with done-callback
            worker_task = asyncio.create_task(
                _spawn_worker(claimed, store, failure_limit=failure_limit),
                name=f"kanban-worker-{claimed.id}",
            )
            _register_worker(worker_task)
            result.spawned.append((claimed.id, claimed.assignee))
            spawned += 1
        except Exception as exc:
            logger.error("Failed to spawn worker for task %s: %s", task.id, exc)
            auto = await store.record_task_failure(
                task.id, str(exc), outcome="spawn_failed", failure_limit=failure_limit
            )
            if auto:
                result.auto_blocked.append(task.id)

    return result


# ---------------------------------------------------------------------------
# Dispatch loop (runs as a background task)
# ---------------------------------------------------------------------------


async def run_dispatch_loop(
    store: KanbanStore,
    *,
    interval_seconds: int = 60,
    max_spawn: int = 3,
    failure_limit: int = DEFAULT_FAILURE_LIMIT,
    ttl_seconds: int = DEFAULT_CLAIM_TTL_SECONDS,
    board: Optional[str] = None,
) -> None:
    """Infinite loop that calls dispatch_once() at regular intervals.

    Intended to be run as an ``asyncio.Task`` from ServiceContainer.
    Shutdown relies on cancelling this task (CancelledError propagation).
    """
    # Clamp interval to prevent CPU spinning
    interval = max(interval_seconds, 5)

    logger.info(
        "Kanban dispatch loop started (interval=%ds, max_spawn=%d)",
        interval,
        max_spawn,
    )
    while True:
        try:
            r = await dispatch_once(
                store,
                max_spawn=max_spawn,
                failure_limit=failure_limit,
                ttl_seconds=ttl_seconds,
                board=board,
            )
            if r.reclaimed or r.promoted or r.spawned or r.auto_blocked or r.timed_out:
                logger.info(
                    "Kanban tick: reclaimed=%d promoted=%d spawned=%d auto_blocked=%d timed_out=%d",
                    r.reclaimed,
                    r.promoted,
                    len(r.spawned),
                    len(r.auto_blocked),
                    len(r.timed_out),
                )
        except asyncio.CancelledError:
            raise  # Let cancellation propagate for graceful shutdown
        except Exception:
            logger.exception("Kanban dispatch tick failed")
        await asyncio.sleep(interval)
