from __future__ import annotations

import json
import logging
from typing import Optional
from contextvars import ContextVar

from langchain_core.tools import tool

logger = logging.getLogger("myclaw.cron_tools")

_cron_service = None

# Auto-injected by message handler before each user interaction
_current_chat_id: ContextVar[str] = ContextVar("_current_chat_id", default="")


def set_current_chat_id(chat_id: str):
    _current_chat_id.set(chat_id)


def set_cron_service(svc):
    global _cron_service
    _cron_service = svc


def _get_service():
    if _cron_service is None:
        try:
            from src.main import _app_instance
            if _app_instance and _app_instance.cron_service:
                set_cron_service(_app_instance.cron_service)
        except Exception:
            pass
    return _cron_service


@tool
async def cron_list() -> str:
    """List all scheduled cron jobs. Returns job IDs, names, schedules, enabled status, and last run info."""
    svc = _get_service()
    if not svc:
        return "Cron service is not available."
    jobs = svc.list_jobs()
    if not jobs:
        return "No cron jobs configured."
    lines = []
    for j in jobs:
        sched = j.schedule
        if sched.kind == "cron":
            sched_str = f"cron '{sched.expr}'"
        elif sched.kind == "every":
            sched_str = f"every {sched.every_seconds}s"
        else:
            sched_str = f"at {sched.at}"
        status = "enabled" if j.enabled else "disabled"
        last = f", last: {j.last_run_status}" if j.last_run_status else ""
        lines.append(f"- [{j.id}] {j.name} ({status}, {sched_str}{last})")
    return "\n".join(lines)


@tool
async def cron_add(
    name: str,
    message: str,
    schedule_kind: str = "cron",
    cron_expr: Optional[str] = None,
    every_seconds: Optional[int] = None,
    run_at: Optional[str] = None,
    enabled: bool = True,
    description: str = "",
) -> str:
    """Create a new cron job. Execution results will be automatically sent to the current chat.

    Args:
        name: Job name (e.g. "daily_summary")
        message: The prompt to send to the agent when the job runs
        schedule_kind: "cron" (cron expression), "every" (interval in seconds), or "at" (one-shot ISO datetime)
        cron_expr: Cron expression like "0 9 * * 1-5" (required if schedule_kind="cron")
        every_seconds: Interval in seconds (required if schedule_kind="every")
        run_at: ISO datetime string like "2026-04-18 11:12:00" (required if schedule_kind="at")
        enabled: Whether the job is active
        description: Optional description
    """
    svc = _get_service()
    if not svc:
        return "Cron service is not available."

    from src.cron.types import CronSchedule, CronJobCreate, CronPayload, CronDelivery

    if schedule_kind == "cron" and not cron_expr:
        return "Error: cron_expr is required when schedule_kind is 'cron'."
    if schedule_kind == "every" and not every_seconds:
        return "Error: every_seconds is required when schedule_kind is 'every'."
    if schedule_kind == "at" and not run_at:
        return "Error: run_at is required when schedule_kind is 'at'."

    schedule = CronSchedule(
        kind=schedule_kind,
        expr=cron_expr,
        every_seconds=every_seconds,
        at=run_at,
    )
    payload = CronPayload(kind="agent_turn", message=message)
    chat_id = _current_chat_id.get("")
    delivery = CronDelivery(mode="announce", to=chat_id) if chat_id else CronDelivery()
    if not chat_id:
        logger.warning("cron_add: no current chat_id, delivery will be disabled")
    create = CronJobCreate(
        name=name,
        description=description,
        enabled=enabled,
        schedule=schedule,
        payload=payload,
        delivery=delivery,
    )
    try:
        job = await svc.add_job(create)
        return f"Job created: [{job.id}] {job.name}"
    except Exception as e:
        return f"Failed to create job: {e}"


@tool
async def cron_delete(job_id: str) -> str:
    """Delete a cron job by its ID.

    Args:
        job_id: The job ID (from cron_list)
    """
    svc = _get_service()
    if not svc:
        return "Cron service is not available."
    try:
        job = await svc.remove_job(job_id)
        if job:
            return f"Job deleted: [{job_id}] {job.name}"
        return f"Job not found: {job_id}"
    except Exception as e:
        return f"Failed to delete job: {e}"


@tool
async def cron_toggle(job_id: str) -> str:
    """Enable or disable a cron job (toggles its current state).

    Args:
        job_id: The job ID (from cron_list)
    """
    svc = _get_service()
    if not svc:
        return "Cron service is not available."
    job = svc.get_job(job_id)
    if not job:
        return f"Job not found: {job_id}"
    from src.cron.types import CronJobPatch
    patch = CronJobPatch(enabled=not job.enabled)
    updated = await svc.update_job(job_id, patch)
    if updated:
        state = "enabled" if updated.enabled else "disabled"
        return f"Job [{job_id}] {updated.name} is now {state}."
    return f"Failed to toggle job: {job_id}"


@tool
async def cron_run(job_id: str) -> str:
    """Immediately trigger a cron job execution.

    Args:
        job_id: The job ID (from cron_list)
    """
    svc = _get_service()
    if not svc:
        return "Cron service is not available."
    job = svc.get_job(job_id)
    if not job:
        return f"Job not found: {job_id}"
    try:
        result = await svc.run_job_now(job_id)
        if result:
            return f"Job [{job_id}] {job.name}: {result.status}" + (f"\n{result.output}" if result.output else "")
        return f"Job [{job_id}] triggered."
    except Exception as e:
        return f"Failed to run job: {e}"
