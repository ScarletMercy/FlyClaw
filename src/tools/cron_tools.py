from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Literal

from src._container import get_container

logger = logging.getLogger("flyclaw.cron_tools")

_current_chat_id: ContextVar[str] = ContextVar("_current_chat_id", default="")


def set_current_chat_id(chat_id: str):
    _current_chat_id.set(chat_id)


def _get_service():
    return get_container().cron_service


def _clean_job_id(job_id: str) -> str:
    return job_id.strip().strip("[]()`").strip()


async def cron_list() -> str:
    """查看所有定时任务。

    Returns:
        所有定时任务的列表，包含名称、ID、状态、调度方式和上次执行结果。
    """
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
        dep_str = f", depends_on: {','.join(j.depends_on)}" if j.depends_on else ""
        lines.append(f"- {j.name}  id={j.id}  ({status}, {sched_str}{last}{dep_str})")
    return "\n".join(lines)


async def cron_create(
    name: str,
    message: str,
    schedule_kind: Literal["cron", "every", "at"] = "cron",
    cron_expr: str = "",
    every_seconds: int = 0,
    at_time: str = "",
    enabled: bool = True,
    description: str = "",
    depends_on: str = "",
) -> str:
    """创建定时任务。schedule_kind="cron" 创建循环任务（不会自动删除），需提供 cron_expr；不要用 cron 创建一次性任务。schedule_kind="every" 创建固定间隔任务，需提供 every_seconds。schedule_kind="at" 创建一次性任务（执行后自动删除），需提供 at_time。

    Args:
        name: 任务名称
        message: 任务触发时发送给 agent 的提示词
        schedule_kind: 调度类型：cron（循环）、every（间隔）、at（一次性）
        cron_expr: Cron 表达式，如 "0 9 * * 1-5"（schedule_kind=cron 时必填）
        every_seconds: 间隔秒数（schedule_kind=every 时必填）
        at_time: 执行时间，ISO 格式如 "2026-05-19 23:12:00"（schedule_kind=at 时必填）
        enabled: 是否启用，默认 true
        description: 任务描述
        depends_on: 逗号分隔的前置任务 ID
    """
    svc = _get_service()
    if not svc:
        return "Cron service is not available."
    if not name:
        return "Error: name is required."
    if not message:
        return "Error: message is required."
    if schedule_kind == "cron" and not cron_expr:
        return "Error: cron_expr is required when schedule_kind is 'cron'."
    if schedule_kind == "every" and not every_seconds:
        return "Error: every_seconds is required when schedule_kind is 'every'."
    if schedule_kind == "at" and not at_time:
        return "Error: at_time is required when schedule_kind is 'at'."

    from src.cron.types import CronSchedule, CronJobCreate, CronPayload, CronDelivery

    schedule = CronSchedule(
        kind=schedule_kind,
        expr=cron_expr or None,
        every_seconds=every_seconds or None,
        at=at_time or None,
    )
    payload = CronPayload(kind="agent_turn", message=message)
    chat_id = _current_chat_id.get("")
    delivery = CronDelivery(mode="announce", to=chat_id) if chat_id else CronDelivery()
    if not chat_id:
        logger.warning("cron_create: no current chat_id, delivery will be disabled")
    deps = [d.strip() for d in depends_on.split(",") if d.strip()] if depends_on else []
    create = CronJobCreate(
        name=name,
        description=description,
        enabled=enabled,
        schedule=schedule,
        payload=payload,
        delivery=delivery,
        depends_on=deps,
    )
    try:
        job = await svc.add_job(create)
        return f"Job created: [{job.id}] {job.name}"
    except Exception as e:
        return f"Failed to create job: {e}"


async def cron_delete(job_id: str) -> str:
    """删除定时任务。

    Args:
        job_id: 要删除的任务 ID
    """
    svc = _get_service()
    if not svc:
        return "Cron service is not available."
    job_id = _clean_job_id(job_id)
    try:
        removed = await svc.remove_job(job_id)
        if removed:
            return f"Job deleted: {job_id}"
        return f"Job not found: {job_id}"
    except Exception as e:
        return f"Failed to delete job: {e}"


async def cron_toggle(job_id: str) -> str:
    """切换定时任务的启停状态（启用↔禁用）。

    Args:
        job_id: 要切换状态的任务 ID
    """
    svc = _get_service()
    if not svc:
        return "Cron service is not available."
    job_id = _clean_job_id(job_id)
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


async def cron_run(job_id: str) -> str:
    """立即触发执行指定定时任务。

    Args:
        job_id: 要立即执行的任务 ID
    """
    svc = _get_service()
    if not svc:
        return "Cron service is not available."
    job_id = _clean_job_id(job_id)
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


def get_tools():
    from src.agent.tooldef import ToolDef
    return [
        ToolDef.from_function(cron_list),
        ToolDef.from_function(cron_create),
        ToolDef.from_function(cron_delete),
        ToolDef.from_function(cron_toggle),
        ToolDef.from_function(cron_run),
    ]
