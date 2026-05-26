from __future__ import annotations

import json
import logging
from typing import Optional
from contextvars import ContextVar

from src._container import get_container
from src.agent.tooldef import ToolDef

logger = logging.getLogger("flyclaw.cron_tools")

_current_chat_id: ContextVar[str] = ContextVar("_current_chat_id", default="")


def set_current_chat_id(chat_id: str):
    _current_chat_id.set(chat_id)


def _get_service():
    return get_container().cron_service


def _clean_job_id(job_id: str) -> str:
    return job_id.strip().strip("[]()`").strip()


_CRON_TOOL_DESCRIPTION = (
    "管理定时任务。用 action 参数指定操作。\n\n"
    "ACTIONS:\n"
    "- list: 查看所有定时任务\n"
    "- create: 创建定时任务。需要 name, message, schedule_kind。\n"
    "  schedule_kind=\"cron\" 创建循环任务（不会自动删除），必须提供 cron_expr。不要用 cron 创建一次性任务。\n"
    "  schedule_kind=\"every\" 创建固定间隔任务，必须提供 every_seconds。\n"
    "  schedule_kind=\"at\" 创建一次性任务（执行后自动删除），必须提供 run_at。\n"
    "  可选参数：enabled, description, depends_on。\n"
    "- delete: 删除任务。需要 job_id\n"
    "- toggle: 切换启停状态。需要 job_id\n"
    "- run: 立即触发执行。需要 job_id\n"
)


async def cronjob(action: str, job_id: str = "", name: str = "", message: str = "",
                  schedule_kind: str = "cron", cron_expr: str = "",
                  every_seconds: int = 0, run_at: str = "",
                  enabled: bool = True, description: str = "",
                  depends_on: str = "") -> str:
    """Manage scheduled cron jobs with a single compressed tool.

    Args:
        action: One of: list, create, delete, toggle, run
        job_id: Job ID (required for delete/toggle/run)
        name: Job name (for create)
        message: The prompt for the agent when the job runs (for create)
        schedule_kind: "cron" (recurring), "every" (interval), or "at" (one-shot)
        cron_expr: Cron expression like "0 9 * * 1-5" (required for cron)
        every_seconds: Interval in seconds (required for every)
        run_at: ISO datetime like "2026-04-18 11:12:00" (required for at)
        enabled: Whether the job is active (for create)
        description: Optional description (for create)
        depends_on: Comma-separated job IDs this job depends on (for create)
    """
    svc = _get_service()
    normalized = (action or "").strip().lower()

    if normalized == "list":
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

    if normalized == "create":
        if not svc:
            return "Cron service is not available."
        if not name:
            return "Error: name is required for create."
        if not message:
            return "Error: message is required for create."
        if schedule_kind == "cron" and not cron_expr:
            return "Error: cron_expr is required when schedule_kind is 'cron'."
        if schedule_kind == "every" and not every_seconds:
            return "Error: every_seconds is required when schedule_kind is 'every'."
        if schedule_kind == "at" and not run_at:
            return "Error: run_at is required when schedule_kind is 'at'."

        from src.cron.types import CronSchedule, CronJobCreate, CronPayload, CronDelivery

        schedule = CronSchedule(
            kind=schedule_kind,
            expr=cron_expr or None,
            every_seconds=every_seconds or None,
            at=run_at or None,
        )
        payload = CronPayload(kind="agent_turn", message=message)
        chat_id = _current_chat_id.get("")
        delivery = CronDelivery(mode="announce", to=chat_id) if chat_id else CronDelivery()
        if not chat_id:
            logger.warning("cronjob create: no current chat_id, delivery will be disabled")
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

    if not job_id:
        return f"Error: job_id is required for action '{action}'"
    job_id = _clean_job_id(job_id)

    if normalized == "delete":
        if not svc:
            return "Cron service is not available."
        try:
            removed = await svc.remove_job(job_id)
            if removed:
                return f"Job deleted: {job_id}"
            return f"Job not found: {job_id}"
        except Exception as e:
            return f"Failed to delete job: {e}"

    if normalized == "toggle":
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

    if normalized == "run":
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

    return f"Unknown action '{action}'. Use: list, create, delete, toggle, run"


def get_tools() -> list[ToolDef]:
    return [
        ToolDef.from_schema(
            name="cronjob",
            description=_CRON_TOOL_DESCRIPTION,
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "create", "delete", "toggle", "run"],
                        "description": "操作类型",
                    },
                    "job_id": {
                        "type": "string",
                        "description": "任务 ID（delete/toggle/run 必填）",
                    },
                    "name": {
                        "type": "string",
                        "description": "任务名称（create 必填）",
                    },
                    "message": {
                        "type": "string",
                        "description": "定时任务执行的提示词（create 必填）",
                    },
                    "schedule_kind": {
                        "type": "string",
                        "enum": ["cron", "every", "at"],
                        "description": "调度类型：cron（循环）、every（间隔）、at（一次性）",
                    },
                    "cron_expr": {
                        "type": "string",
                        "description": "Cron 表达式，如 '0 9 * * 1-5'（schedule_kind=cron 时必填）",
                    },
                    "every_seconds": {
                        "type": "integer",
                        "description": "间隔秒数（schedule_kind=every 时必填）",
                    },
                    "run_at": {
                        "type": "string",
                        "description": "ISO 时间戳，如 '2026-05-19 23:12:00'（schedule_kind=at 时必填，一次性）",
                    },
                    "enabled": {
                        "type": "boolean",
                        "description": "是否启用（默认 true）",
                    },
                    "description": {
                        "type": "string",
                        "description": "任务描述（可选）",
                    },
                    "depends_on": {
                        "type": "string",
                        "description": "逗号分隔的前置任务 ID（可选）",
                    },
                },
                "required": ["action"],
            },
            fn=cronjob,
        ),
    ]
