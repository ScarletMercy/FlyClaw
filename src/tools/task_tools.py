from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from typing import Optional

from src._container import get_container
from src.agent.tooldef import ToolDef

logger = logging.getLogger("myclaw.task_tools")

_current_chat_id: ContextVar[str] = ContextVar("_current_chat_id", default="")
_current_sender_id: ContextVar[str] = ContextVar("_current_sender_id", default="")
_current_thread_id: ContextVar[str] = ContextVar("_current_thread_id", default="")


def set_task_context(chat_id: str = "", sender_id: str = "", thread_id: str = ""):
    if chat_id:
        _current_chat_id.set(chat_id)
    if sender_id:
        _current_sender_id.set(sender_id)
    if thread_id:
        _current_thread_id.set(thread_id)


def _parse_plan_json(raw: str) -> Optional[dict]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import re
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None


def _parse_relative_time(at_str: str) -> Optional[str]:
    at_str = at_str.strip()
    try:
        datetime.fromisoformat(at_str)
        return at_str
    except (ValueError, TypeError):
        pass

    import re
    now = datetime.now(timezone(timedelta(hours=8)))

    m = re.match(r"(\d+)\s*分钟", at_str)
    if m:
        dt = now + timedelta(minutes=int(m.group(1)))
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    m = re.match(r"(\d+)\s*小时", at_str)
    if m:
        dt = now + timedelta(hours=int(m.group(1)))
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    m = re.match(r"(\d+)\s*天", at_str)
    if m:
        dt = now + timedelta(days=int(m.group(1)))
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    m = re.match(r"(\d+)\s*分?", at_str)
    if m:
        dt = now + timedelta(minutes=int(m.group(1)))
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    m = re.match(r"(\d+)\s*h", at_str, re.IGNORECASE)
    if m:
        dt = now + timedelta(hours=int(m.group(1)))
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    m = re.match(r"(\d+)\s*m", at_str, re.IGNORECASE)
    if m:
        dt = now + timedelta(minutes=int(m.group(1)))
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    return None


async def task_plan(goal: str, plan_json: str) -> str:
    """Create an autonomous task execution plan with checkpoints.

    Call this when autonomous mode is active and the user gives a task.
    Generate a plan with steps and time-based checkpoints, then call this tool.

    Args:
        goal: The original task goal from the user.
        plan_json: JSON object with the plan. Must have format:
            {"steps": ["step1", "step2", ...], "checkpoints": [{"at": "30分钟后", "prompt": "检查进度并继续执行"}, ...]}
            The "at" field supports: ISO datetime, "X分钟/小时/天", "Xm/Xh".
    """
    from src.task.store import get_task_store
    from src.task.types import TaskRun, TaskCheckpoint

    plan = _parse_plan_json(plan_json)
    if not plan:
        return json.dumps({"error": "无法解析计划 JSON，请确保输出合法 JSON"}, ensure_ascii=False)

    steps = plan.get("steps", [])
    raw_cps = plan.get("checkpoints", [])

    if not steps:
        return json.dumps({"error": "计划必须包含至少一个步骤"}, ensure_ascii=False)

    container = get_container()
    chat_id = _current_chat_id.get("")
    sender_id = _current_sender_id.get("")
    thread_id = _current_thread_id.get("")

    store = get_task_store(container.config.task.db_path)

    active = await store.list_by_status("running")
    max_parallel = getattr(container.config.task, "max_parallel", 3)
    if len(active) >= max_parallel:
        return json.dumps({"error": f"已达最大并行任务数 {max_parallel}，请等待现有任务完成或使用 task_cancel 取消"}, ensure_ascii=False)

    now = datetime.now(timezone(timedelta(hours=8)))
    checkpoints = []
    for rc in raw_cps[:20]:
        at_str = str(rc.get("at", ""))
        resolved = _parse_relative_time(at_str)
        if not resolved:
            continue
        try:
            resolved_dt = datetime.fromisoformat(resolved)
            if resolved_dt.tzinfo is None:
                resolved_dt = resolved_dt.replace(tzinfo=timezone(timedelta(hours=8)))
            if resolved_dt <= now:
                continue
        except (ValueError, TypeError):
            continue
        checkpoints.append(TaskCheckpoint(
            at=resolved,
            prompt=rc.get("prompt", "检查任务进度并继续执行"),
        ))

    if not checkpoints:
        default_at = (now + timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M:%S")
        checkpoints.append(TaskCheckpoint(
            at=default_at,
            prompt="检查任务进度并继续执行下一步",
        ))

    run = TaskRun(
        goal=goal,
        steps=steps,
        checkpoints=checkpoints,
        status="running",
        chat_id=chat_id,
        thread_id=thread_id,
        sender_id=sender_id,
    )

    await store.save(run)

    registered_cps = []
    for cp in run.checkpoints:
        try:
            cron_svc = container.cron_service
            if not cron_svc:
                continue

            from src.cron.types import CronSchedule, CronJobCreate, CronPayload, CronDelivery

            schedule = CronSchedule(kind="at", at=cp.at)
            payload = CronPayload(
                kind="agent_turn",
                message=cp.prompt,
                timeout_seconds=container.config.task.default_timeout,
            )
            delivery = CronDelivery(mode="announce", to=chat_id) if chat_id else CronDelivery()

            create = CronJobCreate(
                name=f"task:{run.id}:cp:{cp.id}",
                description=f"Task checkpoint: {goal[:60]}",
                enabled=True,
                schedule=schedule,
                payload=payload,
                delivery=delivery,
                session_target="main",
            )
            job = await cron_svc.add_job(create)
            cp.cron_job_id = job.id
            cp.status = "scheduled"
            registered_cps.append({"id": cp.id, "at": cp.at, "cron_job_id": job.id})
        except Exception as e:
            logger.warning("Failed to register checkpoint cron job: %s", e)
            registered_cps.append({"id": cp.id, "at": cp.at, "error": str(e)})

    await store.save(run)

    return json.dumps({
        "ok": True,
        "run_id": run.id,
        "steps": steps,
        "checkpoints": registered_cps,
    }, ensure_ascii=False)


async def task_status() -> str:
    """Check status of all active autonomous tasks. Call this at checkpoints to see progress."""
    from src.task.store import get_task_store
    container = get_container()
    store = get_task_store(container.config.task.db_path)
    runs = await store.list_by_status("planning", "running")
    if not runs:
        return json.dumps({"active_tasks": 0}, ensure_ascii=False)

    result = []
    for r in runs:
        cps = [{"id": c.id, "status": c.status, "at": c.at} for c in r.checkpoints]
        result.append({
            "run_id": r.id,
            "goal": r.goal,
            "current_step": r.current_step,
            "total_steps": len(r.steps),
            "steps": r.steps,
            "status": r.status,
            "checkpoints": cps,
        })
    return json.dumps({"active_tasks": len(result), "tasks": result}, ensure_ascii=False)


async def task_advance(step_index: int, result_summary: str = "", run_id: str = "") -> str:
    """Mark a task step as completed and advance to the next step.

    Args:
        step_index: The 0-based index of the completed step.
        result_summary: Brief summary of what was accomplished.
        run_id: The task run ID. If provided, advances this specific task. Otherwise advances the first active task.
    """
    from src.task.store import get_task_store
    container = get_container()
    store = get_task_store(container.config.task.db_path)

    if run_id:
        run = await store.get(run_id)
        if not run:
            return json.dumps({"error": f"Task not found: {run_id}"}, ensure_ascii=False)
        if run.status != "running":
            return json.dumps({"error": f"Task {run_id} is not running (status: {run.status})"}, ensure_ascii=False)
    else:
        runs = await store.list_by_status("running")
        if not runs:
            return json.dumps({"error": "No active tasks"}, ensure_ascii=False)
        run = runs[0]

    if step_index < 0 or step_index >= len(run.steps):
        return json.dumps({"error": f"Invalid step_index {step_index}, must be 0-{len(run.steps) - 1}"}, ensure_ascii=False)

    run.current_step = step_index + 1
    if run.current_step >= len(run.steps):
        run.status = "completed"
        for cp in run.checkpoints:
            if cp.cron_job_id and container.cron_service:
                try:
                    await container.cron_service.remove_job(cp.cron_job_id)
                except Exception:
                    pass
    await store.save(run)

    return json.dumps({
        "ok": True,
        "run_id": run.id,
        "completed_step": step_index,
        "current_step": run.current_step,
        "total_steps": len(run.steps),
        "status": run.status,
    }, ensure_ascii=False)


async def task_cancel(run_id: str = "") -> str:
    """Cancel an autonomous task and remove its scheduled checkpoints.

    Args:
        run_id: The task run ID to cancel. If empty, cancels the first active task.
    """
    from src.task.store import get_task_store
    container = get_container()
    store = get_task_store(container.config.task.db_path)

    if run_id:
        run = await store.get(run_id)
        if not run:
            return json.dumps({"error": f"Task not found: {run_id}"}, ensure_ascii=False)
    else:
        runs = await store.list_by_status("running")
        if not runs:
            return json.dumps({"error": "No active tasks to cancel"}, ensure_ascii=False)
        run = runs[0]

    run.status = "cancelled"
    for cp in run.checkpoints:
        if cp.cron_job_id and container.cron_service:
            try:
                await container.cron_service.remove_job(cp.cron_job_id)
            except Exception:
                pass
    await store.save(run)

    return json.dumps({"ok": True, "run_id": run.id, "status": "cancelled"}, ensure_ascii=False)


def get_tools() -> list[ToolDef]:
    return [
        ToolDef.from_function(task_plan),
        ToolDef.from_function(task_status),
        ToolDef.from_function(task_advance),
        ToolDef.from_function(task_cancel),
    ]
