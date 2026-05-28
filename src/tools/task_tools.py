from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from typing import Optional

from src._container import get_container
from src.agent.tooldef import ToolDef

logger = logging.getLogger("flyclaw.task_tools")

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


_TASK_TOOL_DESCRIPTION = (
    "管理自主任务执行计划。用 action 参数指定操作。\n\n"
    "ACTIONS:\n"
    "- plan: 制定任务计划（步骤+检查点）。需要 goal 和 plan_json\n"
    "- status: 查看所有活跃任务\n"
    "- advance: 标记步骤完成。需要 step_index，可选 run_id, result_summary\n"
    "- cancel: 取消任务。可选 run_id\n"
)


async def task_manage(action: str, goal: str = "", plan_json: str = "",
                      step_index: int = -1, result_summary: str = "",
                      run_id: str = "") -> str:
    """Manage autonomous task plans with a single compressed tool.

    Args:
        action: One of: plan, status, advance, cancel
        goal: Task goal (required for plan)
        plan_json: JSON plan with steps and checkpoints (required for plan)
        step_index: 0-based step index to mark complete (required for advance)
        result_summary: Brief summary of what was accomplished (for advance)
        run_id: Task run ID (optional for advance/cancel)
    """
    normalized = (action or "").strip().lower()

    if normalized == "plan":
        return await _task_plan_impl(goal, plan_json)

    if normalized == "status":
        return await _task_status_impl()

    if normalized == "advance":
        return await _task_advance_impl(step_index, result_summary, run_id)

    if normalized == "cancel":
        return await _task_cancel_impl(run_id)

    return json.dumps({"error": f"Unknown action '{action}'. Use: plan, status, advance, cancel"}, ensure_ascii=False)


async def _task_plan_impl(goal: str, plan_json: str) -> str:
    from src.task.store import get_task_store
    from src.task.types import TaskRun, TaskCheckpoint

    if not goal:
        return json.dumps({"error": "goal is required for plan action"}, ensure_ascii=False)

    plan = _parse_plan_json(plan_json)
    if not plan:
        return json.dumps({"error": "无法解析计划 JSON，请确保输出合法 JSON"}, ensure_ascii=False)

    raw_steps = plan.get("steps", [])
    if not raw_steps:
        return json.dumps({"error": "计划必须包含至少一个步骤"}, ensure_ascii=False)

    steps = [
        s.get("description") or s.get("name") or json.dumps(s, ensure_ascii=False)
        if isinstance(s, dict) else str(s)
        for s in raw_steps
    ]

    raw_cps = plan.get("checkpoints", [])

    container = get_container()
    chat_id = _current_chat_id.get("")
    sender_id = _current_sender_id.get("")
    thread_id = _current_thread_id.get("")

    store = get_task_store(container.config.task.db_path)

    active = await store.list_by_status("running")
    max_parallel = getattr(container.config.task, "max_parallel", 3)
    if len(active) >= max_parallel:
        return json.dumps({"error": f"已达最大并行任务数 {max_parallel}，请等待现有任务完成或取消"}, ensure_ascii=False)

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

    try:
        run = TaskRun(
            goal=goal,
            steps=steps,
            checkpoints=checkpoints,
            status="running",
            chat_id=chat_id,
            thread_id=thread_id,
            sender_id=sender_id,
        )
    except Exception as e:
        from pydantic import ValidationError
        if isinstance(e, ValidationError):
            bad = [str(err["loc"][-1]) for err in e.errors()]
            return json.dumps({
                "error": f"计划格式有误，steps 应为字符串数组，例如 [\"步骤1\", \"步骤2\"]。"
                         f"当前 steps 中第 {', '.join(bad)} 项格式不对，请修正后重试",
            }, ensure_ascii=False)
        logger.error("TaskRun 构造失败: %s", e)
        return json.dumps({"error": f"创建任务失败: {e}"}, ensure_ascii=False)

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


async def _task_status_impl() -> str:
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


async def _task_advance_impl(step_index: int, result_summary: str, run_id: str) -> str:
    from src.task.store import get_task_store
    container = get_container()
    store = get_task_store(container.config.task.db_path)

    if step_index < 0:
        return json.dumps({"error": "step_index is required for advance action"}, ensure_ascii=False)

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

    if step_index >= len(run.steps):
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


async def _task_cancel_impl(run_id: str) -> str:
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
        ToolDef.from_schema(
            name="task_manage",
            description=_TASK_TOOL_DESCRIPTION,
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["plan", "status", "advance", "cancel"],
                        "description": "操作类型",
                    },
                    "goal": {
                        "type": "string",
                        "description": "任务目标（plan 必填）",
                    },
                    "plan_json": {
                        "type": "string",
                        "description": "计划 JSON，格式: {\"steps\": [...], \"checkpoints\": [{\"at\": \"30分钟后\", \"prompt\": \"...\"}]}（plan 必填）",
                    },
                    "step_index": {
                        "type": "integer",
                        "default": -1,
                        "description": "完成的步骤索引，0-based（advance 必填）",
                    },
                    "result_summary": {
                        "type": "string",
                        "description": "步骤完成摘要（advance 可选）",
                    },
                    "run_id": {
                        "type": "string",
                        "description": "任务 ID（advance/cancel 可选，默认操作第一个活跃任务）",
                    },
                },
                "required": ["action"],
            },
            fn=task_manage,
        ),
    ]
