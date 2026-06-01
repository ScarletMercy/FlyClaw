from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import random
import re
import time
from typing import Any, Optional

import httpx

from src.agent.state import AgentState
from .types import CronJob, CronRunResult

logger = logging.getLogger("flyclaw.cron.executor")

_TRANSIENT_PATTERNS = [
    r"rate.?limit",
    r"429",
    r"overload",
    r"503",
    r"529",
    r"timeout",
    r"timed?\s*out",
    r"connection",
    r"network",
    r"temporary",
]

_DEFAULT_TIMEOUT = 600
_AGENT_TURN_TIMEOUT = 3600
_SILENT_MARKER = "[SILENT]"

_CRON_EXECUTION_HINT = (
    "\n\n## 定时任务执行指引\n"
    "[重要：你正在执行一个定时任务，不是在跟用户聊天。"
    "你的最终回复将被自动投递给用户。"
    "不要回复'收到'、'好的'之类的废话，直接产出用户期望看到的内容。"
    "例如，如果任务是'提醒我起床'，你应该回复类似'该起床了！'这样的提醒文字。"
    "如果确实没有任何需要报告的内容，回复 [SILENT]（仅此一词）以抑制投递。]"
)


def _is_transient_error(error_str: str) -> bool:
    lower = error_str.lower()
    return any(re.search(p, lower) for p in _TRANSIENT_PATTERNS)


async def _is_safe_webhook_url(url: str) -> tuple[bool, str]:
    from src.security.url_safety import is_safe_url

    return await asyncio.get_running_loop().run_in_executor(None, is_safe_url, url)


def _extract_assistant_text(messages: list[dict]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            return msg["content"]
    return ""


async def execute_cron_job(
    job: CronJob,
    agent_loop: Any,
    config: Any,
    channel: Any = None,
    started_at: Optional[float] = None,
) -> CronRunResult:
    if started_at is None:
        started_at = time.time()

    thread_id = f"cron:{job.id}"
    if job.session_target == "main":
        thread_id = "main"

    system_prompt = config.agents.system_prompt

    is_task_job = job.name.startswith("task:")
    if not is_task_job:
        system_prompt += _CRON_EXECUTION_HINT
    if is_task_job:
        try:
            if agent_loop.is_thread_busy(thread_id):
                defer_minutes = getattr(config.task, "defer_minutes", 5)
                import zoneinfo
                from datetime import datetime, timedelta

                tz_name = job.schedule.tz or getattr(config.agents, "timezone", "UTC")
                new_at = (datetime.now(zoneinfo.ZoneInfo(tz_name)) + timedelta(minutes=defer_minutes)).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                logger.info("Task checkpoint deferred: thread '%s' is busy, rescheduling to %s", thread_id, new_at)
                return CronRunResult(
                    job_id=job.id,
                    status="deferred",
                    error=f"Thread busy, deferred {defer_minutes}m",
                    started_at=started_at,
                    finished_at=time.time(),
                    output=json.dumps({"deferred": True, "new_at": new_at, "defer_minutes": defer_minutes}),
                )
        except Exception:
            pass

        task_context = (
            "\n\n## 自主任务检查点\n"
            '这是一个自主任务的检查点触发。请先调用 task_manage(action="status") 查看当前任务进度，'
            '然后继续执行下一步。完成后调用 task_manage(action="advance") 标记步骤完成。'
        )
        system_prompt += task_context

        parts = job.name.split(":")
        if len(parts) >= 4:
            try:
                run_id = parts[1]
                cp_id = parts[3]
                from src.task.store import get_task_store

                task_store = get_task_store(getattr(config.task, "db_path", "~/.flyclaw/data/task_runs.db"))
                run = await task_store.get(run_id)
                if run:
                    task_detail = (
                        f"\n\n## 当前任务详情\n"
                        f"任务ID: {run.id}\n"
                        f"目标: {run.goal}\n"
                        f"当前步骤: {run.current_step}/{len(run.steps)}\n"
                        f"下一步: {run.steps[run.current_step] if run.current_step < len(run.steps) else '全部完成'}\n"
                        f"检查点提示: {job.payload.message or '检查任务进度并继续执行'}\n"
                    )
                    system_prompt += task_detail
                    logger.info("Injected task context for run %s step %d/%d", run_id, run.current_step, len(run.steps))
            except Exception:
                logger.warning("Failed to load task context for job %s", job.name, exc_info=True)

    timeout = job.payload.timeout_seconds or _DEFAULT_TIMEOUT
    if job.payload.kind == "agent_turn":
        timeout = max(timeout, _AGENT_TURN_TIMEOUT)

    if job.payload.kind == "direct":
        output = job.payload.message or ""
        finished_at = time.time()
        if output.strip():
            await _deliver_result(job, output, channel)
        return CronRunResult(
            job_id=job.id,
            status="success",
            output=output,
            started_at=started_at,
            finished_at=finished_at,
        )

    input_state = None
    if job.payload.kind == "agent_turn":
        new_msg = {"role": "user", "content": f"[定时任务触发]\n{job.payload.message or ''}"}
        input_state = AgentState(
            messages=[new_msg],
            system_prompt=system_prompt,
            sender_id=f"cron:{job.id}",
            chat_id=thread_id,
            chat_type="p2p",
            message_id=f"cron:{job.id}:{started_at}",
        )
        store = agent_loop.get_store()
        existing = await store.load(thread_id)
        if existing:
            input_state.messages = existing.messages + [new_msg]
    elif job.payload.kind == "system_event":
        new_msg = {"role": "system", "content": f"[Scheduled Event] {job.payload.text or ''}"}
        input_state = AgentState(
            messages=[new_msg],
            system_prompt=system_prompt,
            sender_id="system",
            chat_id=thread_id,
            chat_type="p2p",
            message_id=f"cron:{job.id}:{started_at}",
        )
        store = agent_loop.get_store()
        existing = await store.load(thread_id)
        if existing:
            input_state.messages = existing.messages + [new_msg]
    else:
        return CronRunResult(
            job_id=job.id,
            status="error",
            error="unknown payload kind",
            started_at=started_at,
            finished_at=time.time(),
        )

    from src.tools.exec import _current_thread_id as _exec_thread_id

    _tid_token = _exec_thread_id.set(thread_id)
    try:
        result = await asyncio.wait_for(
            agent_loop.run(input_state, thread_id),
            timeout=timeout,
        )
        output = _extract_assistant_text(result.messages)
        if output and output.strip().upper().startswith(_SILENT_MARKER):
            logger.info("Job '%s': agent returned [SILENT], skipping delivery", job.name)
            output = ""
        finished_at = time.time()

        cr = CronRunResult(
            job_id=job.id,
            status="success",
            output=output,
            started_at=started_at,
            finished_at=finished_at,
        )

        await _deliver_result(job, output, channel)
        return cr

    except asyncio.TimeoutError:
        finished_at = time.time()
        error_msg = f"Timeout after {timeout}s"
        logger.error("Job '%s' %s", job.name, error_msg)
        return CronRunResult(
            job_id=job.id,
            status="timeout",
            error=error_msg,
            started_at=started_at,
            finished_at=finished_at,
        )
    except Exception as e:
        finished_at = time.time()
        error_msg = f"{type(e).__name__}: {e}"
        logger.error("Job '%s' failed: %s", job.name, error_msg, exc_info=True)
        return CronRunResult(
            job_id=job.id,
            status="error",
            error=error_msg,
            started_at=started_at,
            finished_at=finished_at,
        )
    finally:
        _exec_thread_id.reset(_tid_token)


async def execute_with_retry(
    job: CronJob,
    execute_fn: Any,
    max_retries: int = 3,
) -> CronRunResult:
    last_result = None
    base_delay = 2  # Base delay in seconds for exponential backoff
    max_delay = 600  # Maximum delay of 10 minutes

    for attempt in range(max_retries + 1):
        result = await execute_fn(job)
        last_result = result

        if result.status == "success":
            return result

        if not _is_transient_error(result.error or ""):
            return result

        if attempt < max_retries:
            # Exponential backoff with jitter: delay = base * 2^attempt + random_jitter
            delay = min(base_delay * (2**attempt), max_delay)
            jitter = random.uniform(0, 1) * delay * 0.1  # Add 0-10% jitter
            delay = delay + jitter
            logger.info(
                "Job '%s' transient error, retry %d/%d in %.1fs: %s",
                job.name,
                attempt + 1,
                max_retries,
                delay,
                result.error,
            )
            await asyncio.sleep(delay)

    return last_result


async def _deliver_result(job: CronJob, output: str, channel: Any = None):
    if not output or not output.strip():
        return
    delivery = job.delivery

    if delivery.mode == "none":
        logger.info("Job '%s': delivery mode is 'none', skipping delivery", job.name)
        return

    try:
        if delivery.mode == "announce" and channel is not None:
            target = delivery.to or delivery.channel or ""
            if target:
                await channel.send_text(target, output)
                logger.info("Delivered cron output to channel: %s", target)
            else:
                logger.warning("No delivery target for cron job '%s'", job.name)

        elif delivery.mode == "webhook" and delivery.webhook_url:
            safe, reason = await _is_safe_webhook_url(delivery.webhook_url)
            if not safe:
                logger.error("Webhook URL blocked (SSRF): %s — %s", delivery.webhook_url, reason)
                if not delivery.best_effort:
                    raise ValueError(f"Unsafe webhook URL: {reason}")
                return

            payload = json.dumps({"job_id": job.id, "job_name": job.name, "output": output})
            headers = {"Content-Type": "application/json"}
            webhook_secret = getattr(delivery, "webhook_secret", "") or ""
            if webhook_secret:
                signature = hmac.new(webhook_secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
                headers["X-flyclaw-Signature"] = f"sha256={signature}"
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    delivery.webhook_url,
                    content=payload,
                    headers=headers,
                )
                logger.info(
                    "Delivered cron output to webhook: %s (status=%d)",
                    delivery.webhook_url,
                    resp.status_code,
                )

    except Exception as e:
        logger.error("Delivery failed for job '%s': %s", job.name, e)
        if not delivery.best_effort:
            raise
