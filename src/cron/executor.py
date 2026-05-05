from __future__ import annotations

import asyncio
import hashlib
import hmac
import ipaddress
import json
import logging
import random
import re
import socket
import time
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .types import CronJob, CronRunResult

logger = logging.getLogger("myclaw.cron.executor")

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


def _is_transient_error(error_str: str) -> bool:
    lower = error_str.lower()
    return any(re.search(p, lower) for p in _TRANSIENT_PATTERNS)


async def _is_safe_webhook_url(url: str) -> tuple[bool, str]:
    try:
        parsed = urlparse(url)
    except Exception:
        return False, "invalid URL"

    if parsed.scheme not in ("https", "http"):
        return False, f"unsupported scheme: {parsed.scheme}"

    hostname = parsed.hostname
    if not hostname:
        return False, "missing hostname"

    try:
        addr = ipaddress.ip_address(hostname)
        if addr.is_private:
            return False, f"private IP: {hostname}"
        if addr.is_loopback:
            return False, f"loopback: {hostname}"
        if addr.is_link_local:
            return False, f"link-local: {hostname}"
        if addr.is_reserved:
            return False, f"reserved: {hostname}"
    except ValueError:
        # hostname is a domain — resolve DNS to check for private IPs (prevent DNS rebinding)
        try:
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            addrs = await asyncio.get_event_loop().run_in_executor(
                None, socket.getaddrinfo, hostname, port, socket.AF_UNSPEC, socket.SOCK_STREAM
            )
            for _family, _type, _proto, _canonname, sockaddr in addrs:
                addr = ipaddress.ip_address(sockaddr[0])
                if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
                    return False, f"resolved to private/reserved IP: {addr}"
        except (socket.gaierror, OSError):
            return False, f"cannot resolve hostname: {hostname}"

    return True, ""


def _extract_assistant_text(result: dict) -> str:
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
    return ""


async def execute_cron_job(
    job: CronJob,
    graph: Any,
    config: Any,
    feishu_channel: Any = None,
    started_at: Optional[float] = None,
) -> CronRunResult:
    if started_at is None:
        started_at = time.time()

    thread_id = f"cron:{job.id}"
    if job.session_target == "main":
        thread_id = "main"

    run_config = {"configurable": {"thread_id": thread_id}}
    system_prompt = config.agents.system_prompt

    timeout = job.payload.timeout_seconds or _DEFAULT_TIMEOUT
    if job.payload.kind == "agent_turn":
        timeout = max(timeout, _AGENT_TURN_TIMEOUT)

    input_state = None
    if job.payload.kind == "agent_turn":
        input_state = {
            "messages": [HumanMessage(content=job.payload.message or "")],
            "system_prompt": system_prompt,
            "sender_id": f"cron:{job.id}",
            "chat_id": thread_id,
            "chat_type": "p2p",
            "message_id": f"cron:{job.id}:{started_at}",
        }
    elif job.payload.kind == "system_event":
        input_state = {
            "messages": [SystemMessage(content=f"[Scheduled Event] {job.payload.text or ''}")],
            "system_prompt": system_prompt,
            "sender_id": "system",
            "chat_id": thread_id,
            "chat_type": "p2p",
            "message_id": f"cron:{job.id}:{started_at}",
        }
    else:
        return CronRunResult(
            job_id=job.id,
            status="error",
            error="unknown payload kind",
            started_at=started_at,
            finished_at=time.time(),
        )

    try:
        result = await asyncio.wait_for(
            graph.ainvoke(input_state, run_config),
            timeout=timeout,
        )
        output = _extract_assistant_text(result)
        finished_at = time.time()

        cr = CronRunResult(
            job_id=job.id,
            status="success",
            output=output,
            started_at=started_at,
            finished_at=finished_at,
        )

        await _deliver_result(job, output, feishu_channel)
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


async def _deliver_result(job: CronJob, output: str, feishu_channel: Any = None):
    if not output or not output.strip():
        return
    delivery = job.delivery

    if delivery.mode == "none":
        logger.info("Job '%s': delivery mode is 'none', skipping delivery", job.name)
        return

    try:
        if delivery.mode == "announce" and feishu_channel is not None:
            target = delivery.to or delivery.channel or ""
            if target:
                await feishu_channel.send_text(target, output)
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
                headers["X-MyClaw-Signature"] = f"sha256={signature}"
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
