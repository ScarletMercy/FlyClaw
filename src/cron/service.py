from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.jobstores.base import JobLookupError

from .store import CronStore
from .types import CronJob, CronJobCreate, CronJobPatch, CronRunResult

logger = logging.getLogger("myclaw.cron")

_MAX_CONSECUTIVE_ERRORS = 5
_ERROR_BACKOFF_SECONDS = [30, 60, 300, 900, 3600]


class CronService:
    def __init__(
        self,
        store: CronStore,
        execute_fn: Callable,
        config: Any = None,
        channel: Any = None,
    ):
        self.store = store
        self.execute_fn = execute_fn
        self._config = config
        self._channel = channel
        self._scheduler: Optional[AsyncIOScheduler] = None
        self._jobs: dict[str, CronJob] = {}
        self._failure_alert_after = 2
        self._last_failure_alert: dict[str, float] = {}
        self._running_jobs: set[str] = set()
        if config:
            self._max_transient_retries = config.cron.max_transient_retries
            self._failure_alert_after = config.cron.failure_alert_after
        else:
            self._max_transient_retries = 3

    async def _execute_fn_raw(self, job):
        return await self.execute_fn(job)

    async def _send_failure_alert(self, job: CronJob):
        last_alert = self._last_failure_alert.get(job.id, 0)
        if time.time() - last_alert < 3600:
            return
        self._last_failure_alert[job.id] = time.time()

        alert_text = (
            f"⚠️ Cron job failure alert\n"
            f"Job: {job.name} ({job.id})\n"
            f"Consecutive errors: {job.consecutive_errors}\n"
            f"Last error: {job.last_error or 'unknown'}"
        )
        logger.warning("Failure alert for job '%s': %d consecutive errors", job.name, job.consecutive_errors)

        if self._channel and job.delivery.mode == "announce":
            target = job.delivery.to or job.delivery.channel or ""
            if target:
                try:
                    await self._channel.send_text(target, alert_text)
                except Exception as e:
                    logger.error("Failed to send failure alert: %s", e)

    async def start(self):
        import zoneinfo

        try:
            tz = zoneinfo.ZoneInfo("Asia/Shanghai")
        except Exception:
            tz = "UTC"
        self._scheduler = AsyncIOScheduler(timezone=tz)
        jobs = await self.store.load_jobs()
        for job in jobs:
            self._jobs[job.id] = job
            if job.enabled:
                self._schedule_job(job)
        self._scheduler.start()
        logger.info(
            "Cron service started with %d jobs (%d enabled)",
            len(jobs),
            sum(1 for j in jobs if j.enabled),
        )

    async def stop(self):
        if self._scheduler:
            self._scheduler.shutdown(wait=False)
            self._scheduler = None
        await self.store.close()
        logger.info("Cron service stopped")

    def _schedule_job(self, job: CronJob):
        if not self._scheduler:
            logger.warning("Cannot schedule job '%s': scheduler not initialized", job.name)
            return
        job_id = f"cron_{job.id}"
        try:
            self._scheduler.remove_job(job_id)
        except JobLookupError:
            pass
        try:
            trigger = job.schedule.to_apscheduler_trigger()
            self._scheduler.add_job(
                self._run_job,
                trigger=trigger,
                id=job_id,
                args=[job.id],
                misfire_grace_time=3600,
                max_instances=1,
            )
            try:
                next_time = trigger.get_next_fire_time(None, datetime.now())
                job.next_run_at = next_time.timestamp() if next_time else None
            except Exception:
                job.next_run_at = None
            logger.info(
                "Scheduled job '%s' (id=%s, next=%s, scheduler_running=%s)",
                job.name,
                job.id,
                job.next_run_at,
                self._scheduler.running,
            )
        except Exception as e:
            logger.error("Failed to schedule job '%s': %s", job.name, e)

    def _unschedule_job(self, job_id: str):
        if not self._scheduler:
            return
        try:
            self._scheduler.remove_job(f"cron_{job_id}")
        except JobLookupError:
            pass

    async def _run_job(self, job_id: str):
        job = self._jobs.get(job_id)
        if job is None:
            logger.error("Job %s not found in memory", job_id)
            return
        if not job.enabled:
            return
        if job_id in self._running_jobs:
            logger.warning("Job '%s' is already running, skipping", job.id)
            return

        # Check dependencies — all depends_on jobs must have completed successfully
        if job.depends_on:
            unmet = []
            for dep_id in job.depends_on:
                dep = self._jobs.get(dep_id)
                if dep is None:
                    logger.warning(
                        "Job '%s' depends on '%s' which does not exist, skipping execution",
                        job.name,
                        dep_id,
                    )
                    return
                if dep.last_run_status != "success":
                    unmet.append(dep_id)
            if unmet:
                logger.info(
                    "Job '%s' skipped — dependencies not satisfied: %s",
                    job.name,
                    ", ".join(unmet),
                )
                return

        self._running_jobs.add(job_id)
        try:
            logger.info("Executing cron job '%s' (id=%s)", job.name, job.id)
            started_at = time.time()
            try:
                from .executor import execute_with_retry

                max_retries = getattr(self, "_max_transient_retries", 3)
                result = await execute_with_retry(job, self.execute_fn, max_retries=max_retries)
                if result.status == "success":
                    job.consecutive_errors = 0
                    job.last_run_status = "success"
                    job.last_error = None
                elif result.status == "deferred":
                    logger.info("Job '%s' deferred, rescheduling", job.name)
                    job.last_run_status = "deferred"
                    job.last_error = None
                    try:
                        import json
                        defer_info = json.loads(result.output or "{}")
                        new_at = defer_info.get("new_at", "")
                        if new_at and job.name.startswith("task:"):
                            from .executor import execute_cron_job
                            from .types import CronSchedule, CronJobCreate, CronPayload, CronDelivery
                            schedule = CronSchedule(kind="at", at=new_at)
                            payload = job.payload
                            delivery = job.delivery
                            create = CronJobCreate(
                                name=job.name,
                                description=job.description,
                                enabled=True,
                                schedule=schedule,
                                payload=payload,
                                delivery=delivery,
                                session_target=job.session_target,
                            )
                            new_job = await self.add_job(create)
                            logger.info("Rescheduled task checkpoint to %s (new job: %s)", new_at, new_job.id)

                            parts = job.name.split(":")
                            if len(parts) >= 4:
                                run_id = parts[1]
                                cp_id = parts[3]
                                from src.task.store import get_task_store
                                task_store = get_task_store(getattr(self._config.task, "db_path", "~/.myclaw/data/task_runs.db"))
                                run = await task_store.get(run_id)
                                if run:
                                    for cp in run.checkpoints:
                                        if cp.id == cp_id:
                                            cp.cron_job_id = new_job.id
                                            logger.info("Updated checkpoint %s cron_job_id to %s", cp_id, new_job.id)
                                            break
                                    await task_store.save(run)
                    except Exception as e:
                        logger.warning("Failed to reschedule deferred job: %s", e)
                    finally:
                        await self.store.remove_job(job.id)
                        self._unschedule_job(job.id)
                        if job.id in self._jobs:
                            del self._jobs[job.id]
                        return
                else:
                    job.consecutive_errors += 1
                    job.last_run_status = result.status
                    job.last_error = result.error or result.status
                    logger.error(
                        "Job '%s' failed (consecutive=%d): %s",
                        job.name,
                        job.consecutive_errors,
                        job.last_error,
                    )

                    if job.consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                        logger.warning(
                            "Job '%s' auto-disabled after %d consecutive errors",
                            job.name,
                            job.consecutive_errors,
                        )
                        job.enabled = False
                        self._unschedule_job(job.id)

                    if job.consecutive_errors >= self._failure_alert_after:
                        await self._send_failure_alert(job)

                logger.info("Job '%s' completed: %s", job.name, result.status)
            except Exception as e:
                job.consecutive_errors += 1
                job.last_run_status = "error"
                job.last_error = f"{type(e).__name__}: {e}"
                logger.error("Job '%s' failed (consecutive=%d): %s", job.name, job.consecutive_errors, e)

                if job.consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                    job.enabled = False
                    self._unschedule_job(job.id)

            job.last_run_at = started_at
            await self.store.save_job(job)

            # Trigger downstream jobs that depend on this one
            if job.last_run_status == "success":
                await self._trigger_dependents(job_id)

            if job.schedule.kind == "at" and job.delete_after_run:
                await self.remove_job(job.id)
        finally:
            self._running_jobs.discard(job_id)

    def list_jobs(self) -> list[CronJob]:
        return list(self._jobs.values())

    def get_job(self, job_id: str) -> Optional[CronJob]:
        return self._jobs.get(job_id)

    async def add_job(self, create: CronJobCreate) -> CronJob:
        job = CronJob(
            name=create.name,
            description=create.description,
            enabled=create.enabled,
            delete_after_run=create.delete_after_run,
            schedule=create.schedule,
            payload=create.payload,
            delivery=create.delivery,
            session_target=create.session_target,
            depends_on=create.depends_on,
        )
        if job.schedule.kind == "at":
            job.delete_after_run = True
        self._jobs[job.id] = job
        await self.store.save_job(job)
        if job.enabled:
            self._schedule_job(job)
        logger.info("Added cron job '%s' (id=%s)", job.name, job.id)
        return job

    async def update_job(self, job_id: str, patch: CronJobPatch) -> Optional[CronJob]:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        for field in ("name", "description", "enabled", "schedule", "payload", "delivery", "session_target", "depends_on"):
            val = getattr(patch, field, None)
            if val is not None:
                setattr(job, field, val)

        await self.store.save_job(job)
        self._unschedule_job(job.id)
        if job.enabled:
            self._schedule_job(job)
        logger.info("Updated cron job '%s' (id=%s)", job.name, job.id)
        return job

    async def remove_job(self, job_id: str) -> bool:
        job = self._jobs.pop(job_id, None)
        if job is None:
            return False
        self._unschedule_job(job_id)
        await self.store.remove_job(job_id)
        logger.info("Removed cron job '%s' (id=%s)", job.name, job.id)
        return True

    async def run_job_now(self, job_id: str) -> Optional[CronRunResult]:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        if job_id in self._running_jobs:
            from .types import CronRunResult

            return CronRunResult(
                job_id=job_id,
                status="error",
                error="job is already running",
                started_at=time.time(),
                finished_at=time.time(),
            )
        self._running_jobs.add(job_id)
        try:
            return await self.execute_fn(job)
        finally:
            self._running_jobs.discard(job_id)

    async def _trigger_dependents(self, completed_job_id: str):
        """Trigger jobs that depend on the completed job, if all their dependencies are satisfied."""
        for other_id, other_job in self._jobs.items():
            if completed_job_id not in other_job.depends_on:
                continue
            if not other_job.enabled or other_id in self._running_jobs:
                continue
            # Check all dependencies for this job
            all_met = all(
                (dep := self._jobs.get(d)) is not None and dep.last_run_status == "success"
                for d in other_job.depends_on
            )
            if all_met:
                logger.info(
                    "Triggering dependent job '%s' (id=%s) after '%s' completed",
                    other_job.name,
                    other_id,
                    completed_job_id,
                )
                asyncio.create_task(self._run_job(other_id))

    def status(self) -> dict:
        enabled = sum(1 for j in self._jobs.values() if j.enabled)
        return {
            "running": self._scheduler is not None and self._scheduler.running,
            "total_jobs": len(self._jobs),
            "enabled_jobs": enabled,
        }
