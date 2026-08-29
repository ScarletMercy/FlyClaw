from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Callable, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.base import JobLookupError

from .store import CronStore
from .types import CronJob, CronJobCreate, CronJobPatch, CronRunResult

logger = logging.getLogger("flyclaw.cron")

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
        self._running_tasks: dict[str, asyncio.Task] = {}
        # start() 崩溃恢复扫描收集的"被中断"任务,由 app 在 channel 就绪后
        # 调 flush_crash_alerts() 发提醒(start() 早于 channel 启动,不能当场发)。
        self._pending_crash_alerts: list[CronJob] = []
        # Incremented on every stop/reschedule so stale finally-blocks from
        # cancelled tasks know not to mutate the new state.
        self._epoch: int = 0
        if config:
            self._max_transient_retries = config.cron.max_transient_retries
            self._failure_alert_after = config.cron.failure_alert_after
            self._shutdown_timeout = config.cron.shutdown_timeout_seconds
        else:
            self._max_transient_retries = 3
            self._shutdown_timeout = 30.0

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

    def _get_scheduler_tz(self):
        """Get the scheduler timezone from config (defaults to system local)."""
        from src.utils.tz import get_tz

        tz_name = self._config.agents.timezone if self._config else None
        return get_tz(tz_name)

    async def start(self):
        tz = self._get_scheduler_tz()
        self._scheduler = AsyncIOScheduler(timezone=tz)
        jobs = await self.store.load_jobs()

        # Crash recovery: detect jobs that were running when process died.
        # Only recover jobs whose running_at is older than a minimum stale
        # threshold — brief pauses (debugger, system sleep) should NOT be
        # treated as crashes.
        _MIN_STALE_SECONDS = 120  # 2 minutes grace window
        now = time.time()
        recovered = 0
        skipped = 0
        for job in jobs:
            if job.running_at is not None:
                stale_seconds = now - job.running_at
                # Use the greater of the minimum threshold and the job's own
                # timeout as the stale threshold.  A job that hasn't exceeded
                # its normal timeout is very likely still running fine.
                threshold = max(_MIN_STALE_SECONDS, job.payload.timeout_seconds or 0)
                if stale_seconds < threshold:
                    logger.info(
                        "Skipping recovery for job '%s' (id=%s, stale %.0fs < threshold %.0fs)",
                        job.name,
                        job.id,
                        stale_seconds,
                        threshold,
                    )
                    # Clear running_at so the scheduler can pick it up normally
                    job.running_at = None
                    await self.store.save_job(job)
                    skipped += 1
                    continue
                logger.warning(
                    "Recovering stale job '%s' (id=%s, stale %.0fs)",
                    job.name,
                    job.id,
                    stale_seconds,
                )
                job.running_at = None
                job.last_run_status = "interrupted"
                job.last_error = f"Process crashed while running (stale {stale_seconds:.0f}s)"
                job.consecutive_errors += 1
                await self.store.save_job(job)
                self._pending_crash_alerts.append(job)
                recovered += 1
        if recovered:
            logger.info("Crash recovery: reset %d stale jobs", recovered)
        if skipped:
            logger.info("Crash recovery: skipped %d jobs within grace window", skipped)

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

    def drain_pending_crash_alerts(self) -> list[CronJob]:
        """返回并清空崩溃恢复收集的任务列表(检查/测试用)。"""
        alerts = self._pending_crash_alerts
        self._pending_crash_alerts = []
        return alerts

    async def flush_crash_alerts(self, send_timeout: float = 10.0) -> None:
        """发送 start() 崩溃恢复扫描收集的任务崩溃提醒。

        在 channel 就绪后由 app 调用(start() 早于 _start_channels,当场发不出去)。
        走 job.delivery 目标;delivery=none 的仅记日志(方案 A 盲区:程序化建的后台任务)。
        """
        for job in self.drain_pending_crash_alerts():
            text = f"系统提示：{job.name} 任务崩溃了"
            delivery = job.delivery
            # 与 _deliver_result/_send_failure_alert 一致:to-or-channel 解析目标,
            # 否则仅设了 channel 的 announce job 会漏发崩溃提醒。
            target = delivery.to or delivery.channel
            if delivery.mode != "announce" or not target or not self._channel:
                logger.warning("Crash alert skipped (no delivery target): job='%s'", job.name)
                continue
            try:
                # wait_for 防 channel.send_text 挂死拖垮启动(Win 上 httpx 对部分
                # TLS 接口永久挂死、timeout 不触发,见 project_windows-async-httpx-hang)。
                await asyncio.wait_for(self._channel.send_text(target, text), timeout=send_timeout)
                logger.info("Crash alert sent: job='%s' -> %s", job.name, target)
            except asyncio.TimeoutError:
                logger.warning("Crash alert send timed out (10s): job='%s'", job.name)
            except Exception as e:
                logger.warning("Crash alert send failed: job='%s' %s", job.name, e)

    async def stop(self):
        # 1. Stop the scheduler (prevents new job submissions)
        if self._scheduler:
            self._scheduler.shutdown(wait=False)
            self._scheduler = None
        # 2. Drain running tasks
        await self._drain_running_tasks()
        # 3. Close store
        await self.store.close()
        logger.info("Cron service stopped")

    async def _drain_running_tasks(self):
        """Wait for running job tasks to complete, with timeout and state persistence."""
        if not self._running_tasks:
            return
        # Bump epoch so stale finally-blocks from cancelled tasks won't
        # corrupt state created after reschedule.
        self._epoch += 1
        timeout = self._shutdown_timeout
        logger.info("Draining %d running cron jobs (timeout=%.1fs)", len(self._running_tasks), timeout)
        try:
            tasks = list(self._running_tasks.values())
            done, pending = await asyncio.wait(tasks, timeout=timeout)
            if pending:
                logger.warning("%d cron jobs did not finish in time, cancelling", len(pending))
                for t in pending:
                    t.cancel()
                _, still_pending = await asyncio.wait(pending, timeout=5.0)
                if still_pending:
                    logger.warning(
                        "%d cron jobs survived 5s cancel grace period, abandoning",
                        len(still_pending),
                    )
        except Exception as e:
            logger.error("Error draining cron tasks: %s", e)
        # Persist state for any jobs still marked as running (for crash recovery on restart)
        for job_id in list(self._running_jobs):
            job = self._jobs.get(job_id)
            if job and job.running_at is not None:
                try:
                    await self.store.save_job(job)
                except Exception as e:
                    logger.error("Failed to save job %s during drain: %s", job_id, e)
        self._running_tasks.clear()
        self._running_jobs.clear()

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
                self._run_job_tracked,
                trigger=trigger,
                id=job_id,
                args=[job.id],
                misfire_grace_time=3600,
                max_instances=1,
            )
            try:
                next_time = trigger.get_next_fire_time(None, datetime.now().astimezone())
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

    async def _run_job_tracked(self, job_id: str):
        """Wrapper that registers the running task for graceful shutdown."""
        task = asyncio.current_task()
        epoch = self._epoch
        if task:
            self._running_tasks[job_id] = task
        try:
            await self._run_job(job_id, epoch=epoch)
        finally:
            # Only mutate shared state if we're in the same epoch.
            # After _drain_running_tasks bumps the epoch, stale tasks must
            # not touch the new _running_tasks / _running_jobs.
            if self._epoch == epoch:
                self._running_tasks.pop(job_id, None)

    async def _run_job(self, job_id: str, *, epoch: int = 0):
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
                if dep.last_run_status not in ("success", "delivery_failed"):
                    unmet.append(dep_id)
            if unmet:
                logger.info(
                    "Job '%s' skipped — dependencies not satisfied: %s",
                    job.name,
                    ", ".join(unmet),
                )
                return

        self._running_jobs.add(job_id)
        job.running_at = time.time()
        # 落盘 running_at:硬崩溃(kill/断电)时内存丢失,重启后 start() 恢复扫描
        # 靠 DB 里的 running_at 识别被中断的任务。否则只有优雅关闭那条路会存。
        # save 失败仅降级(这次 run 不被崩溃检测覆盖):必须吞掉——318 在 try 外,
        # 异常不进 finally,_running_jobs 不 discard → job 永久卡死(下次 skip)。
        try:
            await self.store.save_job(job)
        except Exception as e:
            logger.warning("_run_job: persist running_at failed, crash-detection blind for this run: %s", e)
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
                elif result.status == "delivery_failed":
                    # Execution succeeded but notification delivery failed.
                    # Don't count toward consecutive_errors — the job itself is healthy.
                    job.last_run_status = "delivery_failed"
                    job.last_error = result.error
                    logger.warning(
                        "Job '%s' executed successfully but delivery failed: %s",
                        job.name,
                        result.error,
                    )
                elif result.status == "deferred":
                    logger.info("Job '%s' deferred, rescheduling", job.name)
                    job.last_run_status = "deferred"
                    job.last_error = None
                    try:
                        import json

                        defer_info = json.loads(result.output or "{}")
                        new_at = defer_info.get("new_at", "")
                        if new_at and job.name.startswith("task:"):
                            from .types import CronSchedule, CronJobCreate

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

                                task_store = get_task_store(getattr(self._config.task, "db_path", None))
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
            job.running_at = None  # 清内存标记;此 save 落盘成功即无 false-positive
            # 守护此 save 不抛:抛了会越过下方收尾(依赖触发、at 任务的 remove_job)
            # ——at 任务漏删会被恢复扫描重新调度,真·多跑一轮。
            # 已知降级:save 持续失败时 DB 残留 stale running_at,recurring 任务若恰在
            # 那几分钟窗口内重启,会多一条假"崩溃了"提醒(下次跑自愈,consecutive_errors
            # 一次成功即归零,无害)。不治本,但优于让异常传出 _run_job。
            try:
                await self.store.save_job(job)
            except Exception as e:
                logger.warning("_run_job: failed to persist cleared running_at: %s", e)

            # Trigger downstream jobs that depend on this one.
            # delivery_failed is included because the job *executed* successfully —
            # only the notification channel had issues, which downstream jobs don't depend on.
            if job.last_run_status in ("success", "delivery_failed"):
                await self._trigger_dependents(job_id)

            if job.schedule.kind == "at" and job.delete_after_run:
                await self.remove_job(job.id)
        finally:
            # 兜底清盘:正常路径(except Exception)已 save 清 running_at,此分支不进。
            # 仅 BaseException(CancelledError 等)绕过 except 时进入,见下 if 内注释。
            if job.running_at is not None:
                job.running_at = None
                # 开头已落盘 running_at,清盘必须对称 save:BaseException
                # (CancelledError/KeyboardInterrupt)绕过 except Exception,
                # 只清内存会留 stale running_at -> 下次 start() 误报崩溃。
                # 守卫 _jobs:执行期间被 remove 则不 save,避免复活已删 job。
                if job.id in self._jobs:
                    try:
                        await self.store.save_job(job)
                    except Exception as e:
                        logger.warning("_run_job finally: failed to persist cleared running_at: %s", e)
            # Only discard from the shared set if we're still in the same epoch.
            # Otherwise this is a stale task from before a drain/reschedule and
            # _running_jobs now belongs to the new scheduler incarnation.
            if self._epoch == epoch:
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
        for field in (
            "name",
            "description",
            "enabled",
            "schedule",
            "payload",
            "delivery",
            "session_target",
            "depends_on",
        ):
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
        epoch = self._epoch
        self._running_jobs.add(job_id)
        job.running_at = time.time()
        # save 失败仅降级(见 _run_job 同处注释):吞掉,避免 _running_jobs 残留卡死。
        try:
            await self.store.save_job(job)
        except Exception as e:
            logger.warning("run_job_now: persist running_at failed, crash-detection blind for this run: %s", e)
        try:
            return await self.execute_fn(job)
        finally:
            job.running_at = None
            # 落盘清掉 running_at:开头落盘了执行中状态,此处必须 save 清掉,
            # 否则进程崩溃后 start() 会把这个已完成的 job 误判为"崩溃中"。
            # 守卫 _jobs:执行期间若被 remove_job 删掉(_jobs 移除+DB 删行),
            # 不再 save,避免把已删的 job 又 INSERT 回去(复活)。
            if job.id in self._jobs:
                try:
                    await self.store.save_job(job)
                except Exception as e:
                    logger.warning("run_job_now: failed to persist cleared running_at: %s", e)
            # Only discard if we're still in the same epoch — reschedule()
            # clears _running_jobs for the new incarnation.
            if self._epoch == epoch:
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
                (dep := self._jobs.get(d)) is not None and dep.last_run_status in ("success", "delivery_failed")
                for d in other_job.depends_on
            )
            if all_met:
                logger.info(
                    "Triggering dependent job '%s' (id=%s) after '%s' completed",
                    other_job.name,
                    other_id,
                    completed_job_id,
                )
                asyncio.create_task(self._run_job_tracked(other_id))

    def status(self) -> dict:
        enabled = sum(1 for j in self._jobs.values() if j.enabled)
        return {
            "running": self._scheduler is not None and self._scheduler.running,
            "total_jobs": len(self._jobs),
            "enabled_jobs": enabled,
        }

    async def reschedule(self):
        """Reload jobs from store and reschedule. Used during config reload without closing the store."""
        # Bump epoch first so stale finally-blocks from the old scheduler
        # incarnation won't corrupt the new _running_tasks / _running_jobs.
        self._epoch += 1
        tz = self._get_scheduler_tz()
        self._scheduler = AsyncIOScheduler(timezone=tz)
        jobs = await self.store.load_jobs()
        # Crash recovery scan (same logic as start())
        now = time.time()
        for job in jobs:
            if job.running_at is not None:
                logger.warning("Recovering stale job '%s' during reschedule", job.name)
                job.running_at = None
                job.last_run_status = "interrupted"
                job.last_error = "Job was running during reschedule"
                # Note: consecutive_errors is NOT incremented here because
                # reschedule is an operator-initiated action (config reload),
                # not a job failure. Auto-disable should not be triggered.
                await self.store.save_job(job)
        # Reset running state for the new epoch — old tasks may still be
        # draining but they will be blocked by the epoch guard above.
        self._running_tasks.clear()
        self._running_jobs.clear()
        self._jobs = {j.id: j for j in jobs}
        for job in jobs:
            if job.enabled:
                self._schedule_job(job)
        self._scheduler.start()
        logger.info("Cron service rescheduled with %d jobs", len(jobs))
