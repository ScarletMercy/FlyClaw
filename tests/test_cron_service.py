"""CronService 崩溃恢复与提醒测试。

覆盖:
- 改动1:执行开始落盘 running_at(硬崩溃可被检测的前提)
- 改动2:start() 崩溃恢复扫描收集 _pending_crash_alerts + flush_crash_alerts 发送
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.cron.service import CronService
from src.cron.types import CronDelivery, CronJob, CronPayload, CronRunResult, CronSchedule


def _make_store(jobs):
    store = MagicMock()
    store.load_jobs = AsyncMock(return_value=jobs)
    store.save_job = AsyncMock()
    store.remove_job = AsyncMock(return_value=True)
    store.close = AsyncMock()
    return store


def _make_config():
    config = MagicMock()
    config.agents.timezone = "UTC"
    config.cron.max_transient_retries = 3
    config.cron.failure_alert_after = 2
    config.cron.shutdown_timeout_seconds = 30.0
    return config


def _job(
    *,
    running_at=None,
    name="crashed-job",
    enabled=True,
    timeout=600,
    delivery_mode="announce",
    to="chat123",
    channel=None,
):
    return CronJob(
        name=name,
        enabled=enabled,
        schedule=CronSchedule(kind="cron", expr="*/5 * * * *"),
        payload=CronPayload(kind="agent_turn", message="m", timeout_seconds=timeout),
        delivery=CronDelivery(mode=delivery_mode, to=to, channel=channel),
        running_at=running_at,
    )


def _patch_scheduler(monkeypatch):
    """避免真起 APScheduler:start() / reschedule() 里的 _scheduler 全打 mock。"""
    import src.cron.service as svc_mod

    monkeypatch.setattr(svc_mod, "AsyncIOScheduler", lambda **kw: MagicMock())


@pytest.mark.asyncio
async def test_crash_recovery_collects_alert(monkeypatch):
    """running_at 远超阈值 -> 标 interrupted + 收集到 _pending_crash_alerts。"""
    _patch_scheduler(monkeypatch)
    stale = time.time() - 3600  # 1h 前,远超 max(120s, timeout=600s)
    job = _job(running_at=stale)
    store = _make_store([job])
    svc = CronService(store, AsyncMock(), config=_make_config(), channel=None)
    await svc.start()

    assert job.last_run_status == "interrupted"
    assert job.running_at is None
    alerts = svc.drain_pending_crash_alerts()
    assert len(alerts) == 1
    assert alerts[0].id == job.id
    await svc.stop()


@pytest.mark.asyncio
async def test_grace_window_skips_recovery(monkeypatch):
    """running_at 在 grace 窗口内(<阈值) -> 不收集,清 running_at 让 scheduler 正常 pick。"""
    _patch_scheduler(monkeypatch)
    job = _job(running_at=time.time() - 10)  # 10s < 120s 阈值
    store = _make_store([job])
    svc = CronService(store, AsyncMock(), config=_make_config(), channel=None)
    await svc.start()

    assert job.running_at is None
    assert svc.drain_pending_crash_alerts() == []
    await svc.stop()


@pytest.mark.asyncio
async def test_normal_job_not_collected(monkeypatch):
    """正常 job(running_at=None)不进 alerts。"""
    _patch_scheduler(monkeypatch)
    job = _job(running_at=None)
    store = _make_store([job])
    svc = CronService(store, AsyncMock(), config=_make_config(), channel=None)
    await svc.start()

    assert svc.drain_pending_crash_alerts() == []
    await svc.stop()


@pytest.mark.asyncio
async def test_reschedule_does_not_collect(monkeypatch):
    """reschedule(config reload)路径标记 interrupted 但不收集崩溃提醒。"""
    _patch_scheduler(monkeypatch)
    stale = time.time() - 3600
    job = _job(running_at=stale)
    store = _make_store([job])
    svc = CronService(store, AsyncMock(), config=_make_config(), channel=None)
    await svc.reschedule()

    assert job.last_run_status == "interrupted"
    assert svc.drain_pending_crash_alerts() == []
    await svc.stop()


@pytest.mark.asyncio
async def test_flush_sends_to_delivery_target(monkeypatch):
    """flush: announce+to -> channel.send_text 调用,文案含 job.name 与'系统提示'。"""
    _patch_scheduler(monkeypatch)
    stale = time.time() - 3600
    job = _job(running_at=stale, name="我的提醒任务")
    store = _make_store([job])
    channel = MagicMock()
    channel.send_text = AsyncMock()
    svc = CronService(store, AsyncMock(), config=_make_config(), channel=None)
    await svc.start()
    svc._channel = channel  # 模拟 app 在 channel 就绪后赋值
    await svc.flush_crash_alerts()

    channel.send_text.assert_awaited_once()
    args = channel.send_text.call_args.args
    assert args[0] == "chat123"
    assert "系统提示" in args[1]
    assert "我的提醒任务" in args[1]
    assert svc.drain_pending_crash_alerts() == []  # flush 后清空
    await svc.stop()


@pytest.mark.asyncio
async def test_flush_none_delivery_skipped(monkeypatch):
    """flush: delivery=none -> 不调 send_text,仅日志(方案 A 盲区)。"""
    _patch_scheduler(monkeypatch)
    stale = time.time() - 3600
    job = _job(running_at=stale, delivery_mode="none")
    store = _make_store([job])
    channel = MagicMock()
    channel.send_text = AsyncMock()
    svc = CronService(store, AsyncMock(), config=_make_config(), channel=None)
    await svc.start()
    svc._channel = channel
    await svc.flush_crash_alerts()

    channel.send_text.assert_not_awaited()
    await svc.stop()


@pytest.mark.asyncio
async def test_flush_without_channel_skipped(monkeypatch):
    """flush: channel 未就绪(_channel=None) -> 不发,仅日志,不抛。"""
    _patch_scheduler(monkeypatch)
    stale = time.time() - 3600
    job = _job(running_at=stale)
    store = _make_store([job])
    svc = CronService(store, AsyncMock(), config=_make_config(), channel=None)
    await svc.start()
    # 不赋 _channel,保持 None
    await svc.flush_crash_alerts()  # 不应抛
    await svc.stop()


@pytest.mark.asyncio
async def test_flush_sends_to_channel_only_delivery(monkeypatch):
    """announce + 仅 channel(无 to):与 _deliver_result/_send_failure_alert 一致,
    走 to-or-channel 解析发到 channel。回归 P1:此前 flush 只看 to 会漏发。"""
    _patch_scheduler(monkeypatch)
    stale = time.time() - 3600
    job = _job(running_at=stale, name="channel-only", to=None, channel="group_x")
    store = _make_store([job])
    channel = MagicMock()
    channel.send_text = AsyncMock()
    svc = CronService(store, AsyncMock(), config=_make_config(), channel=None)
    await svc.start()
    svc._channel = channel
    await svc.flush_crash_alerts()

    channel.send_text.assert_awaited_once()
    assert channel.send_text.call_args.args[0] == "group_x"
    assert "channel-only" in channel.send_text.call_args.args[1]
    await svc.stop()


@pytest.mark.asyncio
async def test_run_job_now_persists_running_before_execute(monkeypatch):
    """改动1:run_job_now 开头落盘 running_at,且在 execute_fn 之前。"""
    _patch_scheduler(monkeypatch)
    job = _job(name="manual-run")
    store = _make_store([])
    saved_running_at = []

    async def capture_save(j):
        saved_running_at.append(j.running_at)

    store.save_job = capture_save
    execute_fn = AsyncMock(return_value=CronRunResult(job_id=job.id, status="success", started_at=time.time()))
    svc = CronService(store, execute_fn, config=_make_config(), channel=None)
    svc._jobs[job.id] = job
    await svc.run_job_now(job.id)

    # 开头落盘执行中状态(execute_fn 之前)
    assert saved_running_at[0] is not None
    # finally 清盘:避免进程崩溃后 start() 误判已完成的 job 为"崩溃中"
    assert saved_running_at[-1] is None
    assert len(saved_running_at) >= 2


@pytest.mark.asyncio
async def test_run_job_finally_clears_running_on_base_exception(monkeypatch):
    """_run_job 遇 BaseException(CancelledError 绕过 except Exception)时 finally 也清盘 running_at。

    开头落盘了 running_at,若 finally 只清内存不 save,CancelledError 路径下 DB 残留
    stale running_at -> 下次 start() 误报崩溃。验证 finally 对称 save。
    """
    _patch_scheduler(monkeypatch)
    job = _job(name="cancel-test")
    store = _make_store([])
    saved = []

    async def capture(j):
        saved.append(j.running_at)

    store.save_job = capture

    async def raise_cancel(j):
        raise asyncio.CancelledError()

    svc = CronService(store, raise_cancel, config=_make_config(), channel=None)
    svc._jobs[job.id] = job
    with pytest.raises(asyncio.CancelledError):
        await svc._run_job(job.id)

    assert saved[0] is not None  # 开头落盘
    assert saved[-1] is None  # finally 对称清盘


@pytest.mark.asyncio
async def test_crash_recovery_e2e_real_store(tmp_path, monkeypatch):
    """e2e:真实 CronStore 往返。

    单元测试 mock 了 store,running_at 是否真序列化进 DB 并能 load 回从未验证。
    本测试用真实 sqlite store:落盘 running_at -> 新 service load -> 检测 -> flush。
    若 model 序列化漏了 running_at,本测试会失败(盲点兜底)。
    """
    _patch_scheduler(monkeypatch)
    from src.cron.store import CronStore

    store = CronStore(tmp_path / "cron.db")
    # 模拟"执行中落盘 running_at 后进程崩溃"
    job = _job(name="e2e-crashed")
    job.running_at = time.time() - 3600
    await store.save_job(job)

    # 重启:新 service 从真实 store load_jobs
    channel = MagicMock()
    channel.send_text = AsyncMock()
    svc = CronService(store, AsyncMock(), config=_make_config(), channel=None)
    await svc.start()
    svc._channel = channel
    await svc.flush_crash_alerts()

    channel.send_text.assert_awaited_once()
    assert "e2e-crashed" in channel.send_text.call_args.args[1]
    # DB 里 running_at 已被恢复扫描清掉
    reloaded = await store.load_jobs()
    assert reloaded[0].running_at is None
    await store.close()


@pytest.mark.asyncio
async def test_run_job_now_does_not_resurrect_removed_job(monkeypatch):
    """run_job_now 执行期间 job 被 remove,finally 不得 save 复活已删 job。"""
    _patch_scheduler(monkeypatch)
    job = _job(name="remove-during-run")
    store = _make_store([])
    saved = []

    async def capture(j):
        saved.append(j.id)

    store.save_job = capture
    store.remove_job = AsyncMock(return_value=True)

    async def execute_then_remove(j):
        # 模拟执行期间外部 remove_job 删掉它(_jobs 移除 + DB 删行)
        await svc.remove_job(j.id)
        return CronRunResult(job_id=j.id, status="success", started_at=time.time())

    svc = CronService(store, execute_then_remove, config=_make_config(), channel=None)
    svc._jobs[job.id] = job
    await svc.run_job_now(job.id)

    # 只有开头那次落盘 running_at;finally 因 job 已不在 _jobs,守卫挡住 save(不复活)
    assert len(saved) == 1


@pytest.mark.asyncio
async def test_flush_send_timeout_does_not_block(monkeypatch):
    """send_text 挂死时 flush 的 wait_for 超时不阻塞(Win httpx 挂死防护)。

    test_flush_sends 用 AsyncMock 立即返回,wait_for 超时分支零覆盖。本测试用挂起
    send_text + 极小 send_timeout,验证超时降级(不挂、不抛)。
    """
    _patch_scheduler(monkeypatch)
    job = _job(running_at=time.time() - 3600)
    store = _make_store([job])
    channel = MagicMock()

    async def slow_send(target, text):
        await asyncio.sleep(100)  # 模拟永久挂死

    channel.send_text = slow_send
    svc = CronService(store, AsyncMock(), config=_make_config(), channel=None)
    await svc.start()
    svc._channel = channel

    t0 = time.monotonic()
    await svc.flush_crash_alerts(send_timeout=0.05)
    assert time.monotonic() - t0 < 1.0  # 没被 sleep(100) 阻塞
    await svc.stop()


@pytest.mark.asyncio
async def test_run_job_survives_persist_running_at_failure(monkeypatch):
    """开头 save_job 失败时:吞掉降级,execute 仍执行,_running_jobs 正常清理(不卡死)。"""
    _patch_scheduler(monkeypatch)
    job = _job(name="save-fail")
    store = _make_store([])
    call_count = {"n": 0}

    async def fail_first(j):
        call_count["n"] += 1
        if call_count["n"] == 1:  # 开头那次(318)抛
            raise RuntimeError("db locked")
        # finally/清盘那次(428)成功

    store.save_job = fail_first
    execute_fn = AsyncMock(return_value=CronRunResult(job_id=job.id, status="success", started_at=time.time()))
    svc = CronService(store, execute_fn, config=_make_config(), channel=None)
    svc._jobs[job.id] = job
    await svc._run_job(job.id)

    execute_fn.assert_awaited_once()  # 开头 save 失败没阻断执行
    assert job.id not in svc._running_jobs  # 已清理,没卡死


@pytest.mark.asyncio
async def test_run_job_survives_clear_running_at_failure(monkeypatch):
    """正常路径清盘 save(433)失败时:吞掉降级,_run_job 正常返回不抛,
    后续 last_run_status 等收尾照常完成。

    回归守护(P0):此前 433 未守护,清盘 save 失败会抛出 _run_job,
    把一个跑成功的任务中断成半截,且 DB 残留 stale running_at。
    """
    _patch_scheduler(monkeypatch)
    job = _job(name="clear-fail")
    store = _make_store([])
    call_count = {"n": 0}

    async def fail_second(j):
        call_count["n"] += 1
        if call_count["n"] == 2:  # 433 清盘那次抛(321 开头那次成功)
            raise RuntimeError("db locked")

    store.save_job = fail_second
    execute_fn = AsyncMock(return_value=CronRunResult(job_id=job.id, status="success", started_at=time.time()))
    svc = CronService(store, execute_fn, config=_make_config(), channel=None)
    svc._jobs[job.id] = job
    await svc._run_job(job.id)  # 修复后不抛

    execute_fn.assert_awaited_once()
    assert job.running_at is None  # 内存已清
    assert job.last_run_status == "success"  # 结果照常处理,没被 save 失败中断
    assert job.id not in svc._running_jobs  # 没卡死
