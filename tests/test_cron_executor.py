"""cron 执行器:isolated 线程用完即弃。"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.state import AgentState
from src.cron.executor import execute_cron_job
from src.cron.types import CronDelivery, CronJob, CronPayload, CronSchedule


def _agent_loop_with_store(store):
    loop = MagicMock()
    loop.get_store = MagicMock(return_value=store)
    loop.run = AsyncMock(
        return_value=AgentState(messages=[{"role": "user", "content": "x"}, {"role": "assistant", "content": "done"}])
    )
    return loop


def _make_config() -> MagicMock:
    config = MagicMock()
    config.agents.system_prompt = ""
    config.task.defer_minutes = 5  # deferred 用例需要;非 task 用例不访问 config.task,无副作用
    return config


def _job(session_target="isolated", *, name="test-reminder", payload_kind="agent_turn", message="remind me"):
    return CronJob(
        name=name,
        schedule=CronSchedule(kind="at", at="2030-01-01 00:00:00"),
        payload=CronPayload(kind=payload_kind, message=message),
        delivery=CronDelivery(mode="none"),
        session_target=session_target,
    )


@pytest.mark.parametrize("session_target,expect_delete", [("isolated", True), ("main", False)])
@pytest.mark.asyncio
async def test_cleanup_follows_session_target(session_target, expect_delete):
    """isolated 用完即弃(删线程),main 保留(不删)。"""
    store = MagicMock()
    store.load = AsyncMock(return_value=None)
    store.delete = AsyncMock(return_value=True)

    job = _job(session_target)
    result = await execute_cron_job(job, _agent_loop_with_store(store), _make_config())

    assert result.status == "success"
    if expect_delete:
        store.delete.assert_awaited_once_with(f"cron:{job.id}")
    else:
        store.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_isolated_thread_deleted_on_timeout():
    """超时也走 finally 删除——避免 checkpoints.db 累积死线程。"""
    store = MagicMock()
    store.load = AsyncMock(return_value=None)
    store.delete = AsyncMock(return_value=True)

    job = _job("isolated")
    loop = _agent_loop_with_store(store)
    loop.run = AsyncMock(side_effect=asyncio.TimeoutError())

    result = await execute_cron_job(job, loop, _make_config())

    assert result.status == "timeout"
    store.delete.assert_awaited_once_with(f"cron:{job.id}")


@pytest.mark.asyncio
async def test_isolated_thread_deleted_on_error():
    """agent 抛错也走 finally 删除——线程不残留。"""
    store = MagicMock()
    store.load = AsyncMock(return_value=None)
    store.delete = AsyncMock(return_value=True)

    job = _job("isolated")
    loop = _agent_loop_with_store(store)
    loop.run = AsyncMock(side_effect=RuntimeError("boom"))

    result = await execute_cron_job(job, loop, _make_config())

    assert result.status == "error"
    store.delete.assert_awaited_once_with(f"cron:{job.id}")


@pytest.mark.asyncio
async def test_deferred_task_job_preserves_thread():
    """busy 的 task job 在 try 之前 early-return,不删线程——保住任务进度。"""
    store = MagicMock()
    store.load = AsyncMock(return_value=None)
    store.delete = AsyncMock(return_value=True)

    job = _job("isolated", name="task:run1:cp:step1")
    loop = _agent_loop_with_store(store)
    loop.is_thread_busy = MagicMock(return_value=True)

    result = await execute_cron_job(job, loop, _make_config())

    assert result.status == "deferred"
    store.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_direct_payload_does_not_touch_thread():
    """direct 载荷在 try 之前 return,不创建/不删线程。"""
    store = MagicMock()
    store.load = AsyncMock(return_value=None)
    store.delete = AsyncMock(return_value=True)

    job = _job("isolated", payload_kind="direct", message="ping")
    loop = _agent_loop_with_store(store)

    result = await execute_cron_job(job, loop, _make_config())

    assert result.status == "success"
    assert result.output == "ping"
    store.delete.assert_not_awaited()


def _loop_capturing_chat_context(store, captured):
    """run 期间抓取 chat 上下文的假 agent loop。"""
    from src.tools.chat_tools import _get_context

    async def fake_run(state, tid):
        captured["ctx"] = _get_context()
        return AgentState(messages=[{"role": "user", "content": "x"}, {"role": "assistant", "content": "done"}])

    loop = MagicMock()
    loop.get_store = MagicMock(return_value=store)
    loop.run = fake_run
    return loop


@pytest.mark.asyncio
async def test_agent_run_binds_declared_delivery_context():
    """agent 运行期间 chat 上下文 = job 声明的投递目标;跑完还原调用前状态。

    复现事故形态:外层预置定时链继承来的残值,声明目标必须盖过残值,
    且 token 还原不能污染调用方。
    """
    from src.tools.chat_tools import _current_channel_type, _current_chat_id, _get_context

    captured = {}
    store = MagicMock()
    store.load = AsyncMock(return_value=None)
    store.delete = AsyncMock(return_value=True)
    loop = _loop_capturing_chat_context(store, captured)

    job = CronJob(
        name="bind-ctx",
        schedule=CronSchedule(kind="at", at="2030-01-01 00:00:00"),
        payload=CronPayload(kind="agent_turn", message="m"),
        delivery=CronDelivery(mode="none", to="c2c:mine", channel_type="qq"),
    )

    t1 = _current_channel_type.set("qq")
    t2 = _current_chat_id.set("c2c:stale")
    try:
        result = await execute_cron_job(job, loop, _make_config())
        assert result.status == "success"
        assert captured["ctx"] == ("qq", "c2c:mine")
        assert _get_context() == ("qq", "c2c:stale")
    finally:
        _current_channel_type.reset(t1)
        _current_chat_id.reset(t2)


@pytest.mark.asyncio
async def test_agent_run_clears_inherited_context_without_target():
    """无投递目标的任务:运行期间上下文必须为空,不得借用定时链残值。"""
    from src.tools.chat_tools import _current_channel_type, _current_chat_id, _get_context

    captured = {}
    store = MagicMock()
    store.load = AsyncMock(return_value=None)
    store.delete = AsyncMock(return_value=True)
    loop = _loop_capturing_chat_context(store, captured)

    job = _job("isolated")

    t1 = _current_channel_type.set("qq")
    t2 = _current_chat_id.set("c2c:stale")
    try:
        result = await execute_cron_job(job, loop, _make_config())
        assert result.status == "success"
        assert captured["ctx"] == ("", "")
        assert _get_context() == ("qq", "c2c:stale")
    finally:
        _current_channel_type.reset(t1)
        _current_chat_id.reset(t2)
