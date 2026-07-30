"""Tests for auto_extract fire-and-forget approval task (_auto_extract_approval_task).

auto_extract 的异步审批 task：发审批 → 等 /y 或超时 → allow 才存。
用真实 ApprovalManager（asyncio.Event）测三态：allow_once / deny / timeout。
不写死文案，只断言记忆入库与否 + reply 调用次数 + 会话授权。
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config import AgentConfig, MemoryStoreConfig
from src.message import MessageHandler
from src.tools.approval import ApprovalManager


def _make_container(save_approval_mode="model", keyboard_channel=False):
    """构造带记忆配置 + agent_loop（含 invalidate_memory_cache）的 container。

    keyboard_channel=True 时给 qq 装一个 send_approval_keyboard（返回非 None），
    模拟 QQ C2C/group 交互按钮键盘；False 时 qq 无此方法 → 走文本回退。
    """
    container = MagicMock()
    container.config = MagicMock()
    container.config.agents = AgentConfig(language="zh")
    container.config.memory_store = MemoryStoreConfig(enabled=True, save_approval_mode=save_approval_mode)
    container.agent_loop = MagicMock()
    container.agent_loop.invalidate_memory_cache = MagicMock()
    # chat_id 不以 c2c:/group: 等开头时，_get_channel_for_chat_id 走 weixin or qq。
    # 用 spec 限定 qq 的可用方法：keyboard_channel=False 时无 send_approval_keyboard → hasattr 为 False。
    container.qq = MagicMock(spec=[] if not keyboard_channel else ["send_approval_keyboard"])
    container.weixin = MagicMock(spec=[])  # 无 send_approval_keyboard
    if keyboard_channel:
        container.qq.send_approval_keyboard = AsyncMock(return_value={"msg_id": "kb1"})
    return container


def _patch_save_memory(monkeypatch, saved: dict):
    """mock save_memory（message.py 内函数局部 import，patch 源模块）。"""

    async def _fake_save(content, key="", category="fact"):
        saved["content"] = content
        saved["category"] = category
        return json.dumps({"ok": True, "key": key or "spy"})

    monkeypatch.setattr("src.tools.memory_tools.save_memory", _fake_save)


@pytest.mark.asyncio
async def test_allow_once_stores_and_grants_session(monkeypatch):
    """allow_once → save_memory + 会话授权 + 二次确认回复。"""
    container = _make_container()
    handler = MessageHandler(container)
    reply_fn = AsyncMock()

    mgr = ApprovalManager()
    monkeypatch.setattr("src.tools.approval.get_approval_manager", lambda: mgr)
    saved: dict = {}
    _patch_save_memory(monkeypatch, saved)

    # 先起 task，再 resolve（模拟用户 /y）
    task = asyncio.create_task(
        handler._auto_extract_approval_task(
            content="用户用 Python 3.13",
            category="preference",
            source="我用 Python 3.13 写代码",
            thread_id="qq:dm",
            chat_id="c1",
            reply_fn=reply_fn,
        )
    )
    await asyncio.sleep(0.02)  # 让 task 跑到 request_approval + await_approval

    # 找到 pending 请求并 resolve（模拟 /y 多路分发）
    pending = mgr.list_pending()
    assert len(pending) == 1
    mgr.resolve(pending[0].id, "allow_once")

    await asyncio.wait_for(task, timeout=2)

    assert saved.get("content") == "用户用 Python 3.13"  # 落库
    container.agent_loop.invalidate_memory_cache.assert_called_once()
    # 会话授权：本会话内同内容不再问
    assert mgr.has_session_approval("qq:dm", "auto_extract", "用户用 Python 3.13"[:200])
    # reply 至少 2 次：审批提示 + 保存确认
    assert reply_fn.await_count >= 2


@pytest.mark.asyncio
async def test_deny_does_not_store(monkeypatch):
    """deny → 不落库 + 取消提示回复。"""
    container = _make_container()
    handler = MessageHandler(container)
    reply_fn = AsyncMock()

    mgr = ApprovalManager()
    monkeypatch.setattr("src.tools.approval.get_approval_manager", lambda: mgr)
    saved: dict = {}
    _patch_save_memory(monkeypatch, saved)

    task = asyncio.create_task(
        handler._auto_extract_approval_task(
            content="臆测内容",
            category="fact",
            source="来源",
            thread_id="qq:dm",
            chat_id="c1",
            reply_fn=reply_fn,
        )
    )
    await asyncio.sleep(0.02)

    pending = mgr.list_pending()
    mgr.resolve(pending[0].id, "deny")  # 用户回了非 /y 消息

    await asyncio.wait_for(task, timeout=2)

    assert saved == {}  # 没落库
    container.agent_loop.invalidate_memory_cache.assert_not_called()
    assert not mgr.has_session_approval("qq:dm", "auto_extract", "臆测内容"[:200])  # 未授权
    assert reply_fn.await_count >= 2  # 审批提示 + 取消提示


@pytest.mark.asyncio
async def test_timeout_does_not_store(monkeypatch):
    """120s 超时 → 不落库 + 超时提示回复。"""
    container = _make_container()
    handler = MessageHandler(container)
    reply_fn = AsyncMock()

    mgr = ApprovalManager()
    monkeypatch.setattr("src.tools.approval.get_approval_manager", lambda: mgr)
    saved: dict = {}
    _patch_save_memory(monkeypatch, saved)

    # 用极短 timeout 触发超时路径（_auto_extract_approval_task 的 timeout 来自内部默认，
    # 这里无法直接传参；改为 monkeypatch ApprovalRequest.timeout_seconds 或直接 await 真实超时不可行。
    # 换思路：不 resolve，等 task 自己超时——但默认 120s 太久。
    # 方案：patch await_approval 的 timeout 来源。本测试改用"不 resolve + 短等"验证：
    # task 内 await_approval(timeout=120) 会阻塞，这里用一个可控 timeout 注入。
    #
    # 实际实现：_auto_extract_approval_task 用模块级常量 _AUTO_EXTRACT_TIMEOUT。下面 patch 它。
    import src.message as _msg

    monkeypatch.setattr(_msg, "_AUTO_EXTRACT_TIMEOUT", 1)  # 1s 超时

    task = asyncio.create_task(
        handler._auto_extract_approval_task(
            content="超时内容",
            category="fact",
            source="来源",
            thread_id="qq:dm",
            chat_id="c1",
            reply_fn=reply_fn,
        )
    )
    await asyncio.wait_for(task, timeout=5)  # 等 task 自己 1s 超时

    assert saved == {}  # 超时没落库
    container.agent_loop.invalidate_memory_cache.assert_not_called()
    assert reply_fn.await_count >= 2  # 审批提示 + 超时提示


@pytest.mark.asyncio
async def test_failure_cleans_up_orphan_pending(monkeypatch):
    """task 内部抛异常 → 兜底清理孤儿 pending 请求（避免 /y 路由永久看到无法 resolve 的请求）。"""
    container = _make_container()
    handler = MessageHandler(container)

    # 让 save_memory 之后的 reply_fn 在第二次调用（保存确认）时抛异常
    call_count = {"n": 0}
    reply_fn = AsyncMock()

    async def _flaky_reply(text):
        call_count["n"] += 1
        if call_count["n"] == 2:  # 保存确认回复时炸
            raise RuntimeError("reply 失败")

    reply_fn.side_effect = _flaky_reply

    mgr = ApprovalManager()
    monkeypatch.setattr("src.tools.approval.get_approval_manager", lambda: mgr)
    saved: dict = {}
    _patch_save_memory(monkeypatch, saved)

    task = asyncio.create_task(
        handler._auto_extract_approval_task(
            content="允许内容",
            category="fact",
            source="来源",
            thread_id="qq:dm",
            chat_id="c1",
            reply_fn=reply_fn,
        )
    )
    await asyncio.sleep(0.02)

    pending = mgr.list_pending()
    assert len(pending) == 1
    mgr.resolve(pending[0].id, "allow_once")

    await asyncio.wait_for(task, timeout=2)  # 不应抛（task 内兜底了）

    # 兜底清理：异常后 pending 请求已被 cancel_pending 移除
    assert mgr.list_pending() == []
    assert saved.get("content") == "允许内容"  # save 在异常前已执行


# ── QQ 渠道交互按钮键盘（keyboard）路径 ────────────────────


@pytest.mark.asyncio
async def test_keyboard_channel_sends_keyboard_not_text(monkeypatch):
    """QQ 渠道支持 keyboard → 调 send_approval_keyboard（不发送文本审批消息）。

    chat_id 用 c2c: 前缀，_get_channel_for_chat_id 走 qq 分支。
    """
    container = _make_container(keyboard_channel=True)
    handler = MessageHandler(container)
    reply_fn = AsyncMock()

    mgr = ApprovalManager()
    monkeypatch.setattr("src.tools.approval.get_approval_manager", lambda: mgr)
    saved: dict = {}
    _patch_save_memory(monkeypatch, saved)

    task = asyncio.create_task(
        handler._auto_extract_approval_task(
            content="用户用 Python 3.13",
            category="preference",
            source="我用 Python 3.13 写代码",
            thread_id="qq:c2c:u1",
            chat_id="c2c:u1",
            reply_fn=reply_fn,
        )
    )
    await asyncio.sleep(0.02)

    # keyboard 已发送（带 request_id + 保存记忆 action_note）
    container.qq.send_approval_keyboard.assert_awaited_once()
    kb_kwargs = container.qq.send_approval_keyboard.call_args.kwargs
    assert kb_kwargs["chat_id"] == "c2c:u1"
    assert "保存记忆" in kb_kwargs["action_note"]  # 与 memory_save 文案一致
    assert kb_kwargs["is_dangerous"] is False

    pending = mgr.list_pending()
    mgr.resolve(pending[0].id, "allow_once")
    await asyncio.wait_for(task, timeout=2)

    # 落库 + 会话授权
    assert saved.get("content") == "用户用 Python 3.13"
    assert mgr.has_session_approval("qq:c2c:u1", "auto_extract", "用户用 Python 3.13"[:200])


@pytest.mark.asyncio
async def test_keyboard_failure_falls_back_to_text(monkeypatch):
    """send_approval_keyboard 抛异常 → 回退文本审批消息。"""
    container = _make_container(keyboard_channel=True)
    container.qq.send_approval_keyboard = AsyncMock(side_effect=RuntimeError("QQ 接口故障"))
    handler = MessageHandler(container)
    reply_fn = AsyncMock()

    mgr = ApprovalManager()
    monkeypatch.setattr("src.tools.approval.get_approval_manager", lambda: mgr)
    saved: dict = {}
    _patch_save_memory(monkeypatch, saved)

    task = asyncio.create_task(
        handler._auto_extract_approval_task(
            content="臆测内容",
            category="fact",
            source="来源",
            thread_id="qq:c2c:u1",
            chat_id="c2c:u1",
            reply_fn=reply_fn,
        )
    )
    await asyncio.sleep(0.02)

    # keyboard 试过但失败 → 回退文本（reply_fn 收到审批文案）
    container.qq.send_approval_keyboard.assert_awaited_once()
    # 第一条 reply_fn 应是审批文本（回退路径）
    first_msg = reply_fn.call_args_list[0].args[0]
    assert "臆测内容" in first_msg  # _memory_save_approval_text 文案含 content

    pending = mgr.list_pending()
    mgr.resolve(pending[0].id, "deny")
    await asyncio.wait_for(task, timeout=2)

    assert saved == {}  # deny 没落库


@pytest.mark.asyncio
async def test_always_button_grants_pattern_session_approval(monkeypatch):
    """keyboard 的"⭐ 始终允许"按钮（always）→ pattern 授权，gate 二次提取同内容直接放行。

    模拟 _handle_interaction 的 always 路径：approval_key == tool_name 时走
    approve_session_pattern(thread_id, "auto_extract")。gate 的 has_session_approval
    pattern 分支命中 tool_name → 二次同内容提取不再弹审批。
    """
    container = _make_container(keyboard_channel=True)
    handler = MessageHandler(container)
    reply_fn = AsyncMock()

    mgr = ApprovalManager()
    monkeypatch.setattr("src.tools.approval.get_approval_manager", lambda: mgr)
    saved: dict = {}
    _patch_save_memory(monkeypatch, saved)

    thread_id = "qq:c2c:u1"
    task = asyncio.create_task(
        handler._auto_extract_approval_task(
            content="我喜欢深色主题",
            category="preference",
            source="我喜欢深色主题",
            thread_id=thread_id,
            chat_id="c2c:u1",
            reply_fn=reply_fn,
        )
    )
    await asyncio.sleep(0.02)

    pending = mgr.list_pending()
    assert len(pending) == 1
    # 模拟用户点"⭐ 始终允许"按钮（_handle_interaction always 路径）：
    # approval_key("auto_extract") == tool_name("auto_extract") → approve_session_pattern
    mgr.approve_session_pattern(thread_id, "auto_extract")
    mgr.resolve(pending[0].id, "allow_once")
    await asyncio.wait_for(task, timeout=2)

    # pattern 授权生效：gate 查 has_session_approval 应命中 tool_name pattern
    from src.tools.memory_tools import _gate_auto_extract

    decision, _reason = await _gate_auto_extract("我喜欢深色主题", source_context="我喜欢深色主题")
    assert decision == "allow"  # 二次同内容不再弹审批（pattern 命中）
