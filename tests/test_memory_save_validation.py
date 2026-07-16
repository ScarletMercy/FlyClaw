"""Tests for memory save pre-validation (_validate_memory).

保存前自检：判断候选记忆是否被来源原文支持，可疑则拒绝落库。
降级策略 fail-open：无主模型或调用失败时放行，不阻断记忆系统。
"""

from __future__ import annotations

import json

import pytest

from src.tools import memory_tools
from src.tools.memory_tools import (
    MemorySaveNeedsApproval,
    _validate_memory,
    memory,
    set_memory_dialog_context,
    set_memory_session,
)


def _patch_container_with_client(monkeypatch, client, save_approval_mode="model"):
    """注入带主模型 client 的 container（仿 _extract_relevant_with_llm 获取路径）。

    approval_manager 用 mock（未 approve），让 save 默认走自检路径。
    save_approval_mode 控制memory(save) 审批模式（model/manual）。
    """
    import src._container as _c
    from unittest.mock import MagicMock
    from src.config import MemoryStoreConfig

    _mgr = MagicMock()
    _mgr.has_session_approval.return_value = False

    _ms_cfg = MemoryStoreConfig(save_approval_mode=save_approval_mode)

    class _AgentLoop:
        _client = client

    class _Container:
        agent_loop = _AgentLoop()
        approval_manager = _mgr
        config = MagicMock()

    _Container.config.memory_store = _ms_cfg

    monkeypatch.setattr(_c, "get_container", lambda: _Container())
    # 直接 mock get_approval_manager，避免依赖 approval.py 模块级 import get_container
    # 的绑定时机（跨文件跑时 approval 可能已被前序测试提前 import）
    monkeypatch.setattr("src.tools.approval.get_approval_manager", lambda: _mgr)


# ── 核心裁决解析 ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_validate_allows_when_judge_says_supported(monkeypatch):
    """审核员判定有据 → 放行。"""

    class Client:
        async def chat_simple(self, messages, **extra):
            return "allow 用户原话明确支持"

    _patch_container_with_client(monkeypatch, Client())
    decision, _reason = await _validate_memory("用户用 Python 3.13", "我用 Python 3.13 写代码")
    assert decision == "allow"


@pytest.mark.asyncio
async def test_validate_rejects_when_judge_says_unsupported(monkeypatch):
    """审核员判定臆测（原文不支持）→ 拒绝，且带理由。"""

    class Client:
        async def chat_simple(self, messages, **extra):
            return "reject 原文未提及后端工程师，属臆测"

    _patch_container_with_client(monkeypatch, Client())
    decision, reason = await _validate_memory("用户是后端工程师", "我用 Python 写脚本")
    assert decision == "reject"
    assert reason


# ── 降级：fail-open（不阻断记忆系统）────────────────────────


@pytest.mark.asyncio
async def test_validate_fail_open_when_no_llm_client(monkeypatch):
    """无主模型（get_container 未初始化）→ fail-open 放行。"""
    import src._container as _c

    def _raise_runtime():
        raise RuntimeError("container not initialized")

    monkeypatch.setattr(_c, "get_container", _raise_runtime)
    decision, _reason = await _validate_memory("任意内容", "任意来源")
    assert decision == "allow"


@pytest.mark.asyncio
async def test_validate_fail_open_when_call_fails(monkeypatch):
    """主模型调用抛异常 → fail-open 放行。"""

    class FlakyClient:
        async def chat_simple(self, messages, **extra):
            raise RuntimeError("主模型调用失败")

    _patch_container_with_client(monkeypatch, FlakyClient())
    decision, _reason = await _validate_memory("任意内容", "任意来源")
    assert decision == "allow"


@pytest.mark.asyncio
async def test_validate_allows_without_calling_client_when_no_source(monkeypatch):
    """无来源上下文（拿不到对话原文）→ 直接放行，不调主模型（省调用）。"""

    class Client:
        called = False

        async def chat_simple(self, messages, **extra):
            Client.called = True
            return "reject 不该走到这"

    _patch_container_with_client(monkeypatch, Client())
    decision, _reason = await _validate_memory("任意内容", "")
    assert decision == "allow"
    assert Client.called is False


async def _async_return(val):
    return val


# ── memory(save) 接入自检 ──────────────────────────────────


def _patch_store_with_spy(monkeypatch):
    """mock get_memory_store，返一个记录 remember 调用的 store。"""
    saved: dict = {}

    class _Store:
        async def remember(self, content, key="", category="fact", group_id=""):
            saved["content"] = content
            saved["key"] = key
            return json.dumps({"ok": True, "key": key or "spy_key"})

    monkeypatch.setattr(memory_tools, "get_memory_store", lambda chat_type="p2p": _async_return(_Store()))
    return saved


@pytest.mark.asyncio
async def test_model_mode_reject_returns_rejected_without_approval(monkeypatch):
    """model 模式自检 reject → 直接丢弃（返 rejected result），不抛审批、不存。"""
    set_memory_session("p2p", "")
    set_memory_dialog_context("我用 Python 写脚本")
    saved = _patch_store_with_spy(monkeypatch)

    class Client:
        async def chat_simple(self, messages, **extra):
            return "reject 原文未提及后端工程师"

    _patch_container_with_client(monkeypatch, Client())  # 默认 model 模式

    result = json.loads(await memory(action="save", content="用户是后端工程师"))
    assert result.get("rejected") is True  # 被模型拦
    assert result.get("ok") is not True
    assert saved == {}, "reject 时不应落库"


@pytest.mark.asyncio
async def test_memory_save_stores_when_validation_allows(monkeypatch):
    """自检放行 → 正常落库。"""
    set_memory_session("p2p", "")
    set_memory_dialog_context("我用 Python 3.13")
    saved = _patch_store_with_spy(monkeypatch)

    class Client:
        async def chat_simple(self, messages, **extra):
            return "allow 用户原话支持"

    _patch_container_with_client(monkeypatch, Client())

    result = json.loads(await memory(action="save", content="用户用 Python 3.13"))
    assert result.get("ok") is True
    assert saved.get("content") == "用户用 Python 3.13"


@pytest.mark.asyncio
async def test_memory_save_stores_when_no_source(monkeypatch):
    """无来源上下文 → 自检 fail-open 放行 → 正常落库（不阻断）。"""
    set_memory_session("p2p", "")
    set_memory_dialog_context("")
    saved = _patch_store_with_spy(monkeypatch)

    class Client:
        called = False

        async def chat_simple(self, messages, **extra):
            Client.called = True
            return "reject"

    _patch_container_with_client(monkeypatch, Client())

    result = json.loads(await memory(action="save", content="任意内容"))
    assert result.get("ok") is True
    assert Client.called is False


@pytest.mark.asyncio
async def test_memory_save_skips_validation_when_session_approved(monkeypatch, tmp_path):
    """session 已 approve 该 args（精确 digest）→ 跳过自检直接存（对齐 delete 短路）。

    回归 allow 死循环：allow 后 resume 重新执行 save，session 命中应直接存，
    不再跑自检、不再抛审批。
    """
    set_memory_session("p2p", "")
    set_memory_dialog_context("无关来源")
    saved = _patch_store_with_spy(monkeypatch)

    class Client:
        async def chat_simple(self, messages, **extra):
            return "reject 臆测"

    _patch_container_with_client(monkeypatch, Client())

    # 预先 approve 这条 save 的 args（精确 digest）
    from src.tools.approval import ApprovalManager

    real_mgr = ApprovalManager()
    real_mgr.approve_session("", "memory_save", "臆测内容")
    monkeypatch.setattr("src.tools.approval.get_approval_manager", lambda: real_mgr)

    result = json.loads(await memory(action="save", content="臆测内容"))
    assert result.get("ok") is True
    assert saved.get("content") == "臆测内容"  # 直接存了，没被自检拦


# ── save_approval_mode 配置 ────────────────────────────────


def test_memory_store_config_default_mode_is_model():
    from src.config import MemoryStoreConfig

    assert MemoryStoreConfig().save_approval_mode == "model"


def test_memory_store_config_manual_mode():
    from src.config import MemoryStoreConfig

    assert MemoryStoreConfig(save_approval_mode="manual").save_approval_mode == "manual"


# ── manual 模式：跳过自检，全部人工审批 ────────────────────


@pytest.mark.asyncio
async def test_manual_mode_save_skips_validation_and_requests_approval(monkeypatch):
    """manual 模式：跳过自检，直接转人工审批（即使自检会 allow）。"""
    set_memory_session("p2p", "")
    set_memory_dialog_context("来源")
    saved = _patch_store_with_spy(monkeypatch)

    class Client:
        called = False

        async def chat_simple(self, messages, **extra):
            Client.called = True
            return "allow"

    _patch_container_with_client(monkeypatch, Client(), save_approval_mode="manual")

    with pytest.raises(MemorySaveNeedsApproval) as ei:
        await memory(action="save", content="任何内容")
    assert ei.value.mode == "manual"
    assert Client.called is False  # manual 不自检
    assert saved == {}  # 没存


@pytest.mark.asyncio
async def test_manual_mode_session_approved_stores_directly(monkeypatch, tmp_path):
    """manual 模式 + session 已 approve → 直接存（短路，不弹审批）。"""
    set_memory_session("p2p", "")
    set_memory_dialog_context("来源")
    saved = _patch_store_with_spy(monkeypatch)

    class Client:
        async def chat_simple(self, *a, **k):
            return "allow"

    _patch_container_with_client(monkeypatch, Client(), save_approval_mode="manual")

    from src.tools.approval import ApprovalManager

    real_mgr = ApprovalManager()
    real_mgr.approve_session("", "memory_save", "任何内容")
    monkeypatch.setattr("src.tools.approval.get_approval_manager", lambda: real_mgr)

    result = json.loads(await memory(action="save", content="任何内容"))
    assert result.get("ok") is True
    assert saved.get("content") == "任何内容"


# ── build_dialog_context（source 拼接）──────────────────────


def test_build_dialog_context_joins_recent_turns():
    from src.tools.memory_tools import build_dialog_context

    messages = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好啊"},
        {"role": "user", "content": "我用 Python 3.13"},
    ]
    ctx = build_dialog_context(messages)
    assert "你好" in ctx
    assert "我用 Python 3.13" in ctx


def test_build_dialog_context_skips_empty_content():
    from src.tools.memory_tools import build_dialog_context

    messages = [
        {"role": "user", "content": "实际问题"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "x"}]},
        {"role": "tool", "content": ""},
        {"role": "user", "content": "后续"},
    ]
    ctx = build_dialog_context(messages)
    assert "后续" in ctx
    assert "实际问题" in ctx
    assert "assistant: \n" not in ctx  # 空 content 不应产生空行


def test_build_dialog_context_truncates_long_text():
    from src.tools.memory_tools import build_dialog_context

    long = "x" * 5000
    ctx = build_dialog_context([{"role": "user", "content": long}])
    assert len(ctx) <= 2100  # max_chars=2000 + 少量前缀


# ── memory_save 审批文案 ────────────────────────────────────


def test_memory_save_approval_text_includes_content_and_source():
    from src.tools.memory_tools import _memory_save_approval_text

    text = _memory_save_approval_text("用户是后端工程师", "我用 Python 写脚本", timeout=120, zh=True)
    assert "用户是后端工程师" in text  # 候选记忆展示
    assert "我用 Python 写脚本" in text  # 来源原文展示，供核对
    assert "/y" in text or "确认" in text


def test_memory_save_approval_text_omits_source_line_when_empty():
    from src.tools.memory_tools import _memory_save_approval_text

    text = _memory_save_approval_text("某内容", "", timeout=120, zh=True)
    assert "某内容" in text
    assert "来源原文" not in text  # 无来源时不显示来源行（降级）


def test_memory_save_approval_text_manual_mode_no_speculation():
    """manual 模式文案不说'臆测'（无模型判断），用'确认保存'。"""
    from src.tools.memory_tools import _memory_save_approval_text

    text = _memory_save_approval_text("某内容", "来源", timeout=120, zh=True, model_mode=False)
    assert "某内容" in text
    assert "来源" in text
    assert "臆测" not in text
    assert "确认" in text or "/y" in text


def test_memory_save_approval_text_model_mode_mentions_speculation():
    """model 模式文案提示'疑似臆测'（有模型判断）。"""
    from src.tools.memory_tools import _memory_save_approval_text

    text = _memory_save_approval_text("某内容", "来源", timeout=120, zh=True, model_mode=True)
    assert "臆测" in text


# ── consecutive_denies：memory_save 逐条 deny 不计入终止 ──────


def test_deny_counts_toward_abort_for_non_memory_save():
    """exec/memory_delete 的 deny 计入终止计数（防死循环重试）。"""
    from src.tools.memory_tools import _deny_counts_toward_abort

    assert _deny_counts_toward_abort("exec_command") is True
    assert _deny_counts_toward_abort("memory_delete") is True


def test_deny_does_not_count_for_memory_save():
    """memory_save 逐条审批的 deny 不计入终止（不同 content，非死循环）。"""
    from src.tools.memory_tools import _deny_counts_toward_abort

    assert _deny_counts_toward_abort("memory_save") is False
