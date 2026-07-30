"""Tests for auto_extract_memory pre-save gate (_gate_auto_extract).

入口2（auto_extract 正则提取）的保存前置把关，复用入口1 的 model/manual 审批语义，
但用三出口（allow/reject/manual）适配 fire-and-forget 异步审批。

- allow:  放行，调用方 save_memory
- reject: model 自检判臆测 → 静默丢弃
- manual: 需人工审批（model 模式可疑 / manual 模式全部）
- fail-open：无主模型/调用失败 → allow
- session 短路：本会话已授权该内容 → allow（不再问）
"""

from __future__ import annotations

import pytest

from src.tools import memory_tools
from src.tools.memory_tools import _gate_auto_extract, set_memory_session


def _patch_container_with_client(monkeypatch, client, save_approval_mode="model", session_approved=False):
    """注入带主模型 client 的 container（仿 _gate_auto_extract 获取路径）。

    session_approved 控制 has_session_approval 返回值（auto_extract 短路）。
    """
    import src._container as _c
    from unittest.mock import MagicMock
    from src.config import MemoryStoreConfig

    _mgr = MagicMock()
    _mgr.has_session_approval.return_value = session_approved

    _ms_cfg = MemoryStoreConfig(save_approval_mode=save_approval_mode)

    class _AgentLoop:
        _client = client

    class _Container:
        agent_loop = _AgentLoop()
        approval_manager = _mgr
        config = MagicMock()

    _Container.config.memory_store = _ms_cfg

    monkeypatch.setattr(_c, "get_container", lambda: _Container())
    monkeypatch.setattr("src.tools.approval.get_approval_manager", lambda: _mgr)
    # auto_extract 关卡用 _current_thread_id 取会话授权（默认 ""）；ContextVar.get 不可
    # monkeypatch，但默认值已是 ""，无需额外设置。


# ── model 模式 ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_model_mode_allows_when_supported(monkeypatch):
    """model 模式 + 自检判定有据 → allow。"""
    set_memory_session("p2p", "")

    class Client:
        async def chat_simple(self, messages, **extra):
            return "allow 用户原话明确支持"

    _patch_container_with_client(monkeypatch, Client())
    decision, _reason = await _gate_auto_extract("用户用 Python 3.13", "我用 Python 3.13 写代码")
    assert decision == "allow"


@pytest.mark.asyncio
async def test_model_mode_rejects_when_unsupported(monkeypatch):
    """model 模式 + 自检判定臆测 → reject（静默丢弃，调用方不存）。"""
    set_memory_session("p2p", "")

    class Client:
        async def chat_simple(self, messages, **extra):
            return "reject 原文未提及后端工程师，属臆测"

    _patch_container_with_client(monkeypatch, Client())
    decision, reason = await _gate_auto_extract("用户是后端工程师", "我用 Python 写脚本")
    assert decision == "reject"
    assert reason


@pytest.mark.asyncio
async def test_model_mode_no_source_allows_without_calling_client(monkeypatch):
    """model 模式 + 无来源 → 直接 allow（fail-open，省一次主模型调用）。"""
    set_memory_session("p2p", "")

    class Client:
        called = False

        async def chat_simple(self, messages, **extra):
            Client.called = True
            return "reject 不该走到这"

    _patch_container_with_client(monkeypatch, Client())
    decision, _reason = await _gate_auto_extract("任意内容", "")
    assert decision == "allow"
    assert Client.called is False


# ── manual 模式 ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_manual_mode_returns_manual_without_validation(monkeypatch):
    """manual 模式：跳过自检，全部转 manual（fire-and-forget 审批），即使自检会 allow。"""
    set_memory_session("p2p", "")

    class Client:
        called = False

        async def chat_simple(self, messages, **extra):
            Client.called = True
            return "allow"

    _patch_container_with_client(monkeypatch, Client(), save_approval_mode="manual")
    decision, _reason = await _gate_auto_extract("任何内容", "来源")
    assert decision == "manual"
    assert Client.called is False  # manual 不自检


# ── fail-open ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fail_open_when_no_container(monkeypatch):
    """get_container 未初始化 → fail-open allow（不阻断 auto_extract）。"""
    import src._container as _c

    monkeypatch.setattr(_c, "get_container", lambda: (_ for _ in ()).throw(RuntimeError("not init")))
    set_memory_session("p2p", "")
    decision, _reason = await _gate_auto_extract("任意内容", "任意来源")
    assert decision == "allow"


@pytest.mark.asyncio
async def test_fail_open_when_client_call_fails(monkeypatch):
    """主模型调用抛异常 → fail-open allow。"""
    set_memory_session("p2p", "")

    class FlakyClient:
        async def chat_simple(self, messages, **extra):
            raise RuntimeError("主模型调用失败")

    _patch_container_with_client(monkeypatch, FlakyClient())
    decision, _reason = await _gate_auto_extract("任意内容", "任意来源")
    assert decision == "allow"


# ── 会话级授权短路 ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_session_approved_short_circuits_to_allow(monkeypatch):
    """本会话已授权该内容 → 直接 allow，不调主模型（不再问）。"""
    set_memory_session("p2p", "")

    class Client:
        called = False

        async def chat_simple(self, messages, **extra):
            Client.called = True
            return "reject 臆测"

    _patch_container_with_client(monkeypatch, Client(), session_approved=True)
    decision, _reason = await _gate_auto_extract("已批准过的内容", "无关来源")
    assert decision == "allow"
    assert Client.called is False  # 短路，不自检
