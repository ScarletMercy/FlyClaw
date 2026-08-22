"""Tests for consolidation_state: load/save + wrapper advancement semantics."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.consolidation_state import (
    ConsolidationState,
    load_state,
    save_state,
    run_session_organize,
    run_memory_organize,
)


def _patch_data_dir(tmp_path):
    """把 data_dir() 指向 tmp_path，隔离状态文件。"""
    return patch("src.instance.data_dir", return_value=tmp_path)


@pytest.mark.asyncio
async def test_load_state_missing_file_returns_none(tmp_path):
    with _patch_data_dir(tmp_path):
        st = await load_state()
    assert st.last_session_organize_at is None
    assert st.last_memory_organize_at is None


@pytest.mark.asyncio
async def test_save_load_roundtrip(tmp_path):
    with _patch_data_dir(tmp_path):
        await save_state(ConsolidationState(last_session_organize_at=1000.0, last_memory_organize_at=2000.0))
        st = await load_state()
    assert st.last_session_organize_at == 1000.0
    assert st.last_memory_organize_at == 2000.0


@pytest.mark.asyncio
async def test_load_state_corrupt_file_returns_none(tmp_path):
    (tmp_path / "consolidation_state.json").write_text("{ broken json", encoding="utf-8")
    with _patch_data_dir(tmp_path):
        st = await load_state()
    assert st.last_session_organize_at is None


@pytest.mark.asyncio
async def test_save_state_atomic_replace(tmp_path):
    """save 用 tmp+os.replace，成功后文件存在且内容正确。"""
    with _patch_data_dir(tmp_path):
        await save_state(ConsolidationState(last_session_organize_at=5.0))
        # 无残留 tmp 文件
        tmps = list(tmp_path.glob("*.tmp"))
        assert tmps == []
        assert (tmp_path / "consolidation_state.json").exists()
    with _patch_data_dir(tmp_path):
        st = await load_state()
    assert st.last_session_organize_at == 5.0


# ─── wrapper: 推进语义 ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_session_organize_advances_on_no_errors(tmp_path):
    """errors 为空 → 推进 last_session_organize_at；since 来自旧 last。"""
    with _patch_data_dir(tmp_path):
        await save_state(ConsolidationState(last_session_organize_at=100.0))
        daily_mock = AsyncMock(return_value={"sessions_processed": 3, "sessions_skipped": 0, "errors": []})
        container = MagicMock()
        with patch("src.services.daily_consolidation.run_daily_consolidation", daily_mock):
            result = await run_session_organize(container)
        # since 取自旧 last
        assert daily_mock.call_args.kwargs["since_ts"] == 100.0
        # 推进了
        assert result["last_session_organize_at"] > 100.0
        st = await load_state()
        assert st.last_session_organize_at == result["last_session_organize_at"]


@pytest.mark.asyncio
async def test_run_session_organize_no_advance_on_errors(tmp_path):
    """errors 非空 → 不推进 last。"""
    with _patch_data_dir(tmp_path):
        await save_state(ConsolidationState(last_session_organize_at=100.0))
        daily_mock = AsyncMock(return_value={"sessions_processed": 0, "errors": ["t1: boom"]})
        container = MagicMock()
        with patch("src.services.daily_consolidation.run_daily_consolidation", daily_mock):
            result = await run_session_organize(container)
        assert result["last_session_organize_at"] == 100.0  # 未推进
        st = await load_state()
        assert st.last_session_organize_at == 100.0


@pytest.mark.asyncio
async def test_run_session_organize_default_since_when_unset(tmp_path):
    """字段缺失时 since 兜底 now-24h（保持原凌晨行为）。"""
    with _patch_data_dir(tmp_path):
        daily_mock = AsyncMock(return_value={"errors": []})
        container = MagicMock()
        with patch("src.services.daily_consolidation.run_daily_consolidation", daily_mock):
            await run_session_organize(container)
        since = daily_mock.call_args.kwargs["since_ts"]
        # 约 24h 前
        import time as _t

        assert abs(_t.time() - 24 * 3600 - since) < 5  # 5s 容差


@pytest.mark.asyncio
async def test_run_memory_organize_advances_on_no_errors(tmp_path):
    with _patch_data_dir(tmp_path):
        await save_state(ConsolidationState(last_memory_organize_at=100.0))
        mem_mock = AsyncMock(return_value={"total_memories": 5, "errors": []})
        container = MagicMock()
        with patch("src.services.memory_consolidation.run_memory_consolidation", mem_mock):
            result = await run_memory_organize(container)
        assert result["last_memory_organize_at"] > 100.0
        st = await load_state()
        assert st.last_memory_organize_at == result["last_memory_organize_at"]


@pytest.mark.asyncio
async def test_run_memory_organize_no_advance_on_errors(tmp_path):
    with _patch_data_dir(tmp_path):
        await save_state(ConsolidationState(last_memory_organize_at=100.0))
        mem_mock = AsyncMock(return_value={"errors": ["cat: boom"]})
        container = MagicMock()
        with patch("src.services.memory_consolidation.run_memory_consolidation", mem_mock):
            result = await run_memory_organize(container)
        assert result["last_memory_organize_at"] == 100.0
        st = await load_state()
        assert st.last_memory_organize_at == 100.0


@pytest.mark.asyncio
async def test_run_session_organize_explicit_since_ts(tmp_path):
    """显式 since_ts=0（全量补漏）→ 传给 run_daily_consolidation。"""
    with _patch_data_dir(tmp_path):
        await save_state(ConsolidationState(last_session_organize_at=100.0))
        daily_mock = AsyncMock(return_value={"errors": []})
        container = MagicMock()
        with patch("src.services.daily_consolidation.run_daily_consolidation", daily_mock):
            await run_session_organize(container, since_ts=0.0)
        assert daily_mock.call_args.kwargs["since_ts"] == 0.0


@pytest.mark.asyncio
async def test_concurrent_advance_no_lost_update(tmp_path):
    """session 与 memory 整理并发推进各自字段，_state_lock 保证互不覆盖（无 lost update）。"""
    import asyncio

    from src.services.consolidation_state import _advance_if_clean

    with _patch_data_dir(tmp_path):
        await save_state(ConsolidationState(last_session_organize_at=100.0, last_memory_organize_at=200.0))
        # 两个不同字段并发推进（模拟 session 整理与 memory 整理同时完成落盘）
        await asyncio.gather(
            _advance_if_clean("last_session_organize_at", 1000.0, {"errors": []}),
            _advance_if_clean("last_memory_organize_at", 2000.0, {"errors": []}),
        )
        st = await load_state()
    assert st.last_session_organize_at == 1000.0  # 未被 memory 的写入覆盖
    assert st.last_memory_organize_at == 2000.0  # 未被 session 的写入覆盖
