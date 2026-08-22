"""整理进度状态：last_session_organize_at / last_memory_organize_at。

全局一份，存 data_dir/consolidation_state.json，原子写（tmp + os.replace）。
定时任务与手动命令共用此状态 + organize_lock 防并发。

推进语义：run 正常返回且 result["errors"] 为空才推进 last=now；任一失败不推进，
下轮重扫同一区间——配合每项的 organized 标志，已成功的会被跳过，零重复处理。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger("flyclaw.consolidation_state")

# _session_lock / _memory_lock：防止同类整理并发（避免重复整理同一批数据），两类互不阻塞。
# _state_lock：保护 consolidation_state.json 的读-改-写，防止 session 与 memory 整理并发时
#              互相覆盖对方字段（lost update）。仅在 _advance_if_clean 内短暂持有。
_session_lock = asyncio.Lock()
_memory_lock = asyncio.Lock()
_state_lock = asyncio.Lock()


@dataclass
class ConsolidationState:
    last_session_organize_at: float | None = None
    last_memory_organize_at: float | None = None


def _state_path() -> str:
    from src.instance import data_dir

    return str(data_dir() / "consolidation_state.json")


async def load_state() -> ConsolidationState:
    """读取状态；文件不存在或损坏 → 全 None（首跑）。"""
    path = _state_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return ConsolidationState()
    return ConsolidationState(
        last_session_organize_at=data.get("last_session_organize_at"),
        last_memory_organize_at=data.get("last_memory_organize_at"),
    )


async def save_state(state: ConsolidationState) -> None:
    """原子写：tmp 文件 + os.replace（仿 session/tracker.py 的持久化模式）。"""
    path = _state_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    text = json.dumps(asdict(state), ensure_ascii=False, indent=2)
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    try:
        os.close(fd)
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(text)
        await asyncio.to_thread(os.replace, tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


async def _advance_if_clean(field: str, now: float, result: dict[str, Any]) -> None:
    """无错则推进 last_* 并落盘；有错则不推进。结果注入 result[field]。

    在 _state_lock 内**重读最新状态**再只改本字段：session 与 memory 整理可能并发，
    各自只推进自己的字段；若用调用方早先加载的 state 直接整盘覆盖，会丢掉对方刚写入的字段。
    """
    async with _state_lock:
        state = await load_state()
        if not result.get("errors"):
            setattr(state, field, now)
            try:
                await save_state(state)
            except Exception as e:
                logger.warning("Failed to persist %s: %s", field, e)
        else:
            logger.info("Organize had %d errors, not advancing %s", len(result["errors"]), field)
        result[field] = getattr(state, field)


async def run_session_organize(container: Any, since_ts: float | None = None) -> dict[str, Any]:
    """会话整理统一入口（定时 + 命令共用）。

    区间 = [since_ts, now]；since_ts 为 None 时取 last_session_organize_at，字段缺失兜底 now-24h。
    since_ts=0 表示全量补漏（扫描所有未 organized 的会话）。
    run 正常返回且 errors 为空 → 推进 last=now；否则不推进。
    """
    async with _session_lock:
        from src.services.daily_consolidation import run_daily_consolidation

        state = await load_state()
        now = time.time()
        since = since_ts if since_ts is not None else (state.last_session_organize_at or now - 24 * 3600)

        result = await run_daily_consolidation(container, since_ts=since)
        await _advance_if_clean("last_session_organize_at", now, result)
        result["since_ts"] = since
        return result


async def run_memory_organize(container: Any) -> dict[str, Any]:
    """记忆整理统一入口（定时 + 命令共用）。

    记忆始终全量扫 KV（未归档的），organized 标志控制跳过。
    run 正常返回且 errors 为空 → 推进 last=now（进度记录）；否则不推进。
    """
    async with _memory_lock:
        from src.services.memory_consolidation import run_memory_consolidation

        now = time.time()

        result = await run_memory_consolidation(container)
        await _advance_if_clean("last_memory_organize_at", now, result)
        return result
