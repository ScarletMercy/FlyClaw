"""Sync messages from state store to the session index."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Optional

from .store import SessionIndexStore, parse_thread_id
from src.utils.content import content_to_text

logger = logging.getLogger("flyclaw.session_index.sync")


def _extract_role(msg: dict) -> str:
    role = msg.get("role", "")
    if role == "user":
        return "human"
    if role == "assistant":
        return "ai"
    if role == "tool":
        return "tool"
    if role == "system":
        return "system"
    return "unknown"


def _extract_content(msg: dict) -> str:
    content = msg.get("content", "")
    if isinstance(content, (str, list)):
        # allow_bare_str: 历史消息里 content list 可能混入裸字符串
        return content_to_text(content, allow_bare_str=True)
    # 非 str/list 兜底(理论不出现):保留历史 str() 行为,不走 util 的空串兜底
    return str(content) if content else ""


def _extract_tool_calls(msg: dict) -> Optional[str]:
    calls = msg.get("tool_calls")
    if not calls:
        return None
    try:
        return json.dumps(
            [
                {
                    "name": tc.get("function", {}).get("name", ""),
                    "args": str(tc.get("function", {}).get("arguments", ""))[:200],
                }
                for tc in calls
            ],
            ensure_ascii=False,
        )
    except Exception:
        return None


async def sync_messages(
    store: SessionIndexStore,
    thread_id: str,
    messages: list[dict],
    channel: str,
    sender_id: str,
    chat_id: str,
    chat_type: str,
    tool_max_chars: int = 500,
) -> None:
    if not messages:
        return

    await store.upsert_session(thread_id, channel, sender_id, chat_id, chat_type)

    records = []
    for msg in messages:
        msg_id = msg.get("id") or msg.get("message_id") or uuid.uuid4().hex[:12]
        role = _extract_role(msg)
        content = _extract_content(msg)

        if role == "tool":
            content = content[:tool_max_chars] if content else None

        records.append(
            {
                "message_id": msg_id,
                "role": role,
                "content": content,
                "tool_name": msg.get("name") if role == "tool" else None,
                "tool_calls": _extract_tool_calls(msg) if role == "ai" else None,
                "timestamp": time.time(),
            }
        )

    if records:
        await store.add_messages(thread_id, records)


def _infer_channel_from_thread_id(thread_id: str) -> str:
    if ":" in thread_id:
        return thread_id.split(":")[0]
    return "unknown"


async def startup_sync(
    store: SessionIndexStore,
    state_store,
    tool_max_chars: int = 500,
) -> int:
    try:
        thread_ids = await state_store.list_threads()
    except Exception as e:
        logger.warning("Failed to list threads: %s", e)
        return 0

    if not thread_ids:
        logger.info("No threads found in state store")
        return 0

    already_indexed = set()
    try:
        already_indexed = await store.get_indexed_thread_ids()
    except Exception:
        pass

    to_sync = [t for t in thread_ids if t not in already_indexed]
    if not to_sync:
        logger.info("All %d threads already indexed", len(thread_ids))
        return 0

    logger.info("Startup sync: %d/%d threads need indexing", len(to_sync), len(thread_ids))
    synced = 0
    for tid in to_sync:
        try:
            state = await state_store.load(tid)
            if state is None or not state.messages:
                continue

            meta = parse_thread_id(tid)
            channel = meta["channel"] if meta["channel"] != "unknown" else "unknown"
            if state.channel:
                channel = state.channel

            await sync_messages(
                store,
                thread_id=tid,
                messages=state.messages,
                channel=channel,
                sender_id=state.sender_id or meta.get("sender_id", ""),
                chat_id=state.chat_id,
                chat_type=state.chat_type or meta.get("chat_type", "p2p"),
                tool_max_chars=tool_max_chars,
            )
            synced += 1
        except Exception as e:
            logger.debug("Skipping thread %s: %s", tid, e)

    logger.info("Startup sync complete: %d/%d threads indexed", synced, len(to_sync))
    return synced
