"""Sync messages from LangGraph checkpoints to the session index."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from .store import SessionIndexStore, parse_thread_id

logger = logging.getLogger("myclaw.session_index.sync")


def _extract_role(msg) -> str:
    if isinstance(msg, HumanMessage):
        return "human"
    if isinstance(msg, AIMessage):
        return "ai"
    if isinstance(msg, ToolMessage):
        return "tool"
    if isinstance(msg, SystemMessage):
        return "system"
    return "unknown"


def _extract_content(msg) -> str:
    content = getattr(msg, "content", "")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content) if content else ""


def _extract_tool_calls(msg) -> Optional[str]:
    calls = getattr(msg, "tool_calls", None)
    if not calls:
        return None
    try:
        return json.dumps(
            [{"name": c.get("name", ""), "args": str(c.get("args", ""))[:200]} for c in calls],
            ensure_ascii=False,
        )
    except Exception:
        return None


def sync_messages(
    store: SessionIndexStore,
    thread_id: str,
    messages: list,
    channel: str,
    sender_id: str,
    chat_id: str,
    chat_type: str,
    tool_max_chars: int = 500,
) -> None:
    """Sync messages to the index store. Idempotent via message_id UNIQUE."""
    if not messages:
        return

    store.upsert_session(thread_id, channel, sender_id, chat_id, chat_type)

    records = []
    for msg in messages:
        msg_id = getattr(msg, "id", None)
        if not msg_id:
            continue

        role = _extract_role(msg)
        content = _extract_content(msg)

        if role == "tool":
            content = content[:tool_max_chars] if content else None

        records.append({
            "message_id": msg_id,
            "role": role,
            "content": content,
            "tool_name": getattr(msg, "name", None) if role == "tool" else None,
            "tool_calls": _extract_tool_calls(msg) if role == "ai" else None,
            "timestamp": time.time(),
        })

    if records:
        store.add_messages(thread_id, records)


def _get_all_thread_ids(checkpoints_path: str) -> list[str]:
    """Get all distinct thread_ids from checkpoints.db."""
    if not checkpoints_path or not Path(checkpoints_path).exists():
        return []
    conn = None
    try:
        conn = sqlite3.connect(checkpoints_path)
        rows = conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints"
        ).fetchall()
        return [r[0] for r in rows]
    except Exception as e:
        logger.warning("Failed to read checkpoints.db: %s", e)
        return []
    finally:
        if conn:
            conn.close()


def _infer_channel_from_thread_id(thread_id: str) -> str:
    """Best-effort channel extraction from thread_id."""
    if ":" in thread_id:
        return thread_id.split(":")[0]
    return "unknown"


async def startup_sync(
    store: SessionIndexStore,
    compiled_graph,
    checkpoints_path: str,
    tool_max_chars: int = 500,
) -> int:
    """Sync all threads from checkpoints.db into the index store.

    Uses LangGraph's aget_state() for proper deserialization.
    Skips threads that already have messages in the index.
    Returns count of threads synced.
    """
    thread_ids = _get_all_thread_ids(checkpoints_path)
    if not thread_ids:
        logger.info("No threads found in checkpoints.db")
        return 0

    # Find threads that need syncing (no messages in index yet)
    already_indexed = set()
    try:
        rows = store._db.execute(
            "SELECT DISTINCT thread_id FROM messages"
        ).fetchall()
        already_indexed = {r[0] for r in rows}
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
            config = {"configurable": {"thread_id": tid}}
            state = await compiled_graph.aget_state(config)
            messages = state.values.get("messages", []) if state and state.values else []
            if not messages:
                continue

            meta = parse_thread_id(tid)
            channel = meta["channel"] if meta["channel"] != "unknown" else _infer_channel_from_thread_id(tid)

            # Try to extract channel from state values
            state_channel = state.values.get("channel", "")
            if state_channel:
                channel = state_channel

            sync_messages(
                store,
                thread_id=tid,
                messages=messages,
                channel=channel,
                sender_id=state.values.get("sender_id", meta.get("sender_id", "")),
                chat_id=state.values.get("chat_id", ""),
                chat_type=state.values.get("chat_type", meta.get("chat_type", "p2p")),
                tool_max_chars=tool_max_chars,
            )
            synced += 1
        except Exception as e:
            logger.debug("Skipping thread %s: %s", tid, e)

    logger.info("Startup sync complete: %d/%d threads indexed", synced, len(to_sync))
    return synced
