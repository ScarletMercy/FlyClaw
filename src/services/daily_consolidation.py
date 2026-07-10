"""Daily memory & skill consolidation.

Called by ConsolidationScheduler at 03:00 every day. Three passes so
notifications read "dreaming, dreaming, wake" instead of interleaved
"dreaming, wake, dreaming, wake" per session:

  Pass 1 — for each session passing all three gates (created < 24h ago,
           >= min_messages total messages, >= 3 user messages): send a
           "dreaming" notification and collect it as pending.
  Pass 2 — consolidate each pending session silently: extract memories +
           create/update skills via background review, and save a diary-style
           episodic summary. set_memory_session is re-applied per entry
           because the ContextVar would otherwise hold pass 1's last scope.
  Pass 3 — send one "wake" per chat (once per chat, not per session), merging
           only the summaries of sessions that succeeded with non-empty text.
           A chat whose sessions all failed gets "consolidation failed"; one
           that succeeded but produced no text gets "nothing to consolidate";
           any failed sessions add a failure count to the message.

Finally, open a new session via SessionRegistry for each consolidated
legacy thread (old session preserved for /old) and invalidate the memory
cache.
"""

from __future__ import annotations

import datetime
import logging
import re
import time
from collections import defaultdict
from typing import Any

logger = logging.getLogger("flyclaw.consolidation")

_SUMMARY_PROMPT = "用100字以内概括这个会话的主要内容，包括讨论了什么、完成了什么。只输出摘要文本，不要其他内容。"


async def run_daily_consolidation(container: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "sessions_processed": 0,
        "sessions_skipped": 0,
        "memories_saved": 0,
        "skills_updated": 0,
        "errors": [],
    }

    state_store = container.state_store
    if state_store is None:
        logger.warning("Consolidation: no state store available")
        return result

    agent_loop = container.agent_loop
    if agent_loop is None:
        logger.warning("Consolidation: no agent loop available")
        return result

    config = container.config
    min_messages = getattr(config, "consolidation", None)
    min_messages = getattr(min_messages, "min_messages", 10) if min_messages else 10

    cutoff_ts = time.time() - 24 * 3600

    thread_ids = await state_store.list_threads()

    # Day counter for diary-style keys: {days_ago: next_index}
    day_counter: dict[int, int] = defaultdict(int)
    consolidated_keys: set[str] = set()

    # Three passes so notifications read "dreaming, dreaming, wake" instead of
    # interleaved "dreaming, wake, dreaming, wake" per session:
    #   pass 1 — filter candidates and send all dreaming (one per session)
    #   pass 2 — consolidate each (silent, no notifications)
    #   pass 3 — one wake per chat, merging every session's summary in that
    #            chat into a single message (wake is once per chat, not per session)
    # set_memory_session is re-applied per entry in pass 2 because the
    # ContextVar would otherwise hold the last session's scope from pass 1.
    from src.tools.memory_tools import set_memory_session

    pending: list[dict[str, Any]] = []

    try:
        # Pass 1: filter + announce dreaming
        logger.info(
            "Consolidation Pass1: 扫描 %d 个 thread, cutoff=%s, min_messages=%d",
            len(thread_ids),
            datetime.datetime.fromtimestamp(cutoff_ts).isoformat(),
            min_messages,
        )
        for thread_id in thread_ids:
            try:
                state = await state_store.load(thread_id)
            except Exception:
                continue

            if state is None:
                continue

            chat_id = state.chat_id
            channel_name = state.channel

            ct = getattr(state, "chat_type", "p2p")
            gid = chat_id if ct != "p2p" else ""
            set_memory_session(ct, gid)

            # Decide whether to run expensive consolidation
            should_consolidate = True
            if len(state.messages) < min_messages:
                should_consolidate = False
            elif state.created_at < cutoff_ts:
                should_consolidate = False
            elif len([m for m in state.messages if m.get("role") == "user"]) < 3:
                should_consolidate = False

            if not should_consolidate:
                result["sessions_skipped"] += 1
                continue

            logger.info("💤 dreaming 通知 -> thread=%s chat=%s channel=%s", thread_id, chat_id, channel_name)
            await _send_notification(container, channel_name, chat_id, "\U0001f4a4 dreaming...")
            pending.append({"thread_id": thread_id, "state": state, "summary": "", "failed": False})

        logger.info("Consolidation Pass1 完成: %d 个会话过 gate", len(pending))

        # Pass 2: consolidate each (no notifications)
        logger.info("Consolidation Pass2: 整理 %d 个 pending 会话", len(pending))
        for entry in pending:
            state = entry["state"]
            thread_id = entry["thread_id"]
            chat_id = state.chat_id
            channel_name = state.channel

            ct = getattr(state, "chat_type", "p2p")
            gid = chat_id if ct != "p2p" else ""
            set_memory_session(ct, gid)

            logger.info("会话整理开始: thread=%s (%d 条消息)", thread_id, len(state.messages))
            try:
                summary = await _consolidate_session(
                    agent_loop=agent_loop,
                    config=config,
                    messages=state.messages,
                )
            except Exception as e:
                logger.error("Consolidation failed for session %s: %s", thread_id, e, exc_info=True)
                result["errors"].append(f"{thread_id}: {e}")
                entry["failed"] = True
                continue

            logger.info("会话整理完成: thread=%s summary=%s", thread_id, (summary or "")[:80])

            result["sessions_processed"] += 1
            consolidated_keys.add(re.sub(r":s\d+$", "", thread_id))
            if summary:
                if "memory" in summary.lower() or "saved" in summary.lower():
                    result["memories_saved"] += 1
                if "skill" in summary.lower() or "patched" in summary.lower() or "created" in summary.lower():
                    result["skills_updated"] += 1

            # Save episodic summary (diary-style)
            try:
                await _save_session_summary(config, state.created_at, state.messages, day_counter)
            except Exception as e:
                logger.warning("Session summary failed for %s: %s", thread_id, e)

            entry["summary"] = summary or ""

        # Pass 3: one wake per chat, merging its sessions' summaries
        wake_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for entry in pending:
            state = entry["state"]
            wake_groups.setdefault((state.channel or "", state.chat_id or ""), []).append(entry)

        for (channel_name, chat_id), entries in wake_groups.items():
            if not chat_id:
                continue
            parts = [e["summary"] for e in entries if e["summary"] and not e["failed"]]
            failed_n = sum(1 for e in entries if e["failed"])
            if parts:
                wake_msg = "\u2600\ufe0f wake up! " + " \u00b7 ".join(parts)
            elif failed_n == len(entries):
                # every session in this chat failed
                wake_msg = "\u2600\ufe0f consolidation failed, sessions preserved"
            else:
                # succeeded but produced no summary text (possibly some failed too)
                wake_msg = "\u2600\ufe0f wake up! nothing to consolidate"
            if failed_n:
                wake_msg += f" ({failed_n} \u4e2a\u4f1a\u8bdd\u6574\u7406\u5931\u8d25)"
            logger.info(
                "\u2600\ufe0f wake \u901a\u77e5 -> chat=%s channel=%s failed=%d/%d msg=%s",
                chat_id,
                channel_name,
                failed_n,
                len(entries),
                wake_msg[:80],
            )
            await _send_notification(container, channel_name, chat_id, wake_msg)
    finally:
        registry = getattr(container, "session_registry", None)
        if registry is not None:
            for legacy in consolidated_keys:
                try:
                    sid = await registry.new_session(legacy)
                    logger.info("Consolidation: opened new session %s for %s", sid, legacy)
                except Exception as e:
                    logger.error("Consolidation: failed to open new session for %s: %s", legacy, e)
        if agent_loop:
            agent_loop.invalidate_memory_cache()

    logger.info(
        "Consolidation complete: %d processed, %d skipped, %d errors",
        result["sessions_processed"],
        result["sessions_skipped"],
        len(result["errors"]),
    )
    return result


async def _consolidate_session(
    agent_loop: Any,
    config: Any,
    messages: list[dict],
) -> str:
    from src.skills.review import spawn_background_review

    summary = await spawn_background_review(
        client=agent_loop._client,
        tools=agent_loop._tools,
        config=config,
        messages_snapshot=list(messages),
        review_skills=True,
        review_memory=True,
    )
    return summary or ""


async def _save_session_summary(
    config: Any,
    created_at: float,
    messages: list[dict],
    day_counter: dict[int, int],
) -> None:
    """Compress a session into a ~100-char episodic memory (diary-style)."""
    from src.agent.client import ChatClient
    from src.tools.memory_tools import save_memory

    if not messages:
        return

    # Build conversation excerpt for the LLM
    user_msgs = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if not isinstance(content, str):
            continue
        if role == "user":
            user_msgs.append(f"用户: {content}")
        elif role == "assistant" and content:
            user_msgs.append(f"助手: {content}")
    if not user_msgs:
        return

    excerpt = "\n".join(user_msgs)
    prompt = _SUMMARY_PROMPT + "\n\n" + excerpt

    client = ChatClient(
        base_url=config.model.base_url,
        api_key=config.model.api_key,
        model=config.model.name,
        temperature=0.0,
    )
    try:
        resp = await client.chat([{"role": "user", "content": prompt}])
        summary_text = resp.content.strip() if resp.content else ""
    except Exception as e:
        logger.warning("Session summary LLM call failed: %s", e)
        return
    finally:
        await client.close()

    if not summary_text:
        return

    # Diary-style key: "x天的日记{n}_{date}".
    # The creation date makes each session's entry unique, so nightly runs
    # accumulate instead of overwriting via ON CONFLICT(key) DO UPDATE —
    # day_counter resets every run, so without the date every ~1-day-old
    # session would reuse "1天的日记1" and clobber the previous night's diary.
    days_ago = max(1, int((time.time() - created_at) / 86400))
    day_counter[days_ago] += 1
    date_str = datetime.datetime.fromtimestamp(created_at).strftime("%Y%m%d")
    key = f"{days_ago}天的日记{day_counter[days_ago]}_{date_str}"

    await save_memory(summary_text, key=key, category="episodic")
    logger.info("Saved session summary: %s → %s", key, summary_text[:50])


async def _send_notification(container: Any, channel_name: str, chat_id: str, text: str) -> None:
    if not chat_id:
        return
    try:
        ch = None
        if channel_name == "qq" and container.qq:
            ch = container.qq
        elif channel_name == "weixin" and container.weixin:
            ch = container.weixin
        if ch is None:
            ch = container.qq or container.weixin
        if ch and hasattr(ch, "send_text"):
            await ch.send_text(chat_id, text)
    except Exception as e:
        logger.debug("Consolidation notification failed: %s", e)
