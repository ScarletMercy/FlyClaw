"""Daily memory & skill consolidation.

Called by ConsolidationScheduler at 03:00 every day.
For each active session:
  1. Send "dreaming" notification
  2. Read all messages from the session
  3. Extract memories + create/update skills via background review
  4. Save a diary-style episodic summary of the session
  5. Create a new session via SessionRegistry (old session preserved for /old)
  6. Send "wake" notification with summary
"""

from __future__ import annotations

import logging
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

    for thread_id in thread_ids:
        try:
            state = await state_store.load(thread_id)
        except Exception:
            continue

        if state is None:
            continue

        chat_id = state.chat_id
        channel_name = state.channel

        # Decide whether to run expensive consolidation
        should_consolidate = True
        if len(state.messages) < min_messages:
            should_consolidate = False
        elif state.created_at < cutoff_ts:
            should_consolidate = False
        elif len([m for m in state.messages if m.get("role") == "user"]) < 3:
            should_consolidate = False

        summary = ""
        if should_consolidate:
            await _send_notification(container, channel_name, chat_id, "\U0001f4a4 dreaming...")
            try:
                summary = await _consolidate_session(
                    agent_loop=agent_loop,
                    config=config,
                    messages=state.messages,
                )
            except Exception as e:
                logger.error("Consolidation failed for session %s: %s", thread_id, e, exc_info=True)
                result["errors"].append(f"{thread_id}: {e}")
                await _send_notification(
                    container, channel_name, chat_id, "\u2600\ufe0f consolidation failed, session preserved"
                )
                summary = ""
            else:
                result["sessions_processed"] += 1
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

                wake_msg = (
                    f"\u2600\ufe0f wake up! {summary}" if summary else "\u2600\ufe0f wake up! nothing to consolidate"
                )
                await _send_notification(container, channel_name, chat_id, wake_msg)
        else:
            result["sessions_skipped"] += 1

        # Always rotate session \u2014 prevents sessions from getting stuck forever
        await _create_new_session(container, thread_id, channel_name)

        if agent_loop:
            agent_loop.invalidate_memory_cache()

    logger.info(
        "Consolidation complete: %d processed, %d skipped, %d errors",
        result["sessions_processed"],
        result["sessions_skipped"],
        len(result["errors"]),
    )
    return result


async def _create_new_session(container: Any, thread_id: str, channel_name: str) -> str | None:
    registry = getattr(container, "session_registry", None)
    if registry is None:
        logger.warning("Consolidation: no session registry, skipping new session creation")
        return None

    parts = thread_id.split(":")
    if len(parts) >= 3:
        user_hash = parts[-1]
    else:
        user_hash = thread_id

    channel_prefix = channel_name or (parts[0] if parts else "unknown")
    legacy_key = f"{channel_prefix}:user:{user_hash}"

    try:
        sid = await registry.new_session(legacy_key, channel_prefix, user_hash)
        logger.info("Consolidation: created new session %s for %s (old: %s)", sid, legacy_key, thread_id)
        return sid
    except Exception as e:
        logger.error("Consolidation: failed to create new session for %s: %s", legacy_key, e)
        return None


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

    # Diary-style key: "x天的日记1", "x天的日记2", ...
    days_ago = max(1, int((time.time() - created_at) / 86400))
    day_counter[days_ago] += 1
    key = f"{days_ago}天的日记{day_counter[days_ago]}"

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
