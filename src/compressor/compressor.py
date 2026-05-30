"""Context compressor — LLM-based conversation summarization.

When a conversation grows too long, this module:
1. Cleans internal markers from middle messages
2. Protects recent messages (tail) and first exchange (head)
3. Summarizes middle turns with a small LLM (or static fallback)
4. Returns: [SummaryMessage] + [protected tail messages]
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.agent.client import ChatClient, FallbackChain

logger = logging.getLogger("flyclaw.compressor")

_CHARS_PER_TOKEN = 4

_SUMMARY_SYSTEM = """你是一个会话摘要助手。你需要将一段对话历史压缩成简洁但完整的摘要。

规则：
1. 保留所有关键事实、决策、文件路径、命令结果
2. 保留未完成的任务和待办事项
3. 记录用户偏好和重要上下文
4. 用清晰的中文列出要点
5. 如果已有之前的摘要，将其与新对话合并更新
6. 不要遗漏重要信息，但可以省略细节过程

输出格式：
## 已完成
- ...
## 进行中的任务
- ...
## 重要上下文
- ...
## 关键文件/路径
- ..."""

_CACHE_PATH_RE = re.compile(r'\. Full content saved to: `[^`]+`')


def _clean_for_summary(messages: list[dict]) -> list[dict]:
    cleaned = []
    for m in messages:
        if "_truncated" not in m:
            content = m.get("content", "")
            if not (isinstance(content, str) and _CACHE_PATH_RE.search(content)):
                cleaned.append(m)
                continue
        new_m = {k: v for k, v in m.items() if k != "_truncated"}
        content = new_m.get("content", "")
        if isinstance(content, str):
            new_m["content"] = _CACHE_PATH_RE.sub('.', content)
        cleaned.append(new_m)
    return cleaned


def _find_safe_cut(non_system: list[dict], desired_tail_count: int) -> int:
    """Find a cut point that doesn't split assistant+tool_results groups.

    A group is: an assistant message with tool_calls + all its tool_results.
    If the desired cut falls inside a group, the cut is moved to the group's
    start so the group stays entirely in the tail.

    After group adjustment, the cut is moved forward to the nearest user
    message so that the tail always starts with a user turn.
    """
    n = len(non_system)
    cut = n - desired_tail_count
    if cut <= 0:
        return 0

    tc_id_to_asst_idx: dict[str, int] = {}
    for i, m in enumerate(non_system):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                tid = tc.get("id", "")
                if tid:
                    tc_id_to_asst_idx[tid] = i

    asst_to_max_result: dict[int, int] = {}
    for i, m in enumerate(non_system):
        if m.get("role") == "tool" and m.get("tool_call_id"):
            asst_idx = tc_id_to_asst_idx.get(m["tool_call_id"])
            if asst_idx is not None:
                prev = asst_to_max_result.get(asst_idx, asst_idx)
                asst_to_max_result[asst_idx] = max(prev, i)

    # Iterate until cut stabilizes — adjusting for one group may expose
    # another group that now spans across the new cut boundary.
    # Guaranteed to converge: cut strictly decreases and is bounded by 0.
    for _ in range(len(asst_to_max_result) + 1):
        prev_cut = cut
        for asst_idx in sorted(asst_to_max_result):
            max_result = asst_to_max_result[asst_idx]
            if asst_idx < cut <= max_result:
                cut = asst_idx
                break
        if cut == prev_cut:
            break

    # Ensure cut lands on a user message.
    # This guarantees the tail starts with a user turn.
    if cut < len(non_system) and non_system[cut].get("role") != "user":
        found = False
        for i in range(cut, len(non_system)):
            if non_system[i].get("role") == "user":
                cut = i
                found = True
                break
        if not found:
            return 0

    return max(0, cut)


def _estimate_tokens(messages: list[dict]) -> int:
    total = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            total += len(content) // _CHARS_PER_TOKEN
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    total += len(part["text"]) // _CHARS_PER_TOKEN
                elif isinstance(part, str):
                    total += len(part) // _CHARS_PER_TOKEN
        total += 10
    return total


class ContextCompressor:
    """Compresses long conversations using the current model for summarization."""

    def __init__(self, config, client: ChatClient | FallbackChain | None = None):
        self.config = config
        self._client = client
        self._previous_summary: Optional[str] = None
        self._compression_count = 0

    def should_compress(self, messages: list[dict], context_window_tokens: int = 100000) -> bool:
        estimated = _estimate_tokens(messages)
        threshold = int(context_window_tokens * self.config.threshold_percent)
        return estimated > threshold

    async def compress(
        self,
        messages: list[dict],
        context_window_tokens: int = 100000,
    ) -> list[dict]:
        if not self.config.enabled:
            return self._compact(messages, context_window_tokens)

        estimated = _estimate_tokens(messages)
        threshold = int(context_window_tokens * self.config.threshold_percent)

        if estimated <= threshold:
            return messages

        logger.info(
            "Context compression triggered: %d estimated tokens > %d threshold",
            estimated,
            threshold,
        )

        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        if not non_system:
            return messages

        tail_count = max(self.config.tail_messages, len(non_system) // 4)
        cut = _find_safe_cut(non_system, tail_count)
        tail = non_system[cut:]
        middle = non_system[:cut]

        if not middle:
            return messages

        middle = _clean_for_summary(middle)

        turns_text = self._format_turns(middle)
        if not turns_text.strip():
            return messages

        summary = await self._llm_summarize(turns_text)

        if summary:
            self._previous_summary = summary
            self._compression_count += 1
            if tail and tail[0].get("role") == "assistant":
                existing = tail[0].get("content", "")
                merged = summary if not existing else f"{summary}\n\n{existing}"
                tail[0] = {**tail[0], "content": merged}
                result = system_msgs + tail
            else:
                summary_msg = {"role": "assistant", "content": summary}
                result = system_msgs + [summary_msg] + tail
            new_estimated = _estimate_tokens(result)
            logger.info(
                "Context compressed: %d → %d messages (%d → %d tokens, compression #%d)",
                len(messages),
                len(result),
                estimated,
                new_estimated,
                self._compression_count,
            )
            return result

        return self._compact(messages, context_window_tokens)

    def _format_turns(self, messages: list[dict]) -> str:
        lines = []
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if role == "user":
                text = content if isinstance(content, str) else str(content)[:300]
                lines.append(f"用户: {text[:300]}")
            elif role == "assistant":
                tool_calls = m.get("tool_calls")
                if tool_calls:
                    calls = ", ".join(
                        tc.get("function", {}).get("name", "?")
                        if isinstance(tc.get("function"), dict)
                        else tc.get("name", "?")
                        for tc in tool_calls
                    )
                    text = content if isinstance(content, str) else ""
                    entry = f"AI调用工具: [{calls}]"
                    if text:
                        entry += f" {text[:100]}"
                    lines.append(entry)
                elif content:
                    text = content if isinstance(content, str) else str(content)[:300]
                    lines.append(f"AI: {text[:300]}")
            elif role == "tool":
                c = content if isinstance(content, str) else str(content)[:150]
                lines.append(f"工具结果: {c[:150]}")
            elif role == "system":
                pass
        return "\n".join(lines)

    async def _llm_summarize(self, turns_text: str) -> Optional[str]:
        if self._client is None:
            logger.warning("Compression: no client available, using static fallback")
            return None

        if len(turns_text) > 12000:
            turns_text = turns_text[:12000] + "\n... (更多历史已截断)"

        user_msg = ""
        if self._previous_summary:
            user_msg = f"## 之前的摘要\n{self._previous_summary}\n\n## 新的对话\n{turns_text}"
        else:
            user_msg = f"请摘要以下对话：\n\n{turns_text}"

        try:
            resp = await self._client.chat(
                [
                    {"role": "system", "content": _SUMMARY_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                tools=None,
                max_tokens=self.config.max_summary_tokens,
            )
            return resp.content or None
        except Exception as e:
            logger.warning("LLM summarization failed: %s", e)
            return None

    def _compact(
        self,
        messages: list[dict],
        context_window_tokens: int = 100000,
    ) -> list[dict]:
        max_tokens = int(context_window_tokens * self.config.threshold_percent)
        estimated = _estimate_tokens(messages)
        if estimated <= max_tokens:
            return messages

        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        if not non_system:
            return messages

        keep_recent = max(6, len(non_system) // 3)
        cut = _find_safe_cut(non_system, keep_recent)
        kept = non_system[cut:]
        pruned = non_system[:cut]

        if not pruned:
            return messages

        pruned = _clean_for_summary(pruned)
        summaries = []
        for m in pruned:
            role = m.get("role", "")
            content = m.get("content", "")
            if role == "tool":
                tool_name = m.get("name", "unknown")
                c = content if isinstance(content, str) else str(content)
                summaries.append(f"[Tool({tool_name})]: {c[:150]}")
            elif role == "assistant" and m.get("tool_calls"):
                calls = ", ".join(
                    tc.get("function", {}).get("name", "?")
                    if isinstance(tc.get("function"), dict)
                    else tc.get("name", "?")
                    for tc in m["tool_calls"]
                )
                text = content if isinstance(content, str) else ""
                entry = f"Assistant: called [{calls}]"
                if text:
                    entry += f" {text[:100]}"
                summaries.append(entry)
            elif role == "user":
                text = content if isinstance(content, str) else str(content)[:200]
                summaries.append(f"User: {text[:200]}")
            elif role == "assistant" and content:
                text = content if isinstance(content, str) else str(content)[:200]
                summaries.append(f"Assistant: {text[:200]}")

        summary_text = "\n".join(summaries[-20:])

        if kept and kept[0].get("role") == "assistant":
            existing = kept[0].get("content", "")
            merged = summary_text if not existing else f"{summary_text}\n\n{existing}"
            kept[0] = {**kept[0], "content": merged}
            result = system_msgs + kept
        else:
            summary_msg = {"role": "assistant", "content": summary_text}
            result = system_msgs + [summary_msg] + kept
        logger.info(
            "Static compact: %d → %d messages (%d → %d tokens)",
            len(messages),
            len(result),
            estimated,
            _estimate_tokens(result),
        )
        return result
