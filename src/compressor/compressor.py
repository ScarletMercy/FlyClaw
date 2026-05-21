"""Context compressor — LLM-based conversation summarization.

When a conversation grows too long, this module:
1. Prunes old tool outputs (cheap, no LLM)
2. Protects recent messages (tail) and first exchange (head)
3. Summarizes middle turns with a small LLM
4. Returns: [SummaryMessage] + [protected tail messages]
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.agent.client import ChatClient, FallbackChain

logger = logging.getLogger("myclaw.compressor")

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

_PLACEHOLDER = "[旧工具输出已清除以节省上下文空间]"


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


def _tool_result_summary(name: str, content: str) -> str:
    content_len = len(content) if isinstance(content, str) else 0
    line_count = content.count("\n") + 1 if isinstance(content, str) else 0

    if name in ("exec_command", "terminal"):
        preview = (
            (content[:80] if isinstance(content, str) else "")
            .replace("\n", " ")
        )
        return f"[{name}] {preview}... ({line_count} lines)"

    if name in ("read_file",):
        return f"[{name}] ({content_len} chars)"

    if name in ("web_search", "web_fetch"):
        return f"[{name}] ({content_len} chars)"

    if name.startswith("feishu_"):
        return f"[{name}] ({content_len} chars)"

    if name.startswith("browser_"):
        return f"[{name}] ({content_len} chars)"

    return f"[{name}] ({content_len} chars)"


class ContextCompressor:
    """Compresses long conversations using the current model for summarization."""

    def __init__(self, config, main_config=None, client: ChatClient | FallbackChain | None = None):
        self.config = config
        self._main_config = main_config
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
            return self._static_compact(messages, context_window_tokens)

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
        tail = non_system[-tail_count:]
        middle = non_system[:-tail_count]

        if not middle:
            return messages

        pruned_middle = self._prune_tool_outputs(middle)

        turns_text = self._format_turns(pruned_middle)
        if not turns_text.strip():
            return messages

        summary = await self._llm_summarize(turns_text)

        if summary:
            self._previous_summary = summary
            self._compression_count += 1
            summary_msg = {
                "role": "system",
                "content": f"[对话历史摘要 — 仅供参考]\n{summary}",
            }
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

        return self._static_compact(messages, context_window_tokens)

    def _prune_tool_outputs(self, messages: list[dict]) -> list[dict]:
        call_id_to_name: dict[str, str] = {}
        for m in messages:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    tc_id = tc.get("id", "")
                    fn = tc.get("function", {})
                    tc_name = fn.get("name", "unknown") if isinstance(fn, dict) else tc.get("name", "unknown")
                    if tc_id:
                        call_id_to_name[tc_id] = tc_name

        result = []
        for m in messages:
            if m.get("role") == "tool":
                content = m.get("content", "")
                if isinstance(content, str) and len(content) > 300:
                    tool_name = call_id_to_name.get(
                        m.get("tool_call_id", ""), "unknown"
                    )
                    if m.get("name"):
                        tool_name = m["name"]
                    summary = _tool_result_summary(tool_name, content)
                    result.append({**m, "content": summary})
                else:
                    result.append(m)
            else:
                result.append(m)

        return result

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

    def _static_compact(
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
        kept = non_system[-keep_recent:]
        pruned = non_system[:-keep_recent]

        if not pruned:
            return messages

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

        summary_text = (
            "[Earlier conversation summarized]\n" + "\n".join(summaries[-20:])
        )
        summary_msg = {"role": "system", "content": summary_text}

        result = system_msgs + [summary_msg] + kept
        logger.info(
            "Static compact: %d → %d messages (%d → %d tokens)",
            len(messages),
            len(result),
            estimated,
            _estimate_tokens(result),
        )
        return result
