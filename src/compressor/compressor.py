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
from typing import Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

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


def _estimate_tokens(messages: list[BaseMessage]) -> int:
    total = 0
    for m in messages:
        if isinstance(m.content, str):
            total += len(m.content) // _CHARS_PER_TOKEN
        elif isinstance(m.content, list):
            for part in m.content:
                if isinstance(part, dict) and "text" in part:
                    total += len(part["text"]) // _CHARS_PER_TOKEN
                elif isinstance(part, str):
                    total += len(part) // _CHARS_PER_TOKEN
                elif hasattr(part, "text"):
                    total += len(part.text) // _CHARS_PER_TOKEN
        total += 10
    return total


def _tool_result_summary(name: str, content: str) -> str:
    """Create a 1-line summary of a tool result."""
    content_len = len(content) if isinstance(content, str) else 0
    line_count = content.count("\n") + 1 if isinstance(content, str) else 0

    if name in ("exec_command", "terminal"):
        preview = (content[:80] if isinstance(content, str) else "") .replace("\n", " ")
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
    """Compresses long conversations using LLM summarization."""

    def __init__(self, config, main_config=None):
        self.config = config
        self._main_config = main_config
        self._previous_summary: Optional[str] = None
        self._compression_count = 0

    def _get_model_config(self):
        """Resolve model config, inheriting from main model if needed."""
        if self._main_config is None:
            from src.config import load_config
            self._main_config = load_config()
        main = self._main_config
        model_name = self.config.model or "LongCat-Flash-Lite"
        base_url = self.config.base_url or main.model.base_url
        api_key = self.config.api_key or main.model.api_key
        return model_name, base_url, api_key

    async def compress(
        self,
        messages: list[BaseMessage],
        context_window_tokens: int = 100000,
    ) -> list[BaseMessage]:
        """Compress messages if they exceed threshold.

        Returns original messages if under threshold, or compressed list.
        """
        if not self.config.enabled:
            return self._static_compact(messages, context_window_tokens)

        estimated = _estimate_tokens(messages)
        threshold = int(context_window_tokens * self.config.threshold_percent)

        if estimated <= threshold:
            return messages

        logger.info(
            "Context compression triggered: %d estimated tokens > %d threshold",
            estimated, threshold,
        )

        # 1. Separate system messages
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        non_system = [m for m in messages if not isinstance(m, SystemMessage)]

        if not non_system:
            return messages

        # 2. Protect tail
        tail_count = max(self.config.tail_messages, len(non_system) // 4)
        tail = non_system[-tail_count:]
        middle = non_system[:-tail_count]

        if not middle:
            return messages

        # 3. Prune tool outputs in middle section
        pruned_middle = self._prune_tool_outputs(middle)

        # 4. Build text for LLM summarization
        turns_text = self._format_turns(pruned_middle)
        if not turns_text.strip():
            return messages

        # 5. Call LLM for summary
        summary = await self._llm_summarize(turns_text)

        if summary:
            self._previous_summary = summary
            self._compression_count += 1
            summary_msg = SystemMessage(
                content=f"[对话历史摘要 — 仅供参考]\n{summary}"
            )
            result = system_msgs + [summary_msg] + tail
            new_estimated = _estimate_tokens(result)
            logger.info(
                "Context compressed: %d → %d messages (%d → %d tokens, compression #%d)",
                len(messages), len(result),
                estimated, new_estimated,
                self._compression_count,
            )
            return result

        # Fallback to static compression
        return self._static_compact(messages, context_window_tokens)

    def _prune_tool_outputs(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """Replace large tool outputs with 1-line summaries."""
        # Build tool_call_id → tool_name mapping
        call_id_to_name: dict[str, str] = {}
        for m in messages:
            if isinstance(m, AIMessage) and m.tool_calls:
                for tc in m.tool_calls:
                    tc_id = tc.get("id", "")
                    tc_name = tc.get("name", "unknown")
                    if tc_id:
                        call_id_to_name[tc_id] = tc_name

        result = []
        for m in messages:
            if isinstance(m, ToolMessage):
                content = m.content if isinstance(m.content, str) else str(m.content)
                if len(content) > 300:
                    tool_name = call_id_to_name.get(m.tool_call_id, "unknown")
                    # Also check m.name (LangChain sets this)
                    if hasattr(m, "name") and m.name:
                        tool_name = m.name
                    summary = _tool_result_summary(tool_name, content)
                    result.append(ToolMessage(
                        content=summary,
                        tool_call_id=m.tool_call_id,
                    ))
                else:
                    result.append(m)
            else:
                result.append(m)

        return result

    def _format_turns(self, messages: list[BaseMessage]) -> str:
        """Format messages into text for LLM summarization."""
        lines = []
        for m in messages:
            if isinstance(m, HumanMessage):
                text = m.content if isinstance(m.content, str) else str(m.content)[:300]
                lines.append(f"用户: {text[:300]}")
            elif isinstance(m, AIMessage):
                if m.tool_calls:
                    calls = ", ".join(tc.get("name", "?") for tc in m.tool_calls)
                    text = m.content if isinstance(m.content, str) else ""
                    entry = f"AI调用工具: [{calls}]"
                    if text:
                        entry += f" {text[:100]}"
                    lines.append(entry)
                elif m.content:
                    text = m.content if isinstance(m.content, str) else str(m.content)[:300]
                    lines.append(f"AI: {text[:300]}")
            elif isinstance(m, ToolMessage):
                content = m.content if isinstance(m.content, str) else str(m.content)[:150]
                lines.append(f"工具结果: {content[:150]}")
            elif isinstance(m, SystemMessage):
                pass  # Skip system messages in the summary
        return "\n".join(lines)

    async def _llm_summarize(self, turns_text: str) -> Optional[str]:
        """Call LLM to summarize conversation turns."""
        model_name, base_url, api_key = self._get_model_config()

        if not base_url or not api_key:
            logger.warning("Compression: no model config available, using static fallback")
            return None

        # Truncate input if too long (max ~12K chars for input)
        if len(turns_text) > 12000:
            turns_text = turns_text[:12000] + "\n... (更多历史已截断)"

        user_msg = ""
        if self._previous_summary:
            user_msg = f"## 之前的摘要\n{self._previous_summary}\n\n## 新的对话\n{turns_text}"
        else:
            user_msg = f"请摘要以下对话：\n\n{turns_text}"

        try:
            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
                resp = await client.post(
                    f"{base_url.rstrip('/')}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": model_name,
                        "messages": [
                            {"role": "system", "content": _SUMMARY_SYSTEM},
                            {"role": "user", "content": user_msg},
                        ],
                        "temperature": 0.0,
                        "max_tokens": self.config.max_summary_tokens,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning("LLM summarization failed: %s", e)
            return None

    def _static_compact(
        self,
        messages: list[BaseMessage],
        context_window_tokens: int = 100000,
    ) -> list[BaseMessage]:
        """Fallback static compression (original behavior)."""
        max_tokens = int(context_window_tokens * self.config.threshold_percent)
        estimated = _estimate_tokens(messages)
        if estimated <= max_tokens:
            return messages

        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        non_system = [m for m in messages if not isinstance(m, SystemMessage)]

        if not non_system:
            return messages

        keep_recent = max(6, len(non_system) // 3)
        kept = non_system[-keep_recent:]
        pruned = non_system[:-keep_recent]

        if not pruned:
            return messages

        summaries = []
        for m in pruned:
            if isinstance(m, ToolMessage):
                tool_name = getattr(m, "name", "unknown")
                content = m.content if isinstance(m.content, str) else str(m.content)
                summaries.append(f"[Tool({tool_name})]: {content[:150]}")
            elif isinstance(m, AIMessage) and m.tool_calls:
                calls = ", ".join(tc.get("name", "?") for tc in m.tool_calls)
                text = m.content if isinstance(m.content, str) else ""
                entry = f"Assistant: called [{calls}]"
                if text:
                    entry += f" {text[:100]}"
                summaries.append(entry)
            elif isinstance(m, HumanMessage):
                text = m.content if isinstance(m.content, str) else str(m.content)[:200]
                summaries.append(f"User: {text[:200]}")
            elif isinstance(m, AIMessage) and m.content:
                text = m.content if isinstance(m.content, str) else str(m.content)[:200]
                summaries.append(f"Assistant: {text[:200]}")

        summary_text = "[Earlier conversation summarized]\n" + "\n".join(summaries[-20:])
        summary_msg = SystemMessage(content=summary_text)

        result = system_msgs + [summary_msg] + kept
        logger.info(
            "Static compact: %d → %d messages (%d → %d tokens)",
            len(messages), len(result),
            estimated, _estimate_tokens(result),
        )
        return result
