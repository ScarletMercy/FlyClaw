"""Weekly memory consolidation.

Runs every Sunday at 03:00 (before daily consolidation) via the built-in
ConsolidationScheduler. Loads all memories from the KV store, groups by
category, and for each group asks the LLM to: merge duplicates, delete
outdated/useless entries, keep valid ones.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from datetime import datetime
from typing import Any

logger = logging.getLogger("flyclaw.memory_consolidation")


def _friendly_age(updated_at: str | None, now: datetime) -> str:
    if not updated_at:
        return "未知"
    try:
        dt = datetime.fromisoformat(updated_at)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=now.tzinfo)
        days = (now - dt).days
        if days <= 0:
            return "今天"
        if days == 1:
            return "1天前"
        if days < 30:
            return f"{days}天前"
        if days < 330:
            months = round(days / 30)
            return f"约{months}个月前"
        years = round(days / 365)
        return f"约{years}年前"
    except (ValueError, TypeError):
        return "未知"


_CATEGORY_LABELS: dict[str, str] = {
    "preference": "用户偏好",
    "identity": "身份信息",
    "contact": "联系方式",
    "project": "项目信息",
    "service": "服务配置",
    "fact": "事实信息",
}

_CONSOLIDATION_PROMPT = """\
你是一个记忆整理专家。下面是「{category_label}」分类下的全部记忆条目。
今天是 {today}。每条记忆标注了距今天数（如"32天前"）。

请执行三步操作：

1. **合并重复**：语义高度相似的条目（说同一件事的不同表述），合并为一条更完整准确的表述。
2. **删除无用**：基于以下信号判断过时：
   - **时间过旧**：fact 类一次性事实超过 180 天未更新且不再相关的，优先删除；
     service/project 类配置信息超过 365 天未更新且描述的工具/版本已明显陈旧的，删除。
   - **语义过旧**：提到具体版本号/日期但明显陈旧、描述的工具/框架已被替换、
     内容是临时故障或一次性错误的记录。
   - **注意**：preference/identity/contact 类（用户偏好、身份、联系方式）即使较旧也保留，除非明确矛盾。
3. **保留有效**：其余条目保持不变。

输出严格 JSON（不要输出其他内容）：
{{
  "merge": [
    {{"from_keys": ["key1", "key2"], "to_content": "合并后的内容", "to_category": "{category}"}}
  ],
  "delete": ["key3"],
  "keep": ["key4", "key5"]
}}

如果某个类别无需任何操作，对应数组为空。from_keys 中的 key 必须是下面列表中存在的。

记忆条目：
{items}"""


async def _consolidate_store(
    store: Any,
    client: Any,
    now: datetime,
    today_str: str,
    result: dict[str, Any],
    memories: list[dict],
    group_id: str = "",
) -> None:
    """对单个 store（DM 或特定 group_id）执行记忆合并/删除整理。"""
    from src.tools.memory_tools import GroupMemoryStore

    is_group = isinstance(store, GroupMemoryStore)

    if len(memories) < 5:
        scope = f"group {group_id}" if group_id else "DM"
        logger.info("Memory consolidation: %s has only %d memories, skipping", scope, len(memories))
        return

    by_category: dict[str, list[dict]] = defaultdict(list)
    for mem in memories:
        cat = mem.get("category", "fact") or "fact"
        by_category[cat].append(mem)

    for category, cat_memories in by_category.items():
        if len(cat_memories) < 2:
            continue

        category_label = _CATEGORY_LABELS.get(category, category)
        lines = []
        for m in cat_memories:
            age = _friendly_age(m.get("updated_at"), now)
            lines.append(f"- key: {m['key']}\n  content: {m['content']}\n  updated: {age}")
        items_text = "\n".join(lines)

        prompt = _CONSOLIDATION_PROMPT.format(
            category_label=category_label,
            category=category,
            today=today_str,
            items=items_text,
        )

        try:
            plan = await _ask_llm(client, prompt)
        except Exception as e:
            label = f"{category}[{group_id}]" if group_id else category
            logger.error("Memory consolidation LLM call failed for %s: %s", label, e)
            result["errors"].append(f"{label}: {e}")
            continue

        if plan is None:
            continue

        merge_ops = plan.get("merge", [])
        delete_keys = plan.get("delete", [])
        keep_keys = plan.get("keep", [])

        if not isinstance(merge_ops, list):
            merge_ops = []
        if not isinstance(delete_keys, list):
            delete_keys = []
        if not isinstance(keep_keys, list):
            keep_keys = []

        for op in merge_ops:
            if not isinstance(op, dict):
                continue
            from_keys = op.get("from_keys", [])
            to_content = op.get("to_content", "")
            to_category = op.get("to_category", category)
            if not from_keys or not to_content:
                continue
            try:
                if is_group:
                    result_json = await store.remember(to_content, key="", category=to_category, group_id=group_id)
                else:
                    result_json = await store.remember(to_content, key="", category=to_category)
                parsed = json.loads(result_json)
                if "error" in parsed:
                    logger.warning(
                        "Merge rejected for keys %s: remember returned %s",
                        from_keys,
                        parsed["error"],
                    )
                    continue
                new_key = parsed.get("key", "")
                for fk in from_keys:
                    if fk == new_key:
                        logger.warning("Merge: auto-key '%s' collides with from_key, skipping forget", fk)
                        continue
                    if is_group:
                        await store.forget(fk, group_id=group_id)
                    else:
                        await store.forget(fk)
                result["merged"] += 1
            except Exception as e:
                logger.warning("Merge failed for keys %s: %s", from_keys, e)

        for dk in delete_keys:
            if not isinstance(dk, str) or not dk:
                continue
            try:
                if is_group:
                    await store.forget(dk, group_id=group_id)
                else:
                    await store.forget(dk)
                result["deleted"] += 1
            except Exception as e:
                logger.warning("Delete failed for key %s: %s", dk, e)

        result["kept"] += len(keep_keys)
        result["categories_processed"] += 1


async def run_memory_consolidation(container: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "categories_processed": 0,
        "total_memories": 0,
        "merged": 0,
        "deleted": 0,
        "kept": 0,
        "errors": [],
    }

    config = container.config
    if not getattr(config, "memory_store", None) or not config.memory_store.enabled:
        logger.info("Memory consolidation: memory store not enabled, skipping")
        return result

    from src.agent.client import ChatClient
    from src.tools.memory_tools import GroupMemoryStore, get_memory_store

    now = datetime.now().astimezone()
    today_str = now.strftime("%Y-%m-%d")

    client = ChatClient(
        base_url=config.model.base_url,
        api_key=config.model.api_key,
        model=config.model.name,
        temperature=0.0,
    )

    try:
        dm_store = await get_memory_store()
        dm_memories = await dm_store.list_all(limit=2000)
        result["total_memories"] += len(dm_memories)
        await _consolidate_store(dm_store, client, now, today_str, result, memories=dm_memories)

        group_store = await get_memory_store(chat_type="group")
        if isinstance(group_store, GroupMemoryStore):
            all_group = await group_store.list_all(limit=2000, group_id=None)
            result["total_memories"] += len(all_group)

            by_group: dict[str, list[dict]] = defaultdict(list)
            for mem in all_group:
                by_group[mem.get("group_id", "")].append(mem)

            for gid, gmemories in by_group.items():
                await _consolidate_store(group_store, client, now, today_str, result, memories=gmemories, group_id=gid)
    finally:
        await client.close()

    logger.info(
        "Memory consolidation complete: %d memories, %d merged, %d deleted, %d kept",
        result["total_memories"],
        result["merged"],
        result["deleted"],
        result["kept"],
    )
    return result


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return None


async def _ask_llm(client: Any, prompt: str) -> dict | None:
    try:
        resp = await client.chat([{"role": "user", "content": prompt}])
        result = _extract_json(resp.content)
        if result is None:
            logger.warning("Memory consolidation: LLM returned non-JSON")
        return result
    except Exception as e:
        logger.warning("Memory consolidation: LLM call failed: %s", e)
        raise
