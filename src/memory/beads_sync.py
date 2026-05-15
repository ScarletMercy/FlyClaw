"""Sync Beads memories to curated markdown files (MEMORY.md / USER.md).

This module provides a bridge between the raw key-value Beads store and
human-readable curated memory files. The agent reads these files as bootstrap
context on every session start.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger("myclaw.memory.beads_sync")

_CATEGORY_MAP: dict[str, str] = {
    "preference": "用户偏好",
    "identity": "身份信息",
    "contact": "联系方式",
    "project": "项目信息",
    "service": "服务配置",
    "fact": "事实信息",
}

_MEMORY_MD_HEADER = """\
# MEMORY.md — Agent-curated memory

> This file is auto-generated from Beads memories. Do not edit manually.
> Run `sync_curated_memory()` to refresh.

"""

_USER_MD_HEADER = """\
# USER.md — User-curated memory

> This file contains knowledge about the user. Agent can suggest additions,
> but the user has final control.

"""


def _extract_category(content: str) -> str:
    """Extract category from content like '[preference] ...'."""
    m = re.match(r"\[(\w+)\]\s*", content)
    if m:
        return m.group(1)
    return "fact"


def _format_memories_by_category(memories: list[dict]) -> dict[str, list[str]]:
    """Group memories by category."""
    grouped: dict[str, list[str]] = {cat: [] for cat in _CATEGORY_MAP}
    for mem in memories:
        content = mem.get("content", "")
        if not content:
            continue
        category = _extract_category(content)
        if category not in grouped:
            grouped[category] = []
        # Strip category prefix for display
        clean = re.sub(r"^\[\w+\]\s*", "", content)
        grouped[category].append(clean)
    return grouped


def format_memory_md(grouped: dict[str, list[str]]) -> str:
    """Format grouped memories into MEMORY.md content."""
    lines = [_MEMORY_MD_HEADER]
    for category, items in grouped.items():
        if not items:
            continue
        label = _CATEGORY_MAP.get(category, category)
        lines.append(f"## {label}\n")
        for item in items:
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines)


def format_user_md(grouped: dict[str, list[str]]) -> str:
    """Format grouped memories into USER.md content."""
    lines = [_USER_MD_HEADER]
    # Only include user-centric categories
    user_categories = ["preference", "identity", "contact"]
    for category in user_categories:
        items = grouped.get(category, [])
        if not items:
            continue
        label = _CATEGORY_MAP.get(category, category)
        lines.append(f"## {label}\n")
        for item in items:
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines)


async def sync_beads_to_curated_files(workspace: Path) -> dict[str, str]:
    """将 Beads 记忆同步到策展文件。
    
    Args:
        workspace: 工作区路径
    
    Returns:
        生成的文件路径字典 {"memory_md": "...", "user_md": "..."}
    """
    from src.tools.beads_tools import bd_memories
    import json

    # 1. 获取所有记忆
    result = await bd_memories()
    try:
        memories = json.loads(result)
        if not isinstance(memories, list):
            memories = []
    except (json.JSONDecodeError, TypeError):
        memories = []

    # 2. 按类别分组
    grouped = _format_memories_by_category(memories)

    # 3. 生成 MEMORY.md（自动生成）
    memory_md = workspace / "MEMORY.md"
    memory_md.write_text(format_memory_md(grouped), encoding="utf-8")
    logger.info("MEMORY.md synced: %d categories", len([v for v in grouped.values() if v]))

    # 4. 生成/更新 USER.md（仅当不存在时创建）
    user_md = workspace / "USER.md"
    if not user_md.exists():
        user_md.write_text(format_user_md(grouped), encoding="utf-8")
        logger.info("USER.md created from Beads memories")

    return {
        "memory_md": str(memory_md),
        "user_md": str(user_md),
    }


def get_tools():
    """返回策展记忆同步工具。"""
    from src.agent.tooldef import ToolDef

    async def sync_curated_memory() -> str:
        """手动触发 Beads → 策展文件同步。"""
        from src._container import get_container
        container = get_container()
        workspace = Path(container.config.agents.workspace).expanduser().resolve()
        result = await sync_beads_to_curated_files(workspace)
        return f"Synced: {result['memory_md']}, {result['user_md']}"

    return [
        ToolDef.from_function(sync_curated_memory),
    ]
