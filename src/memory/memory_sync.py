"""Sync memories to curated markdown files (MEMORY.md / USER.md).

This module provides a bridge between the memory store and
human-readable curated memory files. The agent reads these files as bootstrap
context on every session start.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger("flyclaw.memory.memory_sync")

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

> This file is auto-generated from memories. Do not edit manually.

"""

_USER_MD_HEADER = """\
# USER.md — User-curated memory

> This file contains knowledge about the user. Agent can suggest additions,
> but the user has final control.

"""


def _format_memories_by_category(memories: list[dict]) -> dict[str, list[str]]:
    """Group memories by category using DB category column."""
    grouped: dict[str, list[str]] = {cat: [] for cat in _CATEGORY_MAP}
    for mem in memories:
        content = mem.get("content", "")
        if not content:
            continue
        category = mem.get("category", "fact") or "fact"
        if category not in grouped:
            grouped[category] = []
        grouped[category].append(content)
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


async def sync_memories_to_curated_files(workspace: Path) -> dict[str, str]:
    """将记忆同步到策展文件。

    Args:
        workspace: 工作区路径

    Returns:
        生成的文件路径字典 {"memory_md": "...", "user_md": "..."}
    """
    from src.tools.memory_tools import get_memory_store

    s = get_memory_store()
    memories = await s.list_all()

    grouped = _format_memories_by_category(memories)

    memory_md = workspace / "MEMORY.md"
    memory_md.write_text(format_memory_md(grouped), encoding="utf-8")
    logger.info("MEMORY.md synced: %d categories", len([v for v in grouped.values() if v]))

    user_md = workspace / "USER.md"
    if not user_md.exists():
        user_md.write_text(format_user_md(grouped), encoding="utf-8")
        logger.info("USER.md created from memories")

    return {
        "memory_md": str(memory_md),
        "user_md": str(user_md),
    }
