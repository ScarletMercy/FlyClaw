from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import aiofiles

from .types import Skill, SkillMetadata

if TYPE_CHECKING:
    from src.config import AppConfig

logger = logging.getLogger("flyclaw.skills")

_MAX_SKILL_FILE_BYTES = 256 * 1024
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)", re.DOTALL)


def is_skill_disabled(skill_name: str, config: "AppConfig", channel: str | None = None) -> bool:
    """Check if a skill is disabled globally or for a specific channel."""
    if skill_name in config.skills.disabled:
        return True
    if channel and channel in config.skills.channel_disabled:
        if skill_name in config.skills.channel_disabled[channel]:
            return True
    return False


def parse_frontmatter(content: str) -> tuple[dict, str]:
    m = _FRONTMATTER_RE.match(content)
    if not m:
        return {}, content
    yaml_str = m.group(1).strip()
    body = m.group(2)
    try:
        import yaml

        data = yaml.safe_load(yaml_str)
        if isinstance(data, dict):
            return data, body
    except Exception as e:
        logger.warning("Failed to parse YAML frontmatter: %s", e)
    return {}, content


async def load_skill(skill_dir: Path, source: str) -> Optional[Skill]:
    md_path = skill_dir / "SKILL.md"
    if not await asyncio.to_thread(md_path.exists):
        return None

    try:
        resolved = await asyncio.to_thread(md_path.resolve)
        size = await asyncio.to_thread(lambda: resolved.stat().st_size)
        if size > _MAX_SKILL_FILE_BYTES:
            logger.warning("Skill file too large, skipping: %s", md_path)
            return None
        async with aiofiles.open(resolved, encoding="utf-8") as f:
            content = await f.read()
        content = content.replace("\r\n", "\n")
    except FileNotFoundError:
        logger.debug("Skill file not found: %s", md_path)
        return None
    except PermissionError:
        logger.warning("Permission denied reading skill: %s", md_path)
        return None
    except Exception as e:
        logger.error("Failed to read skill %s: %s", md_path, e)
        return None

    frontmatter, body = parse_frontmatter(content)
    name = frontmatter.get("name", skill_dir.name)
    description = frontmatter.get("description", "")

    metadata = SkillMetadata(
        name=name,
        description=description,
        user_invocable=frontmatter.get("user-invocable", True),
        disable_model_invocation=frontmatter.get("disable-model-invocation", False),
        command_dispatch=frontmatter.get("command-dispatch"),
        command_tool=frontmatter.get("command-tool"),
        command_arg_mode=frontmatter.get("command-arg-mode"),
    )

    return Skill(
        name=name,
        description=description,
        file_path=md_path,
        base_dir=skill_dir,
        source=source,
        metadata=metadata,
        body=body.strip(),
    )


async def discover_skills(
    directories: list[tuple[str, Path]],
    config: "AppConfig",
    channel: str | None = None,
) -> list[Skill]:
    skills: dict[str, Skill] = {}
    for source_label, directory in directories:
        if not await asyncio.to_thread(directory.exists):
            continue
        found = await _scan_directory(directory, source_label)
        for skill in found:
            if not is_skill_disabled(skill.name, config, channel):
                skills[skill.name] = skill
    return list(skills.values())


async def _scan_directory(directory: Path, source: str) -> list[Skill]:
    results = []
    candidate = await load_skill(directory, source)
    if candidate:
        results.append(candidate)
        return results

    try:
        entries = await asyncio.to_thread(lambda: sorted(directory.iterdir()))
    except PermissionError:
        return results

    for entry in entries:
        if not await asyncio.to_thread(entry.is_dir):
            continue
        if entry.name.startswith(".") or entry.name.startswith("_"):
            continue
        skill = await load_skill(entry, source)
        if skill:
            results.append(skill)
    return results
