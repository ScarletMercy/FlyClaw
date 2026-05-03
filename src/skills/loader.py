from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from .types import Skill, SkillMetadata

logger = logging.getLogger("myclaw.skills")

_MAX_SKILL_FILE_BYTES = 256 * 1024
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)", re.DOTALL)


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


def load_skill(skill_dir: Path, source: str) -> Optional[Skill]:
    md_path = skill_dir / "SKILL.md"
    if not md_path.exists():
        return None

    try:
        resolved = md_path.resolve()
        if resolved.stat().st_size > _MAX_SKILL_FILE_BYTES:
            logger.warning("Skill file too large, skipping: %s", md_path)
            return None
        content = resolved.read_text(encoding="utf-8")
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


def discover_skills(directories: list[tuple[str, Path]]) -> list[Skill]:
    skills: dict[str, Skill] = {}
    for source_label, directory in directories:
        if not directory.exists():
            continue
        found = _scan_directory(directory, source_label)
        for skill in found:
            skills[skill.name] = skill
    return list(skills.values())


def _scan_directory(directory: Path, source: str) -> list[Skill]:
    results = []
    candidate = load_skill(directory, source)
    if candidate:
        results.append(candidate)
        return results

    try:
        entries = sorted(directory.iterdir())
    except PermissionError:
        return results

    for entry in entries:
        if not entry.is_dir():
            continue
        if entry.name.startswith(".") or entry.name.startswith("_"):
            continue
        skill = load_skill(entry, source)
        if skill:
            results.append(skill)
    return results
