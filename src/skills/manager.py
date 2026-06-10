"""Skill manager for agent-driven skill creation, editing, and deletion.

Allows the agent to create new skills (SKILL.md + directory structure),
edit existing skills, and manage supporting files (references, templates, etc.).
"""

from __future__ import annotations

import asyncio
import json
import os
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import aiofiles
import aiofiles.os

from src.skills.loader import load_skill
from src.skills.types import Skill, SkillMetadata

logger = logging.getLogger("flyclaw.skills.manager")

_WIN_RETRIES = 5
_WIN_DELAY = 0.15


async def _safe_unlink(path: Path) -> None:
    """Delete a file, retrying on PermissionError (WinError 32)."""
    for i in range(1, _WIN_RETRIES + 1):
        try:
            await asyncio.to_thread(path.unlink)
            return
        except PermissionError:
            if i == _WIN_RETRIES:
                raise
            await asyncio.sleep(_WIN_DELAY * (2 ** (i - 1)))


async def _safe_rmtree(directory: Path) -> None:
    """Delete a directory tree, retrying on PermissionError (WinError 32)."""
    import shutil

    for i in range(1, _WIN_RETRIES + 1):
        try:
            await asyncio.to_thread(shutil.rmtree, directory)
            return
        except PermissionError:
            if i == _WIN_RETRIES:
                raise
            await asyncio.sleep(_WIN_DELAY * (2 ** (i - 1)))


def _user_skills_dir() -> Path:
    from src.instance import skills_dir

    return skills_dir()


# Validation limits
_MAX_NAME_LENGTH = 64
_MAX_DESCRIPTION_LENGTH = 1024
# Security patterns for injection detection
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(your\s+)?instructions", re.IGNORECASE),
    re.compile(r"you\s+are\s+(now\s+)?", re.IGNORECASE),
    re.compile(r"<\s*system\s*>", re.IGNORECASE),
    re.compile(r"\]\]>", re.IGNORECASE),
]


def _validate_skill_name(name: str) -> Optional[str]:
    """Validate skill name. Returns error message or None."""
    if not name:
        return "Skill name is required"
    if len(name) > _MAX_NAME_LENGTH:
        return f"Skill name too long (max {_MAX_NAME_LENGTH} chars)"
    if not re.match(r"^[a-zA-Z0-9_-]+$", name):
        return "Skill name must contain only letters, numbers, hyphens, and underscores"
    return None


def _scan_for_injection(content: str) -> Optional[str]:
    """Scan content for prompt injection patterns. Returns warning or None."""
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(content):
            return f"Potential injection detected: {pattern.pattern}"
    return None


def _format_skill_frontmatter(name: str, description: str, category: str = "") -> str:
    """Generate YAML frontmatter for SKILL.md."""
    import yaml

    meta: dict = {"name": name, "description": description}
    if category:
        meta["category"] = category
    # yaml.dump adds trailing newline; strip it so we control formatting
    yaml_block = yaml.dump(meta, allow_unicode=True, default_flow_style=False).rstrip("\n")
    return f"---\n{yaml_block}\n---\n"


SUPPORTING_DIRS = frozenset({"references", "templates", "scripts", "assets"})


class SkillManager:
    """Manages skill CRUD operations."""

    def __init__(self, skills_dir: Path | None = None):
        if skills_dir is None:
            skills_dir = _user_skills_dir()
        self.skills_dir = skills_dir
        self.skills_dir.mkdir(parents=True, exist_ok=True)

    async def create_skill(
        self,
        name: str,
        description: str,
        content: str,
        category: str = "",
        created_by: str = "agent",
    ) -> tuple[Skill, Optional[str]]:
        """创建新技能。

        Returns:
            (Skill, error_message) - error_message is None on success
        """
        # 验证名称
        error = _validate_skill_name(name)
        if error:
            return None, error

        # 检查是否已存在
        skill_dir = self.skills_dir / name
        if await aiofiles.os.path.exists(skill_dir):
            return None, f"Skill already exists: {name}"

        # 安全检查
        injection_warning = _scan_for_injection(content)
        if injection_warning:
            logger.warning("Injection warning for skill '%s': %s", name, injection_warning)

        # 创建目录结构
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "references").mkdir(exist_ok=True)
        (skill_dir / "templates").mkdir(exist_ok=True)

        # 创建 SKILL.md
        skill_md = skill_dir / "SKILL.md"
        frontmatter = _format_skill_frontmatter(name, description, category)
        async with aiofiles.open(skill_md, "w", encoding="utf-8") as f:
            await f.write(frontmatter + content)

        # 加载并返回
        skill = await load_skill(skill_dir, "user")
        if skill:
            # 记录创建信息
            await self._record_usage(name, created_by=created_by)
            return skill, None
        return None, "Failed to load created skill"

    async def edit_skill(
        self,
        name: str,
        content: str,
        description: Optional[str] = None,
    ) -> tuple[Skill, Optional[str]]:
        """编辑现有技能。

        Returns:
            (Skill, error_message) - error_message is None on success
        """
        error = _validate_skill_name(name)
        if error:
            return None, error  # type: ignore[return-value]

        skill_dir = self.skills_dir / name
        skill_md = skill_dir / "SKILL.md"

        if not await aiofiles.os.path.exists(skill_md):
            return None, f"Skill not found: {name}"

        # 安全检查
        injection_warning = _scan_for_injection(content)
        if injection_warning:
            logger.warning("Injection warning for skill edit '%s': %s", name, injection_warning)

        # 读取现有 frontmatter
        existing = await load_skill(skill_dir, "user")
        if not existing:
            return None, "Failed to load existing skill"

        # 更新内容
        desc = description or existing.description
        frontmatter = _format_skill_frontmatter(name, desc, "")
        async with aiofiles.open(skill_md, "w", encoding="utf-8") as f:
            await f.write(frontmatter + content)

        # 重新加载
        skill = await load_skill(skill_dir, "user")
        if skill:
            await self._record_usage(name, action="edited")
            return skill, None
        return None, "Failed to reload edited skill"

    async def delete_skill(self, name: str) -> tuple[bool, Optional[str]]:
        """删除技能（仅用户创建的）。

        Returns:
            (success, error_message)
        """
        error = _validate_skill_name(name)
        if error:
            return False, error

        skill_dir = (self.skills_dir / name).resolve()
        base = str(self.skills_dir.resolve()) + os.sep
        if not (str(skill_dir) + os.sep).startswith(base):
            return False, f"Invalid skill name: {name}"
        if not await aiofiles.os.path.exists(skill_dir):
            return False, f"Skill not found: {name}"

        # 检查来源（只允许删除用户技能）
        skill = await load_skill(skill_dir, "user")
        if skill and skill.source not in ("user", "agents-project"):
            return False, f"Cannot delete system skill: {name}"

        try:
            await _safe_rmtree(skill_dir)
            await self._record_usage(name, action="deleted")
            return True, None
        except Exception as e:
            return False, f"Failed to delete skill: {str(e)}"

    async def list_skills(self) -> list[Skill]:
        """列出所有用户技能。"""
        from src.skills.loader import discover_skills
        from src._container import get_container

        if not await aiofiles.os.path.exists(self.skills_dir):
            return []

        container = get_container()
        skills = await discover_skills([("user", self.skills_dir)], container.config)
        return skills

    async def patch_skill(
        self,
        name: str,
        old_string: str,
        new_string: str,
        file_path: str = "",
        replace_all: bool = False,
    ) -> tuple[bool, Optional[str]]:
        """Targeted find-and-replace in SKILL.md or a supporting file."""
        error = _validate_skill_name(name)
        if error:
            return False, error

        skill_dir = self.skills_dir / name
        if not await aiofiles.os.path.exists(skill_dir):
            return False, f"Skill not found: {name}"

        target = skill_dir / "SKILL.md"
        if file_path:
            resolved = (skill_dir / file_path).resolve()
            if not resolved.is_relative_to(skill_dir.resolve()):
                return False, "Path traversal not allowed"
            if not await aiofiles.os.path.exists(resolved.parent):
                resolved.parent.mkdir(parents=True, exist_ok=True)
            target = resolved
        if not await aiofiles.os.path.exists(target):
            return False, f"File not found: {file_path or 'SKILL.md'}"

        async with aiofiles.open(target, encoding="utf-8") as f:
            content = await f.read()
        if old_string not in content:
            return False, f"old_string not found in {target.name}"

        count = content.count(old_string)
        if count > 1 and not replace_all:
            return False, f"Found {count} matches; set replace_all=True or provide more context"

        new_content = content.replace(old_string, new_string)
        async with aiofiles.open(target, "w", encoding="utf-8") as f:
            await f.write(new_content)
        await self.bump_patch(name)
        return True, None

    async def write_supporting_file(
        self,
        name: str,
        file_path: str,
        file_content: str,
    ) -> tuple[bool, Optional[str]]:
        """Write a supporting file under a skill (references/templates/scripts/assets)."""
        error = _validate_skill_name(name)
        if error:
            return False, error

        skill_dir = self.skills_dir / name
        if not await aiofiles.os.path.exists(skill_dir):
            return False, f"Skill not found: {name}"
        parts = Path(file_path).parts
        if not parts or parts[0] not in SUPPORTING_DIRS:
            return False, f"file_path must start with one of: {', '.join(sorted(SUPPORTING_DIRS))}"
        target = (skill_dir / file_path).resolve()
        if not target.is_relative_to(skill_dir.resolve()):
            return False, "Path traversal not allowed"
        target.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(target, "w", encoding="utf-8") as f:
            await f.write(file_content)
        await self.bump_patch(name)
        return True, None

    async def remove_supporting_file(self, name: str, file_path: str) -> tuple[bool, Optional[str]]:
        """Remove a supporting file from a skill."""
        error = _validate_skill_name(name)
        if error:
            return False, error

        skill_dir = self.skills_dir / name
        if not await aiofiles.os.path.exists(skill_dir):
            return False, f"Skill not found: {name}"
        target = (skill_dir / file_path).resolve()
        if not target.is_relative_to(skill_dir.resolve()):
            return False, "Path traversal not allowed"
        if not await aiofiles.os.path.exists(target):
            return False, f"File not found: {file_path}"
        await _safe_unlink(target)
        return True, None

    async def _read_usage(self) -> dict:
        """异步读取并解析 .usage.json。"""
        usage_file = self.skills_dir / ".usage.json"
        if not await aiofiles.os.path.exists(usage_file):
            return {}
        try:
            async with aiofiles.open(usage_file, encoding="utf-8") as f:
                return json.loads(await f.read())
        except Exception:
            return {}

    async def _write_usage(self, usage_data: dict) -> None:
        """异步序列化并写入 .usage.json。"""
        usage_file = self.skills_dir / ".usage.json"
        try:
            async with aiofiles.open(usage_file, "w", encoding="utf-8") as f:
                await f.write(json.dumps(usage_data, indent=2, ensure_ascii=False))
        except Exception as e:
            logger.warning("Failed to write skill usage: %s", e)

    async def _record_usage(self, skill_name: str, action: str = "created", created_by: str = "agent") -> None:
        """记录技能使用信息到 .usage.json。"""
        usage_data = await self._read_usage()

        if skill_name not in usage_data:
            usage_data[skill_name] = self._empty_record()
            usage_data[skill_name]["created_by"] = created_by

        record = usage_data[skill_name]
        record["last_used_at"] = datetime.now().isoformat()

        if action == "created":
            record["created_by"] = created_by
        elif action == "edited":
            record["patch_count"] = record.get("patch_count", 0) + 1
            record["last_patched_at"] = datetime.now().isoformat()
        elif action == "viewed":
            record["view_count"] = record.get("view_count", 0) + 1
            record["last_viewed_at"] = datetime.now().isoformat()

        await self._write_usage(usage_data)

    async def get_usage(self, skill_name: str) -> Optional[dict]:
        """获取技能使用统计。"""
        usage_data = await self._read_usage()
        return usage_data.get(skill_name)

    async def bump_view(self, skill_name: str) -> None:
        await self._mutate_usage(
            skill_name,
            lambda r: (r.update(view_count=r.get("view_count", 0) + 1, last_viewed_at=datetime.now().isoformat()),),
        )

    async def bump_use(self, skill_name: str) -> None:
        await self._mutate_usage(
            skill_name,
            lambda r: (r.update(use_count=r.get("use_count", 0) + 1, last_used_at=datetime.now().isoformat()),),
        )

    async def bump_patch(self, skill_name: str) -> None:
        await self._mutate_usage(
            skill_name,
            lambda r: (r.update(patch_count=r.get("patch_count", 0) + 1, last_patched_at=datetime.now().isoformat()),),
        )

    async def mark_agent_created(self, skill_name: str) -> None:
        await self._mutate_usage(skill_name, lambda r: r.update(created_by="agent"))

    async def _mutate_usage(self, skill_name: str, mutator) -> None:
        usage_data = await self._read_usage()
        if skill_name not in usage_data:
            usage_data[skill_name] = self._empty_record()
        mutator(usage_data[skill_name])
        await self._write_usage(usage_data)

    def _empty_record(self) -> dict:
        return {
            "use_count": 0,
            "view_count": 0,
            "patch_count": 0,
            "last_used_at": None,
            "last_viewed_at": None,
            "last_patched_at": None,
            "created_by": None,
            "state": "active",
            "pinned": False,
            "created_at": datetime.now().isoformat(),
            "archived_at": None,
        }


_SKILL_TOOL_NAMES = frozenset({"skill_view", "skill_manage", "skill_hub"})


def _normalize_skill_name(name: str) -> str:
    """Strip markdown bold markers and whitespace from a skill name.

    Models seeing ``**my-skill**`` in the system prompt may copy the
    asterisks verbatim into the tool call.  This helper removes leading /
    trailing ``*`` characters and whitespace so the lookup still succeeds.
    """
    cleaned = name.strip()
    # Remove leading/trailing asterisks (markdown bold: **name** or *name*)
    while cleaned.startswith("*"):
        cleaned = cleaned[1:]
    while cleaned.endswith("*"):
        cleaned = cleaned[:-1]
    cleaned = cleaned.strip()
    return cleaned


def _find_skill(skills: list["Skill"], raw_name: str) -> "Skill | None":
    """Look up a skill by name with tolerant matching.

    Strategy:
    1. Exact match on normalized name (stripped, de-asterisked).
    2. Case-insensitive fallback on normalized name.
    """
    norm = _normalize_skill_name(raw_name)
    if not norm:
        return None

    # 1. Exact match
    for s in skills:
        if s.name == norm:
            return s

    # 2. Case-insensitive fallback
    norm_lower = norm.casefold()
    for s in skills:
        if s.name.casefold() == norm_lower:
            return s

    return None


def _format_skill_result(
    skill: "Skill",
    **_extra: object,
) -> str:
    """Format a skill as plain text for the model.

    The body is the top-level content so the model treats it as direct
    instruction text rather than having to parse a JSON envelope.
    A lightweight metadata footer is appended.
    """
    parts: list[str] = []
    if skill.body:
        parts.append(skill.body)
    parts.append("")
    parts.append("---")
    parts.append(f"*Skill: {skill.name}*")
    if skill.description:
        parts.append(f"*{skill.description}*")
    parts.append(f"*Source: {skill.source} · Path: {skill.base_dir}*")
    if _extra.get("view_count") or _extra.get("use_count"):
        parts.append(f"*Views: {_extra.get('view_count', 0)} · Uses: {_extra.get('use_count', 0)}*")
    parts.append("---")
    return "\n".join(parts)


def get_tools() -> list:
    """返回技能管理工具集（3 个独立工具）。"""
    from src.agent.tooldef import ToolDef

    manager = SkillManager()

    async def skill_view(
        name: str,
        file_path: str = "",
    ) -> str:
        """Load the full instruction content of a specified skill. Must use this (not read_file) to read skills.

        Args:
            name: Skill name to view.
            file_path: Optional supporting file path within the skill directory.
        """
        if not name:
            return json.dumps({"error": "name is required"})
        from src._container import get_container

        container = get_container()
        skills = container.skills_cache or []
        s = _find_skill(skills, name)
        if s is not None:
            await manager.bump_view(s.name)
            usage = await manager.get_usage(s.name) or {}
            return _format_skill_result(s, **usage)
        return json.dumps({"error": f"Skill not found: {name}"})

    async def skill_manage(
        action: Literal["create", "edit", "patch", "delete", "toggle", "write_file", "remove_file"],
        name: str = "",
        content: str = "",
        description: str = "",
        category: str = "",
        old_string: str = "",
        new_string: str = "",
        file_path: str = "",
        file_content: str = "",
        replace_all: bool = False,
        enabled: bool = True,
        channel: str = "",
        absorbed_into: str = "",
    ) -> str:
        """Create, edit, patch, delete, toggle, or manage supporting files for skills.

        Args:
            action: Operation type (create, edit, patch, delete, toggle, write_file, remove_file)
            name: Skill name (required for all actions)
            content: Full skill content for create/edit
            description: Skill description
            category: Skill category (for create)
            old_string: Text to find for patch
            new_string: Replacement text for patch
            file_path: Supporting file path for patch/write_file/remove_file
            file_content: File content for write_file
            replace_all: Replace all occurrences for patch
            enabled: Whether to enable for toggle
            channel: Channel name for toggle (e.g. qq)
            absorbed_into: Target umbrella skill name when deleting a merged skill
        """
        from src._container import get_container

        container = get_container()

        if action == "create":
            if not name or not content:
                return json.dumps({"error": "name and content are required for create"})
            skill, error = await manager.create_skill(name, description or name, content, category)
            if error:
                return json.dumps({"error": error})
            from src.skills.provenance import is_background_review

            if is_background_review():
                await manager.mark_agent_created(name)
            await _reload_skills(container)
            return json.dumps(
                {
                    "success": True,
                    "action": "created",
                    "skill": {"name": skill.name, "description": skill.description, "source": skill.source},
                },
                ensure_ascii=False,
            )

        elif action == "edit":
            if not name or not content:
                return json.dumps({"error": "name and content are required for edit"})
            skill, error = await manager.edit_skill(name, content, description or None)
            if error:
                return json.dumps({"error": error})
            await _reload_skills(container)
            return json.dumps(
                {
                    "success": True,
                    "action": "edited",
                    "skill": {"name": skill.name, "description": skill.description},
                },
                ensure_ascii=False,
            )

        elif action == "patch":
            if not name or not old_string or not new_string:
                return json.dumps({"error": "name, old_string, and new_string are required for patch"})
            ok, error = await manager.patch_skill(
                name,
                old_string,
                new_string,
                file_path=file_path,
                replace_all=replace_all,
            )
            if error:
                return json.dumps({"error": error})
            await _reload_skills(container)
            return json.dumps({"success": True, "action": "patched", "skill": name})

        elif action == "delete":
            if not name:
                return json.dumps({"error": "name is required for delete"})
            success, error = await manager.delete_skill(name)
            if error:
                return json.dumps({"error": error})
            await _reload_skills(container)
            return json.dumps({"success": True, "action": "deleted", "skill": name})

        elif action == "toggle":
            if not name:
                return json.dumps({"error": "name is required for toggle"})
            from src.config import save_config

            # Look up in cache first, then fall back to disk.
            # Disabled skills are excluded from the cache by discover_skills(),
            # but toggle must be able to re-enable them.
            skills = container.skills_cache or []
            found = any(s.name == name for s in skills)
            if not found:
                skill_dir = manager.skills_dir / name
                skill_md = skill_dir / "SKILL.md"
                if await aiofiles.os.path.exists(skill_md):
                    found = True
                else:
                    return json.dumps({"error": f"Skill not found: {name}"})
            config = container.config
            action_result = "no change"
            if channel:
                if channel not in config.skills.channel_disabled:
                    config.skills.channel_disabled[channel] = []
                disabled_list = config.skills.channel_disabled[channel]
                if enabled and name in disabled_list:
                    disabled_list.remove(name)
                    action_result = "enabled"
                elif not enabled and name not in disabled_list:
                    disabled_list.append(name)
                    action_result = "disabled"
            else:
                if enabled and name in config.skills.disabled:
                    config.skills.disabled.remove(name)
                    action_result = "enabled"
                elif not enabled and name not in config.skills.disabled:
                    config.skills.disabled.append(name)
                    action_result = "disabled"
            try:
                await asyncio.to_thread(save_config, config)
                await _reload_skills(container)
            except Exception as e:
                return json.dumps({"error": f"Failed to save config: {str(e)}"})
            return json.dumps(
                {
                    "success": True,
                    "skill": name,
                    "action": action_result,
                    "channel": channel or "global",
                }
            )

        elif action == "write_file":
            if not name or not file_path or not file_content:
                return json.dumps({"error": "name, file_path, and file_content are required for write_file"})
            ok, error = await manager.write_supporting_file(name, file_path, file_content)
            if error:
                return json.dumps({"error": error})
            return json.dumps({"success": True, "action": "wrote_file", "skill": name, "file": file_path})

        elif action == "remove_file":
            if not name or not file_path:
                return json.dumps({"error": "name and file_path are required for remove_file"})
            ok, error = await manager.remove_supporting_file(name, file_path)
            if error:
                return json.dumps({"error": error})
            return json.dumps({"success": True, "action": "removed_file", "skill": name, "file": file_path})

        else:
            return json.dumps(
                {
                    "error": f"Unknown action: {action}. Valid actions: create, edit, patch, delete, toggle, write_file, remove_file"
                }
            )

    async def skill_hub(
        action: Literal["search_hub", "inspect_hub", "install_hub", "scan_hub", "install", "uninstall"],
        query: str = "",
        identifier: str = "",
        name: str = "",
        source: str = "",
        force: bool = False,
    ) -> str:
        """Search, inspect, install, scan skills from remote hubs, or install/uninstall from URL or local path.

        Actions:
            search_hub: Search remote skill hubs by query.
            inspect_hub: View skill details from search results.
            install_hub: Download and install a skill from remote hub (with guard scan).
            scan_hub: Run security scan on a locally installed skill.
            install: Install a skill from a local path or URL. Supports:
                - Local directory containing SKILL.md
                - Local .zip file containing SKILL.md
                - Remote URL pointing to a .zip file
                When user sends a skill package (zip), save it first then use this action.
            uninstall: Remove a locally installed skill.

        Args:
            action: Operation type (search_hub, inspect_hub, install_hub, scan_hub, install, uninstall)
            query: Search query (for search_hub)
            identifier: Skill identifier from search results (for inspect_hub, install_hub)
            name: Skill name (for scan_hub, uninstall)
            source: Source path or URL (for install). Accepts local directory, local .zip, or remote .zip URL.
            force: Force install despite blocked scan verdict (for install_hub)
        """
        from src._container import get_container

        container = get_container()

        if action == "search_hub":
            if not query:
                return json.dumps({"error": "query is required for search_hub"})
            if not getattr(container.config.skills.hub, "enabled", True):
                return json.dumps({"error": "Hub is disabled in configuration"})
            try:
                from src.skills.hub import create_sources, parallel_search

                sources = create_sources()
                results = await parallel_search(
                    sources,
                    query,
                    limit=20,
                    source_filter=source or "all",
                )
                items = []
                for r in results:
                    items.append(
                        {
                            "name": r.name,
                            "description": (r.description or "")[:200],
                            "source": r.source,
                            "identifier": r.identifier,
                            "trust_level": r.trust_level,
                        }
                    )
                return json.dumps({"results": items}, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error("Hub search failed: %s", e)
                return json.dumps({"error": f"Search failed: {str(e)}"})

        elif action == "inspect_hub":
            if not identifier:
                return json.dumps({"error": "identifier is required for inspect_hub"})
            if not getattr(container.config.skills.hub, "enabled", True):
                return json.dumps({"error": "Hub is disabled in configuration"})
            try:
                from src.skills.hub import create_sources, resolve_source

                sources = create_sources()
                src = resolve_source(identifier, sources)
                if not src:
                    return json.dumps({"error": f"No source found for identifier: {identifier}"})
                meta = await src.inspect(identifier)
                if not meta:
                    return json.dumps({"error": f"Skill not found: {identifier}"})
                return json.dumps(
                    {
                        "name": meta.name,
                        "description": meta.description,
                        "source": meta.source,
                        "identifier": meta.identifier,
                        "trust_level": meta.trust_level,
                        "repo": meta.repo,
                        "tags": meta.tags,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            except Exception as e:
                logger.error("Hub inspect failed: %s", e)
                return json.dumps({"error": f"Inspect failed: {str(e)}"})

        elif action == "install_hub":
            if not identifier:
                return json.dumps({"error": "identifier is required for install_hub"})
            if not getattr(container.config.skills.hub, "enabled", True):
                return json.dumps({"error": "Hub is disabled in configuration"})
            quarantine_path = None
            try:
                from src.skills.hub import (
                    create_sources,
                    resolve_source,
                    install_from_quarantine,
                )
                from src.skills.guard import should_allow_install, format_scan_report

                sources = create_sources()
                src = resolve_source(identifier, sources)
                if not src:
                    return json.dumps({"error": f"No source found for identifier: {identifier}"})
                bundle = await src.fetch(identifier)
                if not bundle and src.source_id() == "skills-sh":
                    clawhub = next((s for s in sources if s.source_id() == "clawhub"), None)
                    if clawhub:
                        bundle = await clawhub.fetch(identifier)
                if not bundle:
                    return json.dumps({"error": f"Failed to fetch skill: {identifier}"})
                if not bundle.name:
                    return json.dumps({"error": "Could not determine skill name from remote content"})

                guard_enabled = getattr(container.config.skills.hub, "guard_enabled", True)

                quarantine_path, scan_result = await asyncio.to_thread(
                    _hub_prepare_and_scan,
                    bundle,
                    guard_enabled,
                )

                if guard_enabled:
                    allowed, reason = should_allow_install(scan_result, force=force)
                    if not allowed:
                        report = format_scan_report(scan_result)
                        return json.dumps(
                            {
                                "error": f"Install blocked: {reason}",
                                "scan_report": report,
                            },
                            ensure_ascii=False,
                        )

                install_dir = await asyncio.to_thread(
                    install_from_quarantine,
                    quarantine_path,
                    bundle.name,
                    bundle,
                    scan_result,
                )
                await _reload_skills_from_manager()

                return json.dumps(
                    {
                        "success": True,
                        "action": "installed",
                        "skill": {
                            "name": bundle.name,
                            "source": bundle.source,
                            "trust_level": bundle.trust_level,
                            "scan_verdict": scan_result.verdict,
                        },
                        "message": f"技能 '{bundle.name}' 安装成功，可用 skill_view(name=\"{bundle.name}\") 查看和使用。",
                    },
                    ensure_ascii=False,
                )
            except Exception as e:
                logger.error("Hub install failed: %s", e)
                return json.dumps({"error": f"Install failed: {str(e)}"})
            finally:
                if quarantine_path and quarantine_path.exists():
                    import shutil

                    await asyncio.to_thread(shutil.rmtree, quarantine_path, ignore_errors=True)

        elif action == "scan_hub":
            if not name:
                return json.dumps({"error": "name is required for scan_hub"})
            if not getattr(container.config.skills.hub, "enabled", True):
                return json.dumps({"error": "Hub is disabled in configuration"})
            try:
                from src.skills.guard import scan_skill, format_scan_report

                skill_dir, skill_source = await asyncio.to_thread(
                    _hub_lookup_skill_dir,
                    name,
                    container.skills_cache,
                )
                if not skill_dir:
                    return json.dumps({"error": f"Skill not found: {name}"})
                result = await asyncio.to_thread(scan_skill, skill_dir, source=skill_source)
                report = format_scan_report(result)
                return json.dumps(
                    {
                        "name": name,
                        "verdict": result.verdict,
                        "findings_count": len(result.findings),
                        "report": report,
                    },
                    ensure_ascii=False,
                )
            except Exception as e:
                logger.error("Hub scan failed: %s", e)
                return json.dumps({"error": f"Scan failed: {str(e)}"})

        elif action == "install":
            if not source:
                return json.dumps({"error": "source (URL or path) is required for install"})
            try:
                if source.startswith(("http://", "https://")):
                    result = await asyncio.to_thread(_install_from_url, source)
                else:
                    result = await asyncio.to_thread(_install_from_path, source)
                await _reload_skills_from_manager()
                return result
            except Exception as e:
                logger.error("Skill install failed: %s", e)
                return json.dumps({"error": f"Install failed: {str(e)}"})

        elif action == "uninstall":
            if not name:
                return json.dumps({"error": "name is required for uninstall"})
            from src.config import save_config

            skills = container.skills_cache or []
            skill_dir = None
            for s in skills:
                if s.name == name:
                    skill_dir = s.base_dir
                    break
            if not skill_dir:
                return json.dumps({"error": f"Skill not found: {name}"})
            if not str(skill_dir).startswith(str(manager.skills_dir)):
                return json.dumps({"error": f"Can only uninstall user skills in {manager.skills_dir}"})
            try:
                await _safe_rmtree(skill_dir)
            except Exception as e:
                return json.dumps({"error": f"Failed to remove skill directory: {str(e)}"})
            config = container.config
            if name in config.skills.disabled:
                config.skills.disabled.remove(name)
            for ch in config.skills.channel_disabled:
                if name in config.skills.channel_disabled[ch]:
                    config.skills.channel_disabled[ch].remove(name)
            try:
                await asyncio.to_thread(save_config, config)
                await _reload_skills(container)
            except Exception as e:
                return json.dumps({"error": f"Failed to save config: {str(e)}"})
            try:
                await asyncio.to_thread(_hub_uninstall_lock_cleanup, name)
            except Exception:
                pass
            return json.dumps({"success": True, "uninstalled": name})

        else:
            return json.dumps(
                {
                    "error": f"Unknown action: {action}. Valid actions: search_hub, inspect_hub, install_hub, scan_hub, install, uninstall"
                }
            )

    return [
        ToolDef.from_function(skill_view),
        ToolDef.from_function(skill_manage),
        ToolDef.from_function(skill_hub),
    ]


def _hub_prepare_and_scan(bundle, guard_enabled):
    """Sync helper: ensure dirs, quarantine bundle, scan skill.

    Returns (quarantine_path, scan_result).
    """
    from src.skills.hub import ensure_hub_dirs, quarantine_bundle
    from src.skills.guard import scan_skill

    ensure_hub_dirs()
    qpath = quarantine_bundle(bundle)

    if guard_enabled:
        scan_result = scan_skill(qpath, source=bundle.source)
    else:
        from src.skills.types import ScanResult

        scan_result = ScanResult(
            skill_name=bundle.name,
            source=bundle.source,
            trust_level=bundle.trust_level,
            verdict="safe",
            scanned_at="",
            summary="Guard disabled",
        )
    return qpath, scan_result


def _hub_lookup_skill_dir(name, skills_cache):
    """Sync helper: find skill directory from cache or lock file.

    Returns (skill_dir, skill_source) or (None, None) if not found.
    """
    for s in skills_cache or []:
        if s.name == name:
            return s.base_dir, getattr(s, "source", "community")

    from src.skills.hub import HubLockFile, _skills_dir

    lock = HubLockFile()
    entry = lock.get_installed(name)
    if not entry:
        return None, None
    install_path = entry.get("install_path")
    if not install_path:
        return None, None
    candidate = (_skills_dir() / install_path).resolve()
    if not candidate.is_relative_to(_skills_dir().resolve()):
        return None, None
    return candidate, entry.get("trust_level", "community")


def _hub_uninstall_lock_cleanup(name):
    """Sync helper: update lock file and audit log for uninstall."""
    from src.skills.hub import HubLockFile, append_audit_log

    lock = HubLockFile()
    entry = lock.get_installed(name)
    if entry:
        lock.record_uninstall(name)
        append_audit_log("UNINSTALL", name, entry["source"], entry["trust_level"], "n/a", "user_request")


async def _reload_skills(container) -> None:
    """Reload skills and update all dependent components."""
    from src.skills.loader import discover_skills
    from src.skills.prompt import build_skills_prompt
    from src.prompt import _build_skills_section

    dirs = container._build_skill_directories()
    skills = await discover_skills(dirs, container.config)
    container.skills_cache = skills
    if container.agent_loop:
        container.agent_loop._skills_prompt = build_skills_prompt(skills)
        hub_on = getattr(container.config.skills.hub, "enabled", True)
        container.agent_loop._prompt_skills = "\n".join(
            _build_skills_section(container.agent_loop._skills_prompt, hub_enabled=hub_on)
        )
    dispatcher = getattr(container, "dispatcher", None)
    if dispatcher is not None:
        dispatcher._reload_skills(skills)


def _install_from_url(url: str) -> str:
    """Install a skill from a URL (zip file)."""
    import tempfile
    import zipfile
    from urllib.parse import urlparse
    import httpx

    parsed = urlparse(url)
    filename = Path(parsed.path).name
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / filename
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(url, follow_redirects=True)
            resp.raise_for_status()
            tmp_path.write_bytes(resp.content)
        if filename.endswith(".zip"):
            return _install_from_zip(tmp_path)
        else:
            return json.dumps({"error": "URL must point to a .zip file"})


def _install_from_path(path: str) -> str:
    """Install a skill from a local path (directory or zip)."""
    import shutil
    import zipfile

    p = Path(path).expanduser().resolve()
    if not p.exists():
        return json.dumps({"error": f"Path not found: {path}"})
    if p.is_dir():
        skill_md = p / "SKILL.md"
        if not skill_md.exists():
            return json.dumps({"error": f"Directory does not contain SKILL.md: {p}"})
        dest = _user_skills_dir() / p.name
        if dest.exists():
            return json.dumps({"error": f"Skill already exists: {dest}"})
        shutil.copytree(p, dest)
        return json.dumps({"success": True, "installed": str(dest)})
    elif p.suffix == ".zip":
        return _install_from_zip(p)
    else:
        return json.dumps({"error": "Path must be a directory or .zip file"})


def _install_from_zip(zip_path: Path) -> str:
    """Install a skill from a zip file."""
    import zipfile

    if not zipfile.is_zipfile(zip_path):
        return json.dumps({"error": f"Not a valid zip file: {zip_path}"})
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        skill_md_in_zip = [n for n in names if n.endswith("/SKILL.md") or n == "SKILL.md"]
        if not skill_md_in_zip:
            return json.dumps({"error": "Zip file does not contain SKILL.md"})
        first_skill = skill_md_in_zip[0]
        skill_dir_name = first_skill.split("/")[0] if "/" in first_skill else Path(zip_path).stem
        dest = _user_skills_dir() / skill_dir_name
        if dest.exists():
            return json.dumps({"error": f"Skill already exists: {dest}"})
        for name in names:
            if name.startswith("/") or ".." in Path(name).parts:
                return json.dumps({"error": f"Zip entry contains unsafe path: {name}"})
        zf.extractall(dest)
    return json.dumps({"success": True, "installed": str(dest)})


async def _reload_skills_from_manager() -> None:
    """Reload skills (used by install functions that don't have container access)."""
    try:
        from src._container import get_container

        container = get_container()
        await _reload_skills(container)
    except Exception as e:
        logger.warning("Failed to reload skills after install: %s", e)
