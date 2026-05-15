"""Skill manager for agent-driven skill creation, editing, and deletion.

Allows the agent to create new skills (SKILL.md + directory structure),
edit existing skills, and manage supporting files (references, templates, etc.).
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.skills.loader import load_skill
from src.skills.types import Skill, SkillMetadata

logger = logging.getLogger("myclaw.skills.manager")

_USER_SKILLS_DIR = Path.home() / ".myclaw" / "skills"

# Validation limits
_MAX_NAME_LENGTH = 64
_MAX_DESCRIPTION_LENGTH = 1024
_MAX_SKILL_CONTENT_BYTES = 256 * 1024

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
    lines = [
        "---",
        f"name: {name}",
        f'description: "{description}"',
    ]
    if category:
        lines.append(f"category: {category}")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


class SkillManager:
    """Manages skill CRUD operations."""

    def __init__(self, skills_dir: Path = _USER_SKILLS_DIR):
        self.skills_dir = skills_dir
        self.skills_dir.mkdir(parents=True, exist_ok=True)

    def create_skill(
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
        if skill_dir.exists():
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
        skill_md.write_text(frontmatter + content, encoding="utf-8")

        # 加载并返回
        skill = load_skill(skill_dir, "user")
        if skill:
            # 记录创建信息
            self._record_usage(name, created_by=created_by)
            return skill, None
        return None, "Failed to load created skill"

    def edit_skill(
        self,
        name: str,
        content: str,
        description: Optional[str] = None,
    ) -> tuple[Skill, Optional[str]]:
        """编辑现有技能。
        
        Returns:
            (Skill, error_message) - error_message is None on success
        """
        skill_dir = self.skills_dir / name
        skill_md = skill_dir / "SKILL.md"

        if not skill_md.exists():
            return None, f"Skill not found: {name}"

        # 安全检查
        injection_warning = _scan_for_injection(content)
        if injection_warning:
            logger.warning("Injection warning for skill edit '%s': %s", name, injection_warning)

        # 读取现有 frontmatter
        existing = load_skill(skill_dir, "user")
        if not existing:
            return None, "Failed to load existing skill"

        # 更新内容
        desc = description or existing.description
        frontmatter = _format_skill_frontmatter(name, desc, "")
        skill_md.write_text(frontmatter + content, encoding="utf-8")

        # 重新加载
        skill = load_skill(skill_dir, "user")
        if skill:
            self._record_usage(name, action="edited")
            return skill, None
        return None, "Failed to reload edited skill"

    def delete_skill(self, name: str) -> tuple[bool, Optional[str]]:
        """删除技能（仅用户创建的）。
        
        Returns:
            (success, error_message)
        """
        skill_dir = self.skills_dir / name

        if not skill_dir.exists():
            return False, f"Skill not found: {name}"

        # 检查来源（只允许删除用户技能）
        skill = load_skill(skill_dir, "user")
        if skill and skill.source not in ("user", "agents-project"):
            return False, f"Cannot delete system skill: {name}"

        import shutil
        try:
            shutil.rmtree(skill_dir)
            self._record_usage(name, action="deleted")
            return True, None
        except Exception as e:
            return False, f"Failed to delete skill: {str(e)}"

    def list_skills(self) -> list[Skill]:
        """列出所有用户技能。"""
        from src.skills.loader import discover_skills

        if not self.skills_dir.exists():
            return []

        skills = discover_skills([("user", self.skills_dir)])
        return skills

    def _record_usage(self, skill_name: str, action: str = "created", created_by: str = "agent") -> None:
        """记录技能使用信息到 .usage.json。"""
        usage_file = self.skills_dir / ".usage.json"
        usage_data = {}

        if usage_file.exists():
            try:
                usage_data = json.loads(usage_file.read_text(encoding="utf-8"))
            except Exception:
                usage_data = {}

        if skill_name not in usage_data:
            usage_data[skill_name] = {
                "use_count": 0,
                "view_count": 0,
                "patch_count": 0,
                "last_used_at": None,
                "last_viewed_at": None,
                "last_patched_at": None,
                "created_by": created_by,
                "state": "active",
                "pinned": False,
                "created_at": datetime.now().isoformat(),
                "archived_at": None,
            }

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

        try:
            usage_file.write_text(json.dumps(usage_data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to save skill usage: %s", e)

    def get_usage(self, skill_name: str) -> Optional[dict]:
        """获取技能使用统计。"""
        usage_file = self.skills_dir / ".usage.json"
        if not usage_file.exists():
            return None

        try:
            usage_data = json.loads(usage_file.read_text(encoding="utf-8"))
            return usage_data.get(skill_name)
        except Exception:
            return None


def get_tools() -> list:
    """返回技能管理工具。"""
    from src.agent.tooldef import ToolDef

    manager = SkillManager()

    async def skill_manage(
        action: str,
        name: str = "",
        content: str = "",
        description: str = "",
        category: str = "",
    ) -> str:
        """管理技能（创建/编辑/删除/查看）。
        
        Args:
            action: 操作类型 (create, edit, delete, list, usage)
            name: 技能名称
            content: 技能内容
            description: 技能描述
            category: 技能分类
        """
        if action == "create":
            if not name or not content:
                return json.dumps({"error": "name and content are required for create"})
            skill, error = manager.create_skill(name, description or name, content, category)
            if error:
                return json.dumps({"error": error})
            return json.dumps({
                "success": True,
                "action": "created",
                "skill": {
                    "name": skill.name,
                    "description": skill.description,
                    "source": skill.source,
                },
            }, ensure_ascii=False)

        elif action == "edit":
            if not name or not content:
                return json.dumps({"error": "name and content are required for edit"})
            skill, error = manager.edit_skill(name, content, description or None)
            if error:
                return json.dumps({"error": error})
            return json.dumps({
                "success": True,
                "action": "edited",
                "skill": {
                    "name": skill.name,
                    "description": skill.description,
                },
            }, ensure_ascii=False)

        elif action == "delete":
            if not name:
                return json.dumps({"error": "name is required for delete"})
            success, error = manager.delete_skill(name)
            if error:
                return json.dumps({"error": error})
            return json.dumps({"success": True, "action": "deleted", "skill": name})

        elif action == "list":
            skills = manager.list_skills()
            return json.dumps({
                "skills": [
                    {
                        "name": s.name,
                        "description": s.description,
                        "source": s.source,
                    }
                    for s in skills
                ]
            }, ensure_ascii=False)

        elif action == "usage":
            if not name:
                return json.dumps({"error": "name is required for usage"})
            usage = manager.get_usage(name)
            if usage:
                return json.dumps({"skill": name, "usage": usage})
            return json.dumps({"skill": name, "usage": None})

        else:
            return json.dumps({"error": f"Unknown action: {action}"})

    return [
        ToolDef.from_function(skill_manage),
    ]
