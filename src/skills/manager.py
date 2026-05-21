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
        from src._container import get_container

        if not self.skills_dir.exists():
            return []

        container = get_container()
        skills = discover_skills([("user", self.skills_dir)], container.config)
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
    """返回统一的技能管理工具。"""
    from src.agent.tooldef import ToolDef

    manager = SkillManager()

    async def skill_manage(
        action: str,
        name: str = "",
        content: str = "",
        description: str = "",
        category: str = "",
        enabled: bool = True,
        channel: str = "",
        source: str = "",
        query: str = "",
        identifier: str = "",
    ) -> str:
        """Skill management tool. Must use this (not read_file) to load skills.

        Args:
            action: Operation type (list, view, create, edit, delete, usage, toggle, install, uninstall, search_hub, inspect_hub, install_hub, scan_hub)
                    action="view" loads the full instruction content of a specified skill
            name: Skill name (required for view)
            content: Skill content (needed for create/edit)
            description: Skill description
            category: Skill category
            enabled: Whether to enable (needed for toggle)
            channel: Channel name (optional for toggle, e.g. qq)
            source: Install source (needed for install; URL or local path)
            query: Search query (for search_hub)
            identifier: Skill identifier from search results (for inspect_hub, install_hub)
        """
        if action == "list":
            from src._container import get_container
            container = get_container()
            skills = container.skills_cache or []
            config = container.config
            result = []
            for s in skills:
                is_disabled = s.name in config.skills.disabled
                channel_disabled = {}
                for ch, chans in config.skills.channel_disabled.items():
                    if s.name in chans:
                        channel_disabled[ch] = True
                result.append({
                    "name": s.name,
                    "description": s.description,
                    "source": s.source,
                    "user_invocable": s.metadata.user_invocable,
                    "disable_model_invocation": s.metadata.disable_model_invocation,
                    "disabled": is_disabled,
                    "channel_disabled": channel_disabled,
                })
            return json.dumps({"skills": result}, ensure_ascii=False, indent=2)

        elif action == "view":
            if not name:
                return json.dumps({"error": "name is required for view"})
            from src._container import get_container
            container = get_container()
            skills = container.skills_cache or []
            for s in skills:
                if s.name == name:
                    return json.dumps({
                        "name": s.name,
                        "description": s.description,
                        "source": s.source,
                        "file_path": str(s.file_path),
                        "body": s.body,
                    }, ensure_ascii=False, indent=2)
            return json.dumps({"error": f"Skill not found: {name}"})

        elif action == "create":
            if not name or not content:
                return json.dumps({"error": "name and content are required for create"})
            skill, error = manager.create_skill(name, description or name, content, category)
            if error:
                return json.dumps({"error": error})
            return json.dumps({
                "success": True,
                "action": "created",
                "skill": {"name": skill.name, "description": skill.description, "source": skill.source},
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
                "skill": {"name": skill.name, "description": skill.description},
            }, ensure_ascii=False)

        elif action == "delete":
            if not name:
                return json.dumps({"error": "name is required for delete"})
            success, error = manager.delete_skill(name)
            if error:
                return json.dumps({"error": error})
            return json.dumps({"success": True, "action": "deleted", "skill": name})

        elif action == "usage":
            if not name:
                return json.dumps({"error": "name is required for usage"})
            usage = manager.get_usage(name)
            if usage:
                return json.dumps({"skill": name, "usage": usage})
            return json.dumps({"skill": name, "usage": None})

        elif action == "toggle":
            if not name:
                return json.dumps({"error": "name is required for toggle"})
            from src._container import get_container
            from src.config import save_config
            container = get_container()
            skills = container.skills_cache or []
            if not any(s.name == name for s in skills):
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
                save_config(config)
                _reload_skills(container)
            except Exception as e:
                return json.dumps({"error": f"Failed to save config: {str(e)}"})
            return json.dumps({
                "success": True,
                "skill": name,
                "action": action_result,
                "channel": channel or "global",
            })

        elif action == "install":
            if not source:
                return json.dumps({"error": "source (URL or path) is required for install"})
            try:
                if source.startswith(("http://", "https://")):
                    return _install_from_url(source)
                else:
                    return _install_from_path(source)
            except Exception as e:
                logger.error("Skill install failed: %s", e)
                return json.dumps({"error": f"Install failed: {str(e)}"})

        elif action == "uninstall":
            if not name:
                return json.dumps({"error": "name is required for uninstall"})
            from src._container import get_container
            from src.config import save_config
            container = get_container()
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
                import shutil
                shutil.rmtree(skill_dir)
            except Exception as e:
                return json.dumps({"error": f"Failed to remove skill directory: {str(e)}"})
            config = container.config
            if name in config.skills.disabled:
                config.skills.disabled.remove(name)
            for ch in config.skills.channel_disabled:
                if name in config.skills.channel_disabled[ch]:
                    config.skills.channel_disabled[ch].remove(name)
            try:
                save_config(config)
                _reload_skills(container)
            except Exception as e:
                return json.dumps({"error": f"Failed to save config: {str(e)}"})
            try:
                from src.skills.hub import HubLockFile, append_audit_log
                lock = HubLockFile()
                entry = lock.get_installed(name)
                if entry:
                    lock.record_uninstall(name)
                    append_audit_log("UNINSTALL", name, entry["source"], entry["trust_level"], "n/a", "user_request")
            except Exception:
                pass
            return json.dumps({"success": True, "uninstalled": name})

        elif action == "search_hub":
            if not query:
                return json.dumps({"error": "query is required for search_hub"})
            try:
                from src.skills.hub import create_sources, parallel_search
                sources = create_sources()
                results = parallel_search(
                    sources, query, limit=20,
                    source_filter=source or "all",
                )
                items = []
                for r in results:
                    items.append({
                        "name": r.name,
                        "description": r.description[:200],
                        "source": r.source,
                        "identifier": r.identifier,
                        "trust_level": r.trust_level,
                    })
                return json.dumps({"results": items}, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error("Hub search failed: %s", e)
                return json.dumps({"error": f"Search failed: {str(e)}"})

        elif action == "inspect_hub":
            if not identifier:
                return json.dumps({"error": "identifier is required for inspect_hub"})
            try:
                from src.skills.hub import create_sources, resolve_source
                sources = create_sources()
                src = resolve_source(identifier, sources)
                if not src:
                    return json.dumps({"error": f"No source found for identifier: {identifier}"})
                meta = src.inspect(identifier)
                if not meta:
                    return json.dumps({"error": f"Skill not found: {identifier}"})
                return json.dumps({
                    "name": meta.name,
                    "description": meta.description,
                    "source": meta.source,
                    "identifier": meta.identifier,
                    "trust_level": meta.trust_level,
                    "repo": meta.repo,
                    "tags": meta.tags,
                }, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error("Hub inspect failed: %s", e)
                return json.dumps({"error": f"Inspect failed: {str(e)}"})

        elif action == "install_hub":
            if not identifier:
                return json.dumps({"error": "identifier is required for install_hub"})
            try:
                from src.skills.hub import (
                    create_sources, resolve_source,
                    quarantine_bundle, install_from_quarantine,
                    ensure_hub_dirs,
                )
                from src.skills.guard import scan_skill, should_allow_install, format_scan_report
                from src._container import get_container

                sources = create_sources()
                src = resolve_source(identifier, sources)
                if not src:
                    return json.dumps({"error": f"No source found for identifier: {identifier}"})
                bundle = src.fetch(identifier)
                if not bundle:
                    return json.dumps({"error": f"Failed to fetch skill: {identifier}"})
                if not bundle.name:
                    return json.dumps({"error": "Could not determine skill name from remote content"})

                ensure_hub_dirs()
                quarantine_path = quarantine_bundle(bundle)

                container = get_container()
                guard_enabled = getattr(container.config.skills.hub, "guard_enabled", True)

                if guard_enabled:
                    scan_result = scan_skill(quarantine_path, source=bundle.source)
                    allowed, reason = should_allow_install(scan_result)
                    if not allowed:
                        import shutil
                        shutil.rmtree(quarantine_path, ignore_errors=True)
                        report = format_scan_report(scan_result)
                        return json.dumps({
                            "error": f"Install blocked: {reason}",
                            "scan_report": report,
                        }, ensure_ascii=False)
                else:
                    from src.skills.types import ScanResult
                    scan_result = ScanResult(
                        skill_name=bundle.name, source=bundle.source,
                        trust_level=bundle.trust_level, verdict="safe",
                        scanned_at="", summary="Guard disabled",
                    )

                install_dir = install_from_quarantine(
                    quarantine_path, bundle.name, bundle, scan_result,
                )
                _reload_skills_from_manager()

                return json.dumps({
                    "success": True,
                    "action": "installed",
                    "skill": {
                        "name": bundle.name,
                        "source": bundle.source,
                        "trust_level": bundle.trust_level,
                        "scan_verdict": scan_result.verdict,
                        "install_path": str(install_dir),
                    },
                }, ensure_ascii=False)
            except Exception as e:
                logger.error("Hub install failed: %s", e)
                return json.dumps({"error": f"Install failed: {str(e)}"})

        elif action == "scan_hub":
            if not name:
                return json.dumps({"error": "name is required for scan_hub"})
            try:
                from src._container import get_container
                from src.skills.guard import scan_skill, format_scan_report
                container = get_container()
                skills = container.skills_cache or []
                skill_dir = None
                for s in skills:
                    if s.name == name:
                        skill_dir = s.base_dir
                        break
                if not skill_dir:
                    return json.dumps({"error": f"Skill not found: {name}"})
                result = scan_skill(skill_dir, source="community")
                report = format_scan_report(result)
                return json.dumps({
                    "name": name,
                    "verdict": result.verdict,
                    "findings_count": len(result.findings),
                    "report": report,
                }, ensure_ascii=False)
            except Exception as e:
                logger.error("Hub scan failed: %s", e)
                return json.dumps({"error": f"Scan failed: {str(e)}"})

        else:
            return json.dumps({"error": f"Unknown action: {action}. Valid actions: list, view, create, edit, delete, usage, toggle, install, uninstall, search_hub, inspect_hub, install_hub, scan_hub"})

    return [
        ToolDef.from_function(skill_manage),
    ]


def _reload_skills(container) -> None:
    """Reload skills and update all dependent components."""
    from src.skills.loader import discover_skills
    from src.skills.prompt import build_skills_prompt
    dirs = container._build_skill_directories()
    skills = discover_skills(dirs, container.config)
    container.skills_cache = skills
    container.agent_loop._skills_prompt = build_skills_prompt(skills)
    dispatcher = getattr(container, 'dispatcher', None)
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
        dest = _USER_SKILLS_DIR / p.name
        if dest.exists():
            return json.dumps({"error": f"Skill already exists: {dest}"})
        shutil.copytree(p, dest)
        _reload_skills_from_manager()
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
        dest = _USER_SKILLS_DIR / skill_dir_name
        if dest.exists():
            return json.dumps({"error": f"Skill already exists: {dest}"})
        zf.extractall(dest)
    _reload_skills_from_manager()
    return json.dumps({"success": True, "installed": str(dest)})


def _reload_skills_from_manager() -> None:
    """Reload skills (used by install functions that don't have container access)."""
    try:
        from src._container import get_container
        container = get_container()
        _reload_skills(container)
    except Exception:
        pass
