from __future__ import annotations

import json
import logging
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from src._container import get_container
from src.agent.tooldef import ToolDef
from src.config import save_config
from src.skills.loader import discover_skills, load_skill
from src.skills.prompt import build_skills_prompt

logger = logging.getLogger("myclaw.skills_tools")

_USER_SKILLS_DIR = Path.home() / ".myclaw" / "skills"


def _get_container():
    return get_container()


def _reload_all():
    """Reload skills and update all dependent components."""
    container = _get_container()
    dirs = container._build_skill_directories()
    skills = discover_skills(dirs, container.config)
    container.skills_cache = skills
    container.agent_loop._skills_prompt = build_skills_prompt(skills)
    dispatcher = getattr(container, 'dispatcher', None)
    if dispatcher is not None:
        dispatcher._reload_skills(skills)


def skills_list_tool() -> ToolDef:
    def handler(args: dict) -> str:
        container = _get_container()
        skills = container.skills_cache or []
        config = container.config

        result = []
        for s in skills:
            is_disabled = s.name in config.skills.disabled
            channel_disabled = {}
            for ch, chans in config.skills.channel_disabled.items():
                if s.name in chans:
                    channel_disabled[ch] = True

            result.append(
                {
                    "name": s.name,
                    "description": s.description,
                    "source": s.source,
                    "user_invocable": s.metadata.user_invocable,
                    "disable_model_invocation": s.metadata.disable_model_invocation,
                    "disabled": is_disabled,
                    "channel_disabled": channel_disabled,
                }
            )

        return json.dumps({"skills": result}, ensure_ascii=False, indent=2)

    return ToolDef(
        name="skills_list",
        description="List all available skills with metadata and enabled/disabled state",
        parameters={"type": "object", "properties": {}},
        fn=handler,
    )


def skill_view_tool() -> ToolDef:
    def handler(args: dict) -> str:
        skill_name = args.get("skill_name")
        if not skill_name:
            return json.dumps({"error": "skill_name is required"})

        container = _get_container()
        skills = container.skills_cache or []

        for s in skills:
            if s.name == skill_name:
                return json.dumps(
                    {
                        "name": s.name,
                        "description": s.description,
                        "source": s.source,
                        "file_path": str(s.file_path),
                        "body": s.body,
                    },
                    ensure_ascii=False,
                    indent=2,
                )

        return json.dumps({"error": f"Skill not found: {skill_name}"})

    return ToolDef(
        name="skill_view",
        description="View full content of a specific skill",
        parameters={
            "type": "object",
            "properties": {
                "skill_name": {"type": "string", "description": "Name of the skill to view"},
            },
            "required": ["skill_name"],
        },
        fn=handler,
    )


def skill_toggle_tool() -> ToolDef:
    def handler(args: dict) -> str:
        skill_name = args.get("skill_name")
        enabled = args.get("enabled", True)
        channel = args.get("channel")

        if not skill_name:
            return json.dumps({"error": "skill_name is required"})

        container = _get_container()
        skills = container.skills_cache or []
        if not any(s.name == skill_name for s in skills):
            return json.dumps({"error": f"Skill not found: {skill_name}"})

        config = container.config
        action = "no change"

        if channel:
            if channel not in config.skills.channel_disabled:
                config.skills.channel_disabled[channel] = []

            disabled_list = config.skills.channel_disabled[channel]
            if enabled and skill_name in disabled_list:
                disabled_list.remove(skill_name)
                action = "enabled"
            elif not enabled and skill_name not in disabled_list:
                disabled_list.append(skill_name)
                action = "disabled"
        else:
            if enabled and skill_name in config.skills.disabled:
                config.skills.disabled.remove(skill_name)
                action = "enabled"
            elif not enabled and skill_name not in config.skills.disabled:
                config.skills.disabled.append(skill_name)
                action = "disabled"

        try:
            save_config(config)
            _reload_all()
        except Exception as e:
            return json.dumps({"error": f"Failed to save config: {str(e)}"})

        return json.dumps(
            {
                "success": True,
                "skill": skill_name,
                "action": action,
                "channel": channel or "global",
            }
        )

    return ToolDef(
        name="skill_toggle",
        description="Enable or disable a skill globally or for a specific channel",
        parameters={
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "Name of the skill to toggle",
                },
                "enabled": {
                    "type": "boolean",
                    "description": "True to enable, false to disable",
                },
                "channel": {
                    "type": "string",
                    "description": "Channel name (feishu, qq) or omit for global",
                },
            },
            "required": ["skill_name", "enabled"],
        },
        fn=handler,
    )


def skill_install_tool() -> ToolDef:
    def handler(args: dict) -> str:
        source = args.get("source")
        if not source:
            return json.dumps({"error": "source (URL or path) is required"})

        _USER_SKILLS_DIR.mkdir(parents=True, exist_ok=True)

        try:
            if source.startswith(("http://", "https://")):
                return _install_from_url(source)
            else:
                return _install_from_path(source)
        except Exception as e:
            logger.error("Skill install failed: %s", e)
            return json.dumps({"error": f"Install failed: {str(e)}"})

    return ToolDef(
        name="skill_install",
        description="Install a skill from URL or local path",
        parameters={
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "URL to skill zip file or local path to skill directory/zip",
                },
            },
            "required": ["source"],
        },
        fn=handler,
    )


def _install_from_url(url: str) -> str:
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
        _reload_all()
        return json.dumps({"success": True, "installed": str(dest)})

    elif p.suffix == ".zip":
        return _install_from_zip(p)
    else:
        return json.dumps({"error": "Path must be a directory or .zip file"})


def _install_from_zip(zip_path: Path) -> str:
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
        _reload_all()
        return json.dumps({"success": True, "installed": str(dest)})


def skill_uninstall_tool() -> ToolDef:
    def handler(args: dict) -> str:
        skill_name = args.get("skill_name")
        if not skill_name:
            return json.dumps({"error": "skill_name is required"})

        container = _get_container()
        skills = container.skills_cache or []

        skill_dir = None
        for s in skills:
            if s.name == skill_name:
                skill_dir = s.base_dir
                break

        if not skill_dir:
            return json.dumps({"error": f"Skill not found: {skill_name}"})

        if not str(skill_dir).startswith(str(_USER_SKILLS_DIR)):
            return json.dumps(
                {"error": f"Can only uninstall user skills in {_USER_SKILLS_DIR}"}
            )

        try:
            shutil.rmtree(skill_dir)
        except Exception as e:
            return json.dumps({"error": f"Failed to remove skill directory: {str(e)}"})

        config = container.config
        if skill_name in config.skills.disabled:
            config.skills.disabled.remove(skill_name)
        for ch in config.skills.channel_disabled:
            if skill_name in config.skills.channel_disabled[ch]:
                config.skills.channel_disabled[ch].remove(skill_name)

        try:
            save_config(config)
            _reload_all()
        except Exception as e:
            return json.dumps({"error": f"Failed to save config: {str(e)}"})

        return json.dumps({"success": True, "uninstalled": skill_name})

    return ToolDef(
        name="skill_uninstall",
        description="Uninstall a user skill from ~/.myclaw/skills/",
        parameters={
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "Name of the skill to uninstall",
                },
            },
            "required": ["skill_name"],
        },
        fn=handler,
    )


def get_tools() -> list[ToolDef]:
    return [
        skills_list_tool(),
        skill_view_tool(),
        skill_toggle_tool(),
        skill_install_tool(),
        skill_uninstall_tool(),
    ]
