from __future__ import annotations

import logging
from pathlib import Path

from .types import Skill, SkillCommandSpec

logger = logging.getLogger("myclaw.skills.prompt")

_DEFAULT_BUDGET = 30000


def format_skills_full(skills: list[Skill]) -> str:
    if not skills:
        return ""
    parts = ["<skills>"]
    for s in skills:
        if s.metadata.disable_model_invocation:
            continue
        parts.append(f"  <skill>")
        parts.append(f"    <name>{_esc(s.name)}</name>")
        parts.append(f"    <description>{_esc(s.description)}</description>")
        parts.append(f"    <path>{_esc(str(s.file_path))}</path>")
        parts.append(f"  </skill>")
    parts.append("</skills>")
    parts.append("")
    parts.append("To use a skill, read its SKILL.md file with the appropriate tool to load full instructions.")
    return "\n".join(parts)


def format_skills_compact(skills: list[Skill]) -> str:
    if not skills:
        return ""
    parts = ["<skills> (compact)"]
    for s in skills:
        if s.metadata.disable_model_invocation:
            continue
        p = _compact_path(s.file_path)
        parts.append(f"  {_esc(s.name)}: {_esc(p)}")
    parts.append("")
    parts.append("To use a skill, read its SKILL.md file with the appropriate tool to load full instructions.")
    return "\n".join(parts)


def format_skills_truncated(skills: list[Skill], budget: int) -> str:
    lo, hi = 1, len(skills)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        text = format_skills_compact(skills[:mid])
        if len(text) <= budget:
            lo = mid
        else:
            hi = mid - 1
    text = format_skills_compact(skills[:lo])
    text += f"\n(included {lo} of {len(skills)} skills, compact format)"
    return text


def build_skills_prompt(skills: list[Skill], budget: int = _DEFAULT_BUDGET) -> str:
    if not skills:
        return ""

    active = [s for s in skills if not s.metadata.disable_model_invocation]
    if not active:
        return ""

    full = format_skills_full(active)
    if len(full) <= budget:
        return full

    compact = format_skills_compact(active)
    if len(compact) <= budget:
        return compact

    return format_skills_truncated(active, budget)


def build_skill_commands(skills: list[Skill]) -> list[SkillCommandSpec]:
    specs = []
    for s in skills:
        if not s.metadata.user_invocable:
            continue
        cmd_name = _sanitize_command_name(s.name)
        desc = (s.description or "")[:100]
        specs.append(
            SkillCommandSpec(
                name=cmd_name,
                skill_name=s.name,
                description=desc,
                dispatch_tool=s.metadata.command_tool if s.metadata.command_dispatch == "tool" else None,
                arg_mode=s.metadata.command_arg_mode,
            )
        )
    return specs


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _compact_path(p: Path) -> str:
    s = str(p)
    home = str(Path.home())
    if s.startswith(home):
        s = "~" + s[len(home) :]
    return s


def _sanitize_command_name(name: str) -> str:
    cleaned = name.lower().strip()
    cleaned = "".join(c if c.isalnum() or c == "_" else "_" for c in cleaned)
    cleaned = cleaned[:32]
    return cleaned
