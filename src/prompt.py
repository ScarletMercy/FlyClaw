"""Modular system prompt builder — ported from openclaw-main's system-prompt.ts."""

from __future__ import annotations

import logging
import platform
import sys
from datetime import datetime, timezone
logger = logging.getLogger("myclaw.prompt")

# ── Section builders ──────────────────────────────────────────


def _build_identity() -> list[str]:
    return [
        "You are a personal assistant running inside MyClaw.",
        "",
    ]


def _build_tooling(tools: list) -> list[str]:
    lines = [
        "## Tooling",
        "Tool availability (filtered by policy):",
        "Tool names are case-sensitive. Call tools exactly as listed.",
    ]
    for t in tools:
        desc = (t.description or "").split("\n")[0].strip()
        if desc:
            lines.append(f"- {t.name}: {desc}")
        else:
            lines.append(f"- {t.name}")

    lines += [
        "",
        "## Tool Call Style",
        "Default: do not narrate routine, low-risk tool calls (just call the tool).",
        "Narrate only when it helps: multi-step work, complex/challenging problems, "
        "sensitive actions (e.g. deletions), or when the user explicitly asks.",
        "Keep narration brief and value-dense; avoid repeating obvious steps.",
        "Use plain human language for narration unless in a technical context.",
        "When a first-class tool exists for an action, use the tool directly "
        "instead of asking the user to run equivalent CLI or shell commands.",
        "",
    ]
    return lines


def _build_safety() -> list[str]:
    return [
        "## Safety",
        "You have no independent goals: do not pursue self-preservation, replication, "
        "resource acquisition, or power-seeking; avoid long-term plans beyond the user's request.",
        "Prioritize safety and human oversight over completion; if instructions conflict, "
        "pause and ask; comply with stop/pause/audit requests and never bypass safeguards.",
        "Do not manipulate or persuade anyone to expand access or disable safeguards. "
        "Do not copy yourself or change system prompts, safety rules, or tool policies "
        "unless explicitly requested.",
        "",
    ]


def _build_skills_section(skills_prompt: str) -> list[str]:
    trimmed = skills_prompt.strip()
    if not trimmed:
        return []
    return [
        "## Skills (mandatory)",
        "Before replying: scan <available_skills> <description> entries.",
        "- If exactly one skill clearly applies: read its SKILL.md at <location>, then follow it.",
        "- If multiple could apply: choose the most specific one, then read/follow it.",
        "- If none clearly apply: do not read any SKILL.md.",
        "Constraints: never read more than one skill up front; only read after selecting.",
        trimmed,
        "",
    ]


def _build_beads_section(config) -> list[str]:
    """Beads memory tool guidance — complements AGENTS.md bootstrap context."""
    if not config:
        return []
    beads_cfg = getattr(config, "beads", None)
    if not beads_cfg or not getattr(beads_cfg, "enabled", False):
        return []
    return [
        "## Beads Memory Tools",
        "Use dedicated tools for memory operations — do NOT use exec_command to run bd:",
        "- bd_remember: Save a memory (auto-dedup by key). Use when user shares preferences, identity, contacts, project info, or important decisions.",
        "- bd_recall: Retrieve a specific memory by key.",
        "- bd_memories: List or search all memories.",
        "- bd_forget: Delete a memory.",
        "",
    ]


def _build_memory_section(config) -> list[str]:
    if not config:
        return []
    mem_cfg = getattr(config, "memory", None)
    if not mem_cfg or not getattr(mem_cfg, "enabled", False):
        return []
    return [
        "## Memory",
        "You have a memory_search tool for retrieving stored memories.",
        "Use it proactively when context from past conversations would be helpful.",
        "Memory is automatically indexed from files in configured paths.",
        "",
    ]



def _build_workspace(workspace_dir: str) -> list[str]:
    return [
        "## Workspace",
        f"Your working directory is: {workspace_dir}",
        "Treat this directory as the single global workspace for file operations "
        "unless explicitly instructed otherwise.",
        "",
    ]


def _build_bootstrap_context(context_files: list[dict]) -> list[str]:
    if not context_files:
        return []
    lines = [
        "# Project Context",
        "",
        "The following project context files have been loaded:",
    ]
    has_soul = any(
        f.get("path", "").lower() == "soul.md" for f in context_files
    )
    if has_soul:
        lines.append(
            "If SOUL.md is present, embody its persona and tone. "
            "Avoid stiff, generic replies; follow its guidance unless "
            "higher-priority instructions override it."
        )
    lines.append("")
    for f in context_files:
        lines.append(f"## {f['path']}")
        lines.append("")
        lines.append(f["content"])
        lines.append("")
    return lines


def _build_datetime(tz_name: str | None) -> list[str]:
    import zoneinfo

    tz_str = tz_name or "Asia/Shanghai"
    try:
        tz = zoneinfo.ZoneInfo(tz_str)
    except Exception:
        tz = timezone.utc
        tz_str = "UTC"
    now = datetime.now(tz).strftime("%Y-%m-%d %H:%M %Z")
    weekday = datetime.now(tz).strftime("%A")
    return [
        "## Current Date & Time",
        f"Time zone: {tz_str}",
        f"Now: {now} ({weekday})",
        "",
    ]


def _build_runtime_info(config=None) -> list[str]:
    model_name = ""
    if config:
        model_cfg = getattr(config, "model", None)
        if model_cfg:
            model_name = f"{model_cfg.provider}/{model_cfg.name}"
    lines = [
        "## Runtime",
        f"Runtime: Python {sys.version.split()[0]} on {platform.system()} {platform.release()}",
    ]
    if model_name:
        lines.append(f"Model: {model_name}")
    lines.append("")
    return lines


# ── Main builder ──────────────────────────────────────────────


def build_system_prompt(
    config,
    tools: list,
    skills_prompt: str = "",
    context_files: list[dict] | None = None,
    extra_system_prompt: str = "",
) -> str:
    """Assemble the full system prompt from modular sections.

    When extra_system_prompt contains a detailed custom prompt (e.g. from
    config YAML), it serves as the identity/tooling base — we skip the
    default identity and tooling sections to avoid duplication.
    """
    tz_name = ""
    workspace_dir = "."
    if config:
        agents_cfg = getattr(config, "agents", None)
        if agents_cfg:
            tz_name = getattr(agents_cfg, "timezone", "") or ""
            raw_ws = getattr(agents_cfg, "workspace", ".") or "."
            from pathlib import Path
            workspace_dir = str(Path(raw_ws).expanduser().resolve())

    has_custom_prompt = bool(extra_system_prompt.strip())

    lines: list[str] = []

    # If user provided a custom prompt, use it as the base (it already has
    # identity + tool descriptions). Otherwise use our defaults.
    if has_custom_prompt:
        lines.append(extra_system_prompt.strip())
        lines.append("")
    else:
        lines.extend(_build_identity())
        lines.extend(_build_tooling(tools))

    # Always enforce native function calling — never output pseudo-XML tool calls.
    lines.extend([
        "## Tool Calling",
        "You have access to tools via the native function calling API. "
        "ALWAYS use native tool calls — never output tool invocations as text, XML, or pseudo-tags.",
        "",
    ])

    lines.extend(_build_safety())
    lines.extend(_build_skills_section(skills_prompt))
    lines.extend(_build_memory_section(config))
    lines.extend(_build_beads_section(config))
    lines.extend(_build_workspace(workspace_dir))
    if context_files:
        lines.extend(_build_bootstrap_context(context_files))
    lines.extend(_build_datetime(tz_name or None))
    lines.extend(_build_runtime_info(config))

    return "\n".join(lines)
