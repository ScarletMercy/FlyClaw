from __future__ import annotations

import logging
from typing import Callable, Optional

from src.skills.types import Skill, SkillCommandSpec

logger = logging.getLogger("flyclaw.commands")


class CommandDispatcher:
    def __init__(self, skills: Optional[list[Skill]] = None, config=None):
        self._commands: dict[str, SkillCommandSpec] = {}
        self._skills: dict[str, Skill] = {}
        self._builtins: dict[str, Callable] = {}
        self._config = config
        if skills:
            self._register_skills(skills)

    def _register_skills(self, skills: list[Skill]) -> None:
        from src.skills.prompt import build_skill_commands

        for spec in build_skill_commands(skills):
            self._commands[spec.name] = spec
        for s in skills:
            self._skills[s.name] = s
            sanitized = _sanitize(s.name)
            if sanitized:
                self._skills[sanitized] = s
        logger.info("Registered %d skill commands", len(self._commands))

    def _reload_skills(self, skills: list[Skill]) -> None:
        """Reload skills after hot-reload, clearing old registry."""
        self._commands.clear()
        self._skills.clear()
        self._register_skills(skills)

    def register_builtin(self, name: str, handler: Callable) -> None:
        self._builtins[name] = handler

    def match(self, text: str) -> Optional[tuple[str, str]]:
        text = text.strip()
        if not text.startswith("/"):
            return None
        rest = text[1:].strip()
        if not rest:
            return None
        parts = rest.split(None, 1)
        command_name = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        if command_name in self._commands or command_name in self._builtins:
            return command_name, args
        return None

    async def dispatch(
        self,
        command_name: str,
        args: str,
        context: Optional[dict] = None,
    ) -> str:
        if command_name in self._builtins:
            handler = self._builtins[command_name]
            result = handler(args, context or {})
            if hasattr(result, "__await__"):
                result = await result
            return str(result)

        spec = self._commands.get(command_name)
        if spec is None:
            return f"Unknown command: /{command_name}"

        skill = self._skills.get(spec.skill_name)
        if skill is None:
            return f"Skill not found: {spec.skill_name}"

        if spec.dispatch_tool:
            return await self._dispatch_via_tool(spec, args, context)

        return await self._dispatch_via_skill_prompt(skill, args, context)

    async def _dispatch_via_tool(
        self,
        spec: SkillCommandSpec,
        args: str,
        context: Optional[dict],
    ) -> str:
        tool_name = spec.dispatch_tool
        try:
            from src.tools.registry import get_tool_registry

            config = self._config
            if config is None:
                from src.config import load_config

                config = load_config()
            all_tools = get_tool_registry().collect()
            tool_map = {t.name: t for t in all_tools}
            tool = tool_map.get(tool_name)
            if tool is None:
                return f"Tool not found: {tool_name}"
            result = await tool.execute({"__arg1": args} if args else {})
            return str(result)
        except Exception as e:
            logger.error("Tool dispatch failed for /%s: %s", spec.name, e)
            return f"[error] {type(e).__name__}: {e}"

    async def _dispatch_via_skill_prompt(
        self,
        skill: Skill,
        args: str,
        context: Optional[dict],
    ) -> str:
        system_text = skill.body or skill.description
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": args or "(no arguments)"},
        ]
        try:
            from src.agent.client import create_chain

            config = self._config
            if config is None:
                from src.config import load_config

                config = load_config()
            client = create_chain(config)
            return await client.chat_simple(messages)
        except Exception as e:
            logger.error("Skill prompt dispatch failed for %s: %s", skill.name, e)
            return f"[error] {type(e).__name__}: {e}"

    def list_commands(self) -> list[dict]:
        result = []
        for name, handler in self._builtins.items():
            result.append({"name": name, "description": "Built-in command", "builtin": True})
        for name, spec in self._commands.items():
            result.append(
                {
                    "name": f"/{name}",
                    "skill": spec.skill_name,
                    "description": spec.description,
                    "dispatch_tool": spec.dispatch_tool,
                    "builtin": False,
                }
            )
        return result


def _sanitize(name: str) -> str:
    cleaned = name.lower().strip()
    return "".join(c if c.isalnum() or c == "_" else "_" for c in cleaned)[:32]


# ── Module-level singleton — delegates to ServiceContainer ──

from src._container import get_container


def get_dispatcher() -> Optional[CommandDispatcher]:
    return get_container().dispatcher
