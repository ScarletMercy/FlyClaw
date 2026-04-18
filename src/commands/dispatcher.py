from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from src.skills.types import Skill, SkillCommandSpec

logger = logging.getLogger("myclaw.commands")


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
            from src.graph import collect_tools

            # Use existing config or load a fresh one (isolated context)
            config = self._config
            if config is None:
                from src.config import load_config

                config = load_config()
            all_tools = collect_tools(config)
            tool_map = {t.name: t for t in all_tools}
            tool = tool_map.get(tool_name)
            if tool is None:
                return f"Tool not found: {tool_name}"
            result = await tool.ainvoke({"__arg1": args} if args else {})
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
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        system_text = skill.body or skill.description
        messages = [
            SystemMessage(content=system_text),
            HumanMessage(content=args or "(no arguments)"),
        ]
        try:
            from src.graph import create_model

            # Use existing config or load a fresh one (isolated context)
            config = self._config
            if config is None:
                from src.config import load_config

                config = load_config()
            model = create_model(config)
            response = await model.ainvoke(messages)
            if isinstance(response.content, str):
                return response.content
            return str(response.content)
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


_dispatcher: Optional[CommandDispatcher] = None


def get_dispatcher() -> Optional[CommandDispatcher]:
    return _dispatcher


def set_dispatcher(dispatcher: CommandDispatcher) -> None:
    global _dispatcher
    _dispatcher = dispatcher


def build_builtin_help(commands: list[dict]) -> str:
    lines = ["Available commands:"]
    for cmd in commands:
        if cmd.get("builtin"):
            lines.append(f"  /{cmd['name']} — {cmd['description']}")
        else:
            tool_info = f" (→ {cmd['dispatch_tool']})" if cmd.get("dispatch_tool") else ""
            lines.append(f"  /{cmd['name']} — {cmd['description']}{tool_info}")
    return "\n".join(lines)
