from __future__ import annotations

import logging

from src._container import get_container
from src.skills.loader import discover_skills
from src.skills.prompt import build_skills_prompt

logger = logging.getLogger("myclaw.skills_tools")


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


def get_tools() -> list:
    """All skill management tools are now provided by src.skills.manager."""
    return []
