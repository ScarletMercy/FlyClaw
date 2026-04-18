from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel


class SkillMetadata(BaseModel):
    name: str
    description: str = ""
    user_invocable: bool = True
    disable_model_invocation: bool = False
    command_dispatch: Optional[Literal["tool"]] = None
    command_tool: Optional[str] = None
    command_arg_mode: Optional[str] = None


class Skill(BaseModel):
    name: str
    description: str
    file_path: Path
    base_dir: Path
    source: str
    metadata: SkillMetadata
    body: str

    model_config = {"arbitrary_types_allowed": True}


class SkillCommandSpec(BaseModel):
    name: str
    skill_name: str
    description: str
    dispatch_tool: Optional[str] = None
    arg_mode: Optional[str] = None
