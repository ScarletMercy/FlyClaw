from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Union

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


@dataclass
class SkillMeta:
    name: str
    description: str
    source: str
    identifier: str
    trust_level: str = "community"
    repo: Optional[str] = None
    path: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillBundle:
    name: str
    files: dict[str, Union[str, bytes]]
    source: str
    identifier: str
    trust_level: str = "community"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Finding:
    pattern_id: str
    severity: str
    category: str
    file: str
    line: int
    match: str
    description: str


@dataclass
class ScanResult:
    skill_name: str
    source: str
    trust_level: str
    verdict: str
    findings: list[Finding] = field(default_factory=list)
    scanned_at: str = ""
    summary: str = ""
