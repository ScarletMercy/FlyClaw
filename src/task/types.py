from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional


class TaskCheckpoint(BaseModel):
    id: str = Field(default_factory=lambda: __import__("uuid").uuid4().hex[:8])
    at: str = ""
    prompt: str = ""
    status: str = "pending"
    cron_job_id: Optional[str] = None


class TaskRun(BaseModel):
    id: str = Field(default_factory=lambda: __import__("uuid").uuid4().hex[:12])
    goal: str = ""
    steps: list[str] = Field(default_factory=list)
    checkpoints: list[TaskCheckpoint] = Field(default_factory=list)
    current_step: int = 0
    status: str = "planning"
    chat_id: str = ""
    thread_id: str = ""
    sender_id: str = ""
    created_at: float = Field(default_factory=__import__("time").time)
    updated_at: float = Field(default_factory=__import__("time").time)
