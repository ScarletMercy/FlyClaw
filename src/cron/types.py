from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


class CronSchedule(BaseModel):
    kind: Literal["at", "every", "cron"]
    at: Optional[str] = None
    every_seconds: Optional[int] = None
    expr: Optional[str] = None
    tz: Optional[str] = Field(default="Asia/Shanghai")

    @model_validator(mode="after")
    def validate_schedule_fields(self):
        if self.kind == "at" and not self.at:
            raise ValueError("Schedule kind 'at' requires 'at' field")
        if self.kind == "every" and self.every_seconds is None:
            raise ValueError("Schedule kind 'every' requires 'every_seconds' field")
        if self.kind == "cron" and not self.expr:
            raise ValueError("Schedule kind 'cron' requires 'expr' field")
        return self

    def to_apscheduler_trigger(self):
        if self.kind == "at":
            from apscheduler.triggers.date import DateTrigger
            import zoneinfo

            raw = self.at or ""
            # Try ISO format first (handles timezone-aware strings)
            try:
                dt = datetime.fromisoformat(raw)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=zoneinfo.ZoneInfo(self.tz or "Asia/Shanghai"))
                return DateTrigger(run_date=dt)
            except ValueError:
                pass
            # Fallback: try "YYYY-MM-DD HH:MM:SS" format
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
                try:
                    dt = datetime.strptime(raw, fmt)
                    dt = dt.replace(tzinfo=zoneinfo.ZoneInfo(self.tz or "Asia/Shanghai"))
                    return DateTrigger(run_date=dt)
                except ValueError:
                    continue
            raise ValueError(f"Cannot parse datetime: {raw}")
        elif self.kind == "every":
            from apscheduler.triggers.interval import IntervalTrigger

            return IntervalTrigger(seconds=self.every_seconds or 60, timezone=self.tz)
        elif self.kind == "cron":
            from apscheduler.triggers.cron import CronTrigger

            parts = (self.expr or "* * * * *").split()
            if len(parts) == 5:
                return CronTrigger(
                    minute=parts[0],
                    hour=parts[1],
                    day=parts[2],
                    month=parts[3],
                    day_of_week=parts[4],
                    timezone=self.tz,
                )
            return CronTrigger.from_crontab(self.expr or "* * * * *", timezone=self.tz)
        raise ValueError(f"Unknown schedule kind: {self.kind}")


class CronPayload(BaseModel):
    kind: Literal["system_event", "agent_turn"] = "agent_turn"
    text: Optional[str] = None
    message: Optional[str] = None
    model: Optional[str] = None
    tools_allow: Optional[list[str]] = None
    thinking: Optional[str] = None
    timeout_seconds: int = 600

    @model_validator(mode="after")
    def validate_payload_fields(self):
        if self.kind == "system_event" and not self.text:
            raise ValueError("Payload kind 'system_event' requires 'text' field")
        if self.kind == "agent_turn" and not self.message:
            raise ValueError("Payload kind 'agent_turn' requires 'message' field")
        return self


class CronDelivery(BaseModel):
    mode: Literal["none", "announce", "webhook"] = "none"
    channel: Optional[str] = None
    to: Optional[str] = None
    webhook_url: Optional[str] = None
    best_effort: bool = False

    @model_validator(mode="after")
    def validate_delivery_fields(self):
        if self.mode == "webhook" and not self.webhook_url:
            raise ValueError("Delivery mode 'webhook' requires 'webhook_url' field")
        if self.mode == "announce" and not (self.to or self.channel):
            raise ValueError("Delivery mode 'announce' requires 'to' or 'channel' field")
        return self


class CronJob(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str = ""
    enabled: bool = True
    delete_after_run: bool = False
    schedule: CronSchedule
    payload: CronPayload
    delivery: CronDelivery = CronDelivery()
    session_target: Literal["main", "isolated"] = "isolated"
    depends_on: list[str] = Field(default_factory=list)
    consecutive_errors: int = 0
    last_run_at: Optional[float] = None
    last_run_status: Optional[str] = None
    last_error: Optional[str] = None
    next_run_at: Optional[float] = None
    created_at: float = Field(default_factory=lambda: datetime.now().timestamp())
    schema_version: Literal["v1"] = "v1"
    version: int = 0


class CronJobCreate(BaseModel):
    name: str
    description: str = ""
    enabled: bool = True
    delete_after_run: bool = False
    schedule: CronSchedule
    payload: CronPayload
    delivery: CronDelivery = CronDelivery()
    session_target: Literal["main", "isolated"] = "isolated"
    depends_on: list[str] = Field(default_factory=list)


class CronJobPatch(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    enabled: Optional[bool] = None
    schedule: Optional[CronSchedule] = None
    payload: Optional[CronPayload] = None
    delivery: Optional[CronDelivery] = None
    session_target: Optional[Literal["main", "isolated"]] = None
    depends_on: Optional[list[str]] = None


class CronRunResult(BaseModel):
    job_id: str
    status: Literal["success", "error", "timeout"]
    output: Optional[str] = None
    error: Optional[str] = None
    started_at: float
    finished_at: Optional[float] = None
