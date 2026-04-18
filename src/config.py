from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field


def _env_substitute(value: str) -> str:
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        env_key = value[2:-1]
        return os.environ.get(env_key, "")
    return value


def _substitute_recursive(obj):
    if isinstance(obj, str):
        return _env_substitute(obj)
    if isinstance(obj, dict):
        return {k: _substitute_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_substitute_recursive(v) for v in obj]
    return obj


class GatewayConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 18080
    auth_token: str = ""


class ModelConfig(BaseModel):
    provider: str = "anthropic"
    name: str = "claude-sonnet-4-6"
    temperature: float = 0.0
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    fallbacks: list[ModelFallback] = Field(default_factory=list)


class ModelFallback(BaseModel):
    provider: str
    name: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None


class AgentSubconfig(BaseModel):
    system_prompt: str = ""
    tools: list[str] = Field(default_factory=list)
    description: str = ""
    model: Optional[str] = None


class AgentConfig(BaseModel):
    system_prompt: str = "You are a helpful AI assistant."
    workspace: str = "."
    max_tool_rounds: int = 15
    subagents: dict[str, AgentSubconfig] = Field(default_factory=dict)


class FeishuConfig(BaseModel):
    enabled: bool = False
    app_id: str = ""
    app_secret: str = ""
    domain: Literal["feishu", "lark"] = "feishu"
    dm_policy: Literal["open", "pairing", "allowlist"] = "open"
    group_policy: Literal["open", "allowlist", "disabled"] = "allowlist"
    allow_from: list[str] = Field(default_factory=list)
    group_allow_from: list[str] = Field(default_factory=list)
    require_mention: bool = True
    streaming: bool = True
    typing_indicator: bool = True


class ChannelsConfig(BaseModel):
    feishu: FeishuConfig = Field(default_factory=FeishuConfig)


class SessionConfig(BaseModel):
    scope: Literal["per_sender", "global"] = "per_sender"
    idle_reset_minutes: int = 120


class ExecToolConfig(BaseModel):
    enabled: bool = True
    timeout_seconds: int = 30
    require_approval: bool = False
    approval_mode: Literal["off", "ask", "on_denylist_miss", "always"] = "off"
    deny_patterns: list[str] = []
    max_output_bytes: int = 102400
    max_concurrent: int = 3
    audit_log: bool = True


class WebSearchToolConfig(BaseModel):
    enabled: bool = False
    api_key: str = ""


class WebFetchToolConfig(BaseModel):
    enabled: bool = False


class FeishuToolConfig(BaseModel):
    doc: bool = True
    chat: bool = True


class ToolsPolicyConfig(BaseModel):
    allow: list[str] = Field(default_factory=lambda: ["*"])
    deny: list[str] = Field(default_factory=list)
    owner_only: list[str] = Field(default_factory=list)


class SecurityConfig(BaseModel):
    enabled: bool = True
    audit_on_startup: bool = True


class LinkUnderstandingConfig(BaseModel):
    enabled: bool = True
    max_previews: int = 3


class ToolsConfig(BaseModel):
    exec: ExecToolConfig = Field(default_factory=ExecToolConfig)
    web_search: WebSearchToolConfig = Field(default_factory=WebSearchToolConfig)
    web_fetch: WebFetchToolConfig = Field(default_factory=WebFetchToolConfig)
    feishu: FeishuToolConfig = Field(default_factory=FeishuToolConfig)
    policy: ToolsPolicyConfig = Field(default_factory=ToolsPolicyConfig)


class CheckpointerConfig(BaseModel):
    type: Literal["sqlite", "memory"] = "sqlite"
    path: str = "data/checkpoints.db"


class CronConfig(BaseModel):
    enabled: bool = True
    max_concurrent_runs: int = 1
    store_path: str = "data/cron.db"
    failure_alert_after: int = 2
    max_transient_retries: int = 3


class SkillsConfig(BaseModel):
    enabled: bool = True
    extra_dirs: list[str] = Field(default_factory=list)
    budget_chars: int = 30000
    watch: bool = True


class PluginsConfig(BaseModel):
    enabled: bool = True
    extra_dirs: list[str] = Field(default_factory=list)


class TimeoutsConfig(BaseModel):
    """Global timeout settings for various operations."""
    tool_short: int = 30  # Short-lived tool operations (e.g., simple commands)
    tool_long: int = 600  # Long-running tool operations (e.g., complex builds)
    session_idle: int = 3600  # Session idle timeout in seconds


class AppConfig(BaseModel):
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    agents: AgentConfig = Field(default_factory=AgentConfig)
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    checkpointer: CheckpointerConfig = Field(default_factory=CheckpointerConfig)
    cron: CronConfig = Field(default_factory=CronConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    plugins: PluginsConfig = Field(default_factory=PluginsConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    link_understanding: LinkUnderstandingConfig = Field(default_factory=LinkUnderstandingConfig)
    timeouts: TimeoutsConfig = Field(default_factory=TimeoutsConfig)
    owner_id: str = ""


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    p = Path(path)
    if not p.exists():
        return AppConfig()
    try:
        with open(p, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not raw:
            return AppConfig()
        substituted = _substitute_recursive(raw)
        return AppConfig(**substituted)
    except Exception as e:
        logger = logging.getLogger("myclaw.config")
        logger.error("Failed to load config from %s: %s", path, e)
        return AppConfig()
