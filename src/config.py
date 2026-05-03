from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Literal, Optional

import yaml

from src.mcp.config_models import MCPConfig
from pydantic import BaseModel, ValidationError, Field

_log = logging.getLogger("myclaw.config")

_ENV_VAR_RE = re.compile(r"\$\{([^}]+)\}")


def _env_substitute(value: str) -> str:
    if not isinstance(value, str):
        return value

    def _replace(m):
        env_key = m.group(1)
        result = os.environ.get(env_key, "")
        if env_key not in os.environ:
            _log.warning("Environment variable '%s' is not set, using empty string", env_key)
        return result

    return _ENV_VAR_RE.sub(_replace, value)


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
    subagent_max_depth: int = 2  # Max nesting depth for sub-agent calls


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


class QQConfig(BaseModel):
    enabled: bool = False
    app_id: str = ""
    client_secret: str = ""
    dm_policy: Literal["open", "allowlist"] = "open"
    group_policy: Literal["open", "allowlist", "disabled"] = "allowlist"
    allow_from: list[str] = Field(default_factory=list)
    group_allow_from: list[str] = Field(default_factory=list)
    require_mention: bool = True
    markdown_support: bool = False


class ChannelsConfig(BaseModel):
    feishu: FeishuConfig = Field(default_factory=FeishuConfig)
    qq: QQConfig = Field(default_factory=QQConfig)


class SessionConfig(BaseModel):
    scope: Literal["per_sender", "global"] = "per_sender"
    idle_reset_minutes: int = 120


class ExecToolConfig(BaseModel):
    enabled: bool = True
    timeout_seconds: int = 30
    no_output_timeout_seconds: int = 60  # Kill process if no output for N seconds, 0=disabled
    require_approval: bool = False
    approval_mode: Literal["off", "ask", "on_denylist_miss", "always"] = "off"
    deny_patterns: list[str] = []
    max_output_bytes: int = 102400
    max_concurrent: int = 3
    audit_log: bool = True
    sandbox_enabled: bool = True
    sandbox_allowed_dirs: list[str] = Field(default_factory=lambda: ["."])  # Working dirs allowed (relative to workspace)
    sandbox_env_whitelist: list[str] = Field(default_factory=lambda: ["PATH", "HOME", "USERPROFILE", "SYSTEMROOT", "LANG", "PYTHONPATH"])  # Env vars allowed to pass through


class WebSearchToolConfig(BaseModel):
    enabled: bool = False
    api_key: str = ""


class WebFetchToolConfig(BaseModel):
    enabled: bool = False


class FeishuToolConfig(BaseModel):
    doc: bool = True
    chat: bool = True


class MediaUnderstandingCapabilityConfig(BaseModel):
    """Config for a single media capability (image/audio/video)."""
    enabled: bool = True
    provider: str = ""       # Empty = inherit from parent
    name: str = ""           # Empty = inherit from parent
    base_url: str = ""       # Empty = inherit from parent
    api_key: str = ""        # Empty = inherit from parent


class MediaUnderstandingConfig(BaseModel):
    """Media understanding (image description, audio transcription, video description)."""
    enabled: bool = False
    provider: str = "openai"
    name: str = ""                    # Empty = auto-detect (gpt-4o-mini for openai)
    base_url: str = ""                # Empty = use provider default
    api_key: str = ""                 # Empty = use main model api_key
    max_image_size: int = 20 * 1024 * 1024   # 20MB
    max_audio_size: int = 25 * 1024 * 1024   # 25MB
    max_video_size: int = 50 * 1024 * 1024   # 50MB
    timeout_seconds: int = 60
    image: MediaUnderstandingCapabilityConfig = Field(default_factory=MediaUnderstandingCapabilityConfig)
    audio: MediaUnderstandingCapabilityConfig = Field(default_factory=MediaUnderstandingCapabilityConfig)
    video: MediaUnderstandingCapabilityConfig = Field(default_factory=MediaUnderstandingCapabilityConfig)


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
    media_understanding: MediaUnderstandingConfig = Field(default_factory=MediaUnderstandingConfig)


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


class TtsConfig(BaseModel):
    enabled: bool = False
    provider: str = "openai"  # "openai" | "elevenlabs" | "azure"
    model: str = "tts-1"
    voice: str = "alloy"
    auto_mode: Literal["off", "always", "tagged"] = "tagged"
    max_chars: int = 2000
    api_key: str = ""      # Empty = use main model api_key
    base_url: str = ""     # Empty = use provider default; for Azure, set to region (e.g. "eastus")


class MemoryConfig(BaseModel):
    enabled: bool = False
    db_path: str = "data/memory.db"
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    chunk_tokens: int = 400
    chunk_overlap: int = 80
    vector_weight: float = 0.7  # 70% vector, 30% BM25
    min_score: float = 0.35
    max_results: int = 6
    fts_tokenizer: str = "unicode61"
    api_key: str = ""
    base_url: str = ""
    extra_paths: list[str] = Field(default_factory=list)
    watch: bool = True  # Auto-watch extra_paths for changes
    auto_session_memory: bool = False  # Auto-write Q&A pairs to memory


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
    tts: TtsConfig = Field(default_factory=TtsConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    timeouts: TimeoutsConfig = Field(default_factory=TimeoutsConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
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
    except ValidationError as e:
        _log.error("Config validation failed for %s: %s", path, e)
        print(f"[config] Validation error in {path}: {e}", file=sys.stderr)
        return AppConfig()
    except Exception as e:
        _log.error("Failed to load config from %s: %s", path, e)
        return AppConfig()
