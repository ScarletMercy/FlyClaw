from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError


_log = logging.getLogger("flyclaw.config")

_ENV_VAR_RE = re.compile(r"\$\{([^}]+)\}")


_warned_env_vars: set[str] = set()


def _env_substitute(value: str) -> str:
    if not isinstance(value, str):
        return value

    def _replace(m):
        env_key = m.group(1)
        result = os.environ.get(env_key, "")
        if env_key not in os.environ and env_key not in _warned_env_vars:
            _warned_env_vars.add(env_key)
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
    provider: str = "openai"
    name: str = "gpt-4o"
    temperature: float = 1.0
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    context_window: int = 1000000
    fallbacks: list[ModelFallback] = Field(default_factory=list)


class ModelFallback(BaseModel):
    provider: str
    name: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    context_window: int = 200000


class AgentSubconfig(BaseModel):
    system_prompt: str = ""
    tools: list[str] = Field(default_factory=list)
    description: str = ""
    model: Optional[str] = None


class AgentConfig(BaseModel):
    system_prompt: str = "You are a helpful AI assistant."
    workspace: str = "~/.flyclaw/workspace"
    max_tool_rounds: int = 100
    subagents: dict[str, AgentSubconfig] = Field(default_factory=lambda: {
        "research": AgentSubconfig(
            system_prompt="You are a research specialist. Find and synthesize information thoroughly.",
            tools=["web_search", "web_fetch"],
            description="Research specialist - finds and synthesizes information",
        ),
        "coder": AgentSubconfig(
            system_prompt="You are a coding specialist. Write, analyze, and debug code.",
            tools=["exec_command"],
            description="Coding specialist - writes and analyzes code",
        ),
        "reviewer": AgentSubconfig(
            system_prompt="You are a critical reviewer. Analyze content, code, or proposals.",
            tools=["*"],
            description="Critical reviewer - analyzes and provides feedback",
        ),
    })
    subagent_max_depth: int = 2
    timezone: str = "Asia/Shanghai"
    language: Literal["zh", "en"] = "zh"
    tool_progress_notifications: bool = False
    busy_input_mode: Literal["interrupt", "queue", "steer"] = "interrupt"
    bootstrap_files: list[str] = Field(
        default_factory=lambda: ["AGENTS.md", "IDENTITY.md", "USER.md"]
    )


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


class WeixinConfig(BaseModel):
    enabled: bool = False
    account_id: str = ""
    token: str = ""
    base_url: str = "https://ilinkai.weixin.qq.com"
    cdn_base_url: str = "https://novac2c.cdn.weixin.qq.com/c2c"
    dm_policy: Literal["open", "allowlist", "disabled"] = "open"
    group_policy: Literal["open", "allowlist", "disabled"] = "disabled"
    allowed_users: list[str] = Field(default_factory=list)
    group_allowed_users: list[str] = Field(default_factory=list)
    split_multiline_messages: bool = False
    home_channel: str = ""


class ChannelsConfig(BaseModel):
    qq: QQConfig = Field(default_factory=QQConfig)
    weixin: WeixinConfig = Field(default_factory=WeixinConfig)


class SessionConfig(BaseModel):
    scope: Literal["per_sender", "global"] = "per_sender"
    idle_reset_minutes: int = 120
    # Session pruning configuration (opt-in, default off)
    auto_prune: bool = False
    retention_days: int = 90
    vacuum_after_prune: bool = True
    min_interval_hours: int = 24


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
    sandbox_allowed_dirs: list[str] = Field(
        default_factory=lambda: ["."]
    )  # Working dirs allowed (relative to workspace)
    sandbox_env_whitelist: list[str] = Field(
        default_factory=lambda: ["PATH", "HOME", "USERPROFILE", "SYSTEMROOT", "LANG", "PYTHONPATH"]
    )  # Env vars allowed to pass through


class WebSearchToolConfig(BaseModel):
    enabled: bool = False
    api_key: str = ""


class WebFetchToolConfig(BaseModel):
    enabled: bool = True


class FeishuToolConfig(BaseModel):
    doc: bool = True
    chat: bool = True


class MediaUnderstandingCapabilityConfig(BaseModel):
    """Config for a single media capability (image/audio)."""

    enabled: bool = True
    provider: str = ""  # Empty = inherit from parent
    name: str = ""  # Empty = inherit from parent
    base_url: str = ""  # Empty = inherit from parent
    api_key: str = ""  # Empty = inherit from parent


class MediaUnderstandingFallback(BaseModel):
    """Fallback model config for media understanding."""

    provider: str = "openai"
    name: str = ""
    base_url: str = ""
    api_key: str = ""


class MediaUnderstandingConfig(BaseModel):
    """Media understanding (image description, audio transcription, video description). Video uses the image model config."""

    enabled: bool = False
    provider: str = ""
    name: str = ""
    base_url: str = ""
    api_key: str = ""
    max_image_size: int = 20 * 1024 * 1024  # 20MB
    max_audio_size: int = 25 * 1024 * 1024  # 25MB
    max_video_size: int = 50 * 1024 * 1024  # 50MB
    timeout_seconds: int = 60
    fallbacks: list[MediaUnderstandingFallback] = Field(default_factory=list)
    image: MediaUnderstandingCapabilityConfig = Field(default_factory=MediaUnderstandingCapabilityConfig)
    audio: MediaUnderstandingCapabilityConfig = Field(default_factory=MediaUnderstandingCapabilityConfig)


class ToolsPolicyConfig(BaseModel):
    allow: list[str] = Field(default_factory=lambda: ["*"])
    deny: list[str] = Field(default_factory=list)
    owner_only: list[str] = Field(default_factory=list)


class SecurityConfig(BaseModel):
    enabled: bool = True
    audit_on_startup: bool = True
    allow_private_urls: bool = False


class LinkUnderstandingConfig(BaseModel):
    enabled: bool = True
    max_previews: int = 3


class BrowserConfig(BaseModel):
    enabled: bool = True
    headless: bool = True
    browser: Literal["chromium", "firefox", "webkit"] = "chromium"
    viewport_width: int = 1280
    viewport_height: int = 720
    timeout_seconds: int = 30
    max_sessions: int = 3
    stealth: bool = True
    cdp_url: str = ""
    user_data_dir: str = ""
    block_urls: list[str] = Field(default_factory=list)


class WindowsUseConfig(BaseModel):
    """Windows desktop automation via pyautogui."""

    enabled: bool = True
    screenshot_dir: str = ""
    default_timeout: float = 5.0
    ocr_lang: str = "ch"


class ToolsConfig(BaseModel):
    exec: ExecToolConfig = Field(default_factory=ExecToolConfig)
    web_search: WebSearchToolConfig = Field(default_factory=WebSearchToolConfig)
    web_fetch: WebFetchToolConfig = Field(default_factory=WebFetchToolConfig)
    feishu: FeishuToolConfig = Field(default_factory=FeishuToolConfig)
    policy: ToolsPolicyConfig = Field(default_factory=ToolsPolicyConfig)
    media_understanding: MediaUnderstandingConfig = Field(default_factory=MediaUnderstandingConfig)
    browser: BrowserConfig = Field(default_factory=BrowserConfig)
    windows_use: WindowsUseConfig = Field(default_factory=WindowsUseConfig)


class SnapshotConfig(BaseModel):
    """File snapshot/rollback configuration (shadow git store)."""
    enabled: bool = True
    store_path: str = "~/.flyclaw/data/snapshots"
    max_per_dir: int = 20
    max_file_size: int = 10_000_000  # 10MB


class VoiceConfig(BaseModel):
    """Voice mode configuration."""
    enabled: bool = False
    voice: str = "zh-CN-YunxiNeural"
    threshold: int = 20


class DelegationConfig(BaseModel):
    """Sub-agent delegation configuration."""
    enabled: bool = True
    max_concurrent: int = 3
    child_timeout_seconds: int = 600
    child_timeout_floor: int = 30
    max_iterations: int = 50
    blocked_tools: list[str] = Field(
        default_factory=lambda: ["delegate_task", "delegate_batch"]
    )


class CheckpointerConfig(BaseModel):
    type: Literal["sqlite", "memory"] = "sqlite"
    path: str = "~/.flyclaw/data/checkpoints.db"


class CronConfig(BaseModel):
    enabled: bool = True
    max_concurrent_runs: int = 1
    store_path: str = "~/.flyclaw/data/cron.db"
    failure_alert_after: int = 2
    max_transient_retries: int = 3


class HubConfig(BaseModel):
    enabled: bool = True
    cache_ttl_seconds: int = 3600
    guard_enabled: bool = True
    search_timeout_seconds: int = 30


class CuratorConfig(BaseModel):
    enabled: bool = True
    interval_hours: int = 168
    min_idle_hours: int = 2
    stale_after_days: int = 30
    archive_after_days: int = 90
    max_review_iterations: int = 16


class SkillsConfig(BaseModel):
    enabled: bool = True
    extra_dirs: list[str] = Field(default_factory=list)
    budget_chars: int = 30000
    disabled: list[str] = Field(default_factory=list)
    channel_disabled: dict[str, list[str]] = Field(default_factory=dict)
    hub: HubConfig = HubConfig()
    creation_nudge_interval: int = 10
    curator: CuratorConfig = CuratorConfig()


class PluginsConfig(BaseModel):
    enabled: bool = True
    extra_dirs: list[str] = Field(default_factory=list)


class MemoryConfig(BaseModel):
    enabled: bool = True
    backend: Literal["sqlite", "lancedb"] = "sqlite"
    db_path: str = "~/.flyclaw/data/memory.db"
    chunk_tokens: int = 400
    chunk_overlap: int = 80
    min_score: float = 0.35
    max_results: int = 6
    fts_tokenizer: str = "unicode61"
    api_key: str = ""
    extra_paths: list[str] = Field(default_factory=list)
    watch: bool = True
    auto_session_memory: bool = False


class MemoryStoreConfig(BaseModel):
    """Memory store configuration."""

    enabled: bool = False
    db_path: str = "~/.flyclaw/data/memories.db"
    workspace: str = ""
    memory_judge_model: str = ""
    memory_judge_base_url: str = ""
    memory_judge_api_key: str = ""


class TaskConfig(BaseModel):
    """Autonomous task mode configuration."""

    enabled: bool = False
    max_parallel: int = 3
    default_timeout: int = 7200
    db_path: str = "~/.flyclaw/data/task_runs.db"
    defer_minutes: int = 5


class AuthConfig(BaseModel):
    """Authentication and access control configuration."""

    enabled: bool = True
    default_role: Literal["owner", "admin", "user", "guest"] = "user"
    pairing_enabled: bool = True
    pairing_ttl_seconds: int = 300  # Pairing code validity
    db_path: str = "~/.flyclaw/data/auth.db"


class TimeoutsConfig(BaseModel):
    """Global timeout settings for various operations."""

    tool_short: int = 30  # Short-lived tool operations (e.g., simple commands)
    tool_long: int = 600  # Long-running tool operations (e.g., complex builds)
    session_idle: int = 3600  # Session idle timeout in seconds


class SessionSearchConfig(BaseModel):
    enabled: bool = True
    index_path: str = "~/.flyclaw/data/session_index.db"
    auto_sync: bool = True
    max_results: int = 10
    tool_content_max_chars: int = 500


class CompressionConfig(BaseModel):
    """Context compression configuration."""
    enabled: bool = True
    threshold_percent: float = 0.6
    tail_messages: int = 20
    max_summary_tokens: int = 2000


class CanvasConfig(BaseModel):
    enabled: bool = False
    root: str = ""
    live_reload: bool = True


class HookConfig(BaseModel):
    """User-defined event hook configuration."""
    event: str = ""
    handler: str = ""
    enabled: bool = True


class HooksConfig(BaseModel):
    """Event hooks configuration."""
    hooks: list[HookConfig] = Field(default_factory=list)


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
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    timeouts: TimeoutsConfig = Field(default_factory=TimeoutsConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    memory_store: MemoryStoreConfig = Field(default_factory=MemoryStoreConfig)
    task: TaskConfig = Field(default_factory=TaskConfig)
    session_search: SessionSearchConfig = Field(default_factory=SessionSearchConfig)
    compression: CompressionConfig = Field(default_factory=CompressionConfig)
    canvas: CanvasConfig = Field(default_factory=CanvasConfig)
    hooks: HooksConfig = Field(default_factory=HooksConfig)
    snapshot: SnapshotConfig = Field(default_factory=SnapshotConfig)
    delegation: DelegationConfig = Field(default_factory=DelegationConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)



def _expand_paths(config: AppConfig) -> AppConfig:
    """Expand ~ in all path fields to absolute paths."""
    # Checkpointer
    config.checkpointer.path = str(Path(config.checkpointer.path).expanduser().resolve())
    
    # Cron
    config.cron.store_path = str(Path(config.cron.store_path).expanduser().resolve())
    
    # Memory
    config.memory.db_path = str(Path(config.memory.db_path).expanduser().resolve())
    
    # Auth
    config.auth.db_path = str(Path(config.auth.db_path).expanduser().resolve())

    # Memory store
    config.memory_store.db_path = str(Path(config.memory_store.db_path).expanduser().resolve())

    # Task
    config.task.db_path = str(Path(config.task.db_path).expanduser().resolve())
    
    # Session search
    config.session_search.index_path = str(Path(config.session_search.index_path).expanduser().resolve())

    # Snapshot store
    config.snapshot.store_path = str(Path(config.snapshot.store_path).expanduser().resolve())

    return config


_FLYCLAW_CONFIG_DIR = Path.home() / ".flyclaw"
_DEFAULT_CONFIG_PATH = _FLYCLAW_CONFIG_DIR / "config.yaml"


def load_config(path: str | Path = None) -> AppConfig:
    if path is None:
        path = _DEFAULT_CONFIG_PATH
    p = Path(path)
    if not p.exists():
        return AppConfig()
    try:
        with open(p, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not raw:
            return AppConfig()
        substituted = _substitute_recursive(raw)
        if "beads" in substituted and "memory_store" not in substituted:
            substituted["memory_store"] = substituted.pop("beads")
        config = AppConfig(**substituted)
        return _expand_paths(config)
    except ValidationError as e:
        _log.warning("Config validation failed for %s, using defaults: %s", path, e)
        return AppConfig()
    except Exception as e:
        _log.warning("Failed to load config from %s, using defaults: %s", path, e)
        return AppConfig()


def save_config(config: AppConfig, path: str | Path = None) -> None:
    """Persist config to YAML file."""
    if path is None:
        path = _DEFAULT_CONFIG_PATH
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = config.model_dump(exclude_unset=False)
    with open(p, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    _log.info("Config saved to %s", p)
