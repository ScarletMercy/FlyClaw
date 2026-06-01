from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar

from pydantic import BaseModel
from watchfiles import awatch

from src.config import load_config

_log = logging.getLogger("flyclaw.config_watcher")


@dataclass
class ConfigChange:
    path: str
    old_value: Any
    new_value: Any


@dataclass
class ReloadAction:
    action: str
    paths: list[str] = field(default_factory=list)


@dataclass
class ReloadPlan:
    actions: list[ReloadAction] = field(default_factory=list)
    requires_restart: bool = False

    HOT_RELOAD_MAP: ClassVar[dict[str, str]] = {
        "model.": "reload_model",
        "agents.system_prompt": "reload_skills",
        "agents.max_tool_rounds": "reload_skills",
        "agents.timezone": "reload_skills",
        "agents.bootstrap_files": "reload_skills",
        "cron.": "reload_cron",
        "tools.": "reload_tools",
        "skills.": "reload_skills",
        "memory.": "reload_memory",
        "link_understanding.": "reload_skills",
        "auth.": "reload_auth",
        "security.": "reload_security",
        "compression.": "reload_model",
    }

    RESTART_PREFIXES: ClassVar[tuple[str, ...]] = (
        "gateway.host",
        "gateway.port",
        "gateway.auth_token",
        "channels.qq.enabled",
        "channels.qq.app_id",
        "channels.qq.client_secret",
        "checkpointer.",
        "session.",
    )

    @classmethod
    def build(cls, changes: list[ConfigChange]) -> ReloadPlan:
        actions_map: dict[str, ReloadAction] = {}
        requires_restart = False

        for change in changes:
            if any(change.path.startswith(rp) for rp in cls.RESTART_PREFIXES):
                requires_restart = True

            action_name = None
            for prefix, action in cls.HOT_RELOAD_MAP.items():
                if change.path.startswith(prefix):
                    action_name = action
                    break

            if action_name:
                if action_name not in actions_map:
                    actions_map[action_name] = ReloadAction(action=action_name)
                actions_map[action_name].paths.append(change.path)

        return cls(
            actions=list(actions_map.values()),
            requires_restart=requires_restart,
        )


class DiffEngine:
    @staticmethod
    def diff(old: BaseModel, new: BaseModel, prefix: str = "") -> list[ConfigChange]:
        changes: list[ConfigChange] = []
        for field_name in type(old).model_fields:
            old_val = getattr(old, field_name)
            new_val = getattr(new, field_name)
            path = f"{prefix}{field_name}" if not prefix else f"{prefix}.{field_name}"

            if isinstance(old_val, BaseModel) and isinstance(new_val, BaseModel):
                changes.extend(DiffEngine.diff(old_val, new_val, path))
            elif old_val != new_val:
                changes.append(ConfigChange(path=path, old_value=old_val, new_value=new_val))

        return changes


class ConfigWatcher:
    def __init__(
        self,
        path: str,
        on_reload: Callable,
        debounce_ms: int = 300,
    ):
        self._path = Path(path).resolve()
        self._on_reload = on_reload
        self._debounce_ms = debounce_ms
        self._current = load_config(self._path)
        self._hash = self._compute_hash()
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()

    @property
    def current(self):
        return self._current

    def _compute_hash(self) -> str:
        if not self._path.exists():
            return ""
        return hashlib.md5(self._path.read_bytes()).hexdigest()

    async def start(self):
        self._stop_event.clear()
        self._task = asyncio.create_task(self._watch())

    async def _watch(self):
        parent_dir = str(self._path.parent)
        target_name = self._path.name

        async for changes in awatch(
            parent_dir,
            stop_event=self._stop_event,
            watch_filter=lambda change_type, path: path.endswith(target_name),
        ):
            await asyncio.sleep(self._debounce_ms / 1000.0)
            await self._apply_reload()

    async def _apply_reload(self):
        new_hash = self._compute_hash()
        if new_hash == self._hash:
            return

        new_config = load_config(self._path)
        changes = DiffEngine.diff(self._current, new_config)
        if not changes:
            self._hash = new_hash
            return

        plan = ReloadPlan.build(changes)
        old_config = self._current

        result = self._on_reload(old_config, new_config, plan)
        if asyncio.iscoroutine(result):
            await result

        # 仅在 reload 成功后才提交状态，确保异常时下次能重试
        self._hash = new_hash
        self._current = new_config

    async def stop(self):
        self._stop_event.set()
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
