from __future__ import annotations

import asyncio
import os
import sys

import pytest
import yaml
from unittest.mock import AsyncMock

from src.config import AppConfig, GatewayConfig, ModelConfig, ModelFallback, CronConfig
from src.config_watcher import ConfigChange, ConfigWatcher, DiffEngine, ReloadPlan


class TestDiffEngine:
    def test_no_changes(self):
        a = AppConfig()
        b = AppConfig()
        assert DiffEngine.diff(a, b) == []

    def test_model_name_change(self):
        a = AppConfig()
        b = AppConfig(model=ModelConfig(name="claude-sonnet-4-6"))
        changes = DiffEngine.diff(a, b)
        paths = [c.path for c in changes]
        assert "model.name" in paths
        change = next(c for c in changes if c.path == "model.name")
        assert change.old_value == "gpt-4o"
        assert change.new_value == "claude-sonnet-4-6"

    def test_nested_change(self):
        a = AppConfig()
        b = AppConfig(gateway=GatewayConfig(host="0.0.0.0", port=9999))
        changes = DiffEngine.diff(a, b)
        paths = [c.path for c in changes]
        assert "gateway.host" in paths
        assert "gateway.port" in paths
        host_change = next(c for c in changes if c.path == "gateway.host")
        assert host_change.old_value == "127.0.0.1"
        assert host_change.new_value == "0.0.0.0"

    def test_cron_enabled_change(self):
        a = AppConfig()
        b = AppConfig(cron=CronConfig(enabled=False))
        changes = DiffEngine.diff(a, b)
        change = next(c for c in changes if c.path == "cron.enabled")
        assert change.old_value is True
        assert change.new_value is False

    def test_list_str_no_changes(self):
        a = AppConfig()
        b = AppConfig()
        changes = DiffEngine.diff(a, b)
        list_changes = [c for c in changes if "[" in c.path or ".length" in c.path]
        assert list_changes == []

    def test_list_str_item_changed(self):
        from src.config import ExecToolConfig, ToolsConfig

        a = AppConfig(tools=ToolsConfig(exec=ExecToolConfig(deny_patterns=["a", "b", "c"])))
        b = AppConfig(tools=ToolsConfig(exec=ExecToolConfig(deny_patterns=["a", "x", "c"])))
        changes = DiffEngine.diff(a, b)
        paths = [c.path for c in changes]
        assert "tools.exec.deny_patterns[1]" in paths
        change = next(c for c in changes if c.path == "tools.exec.deny_patterns[1]")
        assert change.old_value == "b"
        assert change.new_value == "x"

    def test_list_str_appended(self):
        from src.config import ExecToolConfig, ToolsConfig

        a = AppConfig(tools=ToolsConfig(exec=ExecToolConfig(deny_patterns=["a", "b"])))
        b = AppConfig(tools=ToolsConfig(exec=ExecToolConfig(deny_patterns=["a", "b", "c"])))
        changes = DiffEngine.diff(a, b)
        paths = [c.path for c in changes]
        assert "tools.exec.deny_patterns[2]" in paths
        change = next(c for c in changes if c.path == "tools.exec.deny_patterns[2]")
        assert change.old_value is None
        assert change.new_value == "c"

    def test_list_str_removed(self):
        from src.config import ExecToolConfig, ToolsConfig

        a = AppConfig(tools=ToolsConfig(exec=ExecToolConfig(deny_patterns=["a", "b", "c"])))
        b = AppConfig(tools=ToolsConfig(exec=ExecToolConfig(deny_patterns=["a"])))
        changes = DiffEngine.diff(a, b)
        paths = [c.path for c in changes]
        assert "tools.exec.deny_patterns[1]" in paths
        assert "tools.exec.deny_patterns[2]" in paths
        c1 = next(c for c in changes if c.path == "tools.exec.deny_patterns[1]")
        assert c1.old_value == "b"
        assert c1.new_value is None

    def test_list_basemodel_recursive_diff(self):
        a = AppConfig(
            model=ModelConfig(
                fallbacks=[
                    ModelFallback(provider="openai", name="gpt-4o"),
                    ModelFallback(provider="anthropic", name="claude"),
                ]
            )
        )
        b = AppConfig(
            model=ModelConfig(
                fallbacks=[
                    ModelFallback(provider="openai", name="gpt-4o-mini"),
                    ModelFallback(provider="anthropic", name="claude"),
                ]
            )
        )
        changes = DiffEngine.diff(a, b)
        paths = [c.path for c in changes]
        # 只应报告 fallbacks[0].name 变更，不报整个列表
        assert "model.fallbacks[0].name" in paths
        assert "model.fallbacks[1].name" not in paths
        change = next(c for c in changes if c.path == "model.fallbacks[0].name")
        assert change.old_value == "gpt-4o"
        assert change.new_value == "gpt-4o-mini"

    def test_list_basemodel_added_item(self):
        a = AppConfig(
            model=ModelConfig(
                fallbacks=[
                    ModelFallback(provider="openai", name="gpt-4o"),
                ]
            )
        )
        b = AppConfig(
            model=ModelConfig(
                fallbacks=[
                    ModelFallback(provider="openai", name="gpt-4o"),
                    ModelFallback(provider="anthropic", name="claude"),
                ]
            )
        )
        changes = DiffEngine.diff(a, b)
        paths = [c.path for c in changes]
        # 新增 BaseModel 项时，old=None/new=BaseModel，报整项级变更
        assert "model.fallbacks[1]" in paths
        change = next(c for c in changes if c.path == "model.fallbacks[1]")
        assert change.old_value is None
        assert isinstance(change.new_value, ModelFallback)

    def test_list_basemodel_removed_item(self):
        a = AppConfig(
            model=ModelConfig(
                fallbacks=[
                    ModelFallback(provider="openai", name="gpt-4o"),
                    ModelFallback(provider="anthropic", name="claude"),
                ]
            )
        )
        b = AppConfig(
            model=ModelConfig(
                fallbacks=[
                    ModelFallback(provider="openai", name="gpt-4o"),
                ]
            )
        )
        changes = DiffEngine.diff(a, b)
        paths = [c.path for c in changes]
        # 删除 BaseModel 项时，old=BaseModel/new=None，报整项级变更
        assert "model.fallbacks[1]" in paths
        change = next(c for c in changes if c.path == "model.fallbacks[1]")
        assert isinstance(change.old_value, ModelFallback)
        assert change.new_value is None


class TestReloadPlan:
    def test_model_change_is_hot(self):
        changes = [ConfigChange(path="model.name", old_value="a", new_value="b")]
        plan = ReloadPlan.build(changes)
        assert not plan.requires_restart
        action_names = [a.action for a in plan.actions]
        assert "reload_model" in action_names

    def test_gateway_port_requires_restart(self):
        changes = [ConfigChange(path="gateway.port", old_value=18080, new_value=9090)]
        plan = ReloadPlan.build(changes)
        assert plan.requires_restart

    def test_cron_change_is_hot(self):
        changes = [ConfigChange(path="cron.enabled", old_value=True, new_value=False)]
        plan = ReloadPlan.build(changes)
        assert not plan.requires_restart
        action_names = [a.action for a in plan.actions]
        assert "reload_cron" in action_names


class TestConfigWatcher:
    @pytest.mark.asyncio
    async def test_apply_reload_directly(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump({"model": {"name": "claude-sonnet-4-6"}}))

        callback = AsyncMock()
        watcher = ConfigWatcher(str(config_path), on_reload=callback, debounce_ms=50)
        assert watcher.current.model.name == "claude-sonnet-4-6"

        config_path.write_text(yaml.dump({"model": {"name": "gpt-4o"}}))
        await watcher._apply_reload()

        callback.assert_called_once()
        old_cfg, new_cfg, plan = callback.call_args[0]
        assert old_cfg.model.name == "claude-sonnet-4-6"
        assert new_cfg.model.name == "gpt-4o"
        assert any(a.action == "reload_model" for a in plan.actions)

    @pytest.mark.asyncio
    async def test_watch_triggers_callback(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump({"model": {"name": "claude-sonnet-4-6"}}))

        callback = AsyncMock()
        watcher = ConfigWatcher(str(config_path), on_reload=callback, debounce_ms=50)
        assert watcher.current.model.name == "claude-sonnet-4-6"

        await watcher.start()
        await asyncio.sleep(0.5)

        with open(config_path, "w", encoding="utf-8") as f:
            f.write(yaml.dump({"model": {"name": "gpt-4o"}}))
            f.flush()
            os.fsync(f.fileno())

        for _ in range(80):
            await asyncio.sleep(0.1)
            if callback.called:
                break

        await watcher.stop()

        assert callback.called
        old_cfg, new_cfg, plan = callback.call_args[0]
        assert old_cfg.model.name == "claude-sonnet-4-6"
        assert new_cfg.model.name == "gpt-4o"
        assert any(a.action == "reload_model" for a in plan.actions)
