"""Tests for src/plugins/loader.py and registry.py — plugin discovery, loading, hook execution."""

import json
import types
from unittest.mock import MagicMock

import pytest

from src.plugins.loader import (
    PluginManifest,
    PluginRecord,
    HookResult,
    _load_module,
    _extract_tools,
    _resolve_hook_ref,
    load_plugin,
    discover_plugins,
)
from src.plugins.registry import PluginRegistry


# ── PluginManifest ─────────────────────────────────────────


class TestPluginManifest:
    def test_defaults(self):
        m = PluginManifest(id="test")
        assert m.name == ""
        assert m.tools == []
        assert m.hooks == {}

    def test_full_manifest(self):
        m = PluginManifest(
            id="my-plugin",
            name="My Plugin",
            version="1.0.0",
            description="A test plugin",
            tools=["tools.py"],
            hooks={"pre_execute": "hooks.py:on_before"},
        )
        assert m.id == "my-plugin"
        assert m.tools == ["tools.py"]


# ── _load_module ───────────────────────────────────────────


class TestLoadModule:
    def test_load_valid_module(self, tmp_path):
        mod_file = tmp_path / "mod.py"
        mod_file.write_text("VALUE = 42\n")
        mod = _load_module(mod_file, "test_mod_plugin")
        assert mod is not None
        assert mod.VALUE == 42

    def test_load_nonexistent(self, tmp_path):
        mod = _load_module(tmp_path / "nonexistent.py", "test_mod_nonexist")
        assert mod is None

    def test_load_module_with_syntax_error(self, tmp_path):
        mod_file = tmp_path / "bad.py"
        mod_file.write_text("def broken(\n")
        mod = _load_module(mod_file, "test_mod_bad")
        assert mod is None

    def test_load_module_path_traversal(self, tmp_path):
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        outside_file = tmp_path / "outside.py"
        outside_file.write_text("X = 1\n")
        mod = _load_module(outside_file, "test_traversal", allowed_dir=allowed_dir)
        assert mod is None


# ── _extract_tools ─────────────────────────────────────────


class TestExtractTools:
    def test_extracts_tooldef_attrs(self):
        from src.agent.tooldef import ToolDef

        def dummy():
            """desc"""
            pass

        td = ToolDef.from_function(dummy)
        ns = {"my_tool": td}
        mod = types.SimpleNamespace(**ns)
        tools = _extract_tools(mod)
        assert td in tools

    def test_extracts_via_get_tools(self):
        td_mock = MagicMock()
        mod = types.SimpleNamespace(get_tools=lambda: [td_mock])
        tools = _extract_tools(mod)
        assert td_mock in tools

    def test_get_tools_returns_non_list(self):
        mod = types.SimpleNamespace(get_tools=lambda: "not a list")
        tools = _extract_tools(mod)
        assert tools == []

    def test_get_tools_raises(self):
        def bad_get():
            raise RuntimeError("boom")

        mod = types.SimpleNamespace(get_tools=bad_get)
        tools = _extract_tools(mod)
        assert tools == []

    def test_empty_module(self):
        mod = types.SimpleNamespace()
        tools = _extract_tools(mod)
        assert tools == []


# ── _resolve_hook_ref ──────────────────────────────────────


class TestResolveHookRef:
    def test_with_colon(self):
        mod = types.SimpleNamespace(my_func=lambda: "ok")
        result = _resolve_hook_ref("file.py:my_func", mod)
        assert result is not None
        assert result() == "ok"

    def test_without_colon(self):
        mod = types.SimpleNamespace(my_func=lambda: "ok")
        result = _resolve_hook_ref("my_func", mod)
        assert result is not None

    def test_not_found(self):
        mod = types.SimpleNamespace()
        result = _resolve_hook_ref("nonexistent", mod)
        assert result is None


# ── load_plugin ────────────────────────────────────────────


class TestLoadPlugin:
    def test_no_manifest(self, tmp_path):
        result = load_plugin(tmp_path)
        assert result is None

    def test_invalid_manifest(self, tmp_path):
        (tmp_path / "plugin.json").write_text("not json", encoding="utf-8")
        result = load_plugin(tmp_path)
        assert result is None

    def test_valid_manifest_no_tools(self, tmp_path):
        manifest = {"id": "test-plugin", "name": "Test", "version": "1.0.0"}
        (tmp_path / "plugin.json").write_text(json.dumps(manifest), encoding="utf-8")
        record = load_plugin(tmp_path)
        assert record is not None
        assert record.manifest.id == "test-plugin"
        assert record.tools == []

    def test_valid_manifest_with_tools(self, tmp_path):
        manifest = {"id": "test-plugin", "tools": ["tools.py"]}
        (tmp_path / "plugin.json").write_text(json.dumps(manifest), encoding="utf-8")
        (tmp_path / "tools.py").write_text("# no tools\n", encoding="utf-8")
        record = load_plugin(tmp_path)
        assert record is not None
        assert record.tools == []

    def test_manifest_missing_tools_file(self, tmp_path):
        manifest = {"id": "test-plugin", "tools": ["missing.py"]}
        (tmp_path / "plugin.json").write_text(json.dumps(manifest), encoding="utf-8")
        record = load_plugin(tmp_path)
        assert record is not None
        assert record.tools == []

    def test_manifest_with_hook(self, tmp_path):
        manifest = {"id": "test-plugin", "hooks": {"pre_exec": "hooks.py:my_hook"}}
        (tmp_path / "plugin.json").write_text(json.dumps(manifest), encoding="utf-8")
        (tmp_path / "hooks.py").write_text("def my_hook(**kw): return {'block': True}\n", encoding="utf-8")
        record = load_plugin(tmp_path)
        assert record is not None
        assert "pre_exec" in record.hooks
        assert len(record.hooks["pre_exec"]) == 1

    def test_hook_invalid_ref(self, tmp_path):
        manifest = {"id": "test-plugin", "hooks": {"pre_exec": "no_colon"}}
        (tmp_path / "plugin.json").write_text(json.dumps(manifest), encoding="utf-8")
        record = load_plugin(tmp_path)
        assert record is not None
        assert record.hooks == {}


# ── discover_plugins ───────────────────────────────────────


class TestDiscoverPlugins:
    def test_discover_from_extra_dirs(self, tmp_path):
        plugin_dir = tmp_path / "my-plugin"
        plugin_dir.mkdir()
        manifest = {"id": "disc-plugin", "name": "Disc"}
        (plugin_dir / "plugin.json").write_text(json.dumps(manifest), encoding="utf-8")

        records = discover_plugins(extra_dirs=[str(tmp_path)])
        # Find our plugin (bundled plugins may also be discovered)
        found = [r for r in records if r.manifest.id == "disc-plugin"]
        assert len(found) == 1
        assert found[0].manifest.name == "Disc"

    def test_dedup_by_id(self, tmp_path):
        p1 = tmp_path / "plugin-a"
        p1.mkdir()
        (p1 / "plugin.json").write_text(json.dumps({"id": "dup", "name": "A"}), encoding="utf-8")
        p2 = tmp_path / "plugin-b"
        p2.mkdir()
        (p2 / "plugin.json").write_text(json.dumps({"id": "dup", "name": "B"}), encoding="utf-8")

        records = discover_plugins(extra_dirs=[str(tmp_path)])
        dup_records = [r for r in records if r.manifest.id == "dup"]
        assert len(dup_records) == 1

    def test_skip_dot_dirs(self, tmp_path):
        dot_dir = tmp_path / ".hidden"
        dot_dir.mkdir()
        (dot_dir / "plugin.json").write_text(json.dumps({"id": "hidden"}), encoding="utf-8")

        records = discover_plugins(extra_dirs=[str(tmp_path)])
        hidden = [r for r in records if r.manifest.id == "hidden"]
        assert len(hidden) == 0

    def test_skip_non_dir_files(self, tmp_path):
        (tmp_path / "readme.md").write_text("# readme", encoding="utf-8")
        records = discover_plugins(extra_dirs=[str(tmp_path)])
        # readme.md should not be loaded as a plugin
        readme = [r for r in records if r.manifest.id == "readme"]
        assert len(readme) == 0

    def test_nonexistent_extra_dir(self):
        records = discover_plugins(extra_dirs=["/nonexistent/path"])
        assert isinstance(records, list)


# ── PluginRegistry ─────────────────────────────────────────


class TestPluginRegistry:
    def test_register_and_collect_tools(self):
        reg = PluginRegistry()
        record = PluginRecord(
            manifest=PluginManifest(id="p1", name="P1"),
            dir_path="/tmp",
            tools=["tool1"],
            hooks={},
        )
        reg.register_plugin(record)
        assert reg.collect_tools() == ["tool1"]
        assert reg.plugin_count == 1
        assert reg.tool_count == 1

    def test_duplicate_plugin_skipped(self):
        reg = PluginRegistry()
        r1 = PluginRecord(
            manifest=PluginManifest(id="p1"),
            dir_path="/a",
            tools=["t1"],
            hooks={},
        )
        r2 = PluginRecord(
            manifest=PluginManifest(id="p1"),
            dir_path="/b",
            tools=["t2"],
            hooks={},
        )
        reg.register_plugin(r1)
        reg.register_plugin(r2)
        assert reg.plugin_count == 1
        assert reg.tool_count == 1

    def test_list_plugins(self):
        reg = PluginRegistry()
        reg.register_plugin(
            PluginRecord(
                manifest=PluginManifest(id="p1", name="Plugin1", version="2.0.0"),
                dir_path="/tmp",
                tools=["t1"],
                hooks={},
            )
        )
        plugins = reg.list_plugins()
        assert len(plugins) == 1
        assert plugins[0]["id"] == "p1"
        assert plugins[0]["version"] == "2.0.0"

    def test_collect_tools_is_copy(self):
        reg = PluginRegistry()
        tools = reg.collect_tools()
        tools.append("extra")
        assert reg.tool_count == 0

    @pytest.mark.asyncio
    async def test_run_hooks_sync(self):
        reg = PluginRegistry()

        def sync_hook(**kw):
            return {"block": True}

        record = PluginRecord(
            manifest=PluginManifest(id="p1"),
            dir_path="/tmp",
            tools=[],
            hooks={"pre_exec": [sync_hook]},
        )
        reg.register_plugin(record)
        results = await reg.run_hooks("pre_exec", tool_name="test")
        assert len(results) == 1
        assert results[0].block is True

    @pytest.mark.asyncio
    async def test_run_hooks_async(self):
        reg = PluginRegistry()

        async def async_hook(**kw):
            return HookResult(block=False, modified_args={"x": 1})

        record = PluginRecord(
            manifest=PluginManifest(id="p1"),
            dir_path="/tmp",
            tools=[],
            hooks={"pre_exec": [async_hook]},
        )
        reg.register_plugin(record)
        results = await reg.run_hooks("pre_exec")
        assert len(results) == 1
        assert results[0].modified_args == {"x": 1}

    @pytest.mark.asyncio
    async def test_run_hooks_block_stops(self):
        reg = PluginRegistry()

        calls = []

        def blocker(**kw):
            calls.append(1)
            return {"block": True}

        def after_block(**kw):
            calls.append(2)
            return {"block": False}

        record = PluginRecord(
            manifest=PluginManifest(id="p1"),
            dir_path="/tmp",
            tools=[],
            hooks={"pre_exec": [blocker, after_block]},
        )
        reg.register_plugin(record)
        results = await reg.run_hooks("pre_exec")
        assert len(results) == 1
        assert calls == [1]  # second hook not called

    @pytest.mark.asyncio
    async def test_run_hooks_error_continues(self):
        reg = PluginRegistry()

        def bad_hook(**kw):
            raise RuntimeError("boom")

        def good_hook(**kw):
            return {"block": False}

        record = PluginRecord(
            manifest=PluginManifest(id="p1"),
            dir_path="/tmp",
            tools=[],
            hooks={"pre_exec": [bad_hook, good_hook]},
        )
        reg.register_plugin(record)
        results = await reg.run_hooks("pre_exec")
        assert len(results) == 1  # only good_hook result

    @pytest.mark.asyncio
    async def test_run_hooks_no_hooks(self):
        reg = PluginRegistry()
        results = await reg.run_hooks("nonexistent")
        assert results == []

    @pytest.mark.asyncio
    async def test_run_hooks_dict_return(self):
        reg = PluginRegistry()

        def dict_hook(**kw):
            return {"block": False, "require_approval": True, "approval_title": "Test"}

        record = PluginRecord(
            manifest=PluginManifest(id="p1"),
            dir_path="/tmp",
            tools=[],
            hooks={"pre": [dict_hook]},
        )
        reg.register_plugin(record)
        results = await reg.run_hooks("pre")
        assert results[0].require_approval is True
        assert results[0].approval_title == "Test"
