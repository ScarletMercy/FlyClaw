"""50-scenario cross-validation for config hot-reload fixes."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import (
    AppConfig,
    ChannelsConfig,
    CompressionConfig,
    GatewayConfig,
    MemoryConfig,
    MemoryStoreConfig,
    ModelConfig,
    QQConfig,
    SecurityConfig,
    ToolsConfig,
    WeixinConfig,
)
from src.config_watcher import ConfigChange, DiffEngine, ReloadAction, ReloadPlan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_app(**kw):
    app = MagicMock()
    app.config = AppConfig()
    app.agent_loop = MagicMock()
    app.agent_loop._config = app.config
    app.agent_loop._compressor = MagicMock()
    app.agent_loop._tools = []
    app.agent_loop._tool_map = {}
    app.agent_loop._skills_prompt = ""
    app.agent_loop._prompt_skills = ""
    app.dispatcher = MagicMock()
    app.qq = MagicMock()
    app.qq.config = QQConfig()
    app.weixin = MagicMock()
    app.memory_searcher = None
    app.cron_service = None
    app.rbac = None
    app.tool_registry = MagicMock()
    app.skills_cache = []
    app._collect_builtin_tools = MagicMock(return_value=[])
    app._build_skill_directories = MagicMock(return_value=[])
    for k, v in kw.items():
        setattr(app, k, v)
    return app


# ===========================================================================
# SECTION A: Mapping correctness
# ===========================================================================

_HOT_FIELDS = [
    "model.provider",
    "model.name",
    "model.api_key",
    "model.base_url",
    "model.temperature",
    "model.context_window",
    "model.fallbacks",
    "tools.web_search.api_key",
    "tools.web_search.enabled",
    "tools.browser.enabled",
    "tools.media_understanding.enabled",
    "tools.media_understanding.name",
    "tools.media_understanding.api_key",
    "agents.workspace",
    "agents.language",
    "agents.busy_input_mode",
    "agents.subagents",
    "agents.tool_progress_notifications",
    "agents.tool_output_cache_chars",
    "delegation.enabled",
    "hooks.hooks",
    "memory.enabled",
    "memory.db_path",
    "security.enabled",
    "security.allow_private_urls",
    "compression.enabled",
    "compression.threshold_percent",
]

_RESTART_FIELDS = [
    "gateway.host",
    "gateway.port",
    "gateway.auth_token",
    "channels.qq.enabled",
    "channels.qq.app_id",
    "channels.qq.client_secret",
    "channels.qq.dm_policy",
    "channels.qq.group_policy",
    "channels.qq.require_mention",
    "channels.qq.markdown_support",
    "channels.weixin.enabled",
    "channels.weixin.account_id",
    "channels.weixin.token",
    "channels.weixin.dm_policy",
    "memory_store.enabled",
    "memory_store.memory_judge_model",
    "checkpointer.path",
    "session.idle_reset_minutes",
    "plugins.enabled",
    "canvas.enabled",
    "voice.enabled",
    "task.enabled",
    "session_search.enabled",
]


@pytest.mark.parametrize("field", _HOT_FIELDS, ids=_HOT_FIELDS)
def test_hot_field_is_hot(field):
    is_hot = any(field.startswith(p) for p in ReloadPlan.HOT_RELOAD_MAP)
    assert is_hot, f"{field} not in any HOT_RELOAD_MAP"


@pytest.mark.parametrize("field", _HOT_FIELDS, ids=_HOT_FIELDS)
def test_hot_field_is_not_restart(field):
    is_restart = any(field.startswith(p) for p in ReloadPlan.RESTART_PREFIXES)
    assert not is_restart, f"{field} unexpectedly matched RESTART_PREFIXES"


@pytest.mark.parametrize("field", _RESTART_FIELDS, ids=_RESTART_FIELDS)
def test_restart_field_is_restart(field):
    is_restart = any(field.startswith(p) for p in ReloadPlan.RESTART_PREFIXES)
    assert is_restart, f"{field} not in any RESTART_PREFIXES"


# ===========================================================================
# SECTION B: ReloadPlan.build logic
# ===========================================================================


class TestReloadPlanBuild:
    def test_pure_hot_no_restart(self):
        changes = [ConfigChange("model.name", "gpt-4o", "deepseek-chat")]
        plan = ReloadPlan.build(changes)
        assert not plan.requires_restart

    def test_pure_hot_has_reload_model(self):
        changes = [ConfigChange("model.name", "gpt-4o", "deepseek-chat")]
        plan = ReloadPlan.build(changes)
        assert any(a.action == "reload_model" for a in plan.actions)

    def test_pure_restart_requires_restart(self):
        changes = [ConfigChange("gateway.port", 18080, 9090)]
        plan = ReloadPlan.build(changes)
        assert plan.requires_restart

    def test_pure_restart_no_hot_actions(self):
        changes = [ConfigChange("gateway.port", 18080, 9090)]
        plan = ReloadPlan.build(changes)
        assert len(plan.actions) == 0

    def test_mixed_requires_restart(self):
        changes = [ConfigChange("gateway.port", 18080, 9090), ConfigChange("model.name", "a", "b")]
        plan = ReloadPlan.build(changes)
        assert plan.requires_restart

    def test_mixed_still_has_reload_model(self):
        changes = [ConfigChange("gateway.port", 18080, 9090), ConfigChange("model.name", "a", "b")]
        plan = ReloadPlan.build(changes)
        assert any(a.action == "reload_model" for a in plan.actions)

    def test_dedup_single_reload_model(self):
        changes = [ConfigChange("model.name", "a", "b"), ConfigChange("model.temperature", 1.0, 0.5)]
        plan = ReloadPlan.build(changes)
        model_actions = [a for a in plan.actions if a.action == "reload_model"]
        assert len(model_actions) == 1

    @pytest.mark.parametrize(
        "field", ["agents.system_prompt", "agents.workspace", "agents.language", "agents.subagents"]
    )
    def test_agents_field_triggers_action(self, field):
        changes = [ConfigChange(field, "old", "new")]
        plan = ReloadPlan.build(changes)
        assert len(plan.actions) > 0

    def test_qq_dm_policy_requires_restart(self):
        changes = [ConfigChange("channels.qq.dm_policy", "open", "allowlist")]
        plan = ReloadPlan.build(changes)
        assert plan.requires_restart

    def test_unknown_field_no_crash(self):
        changes = [ConfigChange("totally.unknown.field", "a", "b")]
        plan = ReloadPlan.build(changes)
        assert len(plan.actions) == 0 and not plan.requires_restart

    def test_subagent_max_depth_no_action(self):
        changes = [ConfigChange("agents.subagent_max_depth", 2, 5)]
        plan = ReloadPlan.build(changes)
        assert len(plan.actions) == 0 and not plan.requires_restart


# ===========================================================================
# SECTION C: DiffEngine
# ===========================================================================


class TestDiffEngine:
    def test_single_field_diff(self):
        old = AppConfig(model=ModelConfig(name="gpt-4o"))
        new = AppConfig(model=ModelConfig(name="deepseek-chat"))
        changes = DiffEngine.diff(old, new)
        assert len(changes) == 1
        assert changes[0].path == "model.name"

    def test_no_change_empty(self):
        changes = DiffEngine.diff(AppConfig(), AppConfig())
        assert len(changes) == 0

    def test_nested_path_correct(self):
        old = AppConfig(channels=ChannelsConfig(qq=QQConfig(dm_policy="open")))
        new = AppConfig(channels=ChannelsConfig(qq=QQConfig(dm_policy="allowlist")))
        changes = DiffEngine.diff(old, new)
        assert any(c.path == "channels.qq.dm_policy" for c in changes)

    def test_multi_field(self):
        old = AppConfig(model=ModelConfig(name="a", temperature=1.0))
        new = AppConfig(model=ModelConfig(name="b", temperature=0.5))
        changes = DiffEngine.diff(old, new)
        paths = {c.path for c in changes}
        assert "model.name" in paths
        assert "model.temperature" in paths

    def test_default_to_set(self):
        old = AppConfig()
        new = AppConfig(security=SecurityConfig(allow_private_urls=True))
        changes = DiffEngine.diff(old, new)
        assert any(c.path == "security.allow_private_urls" for c in changes)


# ===========================================================================
# SECTION D: execute() behavior
# ===========================================================================

from src.config_reload import ReloadExecutor


class TestExecute:
    async def test_returns_dict(self):
        app = _make_mock_app()
        ex = ReloadExecutor(app)
        r = await ex.execute(ReloadPlan(actions=[], requires_restart=False))
        assert isinstance(r, dict)
        assert "succeeded" in r and "failed" in r

    async def test_restart_plus_action_still_runs(self):
        app = _make_mock_app()
        ex = ReloadExecutor(app)
        r = await ex.execute(ReloadPlan(actions=[ReloadAction(action="reload_security")], requires_restart=True))
        assert "reload_security" in r["succeeded"]

    async def test_missing_handler_no_crash(self):
        app = _make_mock_app()
        ex = ReloadExecutor(app)
        r = await ex.execute(ReloadPlan(actions=[ReloadAction(action="reload_nonexistent")], requires_restart=False))
        assert r["succeeded"] == []
        assert r["failed"] == []

    async def test_exception_goes_to_failed(self):
        app = _make_mock_app()
        ex = ReloadExecutor(app)

        async def boom():
            raise RuntimeError("boom")

        ex._do_reload_model = boom
        r = await ex.execute(ReloadPlan(actions=[ReloadAction(action="reload_model")], requires_restart=False))
        assert "reload_model" in r["failed"]

    async def test_reload_security_succeeds(self):
        app = _make_mock_app()
        ex = ReloadExecutor(app)
        r = await ex.execute(ReloadPlan(actions=[ReloadAction(action="reload_security")], requires_restart=False))
        assert "reload_security" in r["succeeded"]

    async def test_memory_disabled_ok(self):
        app = _make_mock_app()
        app.config = AppConfig(memory=MemoryConfig(enabled=False))
        ex = ReloadExecutor(app)
        r = await ex.execute(ReloadPlan(actions=[ReloadAction(action="reload_memory")], requires_restart=False))
        assert "reload_memory" in r["succeeded"]

    async def test_partial_failure(self):
        app = _make_mock_app()
        ex = ReloadExecutor(app)

        async def boom():
            raise RuntimeError("boom")

        ex._do_reload_model = boom
        r = await ex.execute(
            ReloadPlan(
                actions=[ReloadAction(action="reload_model"), ReloadAction(action="reload_security")],
                requires_restart=False,
            )
        )
        assert "reload_model" in r["failed"]
        assert "reload_security" in r["succeeded"]


# ===========================================================================
# SECTION E: Config propagation
# ===========================================================================

from src.app import ServiceContainer


class TestConfigPropagation:
    async def test_agent_loop_config_updated(self):
        app = ServiceContainer.__new__(ServiceContainer)
        app.config = AppConfig(model=ModelConfig(name="old"))
        app.agent_loop = MagicMock()
        app.agent_loop._config = app.config
        app.agent_loop._compressor = MagicMock()
        app.dispatcher = MagicMock()
        app.qq = MagicMock()
        app.weixin = MagicMock()
        app._reload_executor = MagicMock()
        app._reload_executor.execute = AsyncMock(return_value={"succeeded": [], "failed": []})

        new_cfg = AppConfig(model=ModelConfig(name="new"))
        await app.on_config_reload(app.config, new_cfg, ReloadPlan(actions=[], requires_restart=False))
        assert app.agent_loop._config is new_cfg

    async def test_compressor_config_updated(self):
        app = ServiceContainer.__new__(ServiceContainer)
        app.config = AppConfig(model=ModelConfig(name="old"))
        app.agent_loop = MagicMock()
        app.agent_loop._config = app.config
        app.agent_loop._compressor = MagicMock()
        app.dispatcher = MagicMock()
        app.qq = MagicMock()
        app.weixin = MagicMock()
        app._reload_executor = MagicMock()
        app._reload_executor.execute = AsyncMock(return_value={"succeeded": [], "failed": []})

        new_cfg = AppConfig(model=ModelConfig(name="new"))
        await app.on_config_reload(app.config, new_cfg, ReloadPlan(actions=[], requires_restart=False))
        assert app.agent_loop._compressor.config is new_cfg.compression

    async def test_dispatcher_config_updated(self):
        app = ServiceContainer.__new__(ServiceContainer)
        app.config = AppConfig(model=ModelConfig(name="old"))
        app.agent_loop = MagicMock()
        app.agent_loop._config = app.config
        app.agent_loop._compressor = MagicMock()
        app.dispatcher = MagicMock()
        app.qq = MagicMock()
        app.weixin = MagicMock()
        app._reload_executor = MagicMock()
        app._reload_executor.execute = AsyncMock(return_value={"succeeded": [], "failed": []})

        new_cfg = AppConfig(model=ModelConfig(name="new"))
        await app.on_config_reload(app.config, new_cfg, ReloadPlan(actions=[], requires_restart=False))
        assert app.dispatcher._config is new_cfg

    async def test_qq_config_updated(self):
        app = ServiceContainer.__new__(ServiceContainer)
        app.config = AppConfig(model=ModelConfig(name="old"))
        app.agent_loop = MagicMock()
        app.agent_loop._config = app.config
        app.agent_loop._compressor = MagicMock()
        app.dispatcher = MagicMock()
        app.qq = MagicMock()
        app.weixin = MagicMock()
        app._reload_executor = MagicMock()
        app._reload_executor.execute = AsyncMock(return_value={"succeeded": [], "failed": []})

        new_cfg = AppConfig(model=ModelConfig(name="new"))
        await app.on_config_reload(app.config, new_cfg, ReloadPlan(actions=[], requires_restart=False))
        assert app.qq.config is new_cfg.channels.qq

    async def test_failed_handler_no_raise(self):
        app = ServiceContainer.__new__(ServiceContainer)
        app.config = AppConfig()
        app.agent_loop = MagicMock()
        app.agent_loop._config = app.config
        app.agent_loop._compressor = MagicMock()
        app.dispatcher = MagicMock()
        app.qq = MagicMock()
        app.weixin = MagicMock()
        app._reload_executor = MagicMock()
        app._reload_executor.execute = AsyncMock(return_value={"succeeded": [], "failed": ["reload_memory"]})

        # Should not raise despite failed actions
        await app.on_config_reload(app.config, AppConfig(), ReloadPlan(actions=[], requires_restart=False))


# ===========================================================================
# SECTION F: _apply_reload retry
# ===========================================================================

from src.config_watcher import ConfigWatcher


class TestApplyReloadRetry:
    async def test_failed_reload_does_not_update_current(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("model:\n  name: gpt-4o\n")

        async def failing_reload(old, new, plan):
            raise RuntimeError("deliberate")

        watcher = ConfigWatcher(path=str(cfg), on_reload=failing_reload)
        old_current = watcher._current
        await watcher._apply_reload()
        assert watcher._current is old_current

    async def test_failed_reload_does_not_update_hash(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("model:\n  name: gpt-4o\n")

        async def failing_reload(old, new, plan):
            raise RuntimeError("deliberate")

        watcher = ConfigWatcher(path=str(cfg), on_reload=failing_reload)
        old_hash = watcher._hash

        # Modify file to trigger new hash on next read
        cfg.write_text("model:\n  name: deepseek-chat\n")
        await watcher._apply_reload()
        assert watcher._hash == old_hash

    async def test_success_updates_current(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("model:\n  name: gpt-4o\n")

        async def ok_reload(old, new, plan):
            pass

        watcher = ConfigWatcher(path=str(cfg), on_reload=ok_reload)
        await watcher._apply_reload()
        assert watcher._current is not None

        # Change file and reload again
        cfg.write_text("model:\n  name: deepseek-chat\n")

        async def ok_reload2(old, new, plan):
            pass

        watcher._on_reload = ok_reload2
        await watcher._apply_reload()
        assert watcher._current.model.name == "deepseek-chat"


# ===========================================================================
# SECTION G: Edge cases
# ===========================================================================


class TestEdgeCases:
    async def test_all_subsystems_none_no_crash(self):
        app = ServiceContainer.__new__(ServiceContainer)
        app.config = AppConfig()
        app.agent_loop = None
        app.dispatcher = None
        app.qq = None
        app.weixin = None
        app._reload_executor = MagicMock()
        app._reload_executor.execute = AsyncMock(return_value={"succeeded": [], "failed": []})
        await app.on_config_reload(app.config, AppConfig(), ReloadPlan(actions=[], requires_restart=False))

    async def test_memory_disabled_to_disabled(self):
        app = MagicMock()
        app.config = AppConfig(memory=MemoryConfig(enabled=False))
        app.memory_searcher = None
        ex = ReloadExecutor(app)
        r = await ex.execute(ReloadPlan(actions=[ReloadAction(action="reload_memory")], requires_restart=False))
        assert app.memory_searcher is None
        assert "reload_memory" in r["succeeded"]

    def test_compression_maps_to_reload_model(self):
        changes = [ConfigChange("compression.threshold_percent", 0.6, 0.8)]
        plan = ReloadPlan.build(changes)
        assert any(a.action == "reload_model" for a in plan.actions)

    def test_memory_store_change_requires_restart(self):
        changes = [ConfigChange("memory_store.enabled", False, True)]
        plan = ReloadPlan.build(changes)
        assert plan.requires_restart

    def test_two_restart_fields(self):
        changes = [
            ConfigChange("gateway.host", "127.0.0.1", "0.0.0.0"),
            ConfigChange("channels.qq.enabled", False, True),
        ]
        plan = ReloadPlan.build(changes)
        assert plan.requires_restart
        assert len(plan.actions) == 0

    @pytest.mark.parametrize(
        "sub",
        [
            "exec.enabled",
            "web_search.api_key",
            "browser.enabled",
            "media_understanding.enabled",
            "policy.allow",
            "guardrails.enabled",
        ],
        ids=[
            "exec.enabled",
            "web_search.api_key",
            "browser.enabled",
            "media_understanding.enabled",
            "policy.allow",
            "guardrails.enabled",
        ],
    )
    def test_tools_sub_field_triggers_reload_tools(self, sub):
        changes = [ConfigChange(f"tools.{sub}", "a", "b")]
        plan = ReloadPlan.build(changes)
        assert any(a.action == "reload_tools" for a in plan.actions)

    def test_weixin_dm_policy_requires_restart(self):
        changes = [ConfigChange("channels.weixin.dm_policy", "open", "disabled")]
        plan = ReloadPlan.build(changes)
        assert plan.requires_restart

    def test_model_name_diff_detected(self):
        old = AppConfig(model=ModelConfig(name="gpt-4o"))
        new = AppConfig(model=ModelConfig(name="deepseek-chat"))
        changes = DiffEngine.diff(old, new)
        assert any(c.path == "model.name" for c in changes)

    async def test_old_memory_searcher_closed(self):
        app = MagicMock()
        app.config = AppConfig(memory=MemoryConfig(enabled=False))
        mock_searcher = AsyncMock()
        app.memory_searcher = mock_searcher
        ex = ReloadExecutor(app)
        await ex._do_reload_memory()
        mock_searcher.close.assert_called()
        assert app.memory_searcher is None

    async def test_memory_failure_searcher_stays_none(self):
        app = MagicMock()
        cfg = AppConfig(memory=MemoryConfig(enabled=True))
        app.config = cfg
        app.memory_searcher = None
        ex = ReloadExecutor(app)
        with patch("src.memory.search.MemorySearcher", side_effect=RuntimeError("searcher boom")):
            try:
                await ex._do_reload_memory()
            except Exception:
                pass
        assert app.memory_searcher is None

    def test_empty_changes_no_actions_no_restart(self):
        plan = ReloadPlan.build([])
        assert len(plan.actions) == 0
        assert not plan.requires_restart

    def test_all_map_values_have_handlers(self):
        all_ok = True
        missing = []
        for action_name in set(ReloadPlan.HOT_RELOAD_MAP.values()):
            if not hasattr(ReloadExecutor, f"_do_{action_name}"):
                all_ok = False
                missing.append(action_name)
        assert all_ok, f"Missing handlers: {missing}"

    async def test_qq_config_replaced_with_new(self):
        app = ServiceContainer.__new__(ServiceContainer)
        app.config = AppConfig()
        app.agent_loop = MagicMock()
        app.agent_loop._config = app.config
        app.agent_loop._compressor = MagicMock()
        app.dispatcher = MagicMock()
        app.weixin = MagicMock()
        app._reload_executor = MagicMock()
        app._reload_executor.execute = AsyncMock(return_value={"succeeded": [], "failed": []})

        qq_mock = MagicMock()
        qq_mock.config = QQConfig(dm_policy="open")
        app.qq = qq_mock

        new_cfg = AppConfig(channels=ChannelsConfig(qq=QQConfig(dm_policy="allowlist")))
        await app.on_config_reload(app.config, new_cfg, ReloadPlan(actions=[], requires_restart=False))
        assert qq_mock.config is new_cfg.channels.qq

    async def test_weixin_none_no_crash(self):
        app = ServiceContainer.__new__(ServiceContainer)
        app.config = AppConfig()
        app.agent_loop = MagicMock()
        app.agent_loop._config = app.config
        app.agent_loop._compressor = MagicMock()
        app.dispatcher = MagicMock()
        app.qq = MagicMock()
        app.weixin = None
        app._reload_executor = MagicMock()
        app._reload_executor.execute = AsyncMock(return_value={"succeeded": [], "failed": []})
        await app.on_config_reload(app.config, AppConfig(), ReloadPlan(actions=[], requires_restart=False))
