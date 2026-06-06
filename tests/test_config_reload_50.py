"""50-scenario cross-validation for config hot-reload fixes."""

import sys, os, asyncio, traceback
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from src.config import (
    AppConfig,
    ModelConfig,
    GatewayConfig,
    ChannelsConfig,
    QQConfig,
    WeixinConfig,
    MemoryConfig,
    SecurityConfig,
    CompressionConfig,
    ToolsConfig,
    MemoryStoreConfig,
)
from src.config_watcher import ReloadPlan, ReloadAction, ConfigChange, DiffEngine

passed = 0
failed = 0


def check(test_id, desc, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [{test_id:02d}] PASS  {desc}")
    else:
        failed += 1
        print(f"  [{test_id:02d}] FAIL  {desc}  -- {detail}")


# ============================================================
# SECTION A: Mapping correctness for setup-covered fields (1-28)
# ============================================================
print("\n== SECTION A: Mapping correctness ==")

SETUP_FIELDS = {
    "model.provider": "hot",
    "model.name": "hot",
    "model.api_key": "hot",
    "model.base_url": "hot",
    "model.temperature": "hot",
    "model.context_window": "hot",
    "model.fallbacks": "hot",
    "gateway.host": "restart",
    "gateway.port": "restart",
    "gateway.auth_token": "restart",
    "channels.qq.enabled": "restart",
    "channels.qq.app_id": "restart",
    "channels.qq.client_secret": "restart",
    "channels.qq.dm_policy": "restart",
    "channels.qq.group_policy": "restart",
    "channels.qq.require_mention": "restart",
    "channels.qq.markdown_support": "restart",
    "channels.weixin.enabled": "restart",
    "channels.weixin.account_id": "restart",
    "channels.weixin.token": "restart",
    "channels.weixin.dm_policy": "restart",
    "tools.web_search.api_key": "hot",
    "tools.web_search.enabled": "hot",
    "tools.browser.enabled": "hot",
    "tools.media_understanding.enabled": "hot",
    "tools.media_understanding.name": "hot",
    "tools.media_understanding.api_key": "hot",
    "memory_store.enabled": "restart",
    "memory_store.memory_judge_model": "restart",
}

for i, (field, expected) in enumerate(sorted(SETUP_FIELDS.items()), 1):
    is_hot = any(field.startswith(p) for p in ReloadPlan.HOT_RELOAD_MAP)
    is_restart = any(field.startswith(p) for p in ReloadPlan.RESTART_PREFIXES)
    if expected == "hot":
        check(i, f"SETUP {field} -> hot", is_hot and not is_restart, f"is_hot={is_hot}, is_restart={is_restart}")
    else:
        check(i, f"SETUP {field} -> restart", is_restart, f"is_hot={is_hot}, is_restart={is_restart}")

# Non-setup hot fields
EXTRA_HOT = [
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
for i, field in enumerate(EXTRA_HOT, len(SETUP_FIELDS) + 1):
    is_hot = any(field.startswith(p) for p in ReloadPlan.HOT_RELOAD_MAP)
    check(i, f"EXTRA {field} -> hot", is_hot, "not in any HOT_RELOAD_MAP")

# Non-setup restart fields
EXTRA_RESTART = [
    "checkpointer.path",
    "session.idle_reset_minutes",
    "plugins.enabled",
    "canvas.enabled",
    "voice.enabled",
    "task.enabled",
    "session_search.enabled",
]
for i, field in enumerate(EXTRA_RESTART, len(SETUP_FIELDS) + len(EXTRA_HOT) + 1):
    is_restart = any(field.startswith(p) for p in ReloadPlan.RESTART_PREFIXES)
    check(i, f"EXTRA {field} -> restart", is_restart, "not in any RESTART_PREFIXES")


# ============================================================
# SECTION B: ReloadPlan.build logic (29-36)
# ============================================================
print("\n== SECTION B: ReloadPlan.build logic ==")

# 29: Pure hot-reload change
changes = [ConfigChange("model.name", "gpt-4o", "deepseek-chat")]
plan = ReloadPlan.build(changes)
check(29, "Pure hot: requires_restart=False", not plan.requires_restart)
check(29, "Pure hot: has reload_model", any(a.action == "reload_model" for a in plan.actions))

# 30: Pure restart change
changes = [ConfigChange("gateway.port", 18080, 9090)]
plan = ReloadPlan.build(changes)
check(30, "Pure restart: requires_restart=True", plan.requires_restart)
check(30, "Pure restart: no hot actions", len(plan.actions) == 0)

# 31: Mixed change (Bug #4 scenario)
changes = [ConfigChange("gateway.port", 18080, 9090), ConfigChange("model.name", "a", "b")]
plan = ReloadPlan.build(changes)
check(31, "Mixed: requires_restart=True", plan.requires_restart)
check(31, "Mixed: still has reload_model", any(a.action == "reload_model" for a in plan.actions))

# 32: Multiple hot changes deduplicate
changes = [ConfigChange("model.name", "a", "b"), ConfigChange("model.temperature", 1.0, 0.5)]
plan = ReloadPlan.build(changes)
model_actions = [a for a in plan.actions if a.action == "reload_model"]
check(32, "Dedup: 1 reload_model action", len(model_actions) == 1)

# 33: agents.* specific matching
for field in ["agents.system_prompt", "agents.workspace", "agents.language", "agents.subagents"]:
    changes = [ConfigChange(field, "old", "new")]
    plan = ReloadPlan.build(changes)
    check(33, f"{field} triggers action", len(plan.actions) > 0)

# 34: channels.qq.dm_policy now restart
changes = [ConfigChange("channels.qq.dm_policy", "open", "allowlist")]
plan = ReloadPlan.build(changes)
check(34, "qq.dm_policy: requires_restart", plan.requires_restart)

# 35: Unknown field - no crash
changes = [ConfigChange("totally.unknown.field", "a", "b")]
plan = ReloadPlan.build(changes)
check(35, "Unknown field: no crash", len(plan.actions) == 0 and not plan.requires_restart)

# 36: agents.subagent_max_depth - not in map but safe
changes = [ConfigChange("agents.subagent_max_depth", 2, 5)]
plan = ReloadPlan.build(changes)
check(
    36,
    "subagent_max_depth: no action (covered by _config propagation)",
    len(plan.actions) == 0 and not plan.requires_restart,
)


# ============================================================
# SECTION C: DiffEngine (37-41)
# ============================================================
print("\n== SECTION C: DiffEngine ==")

# 37: Single field
old = AppConfig(model=ModelConfig(name="gpt-4o"))
new = AppConfig(model=ModelConfig(name="deepseek-chat"))
changes = DiffEngine.diff(old, new)
check(37, "Single field diff", len(changes) == 1 and changes[0].path == "model.name")

# 38: No change
changes = DiffEngine.diff(AppConfig(), AppConfig())
check(38, "No change: empty", len(changes) == 0)

# 39: Nested change
old = AppConfig(channels=ChannelsConfig(qq=QQConfig(dm_policy="open")))
new = AppConfig(channels=ChannelsConfig(qq=QQConfig(dm_policy="allowlist")))
changes = DiffEngine.diff(old, new)
check(39, "Nested path correct", any(c.path == "channels.qq.dm_policy" for c in changes))

# 40: Multi-field change
old = AppConfig(model=ModelConfig(name="a", temperature=1.0))
new = AppConfig(model=ModelConfig(name="b", temperature=0.5))
changes = DiffEngine.diff(old, new)
paths = {c.path for c in changes}
check(40, "Multi-field", "model.name" in paths and "model.temperature" in paths)

# 41: Default to set
old = AppConfig()
new = AppConfig(security=SecurityConfig(allow_private_urls=True))
changes = DiffEngine.diff(old, new)
check(41, "Default->set", any(c.path == "security.allow_private_urls" for c in changes))


# ============================================================
# SECTION D: execute() behavior (42-48)
# ============================================================
print("\n== SECTION D: execute() behavior ==")

from src.config_reload import ReloadExecutor


def make_mock_app(**kw):
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


async def run_execute_tests():
    # 42: Returns dict
    app = make_mock_app()
    ex = ReloadExecutor(app)
    r = await ex.execute(ReloadPlan(actions=[], requires_restart=False))
    check(42, "Returns dict", isinstance(r, dict) and "succeeded" in r and "failed" in r)

    # 43: requires_restart does NOT skip (Bug #4)
    app = make_mock_app()
    ex = ReloadExecutor(app)
    r = await ex.execute(ReloadPlan(actions=[ReloadAction(action="reload_security")], requires_restart=True))
    check(
        43,
        "restart+action: action still runs",
        "reload_security" in r["succeeded"],
        f"got succeeded={r['succeeded']} failed={r['failed']}",
    )

    # 44: Missing handler - no crash
    app = make_mock_app()
    ex = ReloadExecutor(app)
    r = await ex.execute(ReloadPlan(actions=[ReloadAction(action="reload_nonexistent")], requires_restart=False))
    check(44, "Missing handler: no crash", r["succeeded"] == [] and r["failed"] == [])

    # 45: Handler exception -> failed list
    app = make_mock_app()
    ex = ReloadExecutor(app)

    async def boom():
        raise RuntimeError("boom")

    ex._do_reload_model = boom
    r = await ex.execute(ReloadPlan(actions=[ReloadAction(action="reload_model")], requires_restart=False))
    check(45, "Exception -> failed", "reload_model" in r["failed"])

    # 46: _do_reload_security succeeds
    app = make_mock_app()
    ex = ReloadExecutor(app)
    r = await ex.execute(ReloadPlan(actions=[ReloadAction(action="reload_security")], requires_restart=False))
    check(46, "reload_security OK", "reload_security" in r["succeeded"])

    # 47: _do_reload_memory disabled - early return OK
    app = make_mock_app()
    app.config = AppConfig(memory=MemoryConfig(enabled=False))
    ex = ReloadExecutor(app)
    r = await ex.execute(ReloadPlan(actions=[ReloadAction(action="reload_memory")], requires_restart=False))
    check(47, "memory disabled: OK", "reload_memory" in r["succeeded"])

    # 48: Partial failure
    app = make_mock_app()
    ex = ReloadExecutor(app)

    async def boom():
        raise RuntimeError("boom")

    ex._do_reload_model = boom
    r = await ex.execute(
        ReloadPlan(
            actions=[
                ReloadAction(action="reload_model"),
                ReloadAction(action="reload_security"),
            ],
            requires_restart=False,
        )
    )
    check(
        48,
        "Partial fail: model fails, security ok",
        "reload_model" in r["failed"] and "reload_security" in r["succeeded"],
    )


asyncio.run(run_execute_tests())


# ============================================================
# SECTION E: Config propagation (49-53)
# ============================================================
print("\n== SECTION E: Config propagation ==")


async def run_propagation_tests():
    from src.app import ServiceContainer

    # 49: agent_loop._config updated
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
    check(49, "agent_loop._config is new_config", app.agent_loop._config is new_cfg)

    # 50: compressor.config updated
    check(50, "compressor.config is new.compression", app.agent_loop._compressor.config is new_cfg.compression)

    # 51: dispatcher._config updated
    check(51, "dispatcher._config is new_config", app.dispatcher._config is new_cfg)

    # 52: qq.config updated
    check(52, "qq.config is new QQConfig", app.qq.config is new_cfg.channels.qq)

    # 53: Raises on failed -> retry possible
    app2 = ServiceContainer.__new__(ServiceContainer)
    app2.config = AppConfig()
    app2.agent_loop = MagicMock()
    app2.agent_loop._config = app2.config
    app2.agent_loop._compressor = MagicMock()
    app2.dispatcher = MagicMock()
    app2.qq = MagicMock()
    app2.weixin = MagicMock()
    app2._reload_executor = MagicMock()
    app2._reload_executor.execute = AsyncMock(return_value={"succeeded": [], "failed": ["reload_memory"]})

    raised = False
    try:
        await app2.on_config_reload(app2.config, AppConfig(), ReloadPlan(actions=[], requires_restart=False))
    except RuntimeError:
        raised = True
    check(53, "Raises on failed handler", raised)


asyncio.run(run_propagation_tests())


# ============================================================
# SECTION F: _apply_reload retry (54-56)
# ============================================================
print("\n== SECTION F: _apply_reload retry ==")


async def run_retry_tests():
    import tempfile

    tmp = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w", encoding="utf-8")
    tmp.write("model:\n  name: gpt-4o\n")
    tmp.close()

    call_count = 0

    async def failing_reload(old, new, plan):
        nonlocal call_count
        call_count += 1
        raise RuntimeError("deliberate")

    from src.config_watcher import ConfigWatcher

    watcher = ConfigWatcher(path=tmp.name, on_reload=failing_reload)
    old_current = watcher._current
    old_hash = watcher._hash

    # 54: Failed reload does NOT update _current
    await watcher._apply_reload()
    check(54, "Fail: _current unchanged", watcher._current is old_current)

    # 55: Failed reload does NOT update _hash
    with open(tmp.name, "w", encoding="utf-8") as f:
        f.write("model:\n  name: deepseek-chat\n")
    await watcher._apply_reload()
    check(55, "Fail: _hash still old", watcher._hash == old_hash)

    # 56: Success DOES update both
    async def ok_reload(old, new, plan):
        pass

    watcher._on_reload = ok_reload
    await watcher._apply_reload()
    check(56, "Success: _current updated", watcher._current.model.name == "deepseek-chat")

    os.unlink(tmp.name)


asyncio.run(run_retry_tests())


# ============================================================
# SECTION G: Edge cases (57-50)
# ============================================================
print("\n== SECTION G: Edge cases ==")


async def run_edge_tests():
    from src.app import ServiceContainer
    from src.config_reload import ReloadExecutor

    # 57: All subsystems None - no crash
    app = ServiceContainer.__new__(ServiceContainer)
    app.config = AppConfig()
    app.agent_loop = None
    app.dispatcher = None
    app.qq = None
    app.weixin = None
    app._reload_executor = MagicMock()
    app._reload_executor.execute = AsyncMock(return_value={"succeeded": [], "failed": []})
    try:
        await app.on_config_reload(app.config, AppConfig(), ReloadPlan(actions=[], requires_restart=False))
        check(57, "All None: no crash", True)
    except Exception as e:
        check(57, "All None: no crash", False, str(e))

    # 58: Memory disabled cycle
    app = MagicMock()
    app.config = AppConfig(memory=MemoryConfig(enabled=False))
    app.memory_searcher = None
    ex = ReloadExecutor(app)
    r = await ex.execute(ReloadPlan(actions=[ReloadAction(action="reload_memory")], requires_restart=False))
    check(
        58,
        "Memory disabled->disabled: OK, searcher stays None",
        app.memory_searcher is None and "reload_memory" in r["succeeded"],
    )

    # 59: Compression change -> reload_model
    changes = [ConfigChange("compression.threshold_percent", 0.6, 0.8)]
    plan = ReloadPlan.build(changes)
    check(59, "compression maps to reload_model", any(a.action == "reload_model" for a in plan.actions))

    # 60: memory_store change -> restart
    changes = [ConfigChange("memory_store.enabled", False, True)]
    plan = ReloadPlan.build(changes)
    check(60, "memory_store change: restart", plan.requires_restart)

    # 61: Two restart fields
    changes = [ConfigChange("gateway.host", "127.0.0.1", "0.0.0.0"), ConfigChange("channels.qq.enabled", False, True)]
    plan = ReloadPlan.build(changes)
    check(61, "Two restart fields: still restart", plan.requires_restart and len(plan.actions) == 0)

    # 62: tools. prefix covers all sub-fields
    for sub in [
        "exec.enabled",
        "web_search.api_key",
        "browser.enabled",
        "media_understanding.enabled",
        "policy.allow",
        "guardrails.enabled",
    ]:
        changes = [ConfigChange(f"tools.{sub}", "a", "b")]
        plan = ReloadPlan.build(changes)
        check(
            62,
            f"tools.{sub} -> reload_tools",
            any(a.action == "reload_tools" for a in plan.actions),
            f"actions={[a.action for a in plan.actions]}",
        )

    # 63: weixin channel change -> restart
    changes = [ConfigChange("channels.weixin.dm_policy", "open", "disabled")]
    plan = ReloadPlan.build(changes)
    check(63, "weixin.dm_policy: restart", plan.requires_restart)

    # 64: Model name change detected in diff
    old = AppConfig(model=ModelConfig(name="gpt-4o"))
    new_cfg = AppConfig(model=ModelConfig(name="deepseek-chat"))
    changes = DiffEngine.diff(old, new_cfg)
    check(
        64,
        "Model name diff detected",
        any(c.path == "model.name" for c in changes),
        f"paths={[c.path for c in changes]}",
    )

    # 65: _do_reload_memory with existing searcher (close called)
    app = MagicMock()
    app.config = AppConfig(memory=MemoryConfig(enabled=False))
    mock_searcher = AsyncMock()
    app.memory_searcher = mock_searcher
    ex = ReloadExecutor(app)
    await ex._do_reload_memory()
    check(65, "Old memory_searcher.close() called", mock_searcher.close.called)
    check(65, "memory_searcher set to None", app.memory_searcher is None)

    # 66: _do_reload_memory store cleanup on failure - force MemorySearcher to fail
    app = MagicMock()
    cfg = AppConfig(memory=MemoryConfig(enabled=True))
    app.config = cfg
    app.memory_searcher = None
    ex = ReloadExecutor(app)
    # Patch at the import source
    with patch("src.memory.search.MemorySearcher", side_effect=RuntimeError("searcher boom")):
        try:
            await ex._do_reload_memory()
        except Exception:
            pass
    check(66, "Memory failure: searcher stays None", app.memory_searcher is None)

    # 67: ReloadPlan with empty changes
    plan = ReloadPlan.build([])
    check(67, "Empty changes: no actions, no restart", len(plan.actions) == 0 and not plan.requires_restart)

    # 68: All HOT_RELOAD_MAP values have handlers
    from src.config_reload import ReloadExecutor

    all_ok = True
    missing = []
    for action_name in set(ReloadPlan.HOT_RELOAD_MAP.values()):
        if not hasattr(ReloadExecutor, f"_do_{action_name}"):
            all_ok = False
            missing.append(action_name)
    check(68, "All MAP values have handlers", all_ok, f"missing: {missing}")

    # 69: qq.config propagation uses correct sub-object
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
    old_qq_config = QQConfig(dm_policy="open")
    qq_mock.config = old_qq_config
    app.qq = qq_mock

    new_cfg = AppConfig(channels=ChannelsConfig(qq=QQConfig(dm_policy="allowlist")))
    await app.on_config_reload(app.config, new_cfg, ReloadPlan(actions=[], requires_restart=False))
    check(
        69,
        "qq.config replaced with new QQConfig",
        qq_mock.config is new_cfg.channels.qq,
        f"same={qq_mock.config is new_cfg.channels.qq}",
    )

    # 70: on_config_reload with weixin=None - no crash
    app.qq = MagicMock()
    app.weixin = None
    try:
        await app.on_config_reload(app.config, AppConfig(), ReloadPlan(actions=[], requires_restart=False))
        check(70, "weixin=None: no crash", True)
    except Exception as e:
        check(70, "weixin=None: no crash", False, str(e))


asyncio.run(run_edge_tests())


# ============================================================
# SUMMARY
# ============================================================
total = passed + failed
print(f"\n{'=' * 60}")
print(f"  TOTAL: {total}  PASSED: {passed}  FAILED: {failed}")
print(f"{'=' * 60}")
sys.exit(1 if failed else 0)
