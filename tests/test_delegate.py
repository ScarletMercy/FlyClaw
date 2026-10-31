"""Tests for sub-agent delegation: timeout resolution, run registry touch, heartbeat."""

import time

import pytest


class TestResolveChildTimeout:
    def _make_config(self, child_timeout_seconds=600, child_timeout_floor=30):
        from src.config import DelegationConfig

        class _Cfg:
            delegation = DelegationConfig(
                child_timeout_seconds=child_timeout_seconds,
                child_timeout_floor=child_timeout_floor,
            )

        return _Cfg()

    def test_default_value(self):
        from src.agents.delegate import _resolve_child_timeout

        cfg = self._make_config()
        assert _resolve_child_timeout(None, cfg) == 600.0

    def test_explicit_param_takes_priority(self):
        from src.agents.delegate import _resolve_child_timeout

        cfg = self._make_config()
        assert _resolve_child_timeout(120, cfg) == 120.0

    def test_env_var_fallback(self, monkeypatch):
        from src.agents.delegate import _resolve_child_timeout

        cfg = self._make_config()
        monkeypatch.setenv("FLYCLAW_CHILD_TIMEOUT_SECONDS", "900")
        assert _resolve_child_timeout(None, cfg) == 900.0

    def test_explicit_overrides_env(self, monkeypatch):
        from src.agents.delegate import _resolve_child_timeout

        cfg = self._make_config()
        monkeypatch.setenv("FLYCLAW_CHILD_TIMEOUT_SECONDS", "900")
        assert _resolve_child_timeout(120, cfg) == 120.0

    def test_floor_applied(self):
        from src.agents.delegate import _resolve_child_timeout

        cfg = self._make_config(child_timeout_floor=30)
        assert _resolve_child_timeout(5, cfg) == 30.0

    def test_ceiling_applied(self):
        from src.agents.delegate import _resolve_child_timeout

        cfg = self._make_config(child_timeout_seconds=600)
        ceiling = max(600 * 3, 1800)
        assert _resolve_child_timeout(99999, cfg) == ceiling

    def test_invalid_env_ignored(self, monkeypatch):
        from src.agents.delegate import _resolve_child_timeout

        cfg = self._make_config()
        monkeypatch.setenv("FLYCLAW_CHILD_TIMEOUT_SECONDS", "not_a_number")
        assert _resolve_child_timeout(None, cfg) == 600.0

    def test_zero_explicit_uses_default(self):
        from src.agents.delegate import _resolve_child_timeout

        cfg = self._make_config()
        assert _resolve_child_timeout(0, cfg) == 600.0

    def test_negative_explicit_uses_default(self):
        from src.agents.delegate import _resolve_child_timeout

        cfg = self._make_config()
        assert _resolve_child_timeout(-10, cfg) == 600.0


class TestRunRegistryTouch:
    @pytest.mark.asyncio
    async def test_start_run_has_last_activity(self):
        from src.agents.run_registry import RunRegistry

        reg = RunRegistry()
        run_id = await reg.start_run("test_agent", "test task")
        run = reg._runs[run_id]
        assert run["last_activity_at"] is not None
        assert abs(run["last_activity_at"] - time.time()) < 2

    @pytest.mark.asyncio
    async def test_touch_updates_activity(self):
        from src.agents.run_registry import RunRegistry

        reg = RunRegistry()
        run_id = await reg.start_run("test_agent", "test task")
        old_ts = reg._runs[run_id]["last_activity_at"]

        time.sleep(0.05)
        reg.touch(run_id)

        new_ts = reg._runs[run_id]["last_activity_at"]
        assert new_ts > old_ts

    @pytest.mark.asyncio
    async def test_touch_completed_run_is_noop(self):
        from src.agents.run_registry import RunRegistry

        reg = RunRegistry()
        run_id = await reg.start_run("test_agent", "test task")
        await reg.complete_run(run_id, result="done")

        old_ts = reg._runs[run_id]["last_activity_at"]
        reg.touch(run_id)
        assert reg._runs[run_id]["last_activity_at"] == old_ts

    @pytest.mark.asyncio
    async def test_touch_nonexistent_run_is_noop(self):
        from src.agents.run_registry import RunRegistry

        reg = RunRegistry()
        reg.touch("nonexistent_id")

    @pytest.mark.asyncio
    async def test_timeout_run_sets_status(self):
        from src.agents.run_registry import RunRegistry

        reg = RunRegistry()
        run_id = await reg.start_run("test_agent", "test task")
        await reg.timeout_run(run_id, error="Timeout after 600s")

        run = reg._runs[run_id]
        assert run["status"] == "timeout"
        assert run["completed_at"] is not None

    @pytest.mark.asyncio
    async def test_get_active_tree_includes_idle_seconds(self):
        from src.agents.run_registry import RunRegistry

        reg = RunRegistry()
        run_id = await reg.start_run("test_agent", "test task")
        tree = await reg.get_active_tree()
        assert len(tree) == 1
        entry = tree[0]
        assert "last_activity_at" in entry
        assert "idle_seconds" in entry
        assert entry["idle_seconds"] < 5


class TestDelegationConfigDefaults:
    def test_default_values(self):
        from src.config import DelegationConfig

        cfg = DelegationConfig()
        assert cfg.child_timeout_seconds == 600
        assert cfg.child_timeout_floor == 30
        assert cfg.max_iterations == 50
        assert cfg.enabled is True
        assert cfg.max_concurrent == 3


class TestDelegateTaskFailureEnvelope:
    """失败信封契约：status != completed 必须返回 ToolResult.error——
    裸字符串会被包装层判成 success，审计与熔断全盲。"""

    @staticmethod
    def _patch_config(monkeypatch):
        from types import SimpleNamespace

        cfg = SimpleNamespace(
            delegation=SimpleNamespace(enabled=True),
            agents=SimpleNamespace(subagent_max_depth=2),
        )
        monkeypatch.setattr("src.config.load_config", lambda: cfg)
        # _run_single 已 mock，registry 实例不会被使用——绕开未初始化的 ServiceContainer
        monkeypatch.setattr("src.agents.run_registry.get_run_registry", lambda: object())

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", ["error", "timeout"])
    async def test_failure_status_returns_toolresult_error(self, monkeypatch, status):
        from src.agents import delegate
        from src.tools.result import ToolResult

        self._patch_config(monkeypatch)

        async def fake_run_single(*args, **kwargs):
            return {
                "agent_name": "probe",
                "status": status,
                "result": f"boom ({status})",
                "duration_seconds": 0.1,
            }

        monkeypatch.setattr(delegate, "_run_single", fake_run_single)

        out = await delegate.delegate_task("probe", "do")

        assert isinstance(out, ToolResult)
        assert out.is_error
        assert "boom" in out.content

    @pytest.mark.asyncio
    async def test_completed_returns_bare_string(self, monkeypatch):
        from src.agents import delegate

        self._patch_config(monkeypatch)

        async def fake_run_single(*args, **kwargs):
            return {"agent_name": "probe", "status": "completed", "result": "all good", "duration_seconds": 0.1}

        monkeypatch.setattr(delegate, "_run_single", fake_run_single)

        out = await delegate.delegate_task("probe", "do")

        assert isinstance(out, str)
        assert out == "all good"


class TestExtractResultText:
    """结果回退读 failed_tool_calls 集合。"""

    @staticmethod
    def _state(messages, failed=()):
        from types import SimpleNamespace

        return SimpleNamespace(messages=messages, failed_tool_calls=set(failed))

    def test_error_tool_result_skipped_even_without_prefix(self):
        from src.agents.delegate import _extract_result_text

        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "tc1", "function": {"name": "web_fetch", "arguments": "{}"}},
                    {"id": "tc2", "function": {"name": "send_file", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "Request timed out after 30s"},
            {"role": "tool", "tool_call_id": "tc2", "content": "The answer is 42"},
        ]
        out = _extract_result_text(self._state(messages, failed={"tc1"}))
        assert out == "The answer is 42"

    def test_assistant_content_wins_over_tool_results(self):
        from src.agents.delegate import _extract_result_text

        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "tc1", "function": {"name": "probe", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "partial"},
            {"role": "assistant", "content": "final answer"},
        ]
        assert _extract_result_text(self._state(messages)) == "final answer"

    def test_all_failed_falls_back_to_digest(self):
        from src.agents.delegate import _extract_result_text

        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "tc1", "function": {"name": "probe", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "Request timed out after 30s"},
        ]
        out = _extract_result_text(self._state(messages, failed={"tc1"}))
        assert out.startswith("[No final summary]")
        assert "Request timed out after 30s" in out
