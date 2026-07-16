"""Tests for AgentLoop — core execution, tool calls, approval, resume, limits."""

import asyncio
import contextlib
import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.client import ChatResponse
from src.agent.loop import AgentLoop, ApprovalPending
from src.agent.state import AgentState, MemoryStateStore
from src.agent.tooldef import ToolDef
from src.tools.memory_tools import MemorySaveNeedsApproval


def _make_tool(name: str, fn=None):
    if fn is None:

        async def fn(**kwargs):
            return f"{name} result"

    return ToolDef.from_function(fn, name=name)


def _make_tc(name: str, args: dict | None = None, tc_id: str | None = None):
    @dataclass
    class TC:
        id: str
        function: Any

    @dataclass
    class Fn:
        name: str
        arguments: str

    return TC(id=tc_id or f"call_{name}", function=Fn(name=name, arguments=json.dumps(args or {})))


def _make_config(**overrides):
    config = MagicMock()
    config.agents = MagicMock()
    config.agents.max_tool_rounds = 50
    config.agents.tool_output_cache_chars = 8000
    config.agents.lock_timeout = 5.0
    config.agents.system_prompt = "You are helpful."
    config.agents.workspace = "."
    config.agents.bootstrap_files = []
    config.agents.timezone = "UTC"
    config.auth = None
    config.tools = MagicMock()
    config.tools.policy = MagicMock()
    config.tools.policy.allow = ["*"]
    config.tools.policy.deny = []
    config.tools.policy.owner_only = []
    config.tools.guardrails = MagicMock()
    config.tools.guardrails.enabled = False
    config.compression = MagicMock()
    config.compression.enabled = False
    for k, v in overrides.items():
        setattr(config, k, v)
    return config


@pytest.fixture
def store():
    return MemoryStateStore()


@pytest.fixture
def config():
    return _make_config()


def _make_loop(store, config, tools=None, client=None):
    if tools is None:
        tools = []
    if client is None:
        client = AsyncMock()
    return AgentLoop(
        client=client,
        tools=tools,
        state_store=store,
        config=config,
    )


class TestSwapClient:
    def test_swap_client_updates_both_client_and_compressor(self, store, config):
        # finding #1: swap_client 必须同步更新 _client 和 _compressor._client
        # (压缩/摘要 _llm_summarize 用 _compressor._client;漏更新则切模型后压缩仍用旧模型)
        old_client = AsyncMock()
        loop = _make_loop(store, config, client=old_client)
        assert loop._compressor._client is old_client  # 构造期缓存

        new_client = AsyncMock()
        loop.swap_client(new_client)

        assert loop._client is new_client
        assert loop._compressor._client is new_client  # 压缩/摘要跟随新 client


class TestAgentLoopSingleTurn:
    @pytest.mark.asyncio
    async def test_no_tool_call_returns_immediately(self, store, config):
        client = AsyncMock()
        client.chat.return_value = ChatResponse(content="Hello!", tool_calls=[])
        loop = _make_loop(store, config, client=client)

        state = AgentState(messages=[{"role": "user", "content": "hi"}])
        result = await loop.run(state, "test_thread")

        assert result.messages[-1]["role"] == "assistant"
        assert result.messages[-1]["content"] == "Hello!"
        client.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_assistant_message_has_correct_structure(self, store, config):
        client = AsyncMock()
        client.chat.return_value = ChatResponse(content="world", tool_calls=[])
        loop = _make_loop(store, config, client=client)

        state = AgentState(messages=[{"role": "user", "content": "hi"}])
        result = await loop.run(state, "t1")

        last = result.messages[-1]
        assert last["role"] == "assistant"
        assert "content" in last


class TestAgentLoopToolCalls:
    @pytest.mark.asyncio
    async def test_single_tool_call_executes_and_loops(self, store, config):
        tool = _make_tool("get_weather")
        client = AsyncMock()

        call_count = 0

        async def fake_chat(messages, tools=None, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ChatResponse(
                    content="",
                    tool_calls=[_make_tc("get_weather", {"city": "Tokyo"})],
                )
            return ChatResponse(content="Tokyo is sunny.", tool_calls=[])

        client.chat.side_effect = fake_chat
        loop = _make_loop(store, config, tools=[tool], client=client)

        state = AgentState(messages=[{"role": "user", "content": "weather?"}])
        result = await loop.run(state, "t2")

        assert call_count == 2
        tool_msgs = [m for m in result.messages if m["role"] == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "call_get_weather"
        assert "Tokyo" in result.messages[-1]["content"] or "get_weather result" in tool_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self, store, config):
        client = AsyncMock()
        client.chat.return_value = ChatResponse(
            content="",
            tool_calls=[_make_tc("nonexistent_tool")],
        )

        call_count = 0

        async def fake_chat(messages, tools=None, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ChatResponse(content="", tool_calls=[_make_tc("nonexistent_tool")])
            return ChatResponse(content="Done", tool_calls=[])

        client.chat.side_effect = fake_chat
        loop = _make_loop(store, config, tools=[], client=client)

        state = AgentState(messages=[{"role": "user", "content": "test"}])
        result = await loop.run(state, "t3")

        tool_msgs = [m for m in result.messages if m["role"] == "tool"]
        assert len(tool_msgs) == 1
        assert "Unknown tool" in tool_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_in_one_turn(self, store, config):
        tool_a = _make_tool("tool_a")
        tool_b = _make_tool("tool_b")
        client = AsyncMock()

        call_count = 0

        async def fake_chat(messages, tools=None, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ChatResponse(
                    content="",
                    tool_calls=[
                        _make_tc("tool_a", tc_id="ca"),
                        _make_tc("tool_b", tc_id="cb"),
                    ],
                )
            return ChatResponse(content="Both done.", tool_calls=[])

        client.chat.side_effect = fake_chat
        loop = _make_loop(store, config, tools=[tool_a, tool_b], client=client)

        state = AgentState(messages=[{"role": "user", "content": "go"}])
        result = await loop.run(state, "t4")

        tool_msgs = [m for m in result.messages if m["role"] == "tool"]
        assert len(tool_msgs) == 2
        assert tool_msgs[0]["tool_call_id"] == "ca"
        assert tool_msgs[1]["tool_call_id"] == "cb"


class TestAgentLoopApproval:
    @pytest.mark.asyncio
    async def test_approval_pending_raised(self, store, config):
        from src.tools.exec import ApprovalNeededError

        async def approval_tool_fn(command: str = ""):
            raise ApprovalNeededError(command=command, denylisted=False)

        tool = _make_tool("exec_command", fn=approval_tool_fn)
        client = AsyncMock()
        client.chat.return_value = ChatResponse(
            content="",
            tool_calls=[_make_tc("exec_command", {"command": "rm -rf /"})],
        )
        loop = _make_loop(store, config, tools=[tool], client=client)

        mock_mgr = MagicMock()
        mock_mgr.cancel_pending = MagicMock()
        with (
            patch("src.agent.loop.AgentLoop._handle_approval") as mock_handle,
            patch("src.tools.approval.get_approval_manager", return_value=mock_mgr),
        ):
            mock_handle.side_effect = ApprovalPending(
                thread_id="t5",
                request_id="r1",
                tool_name="exec_command",
                command_preview="rm -rf /",
                tc_id="tc_123",
            )
            state = AgentState(messages=[{"role": "user", "content": "run"}])
            with pytest.raises(ApprovalPending):
                await loop.run(state, "t5")

    @pytest.mark.asyncio
    async def test_resume_allow_executes_tool(self, store, config):
        client = AsyncMock()

        call_count = 0

        async def fake_chat(messages, tools=None, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ChatResponse(content="", tool_calls=[_make_tc("echo", {"text": "hi"}, tc_id="tc1")])
            return ChatResponse(content="Resumed result.", tool_calls=[])

        client.chat.side_effect = fake_chat

        tool = _make_tool("echo")
        loop = _make_loop(store, config, tools=[tool], client=client)

        state = AgentState(
            messages=[
                {"role": "user", "content": "test"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "tc1", "type": "function", "function": {"name": "echo", "arguments": '{"text":"hi"}'}}
                    ],
                },
            ],
            pending_approval={"request_id": "r1", "tool_call_id": "tc1", "tool_name": "echo", "command_preview": ""},
        )
        await store.save("t6", state)

        result = await loop.resume("t6", "allow_once")
        assert result is not None
        assert result.pending_approval is None

    @pytest.mark.asyncio
    async def test_resume_deny_marks_denied(self, store, config):
        client = AsyncMock()
        client.chat.return_value = ChatResponse(content="Denied result.", tool_calls=[])

        tool = _make_tool("echo")
        loop = _make_loop(store, config, tools=[tool], client=client)

        state = AgentState(
            messages=[
                {"role": "user", "content": "test"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "tc2", "type": "function", "function": {"name": "echo", "arguments": "{}"}}],
                },
            ],
            pending_approval={"request_id": "r2", "tool_call_id": "tc2", "tool_name": "echo", "command_preview": ""},
        )
        await store.save("t7", state)

        result = await loop.resume("t7", "deny")
        tool_msgs = [m for m in result.messages if m["role"] == "tool"]
        assert any("denied" in m.get("content", "").lower() for m in tool_msgs)


class TestAgentLoopStatePersistence:
    @pytest.mark.asyncio
    async def test_state_saved_after_completion(self, store, config):
        client = AsyncMock()
        client.chat.return_value = ChatResponse(content="Saved!", tool_calls=[])
        loop = _make_loop(store, config, client=client)

        state = AgentState(messages=[{"role": "user", "content": "save me"}])
        await loop.run(state, "persist_test")

        loaded = await store.load("persist_test")
        assert loaded is not None
        assert any(m.get("content") == "Saved!" for m in loaded.messages)


class TestToolLoopGuardrails:
    def test_repeat_failure_blocks(self):
        from src.agent.guardrails import ToolLoopGuardrails

        g = ToolLoopGuardrails(repeat_fail_block=3)
        args = {"cmd": "rm -rf /"}
        for _ in range(2):
            g.record("exec_command", args, success=False)
        result = g.check("exec_command", args)
        assert result is None
        g.record("exec_command", args, success=False)
        result = g.check("exec_command", args)
        assert result is not None
        assert result.blocked is True
        assert "repeated failure" in result.reason

    def test_storm_blocks(self):
        from src.agent.guardrails import ToolLoopGuardrails

        g = ToolLoopGuardrails(storm_block=4)
        for i in range(3):
            g.record("exec_command", {"cmd": f"cmd_{i}"}, success=False)
        result = g.check("exec_command", {"cmd": "cmd_3"})
        assert result is None
        g.record("exec_command", {"cmd": "cmd_3"}, success=False)
        result = g.check("exec_command", {"cmd": "cmd_4"})
        assert result is not None
        assert result.blocked is True
        assert "failure storm" in result.reason

    def test_reset_clears_history(self):
        from src.agent.guardrails import ToolLoopGuardrails

        g = ToolLoopGuardrails(repeat_fail_block=2)
        args = {"cmd": "fail"}
        g.record("exec_command", args, success=False)
        g.reset()
        result = g.check("exec_command", args)
        assert result is None

    def test_mixed_success_resets_count(self):
        from src.agent.guardrails import ToolLoopGuardrails

        g = ToolLoopGuardrails(repeat_fail_block=3)
        args = {"cmd": "test"}
        g.record("exec_command", args, success=False)
        g.record("exec_command", args, success=False)
        g.record("exec_command", args, success=True)
        result = g.check("exec_command", args)
        assert result is None


class TestSanitizeSurrogates:
    def test_replaces_lone_surrogates(self):
        from src.agent.loop import AgentLoop

        loop = AgentLoop.__new__(AgentLoop)
        messages = [{"role": "user", "content": "hello\ud800world"}]
        sanitized = loop._sanitize_surrogates(messages)
        assert "\ud800" not in sanitized[0]["content"]
        assert "\ufffd" in sanitized[0]["content"]

    def test_handles_nested_dicts(self):
        from src.agent.loop import AgentLoop

        loop = AgentLoop.__new__(AgentLoop)
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "tc1", "function": {"name": "x", "arguments": '{"a": "\udc00"}'}},
                ],
            }
        ]
        sanitized = loop._sanitize_surrogates(messages)
        assert "\udc00" not in sanitized[0]["tool_calls"][0]["function"]["arguments"]

    def test_preserves_valid_content(self):
        from src.agent.loop import AgentLoop

        loop = AgentLoop.__new__(AgentLoop)
        messages = [{"role": "user", "content": "normal text"}]
        sanitized = loop._sanitize_surrogates(messages)
        assert sanitized[0]["content"] == "normal text"


class TestInterruptFlag:
    def test_interrupt_sets_flag(self):
        from src.agent.interrupt import InterruptFlag

        flag = InterruptFlag()
        flag.interrupt("stop now")
        is_int, msg = flag.check()
        assert is_int is True
        assert msg == "stop now"

    def test_steer_sets_text(self):
        from src.agent.interrupt import InterruptFlag

        flag = InterruptFlag()
        assert flag.steer("focus on X") is True
        assert flag.drain_steer() == "focus on X"
        assert flag.drain_steer() is None

    def test_interrupt_clears_steer(self):
        from src.agent.interrupt import InterruptFlag

        flag = InterruptFlag()
        flag.steer("steer text")
        flag.interrupt("stop")
        assert flag.drain_steer() is None
        is_int, msg = flag.check()
        assert is_int is True

    def test_steer_concatenates(self):
        from src.agent.interrupt import InterruptFlag

        flag = InterruptFlag()
        flag.steer("first")
        flag.steer("second")
        assert flag.drain_steer() == "first\nsecond"

    def test_steer_rejected_during_interrupt(self):
        from src.agent.interrupt import InterruptFlag

        flag = InterruptFlag()
        flag.interrupt()
        assert flag.steer("ignored") is False

    def test_clear_resets_all(self):
        from src.agent.interrupt import InterruptFlag

        flag = InterruptFlag()
        flag.interrupt("msg")
        flag.clear()
        is_int, msg = flag.check()
        assert is_int is False
        assert msg is None
        assert flag.drain_steer() is None

    def test_interrupt_sets_event(self):
        from src.agent.interrupt import InterruptFlag

        flag = InterruptFlag()
        event = flag.get_event()
        assert not event.is_set()
        flag.interrupt("msg")
        assert event.is_set()

    def test_clear_resets_event(self):
        from src.agent.interrupt import InterruptFlag

        flag = InterruptFlag()
        event = flag.get_event()
        flag.interrupt("msg")
        assert event.is_set()
        flag.clear()
        assert not event.is_set()


class TestInterruptInLoop:
    @pytest.mark.asyncio
    async def test_interrupt_stops_loop(self, store, config):
        tool = _make_tool("echo")
        client = AsyncMock()

        call_count = 0

        async def fake_chat(messages, tools=None, **kw):
            nonlocal call_count
            call_count += 1
            return ChatResponse(content="", tool_calls=[_make_tc("echo", {"text": "hi"})])

        client.chat.side_effect = fake_chat
        loop = _make_loop(store, config, tools=[tool], client=client)

        # Set interrupt before running
        flag = store.get_interrupt_flag("int_test")
        flag.interrupt("stop this")

        state = AgentState(messages=[{"role": "user", "content": "go"}])
        result = await loop.run(state, "int_test")

        # Loop should have stopped immediately
        assert call_count == 0
        # Interrupt message should be in state
        assert any(m.get("content") == "stop this" for m in result.messages)

    @pytest.mark.asyncio
    async def test_steer_injected_into_tool_result(self, store, config):
        config.agents.max_tool_rounds = 2
        tool = _make_tool("echo")
        client = AsyncMock()

        call_count = 0
        steer_event = asyncio.Event()

        async def fake_chat(messages, tools=None, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Signal that we're in the loop, then wait for steer
                steer_event.set()
                await asyncio.sleep(0.1)  # Give time for steer to be set
                return ChatResponse(content="", tool_calls=[_make_tc("echo", {"text": "hi"})])
            return ChatResponse(content="Done with steer.", tool_calls=[])

        client.chat.side_effect = fake_chat
        loop = _make_loop(store, config, tools=[tool], client=client)

        state = AgentState(messages=[{"role": "user", "content": "go"}])

        async def run_and_steer():
            await steer_event.wait()
            flag = store.get_interrupt_flag("steer_test")
            flag.steer("change direction")

        steer_task = asyncio.create_task(run_and_steer())
        try:
            result = await loop.run(state, "steer_test")

            # Check that steer text was appended as a user message (preserves KV cache)
            user_msgs = [m for m in result.messages if m.get("role") == "user"]
            assert any("[User guidance]" in m.get("content", "") for m in user_msgs)
        finally:
            steer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await steer_task

    @pytest.mark.asyncio
    async def test_interrupt_during_model_call(self, store, config):
        tool = _make_tool("echo")
        client = AsyncMock()

        async def slow_chat(messages, tools=None, **kw):
            await asyncio.sleep(10)
            return ChatResponse(content="should not reach", tool_calls=[])

        client.chat.side_effect = slow_chat
        loop = _make_loop(store, config, tools=[tool], client=client)

        flag = store.get_interrupt_flag("int_during_model")
        asyncio.get_running_loop().call_later(0.1, lambda: flag.interrupt("stop model"))

        state = AgentState(messages=[{"role": "user", "content": "go"}])
        result = await loop.run(state, "int_during_model")

        assert any(m.get("content") == "stop model" for m in result.messages)
        assert "should not reach" not in str(result.messages)

    @pytest.mark.asyncio
    async def test_interrupt_during_tool_execution(self, store, config):
        async def slow_tool(**kwargs):
            await asyncio.sleep(10)
            return "slow result"

        tool = _make_tool("slow_tool", fn=slow_tool)
        client = AsyncMock()
        call_count = 0

        async def fake_chat(messages, tools=None, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ChatResponse(content="", tool_calls=[_make_tc("slow_tool", {"x": "y"})])
            return ChatResponse(content="done", tool_calls=[])

        client.chat.side_effect = fake_chat
        loop = _make_loop(store, config, tools=[tool], client=client)

        flag = store.get_interrupt_flag("int_during_tool")
        asyncio.get_running_loop().call_later(0.1, lambda: flag.interrupt("stop tool"))

        state = AgentState(messages=[{"role": "user", "content": "go"}])
        result = await loop.run(state, "int_during_tool")

        assert any(m.get("content") == "stop tool" for m in result.messages)
        assert "slow result" not in str(result.messages)

    @pytest.mark.asyncio
    async def test_interrupt_event_clears_after_handling(self, store, config):
        flag = store.get_interrupt_flag("event_clear_test")
        event = flag.get_event()

        assert not event.is_set()

        flag.interrupt("test")
        assert event.is_set()

        is_int, msg = flag.check()
        assert is_int
        flag.clear()
        assert not event.is_set()


class TestToolCache:
    @pytest.fixture(autouse=True)
    def _isolate_cache_root(self, tmp_path, monkeypatch):
        """隔离缓存目录到 tmp_path，避免污染真实 ~/.flyclaw/temp。"""
        from src.agent import tool_cache as tc

        monkeypatch.setattr(tc, "cache_root", lambda: tmp_path)

    @pytest.mark.asyncio
    async def test_small_content_not_truncated(self):
        from src.agent.loop import AgentLoop

        loop = AgentLoop.__new__(AgentLoop)
        loop._config = None
        content = "short text"
        messages = [{"role": "tool", "tool_call_id": "tc1", "content": content}]
        await loop._truncate_large_outputs(messages, "test_thread")
        assert messages[0]["content"] == content

    @pytest.mark.asyncio
    async def test_large_content_truncated(self):
        from src.agent.loop import AgentLoop

        loop = AgentLoop.__new__(AgentLoop)
        loop._config = None
        content = "x" * 10000
        messages = [{"role": "tool", "tool_call_id": "tc1", "content": content}]
        await loop._truncate_large_outputs(messages, "test_thread")
        assert len(messages[0]["content"]) < len(content)
        assert "truncated" in messages[0]["content"]

    @pytest.mark.asyncio
    async def test_non_tool_messages_untouched(self):
        from src.agent.loop import AgentLoop

        loop = AgentLoop.__new__(AgentLoop)
        loop._config = None
        content = "x" * 10000
        messages = [{"role": "user", "content": content}]
        await loop._truncate_large_outputs(messages, "test_thread")
        assert messages[0]["content"] == content


class TestCleanTruncatedMarkers:
    """Tests for _clean_for_summary (moved from loop._clean_truncated_markers)."""

    def test_removes_truncated_marker(self):
        from src.compressor.compressor import _clean_for_summary

        msgs = [{"role": "tool", "content": "result", "_truncated": True}]
        cleaned = _clean_for_summary(msgs)
        assert "_truncated" not in cleaned[0]

    def test_removes_cache_file_path(self):
        from src.compressor.compressor import _clean_for_summary

        content = (
            "x" * 8000
            + "\n... [content truncated, 10000 chars total. Full content saved to: `/home/user/.flyclaw/temp/tool_cache/t/file.txt`]"
        )
        msgs = [{"role": "tool", "content": content, "_truncated": True}]
        cleaned = _clean_for_summary(msgs)
        assert "Full content saved to:" not in cleaned[0]["content"]
        assert "content truncated" in cleaned[0]["content"]

    def test_leaves_normal_content_unchanged(self):
        from src.compressor.compressor import _clean_for_summary

        content = "hello world"
        msgs = [{"role": "user", "content": content}]
        cleaned = _clean_for_summary(msgs)
        assert cleaned[0]["content"] == content

    def test_returns_new_objects_not_aliases(self):
        from src.compressor.compressor import _clean_for_summary

        msgs = [{"role": "user", "content": "hello"}]
        cleaned = _clean_for_summary(msgs)
        assert cleaned is not msgs
        assert cleaned[0] is not msgs[0]

    def test_no_error_on_missing_keys(self):
        from src.compressor.compressor import _clean_for_summary

        msgs = [{"role": "tool", "tool_call_id": "tc1"}]
        cleaned = _clean_for_summary(msgs)
        assert len(cleaned) == 1

    @pytest.mark.asyncio
    async def test_roundtrip_cache_and_strip(self, tmp_path, monkeypatch):
        """cache_large_output produces format that strip_cache_path can remove."""
        # 写到隔离目录，避免污染真实 ~/.flyclaw/temp 留下孤儿缓存目录
        from src.agent import tool_cache as tc

        monkeypatch.setattr(tc, "cache_root", lambda: tmp_path)

        from src.compressor.compressor import strip_cache_path

        big = "abc" * 5000  # 15000 chars > default 8000
        truncated, path = await tc.cache_large_output(big, "test_thread")
        assert path is not None, "should have cached to file"
        assert "Full content saved to:" in truncated

        stripped = strip_cache_path(truncated)
        assert "Full content saved to:" not in stripped
        assert "content truncated" in stripped


class TestFindSafeCut:
    def test_no_tool_calls(self):
        from src.compressor.compressor import _find_safe_cut

        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "bye"},
            {"role": "assistant", "content": "see you"},
        ]
        assert _find_safe_cut(msgs, 2) == 2

    def test_preserves_tool_group(self):
        from src.compressor.compressor import _find_safe_cut

        msgs = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "tc1", "type": "function", "function": {"name": "x", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "result"},
            {"role": "assistant", "content": "done"},
        ]
        # tail_count=2 → cut=2 → group adjust to 1 → forward: no user after 1, returns 0
        assert _find_safe_cut(msgs, 2) == 0

    def test_preserves_parallel_tool_group(self):
        from src.compressor.compressor import _find_safe_cut

        msgs = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "tc1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
                    {"id": "tc2", "type": "function", "function": {"name": "b", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "r1"},
            {"role": "tool", "tool_call_id": "tc2", "content": "r2"},
            {"role": "assistant", "content": "done"},
        ]
        # tail_count=3 → cut=2 → group adjust to 1 → forward: no user after 1, returns 0
        assert _find_safe_cut(msgs, 3) == 0

    def test_cut_before_group(self):
        from src.compressor.compressor import _find_safe_cut

        msgs = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "tc1", "type": "function", "function": {"name": "x", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "result"},
        ]
        # tail_count=3 keeps all, cut=0
        assert _find_safe_cut(msgs, 3) == 0

    def test_cut_after_group(self):
        from src.compressor.compressor import _find_safe_cut

        msgs = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "tc1", "type": "function", "function": {"name": "x", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "result"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "next"},
        ]
        # tail_count=2 → cut=3 (assistant) → forward to idx=4 (user "next")
        assert _find_safe_cut(msgs, 2) == 4


class TestParallelApprovalAllNeedApproval:
    """Regression: when multiple parallel tools all need approval, every one
    of them must eventually be presented — not just the first.

    The bug was that _execute_tools_parallel added "[已跳过]" fake results
    for subsequent ApprovalPending tools, so _resume_inner thought they
    already had results and never re-executed them.
    """

    @pytest.mark.asyncio
    async def test_no_fake_skipped_results_for_parallel_approvals(self, store, config):
        """Subsequent approval-pending tools should NOT get fake results."""
        from src.tools.exec import ApprovalNeededError

        async def needs_approval(**kwargs):
            raise ApprovalNeededError(command="danger", denylisted=False)

        tool = _make_tool("exec_command", fn=needs_approval)
        client = AsyncMock()
        client.chat.return_value = ChatResponse(
            content="",
            tool_calls=[
                _make_tc("exec_command", {"command": "cmd_a"}, tc_id="tc_a"),
                _make_tc("exec_command", {"command": "cmd_b"}, tc_id="tc_b"),
            ],
        )
        loop = _make_loop(store, config, tools=[tool], client=client)

        mock_mgr = MagicMock()
        mock_mgr.cancel_pending = MagicMock()
        with (
            patch("src.agent.loop.AgentLoop._handle_approval") as mock_handle,
            patch("src.tools.approval.get_approval_manager", return_value=mock_mgr),
        ):
            # First call raises ApprovalPending for tc_a
            # Second call (tc_b) also raises, but parallel results are
            # processed in order so only the first one is preserved.
            call_idx = 0

            def handle_side_effect(error, tc, state, thread_id):
                nonlocal call_idx
                tc_id = tc.id if hasattr(tc, "id") else tc.get("id")
                call_idx += 1
                raise ApprovalPending(
                    thread_id=thread_id,
                    request_id=f"r_{tc_id}",
                    tool_name="exec_command",
                    command_preview=f"cmd_{call_idx}",
                    tc_id=tc_id,
                )

            mock_handle.side_effect = handle_side_effect

            state = AgentState(messages=[{"role": "user", "content": "run"}])
            with pytest.raises(ApprovalPending) as exc_info:
                await loop.run(state, "parallel_approval_test")

            exc = exc_info.value

        # 1. First ApprovalPending is raised (tc_a)
        assert exc.tc_id == "tc_a"
        assert exc.request_id.startswith("r_tc_a")

        # 2. Second approval request was cancelled
        mock_mgr.cancel_pending.assert_called()

        # 3. Saved state should NOT contain "[已跳过]" for tc_b
        saved_state = await store.load("parallel_approval_test")
        tool_results = [m for m in saved_state.messages if m.get("role") == "tool"]
        skipped = [m for m in tool_results if "已跳过" in m.get("content", "")]
        assert len(skipped) == 0, f"Expected no fake '[已跳过]' results, but found: {skipped}"

        # 4. tc_b should not have any result at all — so it can be
        #    re-executed by _resume_inner later
        tc_ids_with_results = {m.get("tool_call_id") for m in tool_results}
        assert "tc_b" not in tc_ids_with_results

    @pytest.mark.asyncio
    async def test_successful_results_preserved_before_approval(self, store, config):
        """Tools that succeeded before the first approval should keep their results."""
        from src.tools.exec import ApprovalNeededError

        async def ok_tool(**kwargs):
            return "ok_result"

        async def needs_approval(**kwargs):
            raise ApprovalNeededError(command="danger", denylisted=False)

        tool_ok = _make_tool("ok_tool", fn=ok_tool)
        tool_danger = _make_tool("exec_command", fn=needs_approval)
        client = AsyncMock()
        client.chat.return_value = ChatResponse(
            content="",
            tool_calls=[
                _make_tc("ok_tool", tc_id="tc_ok"),
                _make_tc("exec_command", {"command": "rm"}, tc_id="tc_danger"),
            ],
        )
        loop = _make_loop(store, config, tools=[tool_ok, tool_danger], client=client)

        mock_mgr = MagicMock()
        mock_mgr.cancel_pending = MagicMock()
        with (
            patch("src.agent.loop.AgentLoop._handle_approval") as mock_handle,
            patch("src.tools.approval.get_approval_manager", return_value=mock_mgr),
        ):
            mock_handle.side_effect = ApprovalPending(
                thread_id="t_preserve",
                request_id="r_danger",
                tool_name="exec_command",
                command_preview="rm",
                tc_id="tc_danger",
            )

            state = AgentState(messages=[{"role": "user", "content": "go"}])
            with pytest.raises(ApprovalPending):
                await loop.run(state, "t_preserve")

        saved_state = await store.load("t_preserve")
        tool_results = [m for m in saved_state.messages if m.get("role") == "tool"]

        # The ok_tool result should be preserved
        ok_results = [m for m in tool_results if m.get("tool_call_id") == "tc_ok"]
        assert len(ok_results) == 1
        assert "ok_result" in ok_results[0]["content"]

    @pytest.mark.asyncio
    async def test_resume_re_executes_unresulted_tools(self, store, config):
        """After first approval, _resume_inner should re-execute tools without results."""
        from src.tools.exec import ApprovalNeededError

        call_log: list[str] = []

        async def needs_approval(command: str = ""):
            call_log.append(command)
            raise ApprovalNeededError(command=command, denylisted=False)

        tool = _make_tool("exec_command", fn=needs_approval)
        client = AsyncMock()
        call_count = 0

        async def fake_chat(messages, tools=None, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ChatResponse(
                    content="",
                    tool_calls=[
                        _make_tc("exec_command", {"command": "cmd_a"}, tc_id="tc_a"),
                        _make_tc("exec_command", {"command": "cmd_b"}, tc_id="tc_b"),
                    ],
                )
            return ChatResponse(content="Done.", tool_calls=[])

        client.chat.side_effect = fake_chat
        loop = _make_loop(store, config, tools=[tool], client=client)

        mock_mgr = MagicMock()
        mock_mgr.cancel_pending = MagicMock()
        mock_mgr.approve_session = MagicMock()

        with (
            patch("src.agent.loop.AgentLoop._handle_approval") as mock_handle,
            patch("src.tools.approval.get_approval_manager", return_value=mock_mgr),
        ):
            handle_idx = 0

            def handle_side_effect(error, tc, state, thread_id):
                nonlocal handle_idx
                handle_idx += 1
                tc_id = tc.id if hasattr(tc, "id") else tc.get("id")
                raise ApprovalPending(
                    thread_id=thread_id,
                    request_id=f"r_{tc_id}",
                    tool_name="exec_command",
                    command_preview=f"cmd_{handle_idx}",
                    tc_id=tc_id,
                )

            mock_handle.side_effect = handle_side_effect

            state = AgentState(messages=[{"role": "user", "content": "run"}])
            with pytest.raises(ApprovalPending):
                await loop.run(state, "resume_test")

        # Both tools were called during parallel execution
        assert "cmd_a" in call_log, f"cmd_a should have been attempted, got {call_log}"
        assert "cmd_b" in call_log, f"cmd_b should have been attempted, got {call_log}"

        # Verify saved state: tc_a is pending, tc_b has NO result
        saved_state = await store.load("resume_test")
        assert saved_state is not None
        assert saved_state.pending_approval is not None
        assert saved_state.pending_approval["tool_call_id"] == "tc_a"

        tool_results = [m for m in saved_state.messages if m.get("role") == "tool"]
        tc_ids_with_results = {m.get("tool_call_id") for m in tool_results}
        assert "tc_b" not in tc_ids_with_results, "tc_b should not have a result so _resume_inner will re-execute it"


class TestApprovalPendingPartialResults:
    def test_partial_results_default_empty(self):
        exc = ApprovalPending("t1", "r1", "exec", "cmd", tc_id="")
        assert exc.partial_results == []

    def test_partial_results_stored(self):
        exc = ApprovalPending("t1", "r1", "exec", "cmd", tc_id="", partial_results=[("tc1", "ok")])
        assert exc.partial_results == [("tc1", "ok")]

    def test_partial_results_is_independent_copy(self):
        data = [("tc1", "ok")]
        exc = ApprovalPending("t1", "r1", "exec", "cmd", tc_id="", partial_results=data)
        data.append(("tc2", "extra"))
        assert len(exc.partial_results) == 1


class TestApprovalPendingData:
    def test_tc_id_default_empty(self):
        exc = ApprovalPending("t1", "r1", "exec", "cmd", tc_id="")
        assert exc.tc_id == ""

    def test_tc_id_stored(self):
        exc = ApprovalPending("t1", "r1", "exec", "cmd", tc_id="tc_123")
        assert exc.tc_id == "tc_123"

    def test_to_pending_data_basic(self):
        exc = ApprovalPending("t1", "r1", "exec", "cmd", tc_id="tc_1")
        data = exc.to_pending_data()
        assert data == {
            "request_id": "r1",
            "tool_name": "exec",
            "command_preview": "cmd",
            "tool_call_id": "tc_1",
        }

    def test_to_pending_data_with_keys(self):
        exc = ApprovalPending("t1", "r1", "memory_delete", "k1,k2", keys=["k1", "k2"], tc_id="tc_2")
        data = exc.to_pending_data()
        assert data["memory_keys"] == ["k1", "k2"]
        assert data["tool_call_id"] == "tc_2"

    def test_to_pending_data_no_keys_no_memory_keys_field(self):
        exc = ApprovalPending("t1", "r1", "exec", "cmd", tc_id="tc_3")
        data = exc.to_pending_data()
        assert "memory_keys" not in data


class TestFindSafeCutEdgeCases:
    def test_adjacent_groups_cut_at_boundary(self):
        from src.compressor.compressor import _find_safe_cut

        msgs = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "tc1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "r1"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "tc2", "type": "function", "function": {"name": "b", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "tc2", "content": "r2"},
        ]
        # tail_count=2 → cut=3 (assistant+tc2) → forward: no user after 3, returns 0
        assert _find_safe_cut(msgs, 2) == 0

    def test_orphan_tool_result_ignored(self):
        from src.compressor.compressor import _find_safe_cut

        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "tool", "tool_call_id": "orphan_1", "content": "ghost"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "bye"},
        ]
        # tail_count=2 → cut=2 (assistant) → forward to idx=3 (user "bye")
        assert _find_safe_cut(msgs, 2) == 3

    def test_tool_call_without_result_no_group(self):
        from src.compressor.compressor import _find_safe_cut

        msgs = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "tc_no_result", "type": "function", "function": {"name": "x", "arguments": "{}"}},
                ],
            },
            {"role": "assistant", "content": "follow-up"},
            {"role": "user", "content": "next"},
        ]
        # tail_count=2 → cut=2 (assistant) → forward to idx=3 (user "next")
        assert _find_safe_cut(msgs, 2) == 3

    def test_empty_tool_call_id(self):
        from src.compressor.compressor import _find_safe_cut

        msgs = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "", "type": "function", "function": {"name": "x", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "", "content": "r"},
            {"role": "assistant", "content": "done"},
        ]
        # tail_count=2 → cut=2 (tool) → forward: no user after 2, returns 0
        assert _find_safe_cut(msgs, 2) == 0

    def test_single_element_list(self):
        from src.compressor.compressor import _find_safe_cut

        msgs = [{"role": "user", "content": "hi"}]
        assert _find_safe_cut(msgs, 1) == 0

    def test_tail_count_equals_length(self):
        from src.compressor.compressor import _find_safe_cut

        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "tc1", "type": "function", "function": {"name": "x", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "r"},
        ]
        assert _find_safe_cut(msgs, 2) == 0

    def test_no_user_in_tail_returns_zero(self):
        from src.compressor.compressor import _find_safe_cut

        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "tc1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "r1"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "tc2", "type": "function", "function": {"name": "b", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "tc2", "content": "r2"},
        ]
        assert _find_safe_cut(msgs, 2) == 0

    def test_adjustment_only_once(self):
        from src.compressor.compressor import _find_safe_cut

        msgs = [
            {"role": "user", "content": "u1"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "tc1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "r1"},
            {"role": "user", "content": "u2"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "tc2", "type": "function", "function": {"name": "b", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "tc2", "content": "r2"},
        ]
        # tail_count=2 → cut=4 (assistant+tc2) → forward: no user after 4, returns 0
        assert _find_safe_cut(msgs, 2) == 0

        # tail_count=3 → cut=3 (u2) → already on user, no change
        assert _find_safe_cut(msgs, 3) == 3

        # tail_count=4 → cut=2 → group adjust to 1 → forward to idx=3 (u2)
        assert _find_safe_cut(msgs, 4) == 3


class TestFindSafeCutUserTurnAlignment:
    def test_tail_starts_from_user_message(self):
        from src.compressor.compressor import _find_safe_cut

        msgs = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "u3"},
            {"role": "assistant", "content": "a3"},
        ]
        # tail_count=2 → cut=4 (u3) → already on user, no change
        assert _find_safe_cut(msgs, 2) == 4
        # tail_count=3 → cut=3 (a2) → forward to idx=4 (u3)
        assert _find_safe_cut(msgs, 3) == 4

    def test_tail_starts_from_user_with_tool_calls(self):
        from src.compressor.compressor import _find_safe_cut

        msgs = [
            {"role": "user", "content": "u1"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "tc1", "type": "function", "function": {"name": "x", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "r1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "tc2", "type": "function", "function": {"name": "y", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "tc2", "content": "r2"},
        ]
        # tail_count=3 → cut=4 (u2) → already on user, no change
        assert _find_safe_cut(msgs, 3) == 4
        # tail_count=4 → cut=3 (a1) → forward to idx=4 (u2)
        assert _find_safe_cut(msgs, 4) == 4

    def test_cut_already_on_user_message(self):
        from src.compressor.compressor import _find_safe_cut

        msgs = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
        ]
        # tail_count=2 → cut=2 (user u2) → already on user → no change
        assert _find_safe_cut(msgs, 2) == 2

    def test_multiple_user_messages_between_groups(self):
        from src.compressor.compressor import _find_safe_cut

        msgs = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "u3"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "tc1", "type": "function", "function": {"name": "x", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "r1"},
            {"role": "assistant", "content": "a3"},
            {"role": "user", "content": "u4"},
            {"role": "assistant", "content": "a4"},
        ]
        # tail_count=3 → cut=7 (a3) → forward to idx=8 (u4)
        assert _find_safe_cut(msgs, 3) == 8

    def test_no_user_message_before_cut(self):
        from src.compressor.compressor import _find_safe_cut

        msgs = [
            {"role": "assistant", "content": "a1"},
            {"role": "tool", "tool_call_id": "tc1", "content": "r1"},
            {"role": "assistant", "content": "a2"},
        ]
        # No user messages at all → returns 0 (no compression)
        assert _find_safe_cut(msgs, 2) == 0


# ---------------------------------------------------------------------------
# _memory_summary_queried flag behavior
# ---------------------------------------------------------------------------


class TestMemorySummaryQueried:
    """Tests for the _memory_summary_queried flag and _fetch_memory_summary()."""

    @pytest.mark.asyncio
    async def test_queried_flag_prevents_repeated_calls(self, store, config):
        """Once queried, subsequent calls return cache without hitting store."""
        loop = _make_loop(store, config)

        mock_store = AsyncMock()
        mock_store.list_all.return_value = [
            {"category": "fact", "content": "user likes cats"},
        ]

        with patch("src.tools.memory_tools.get_memory_store", return_value=mock_store):
            result1 = await loop._fetch_memory_summary()
            result2 = await loop._fetch_memory_summary()

        assert "cats" in result1
        assert result2 == result1
        # store.list_all should only be called once
        mock_store.list_all.assert_called_once()
        assert "dm" in loop._memory_summary_queried

    @pytest.mark.asyncio
    async def test_empty_items_sets_flag(self, store, config):
        """Empty memory list still sets flag to prevent repeated queries."""
        loop = _make_loop(store, config)

        mock_store = AsyncMock()
        mock_store.list_all.return_value = []

        with patch("src.tools.memory_tools.get_memory_store", return_value=mock_store):
            result1 = await loop._fetch_memory_summary()

        assert result1 == ""
        assert "dm" in loop._memory_summary_queried

        # Second call should not hit store
        with patch("src.tools.memory_tools.get_memory_store") as mock_get:
            result2 = await loop._fetch_memory_summary()
            mock_get.assert_not_called()
        assert result2 == ""

    @pytest.mark.asyncio
    async def test_exception_leaves_flag_false_for_retry(self, store, config):
        """On exception, flag stays False so next call retries."""
        loop = _make_loop(store, config)

        with patch("src.tools.memory_tools.get_memory_store", side_effect=RuntimeError("db down")):
            result = await loop._fetch_memory_summary()

        assert result == ""
        assert "dm" not in loop._memory_summary_queried

        # Next call should retry (call get_memory_store again)
        mock_store = AsyncMock()
        mock_store.list_all.return_value = [{"category": "fact", "content": "recovered"}]
        with patch("src.tools.memory_tools.get_memory_store", return_value=mock_store):
            result2 = await loop._fetch_memory_summary()
        assert "recovered" in result2
        assert "dm" in loop._memory_summary_queried

    @pytest.mark.asyncio
    async def test_invalidate_resets_both(self, store, config):
        """invalidate_memory_cache clears cache content and resets flag."""
        loop = _make_loop(store, config)
        loop._memory_summary_cache = {"dm": "## cached summary"}
        loop._memory_summary_queried = {"dm"}

        loop.invalidate_memory_cache()

        assert loop._memory_summary_cache.get("dm", "") == ""
        assert "dm" not in loop._memory_summary_queried

    @pytest.mark.asyncio
    async def test_memory_tool_usage_invalidates(self, store, config):
        """After memory tool execution, cache is cleared for fresh data."""
        loop = _make_loop(store, config)
        loop._memory_summary_cache = {"dm": "## old summary"}
        loop._memory_summary_queried = {"dm"}

        # Simulate what happens in _execute_tools_inner after memory tool use
        loop._memory_summary_cache.clear()
        loop._memory_summary_queried.clear()

        assert loop._memory_summary_cache.get("dm", "") == ""
        assert "dm" not in loop._memory_summary_queried

        # Next fetch should re-query store
        mock_store = AsyncMock()
        mock_store.list_all.return_value = [{"category": "fact", "content": "fresh data"}]
        with patch("src.tools.memory_tools.get_memory_store", return_value=mock_store):
            result = await loop._fetch_memory_summary()
        assert "fresh data" in result


class TestMemorySummaryScope:
    """群 scope 与 DM scope 的记忆注入隔离。

    _fetch_memory_summary 是提示词记忆注入的数据源——它返回什么,
    _build_system_prompt 就把什么注入 system prompt。这里专测它的群分支
    (isinstance GroupMemoryStore → list_all(group_id=…)) 与跨 scope 独立缓存,
    补上 TestMemorySummaryQueried 只覆盖 DM 的缺口。
    """

    @pytest.mark.asyncio
    async def test_group_scope_fetches_group_memories_with_group_id(self, store, config):
        """群 scope 下取记忆带 group_id 过滤, scope_key 落群而非 dm。"""
        from src.tools.memory_tools import GroupMemoryStore, set_memory_session

        loop = _make_loop(store, config)
        grp_store = MagicMock(spec=GroupMemoryStore)
        grp_store.list_all = AsyncMock(return_value=[{"category": "fact", "content": "群G1规则:禁止广告"}])

        set_memory_session("group", "G1")
        try:
            with patch("src.tools.memory_tools.get_memory_store", return_value=grp_store):
                result = await loop._fetch_memory_summary()

            grp_store.list_all.assert_awaited_once()
            assert grp_store.list_all.call_args.kwargs.get("group_id") == "G1"
            assert "禁止广告" in result
            assert "G1" in loop._memory_summary_queried
            assert "dm" not in loop._memory_summary_queried
        finally:
            set_memory_session("p2p", "")

    @pytest.mark.asyncio
    async def test_dm_and_group_scopes_cached_and_served_independently(self, store, config):
        """DM 与群两 scope 各自缓存、互不串: DM 注入不含群记忆, 群注入不含 DM 记忆。"""
        from src.tools.memory_tools import GroupMemoryStore, set_memory_session

        loop = _make_loop(store, config)
        dm_store = AsyncMock()
        dm_store.list_all.return_value = [{"category": "fact", "content": "DM专属事实X"}]
        grp_store = MagicMock(spec=GroupMemoryStore)
        grp_store.list_all = AsyncMock(return_value=[{"category": "fact", "content": "群专属事实Y"}])

        set_memory_session("p2p", "")
        with patch("src.tools.memory_tools.get_memory_store", return_value=dm_store):
            dm_result = await loop._fetch_memory_summary()

        set_memory_session("group", "G1")
        try:
            with patch("src.tools.memory_tools.get_memory_store", return_value=grp_store):
                grp_result = await loop._fetch_memory_summary()

            # 注入内容按 scope 隔离
            assert "DM专属事实X" in dm_result and "群专属事实Y" not in dm_result
            assert "群专属事实Y" in grp_result and "DM专属事实X" not in grp_result
            # 两 scope 各自缓存
            assert "dm" in loop._memory_summary_queried
            assert "G1" in loop._memory_summary_queried
        finally:
            set_memory_session("p2p", "")


class TestUsageAccumulation:
    @pytest.mark.asyncio
    async def test_usage_accumulated_across_turns(self, store, config):
        client = AsyncMock()
        call_count = 0

        def _chat_response(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ChatResponse(
                    content="",
                    tool_calls=[_make_tc("echo", {"text": "hi"})],
                    usage={"prompt_tokens": 100, "completion_tokens": 50, "cached_tokens": 80},
                )
            return ChatResponse(
                content="Done",
                tool_calls=[],
                usage={"prompt_tokens": 200, "completion_tokens": 60, "cached_tokens": 150},
            )

        client.chat.side_effect = _chat_response

        tools = [_make_tool("echo")]
        loop = _make_loop(store, config, tools=tools, client=client)

        state = AgentState(messages=[{"role": "user", "content": "hi"}])
        result = await loop.run(state, "usage_test")

        assert result.total_usage is not None
        assert result.total_usage["prompt_tokens"] == 300
        assert result.total_usage["completion_tokens"] == 110
        assert result.total_usage["cached_tokens"] == 230

    @pytest.mark.asyncio
    async def test_usage_none_when_no_usage_data(self, store, config):
        client = AsyncMock()
        client.chat.return_value = ChatResponse(content="Hello!", tool_calls=[], usage=None)

        loop = _make_loop(store, config, client=client)
        state = AgentState(messages=[{"role": "user", "content": "hi"}])
        result = await loop.run(state, "no_usage_test")

        assert result.total_usage is None

    @pytest.mark.asyncio
    async def test_usage_initialized_on_first_response(self, store, config):
        client = AsyncMock()
        client.chat.return_value = ChatResponse(
            content="Hello!",
            tool_calls=[],
            usage={"prompt_tokens": 50, "completion_tokens": 25, "cached_tokens": 0},
        )

        loop = _make_loop(store, config, client=client)
        state = AgentState(messages=[{"role": "user", "content": "hi"}])
        result = await loop.run(state, "init_usage_test")

        assert result.total_usage is not None
        assert result.total_usage["prompt_tokens"] == 50
        assert result.total_usage["completion_tokens"] == 25
        assert result.total_usage["cached_tokens"] == 0


class TestMemorySaveApprovalRouting:
    """memory(save) 自检 reject → MemorySaveNeedsApproval 路由到审批，且 owner 不可绕过。"""

    @pytest.mark.asyncio
    async def test_routes_to_handle_approval(self, store, config):
        async def boom(**kwargs):
            raise MemorySaveNeedsApproval(content="臆测内容", source_context="来源原文")

        tool = _make_tool("memory", boom)
        tc = _make_tc("memory", {"action": "save", "content": "臆测内容"})
        loop = _make_loop(store, config, tools=[tool])

        state = AgentState(messages=[{"role": "user", "content": "hi"}])
        with patch(
            "src.agent.loop.AgentLoop._handle_approval",
            new=AsyncMock(return_value="[pending]"),
        ) as mock_handle:
            result = await loop._execute_tool(tc, state, "t1")

        assert result == "[pending]"  # 走审批分支，而非 [error]
        mock_handle.assert_awaited_once()
        err = mock_handle.call_args.args[0]
        assert isinstance(err, MemorySaveNeedsApproval)

    @pytest.mark.asyncio
    async def test_owner_cannot_bypass_save(self, store, config, monkeypatch):
        """owner（rbac 允许 approval_bypass）也不能绕过 save 审批：内容把关不可跳过。"""
        exec_count = {"n": 0}

        async def boom(**kwargs):
            exec_count["n"] += 1
            raise MemorySaveNeedsApproval(content="臆测", source_context="来源原文")

        tool = _make_tool("memory", boom)
        tc = _make_tc("memory", {"action": "save", "content": "臆测"})
        loop = _make_loop(store, config, tools=[tool])

        owner = MagicMock()
        owner.user_id = "u1"
        owner.role.value = "owner"
        monkeypatch.setattr(loop, "_resolve_user", AsyncMock(return_value=owner))

        rbac = MagicMock()
        rbac.check_approval_bypass = MagicMock(return_value=True)
        monkeypatch.setattr("src.auth.rbac.get_rbac", MagicMock(return_value=rbac))

        state = AgentState(messages=[{"role": "user", "content": "hi"}])
        with patch(
            "src.agent.loop.AgentLoop._handle_approval",
            new=AsyncMock(return_value="[pending]"),
        ) as mock_handle:
            result = await loop._execute_tool(tc, state, "t1")

        # 走审批，未被 owner 绕过；tool 只执行一次（抛异常那次），bypass 未重执行
        mock_handle.assert_awaited_once()
        assert exec_count["n"] == 1
        assert result == "[pending]"

    def test_approval_pending_carries_source_context(self):
        """ApprovalPending 携带来源原文，供 message 文案展示。"""
        exc = ApprovalPending("t1", "r1", "memory_save", "content", tc_id="", source_context="来源原文")
        assert exc.source_context == "来源原文"

    def test_approval_pending_source_context_defaults_empty(self):
        exc = ApprovalPending("t1", "r1", "exec", "cmd", tc_id="")
        assert exc.source_context == ""

    def test_approval_pending_carries_memory_save_mode(self):
        """ApprovalPending 携带 memory_save_mode，供 message 文案区分 model/manual。"""
        exc = ApprovalPending("t1", "r1", "memory_save", "c", tc_id="", memory_save_mode="manual")
        assert exc.memory_save_mode == "manual"

    def test_approval_pending_memory_save_mode_defaults_empty(self):
        exc = ApprovalPending("t1", "r1", "exec", "cmd", tc_id="")
        assert exc.memory_save_mode == ""

    @pytest.mark.asyncio
    async def test_resume_deny_save_reexecutes_other_saves(self, store, config):
        """deny 一个 memory_save，同批其余 save 重新执行（不 associated denial）——逐条审批。"""
        client = AsyncMock()
        client.chat.return_value = ChatResponse(content="done", tool_calls=[])
        tool = _make_tool("memory")
        loop = _make_loop(store, config, tools=[tool], client=client)

        state = AgentState(
            messages=[
                {"role": "user", "content": "存几条"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "tc_s1",
                            "type": "function",
                            "function": {"name": "memory", "arguments": '{"action":"save","content":"A"}'},
                        },
                        {
                            "id": "tc_s2",
                            "type": "function",
                            "function": {"name": "memory", "arguments": '{"action":"save","content":"B"}'},
                        },
                    ],
                },
            ],
            pending_approval={
                "request_id": "r1",
                "tool_call_id": "tc_s1",
                "tool_name": "memory_save",
                "command_preview": "A",
            },
        )
        await store.save("t_b2", state)

        result = await loop.resume("t_b2", "deny")
        tool_msgs = {m["tool_call_id"]: m["content"] for m in result.messages if m.get("role") == "tool"}

        # save1 denied
        assert "denied" in tool_msgs.get("tc_s1", "").lower()
        # save2 不被 associated deny，而是重新执行（默认 fn 返 "memory result"）
        assert "Skipped due to associated" not in tool_msgs.get("tc_s2", ""), (
            f"save2 不应被 associated deny: {tool_msgs.get('tc_s2')!r}"
        )
        assert tool_msgs.get("tc_s2") == "memory result", f"save2 应被重新执行: {tool_msgs.get('tc_s2')!r}"

    @pytest.mark.asyncio
    async def test_resume_deny_save_skips_sibling_delete(self, store, config):
        """混合批次：deny 一个 memory_save，兄弟 memory(delete) 不该被重执，走 associated skip。

        门控按 action 区分：只有兄弟是 save 才逐条重执；delete 不该因兄弟 save 被否
        而连带重执（弹删除审批）。
        """
        client = AsyncMock()
        client.chat.return_value = ChatResponse(content="done", tool_calls=[])
        tool = _make_tool("memory")
        loop = _make_loop(store, config, tools=[tool], client=client)

        state = AgentState(
            messages=[
                {"role": "user", "content": "存一条并删一条"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "tc_s1",
                            "type": "function",
                            "function": {"name": "memory", "arguments": '{"action":"save","content":"A"}'},
                        },
                        {
                            "id": "tc_d1",
                            "type": "function",
                            "function": {"name": "memory", "arguments": '{"action":"delete","keys":["k1"]}'},
                        },
                    ],
                },
            ],
            pending_approval={
                "request_id": "r1",
                "tool_call_id": "tc_s1",
                "tool_name": "memory_save",
                "command_preview": "A",
            },
        )
        await store.save("t_mix1", state)

        result = await loop.resume("t_mix1", "deny")
        tool_msgs = {m["tool_call_id"]: m["content"] for m in result.messages if m.get("role") == "tool"}

        # save denied
        assert "denied" in tool_msgs.get("tc_s1", "").lower()
        # 兄弟 delete 不被重执（不返回 "memory result"），而是 associated skip
        assert tool_msgs.get("tc_d1") != "memory result", (
            f"delete 不该因兄弟 save 被否而重执: {tool_msgs.get('tc_d1')!r}"
        )
        assert "Skipped due to associated" in tool_msgs.get("tc_d1", ""), (
            f"delete 应走 associated skip: {tool_msgs.get('tc_d1')!r}"
        )

    @pytest.mark.asyncio
    async def test_resume_deny_delete_reexecutes_sibling_save(self, store, config):
        """混合批次：deny 一个 memory_delete，兄弟 memory(save) 不该被 associated skip 静默丢弃，应重执。

        反向对称：pending 是 delete 时，兄弟 save 仍逐条独立重执（不连带跳过）。
        """
        client = AsyncMock()
        client.chat.return_value = ChatResponse(content="done", tool_calls=[])
        tool = _make_tool("memory")
        loop = _make_loop(store, config, tools=[tool], client=client)

        state = AgentState(
            messages=[
                {"role": "user", "content": "删一条并存一条"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "tc_s1",
                            "type": "function",
                            "function": {"name": "memory", "arguments": '{"action":"save","content":"A"}'},
                        },
                        {
                            "id": "tc_d1",
                            "type": "function",
                            "function": {"name": "memory", "arguments": '{"action":"delete","keys":["k1"]}'},
                        },
                    ],
                },
            ],
            pending_approval={
                "request_id": "r1",
                "tool_call_id": "tc_d1",
                "tool_name": "memory_delete",
                "command_preview": "- [k1]: ...",
            },
        )
        await store.save("t_mix2", state)

        result = await loop.resume("t_mix2", "deny")
        tool_msgs = {m["tool_call_id"]: m["content"] for m in result.messages if m.get("role") == "tool"}

        # delete denied
        assert "denied" in tool_msgs.get("tc_d1", "").lower()
        # 兄弟 save 不被 associated skip，而是重执（默认 fn 返 "memory result"）
        assert "Skipped due to associated" not in tool_msgs.get("tc_s1", ""), (
            f"save 不该因兄弟 delete 被否而 associated skip: {tool_msgs.get('tc_s1')!r}"
        )
        assert tool_msgs.get("tc_s1") == "memory result", f"save 应被重新执行: {tool_msgs.get('tc_s1')!r}"

    @pytest.mark.asyncio
    async def test_resume_deny_save_reexecutes_sibling_get(self, store, config):
        """混合批次：deny 一个 memory_save，兄弟 memory(get) 是纯读，必须重执返回结果，不能 associated skip。

        回归守护：门控应收窄到"仅 delete 跳过"，否则 get/list 这类只读查询会被连带取消，
        模型拿不到它要的记忆列表。
        """
        client = AsyncMock()
        client.chat.return_value = ChatResponse(content="done", tool_calls=[])
        tool = _make_tool("memory")
        loop = _make_loop(store, config, tools=[tool], client=client)

        state = AgentState(
            messages=[
                {"role": "user", "content": "存一条并列一下"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "tc_s1",
                            "type": "function",
                            "function": {"name": "memory", "arguments": '{"action":"save","content":"A"}'},
                        },
                        {
                            "id": "tc_g1",
                            "type": "function",
                            "function": {"name": "memory", "arguments": '{"action":"list","mode":"recent"}'},
                        },
                    ],
                },
            ],
            pending_approval={
                "request_id": "r1",
                "tool_call_id": "tc_s1",
                "tool_name": "memory_save",
                "command_preview": "A",
            },
        )
        await store.save("t_mix_get", state)

        result = await loop.resume("t_mix_get", "deny")
        tool_msgs = {m["tool_call_id"]: m["content"] for m in result.messages if m.get("role") == "tool"}

        # save denied
        assert "denied" in tool_msgs.get("tc_s1", "").lower()
        # 兄弟 list 是纯读，必须重执返回结果，而非 associated skip
        assert "Skipped due to associated" not in tool_msgs.get("tc_g1", ""), (
            f"get/list 不该因兄弟 save 被否而 associated skip: {tool_msgs.get('tc_g1')!r}"
        )
        assert tool_msgs.get("tc_g1") == "memory result", f"get/list 应被重执: {tool_msgs.get('tc_g1')!r}"

    @pytest.mark.asyncio
    async def test_resume_deny_delete_real_save_raises_approval(self, store, config, monkeypatch):
        """真实 memory()（非 mock）：deny delete → 兄弟 save 重执走 manual 审批完整链路。

        填补 mock 盲区：假工具不抛异常，测不到重执 save 时 MemorySaveNeedsApproval
        → _handle_approval → ApprovalPending 中断 + pending 切到 save 的真实副作用。
        """
        import src._container as _c
        from src.config import MemoryStoreConfig
        from src.tools import memory_tools
        from src.tools.approval import ApprovalManager
        from src.tools.memory_tools import get_tools, set_memory_dialog_context, set_memory_session

        set_memory_session("p2p", "")
        set_memory_dialog_context("来源")

        # 真实 approval manager（request_approval 返真实 ApprovalRequest）+ manual 模式
        real_mgr = ApprovalManager()
        _ms_cfg = MemoryStoreConfig(save_approval_mode="manual")

        class _AgentLoop:
            _client = None  # manual 不自检，主模型不会被调

        class _Container:
            agent_loop = _AgentLoop()
            approval_manager = real_mgr

        _Container.config = MagicMock()
        _Container.config.memory_store = _ms_cfg
        monkeypatch.setattr(_c, "get_container", lambda: _Container())
        monkeypatch.setattr("src.tools.approval.get_approval_manager", lambda: real_mgr)

        # spy store：manual save 抛审批前不落库
        saved: dict = {}

        class _Store:
            async def remember(self, content, key="", category="fact", group_id=""):
                saved["content"] = content
                return json.dumps({"ok": True, "key": key or "k"})

        monkeypatch.setattr(memory_tools, "get_memory_store", AsyncMock(return_value=_Store()))

        tool = get_tools()[0]  # 真实 memory 工具
        client = AsyncMock()
        client.chat.return_value = ChatResponse(content="done", tool_calls=[])
        loop = _make_loop(store, config, tools=[tool], client=client)

        state = AgentState(
            messages=[
                {"role": "user", "content": "删一条存一条"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "tc_s1",
                            "type": "function",
                            "function": {"name": "memory", "arguments": '{"action":"save","content":"A"}'},
                        },
                        {
                            "id": "tc_d1",
                            "type": "function",
                            "function": {"name": "memory", "arguments": '{"action":"delete","keys":["k1"]}'},
                        },
                    ],
                },
            ],
            pending_approval={
                "request_id": "r1",
                "tool_call_id": "tc_d1",
                "tool_name": "memory_delete",
                "command_preview": "- [k1]: x",
            },
        )
        await store.save("t_real", state)

        # deny delete → 重执兄弟 save → manual 抛审批 → 冒泡成 ApprovalPending
        with pytest.raises(ApprovalPending) as ei:
            await loop.resume("t_real", "deny")

        # pending 切到 save（重执 save 的审批挂起）
        assert ei.value.tool_name == "memory_save"
        saved_state = await store.load("t_real")
        assert saved_state.pending_approval["tool_name"] == "memory_save"
        # delete 已 deny；save 未落库（待审批）
        tool_msgs = {m["tool_call_id"]: m["content"] for m in saved_state.messages if m.get("role") == "tool"}
        assert "denied" in tool_msgs.get("tc_d1", "").lower()
        assert saved == {}, "manual save 待审批，不应落库"

    @pytest.mark.asyncio
    async def test_resume_deny_without_pending_tc_id_no_crash(self, store, config):
        """pending_approval 缺 tool_call_id 时 deny 不该 UnboundLocalError。

        回归守护：deny 分支里 pending_tool 仅在 ``if pending_tc_id:`` 内赋值，
        兄弟循环无条件引用——pending_tc_id 为空且有 memory 兄弟时，旧代码崩。
        """
        client = AsyncMock()
        client.chat.return_value = ChatResponse(content="done", tool_calls=[])
        tool = _make_tool("memory")
        loop = _make_loop(store, config, tools=[tool], client=client)

        state = AgentState(
            messages=[
                {"role": "user", "content": "存"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "tc_s1",
                            "type": "function",
                            "function": {"name": "memory", "arguments": '{"action":"save","content":"A"}'},
                        },
                    ],
                },
            ],
            pending_approval={
                "request_id": "r1",
                "tool_name": "memory_save",  # 故意缺 tool_call_id
                "command_preview": "A",
            },
        )
        await store.save("t_noid", state)

        result = await loop.resume("t_noid", "deny")
        tool_msgs = {m["tool_call_id"]: m["content"] for m in result.messages if m.get("role") == "tool"}
        # 不崩；兄弟 save 仍被重执（pending_tool 从 pending.tool_name 兜底）
        assert tool_msgs.get("tc_s1") == "memory result"

    @pytest.mark.asyncio
    async def test_resume_deny_memory_save_uses_save_denied_msg(self, store, config):
        """memory_save deny 的 tool result 用'记忆保存已取消'，非通用 Command 文案。"""
        client = AsyncMock()
        client.chat.return_value = ChatResponse(content="done", tool_calls=[])
        tool = _make_tool("memory")
        loop = _make_loop(store, config, tools=[tool], client=client)

        state = AgentState(
            messages=[
                {"role": "user", "content": "存"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "tc_s1",
                            "type": "function",
                            "function": {"name": "memory", "arguments": '{"action":"save","content":"A"}'},
                        },
                    ],
                },
            ],
            pending_approval={
                "request_id": "r1",
                "tool_call_id": "tc_s1",
                "tool_name": "memory_save",
                "command_preview": "A",
            },
        )
        await store.save("t_dmsg", state)

        result = await loop.resume("t_dmsg", "deny")
        tool_msgs = {m["tool_call_id"]: m["content"] for m in result.messages if m.get("role") == "tool"}
        assert "记忆保存" in tool_msgs["tc_s1"], f"应提示记忆保存已取消: {tool_msgs.get('tc_s1')!r}"
        assert "Command execution" not in tool_msgs["tc_s1"]

    @pytest.mark.asyncio
    async def test_execute_tool_injects_dialog_context(self, store, config):
        """_execute_tool 执行工具时把最近对话注入 contextvar，供 save 自检对照来源。

        接线守门：若 _execute_tool 漏 set contextvar，spy 读到的 ctx 为空。
        """
        from src.tools.memory_tools import get_memory_dialog_context, build_dialog_context

        captured = {}

        async def spy_fn(**kwargs):
            captured["ctx"] = get_memory_dialog_context()
            return "ok"

        tool = _make_tool("spy", spy_fn)
        tc = _make_tc("spy", {})
        loop = _make_loop(store, config, tools=[tool])

        messages = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好啊"},
            {"role": "user", "content": "我用 Python 3.13"},
        ]
        state = AgentState(messages=messages)
        await loop._execute_tool(tc, state, "t1")

        assert captured.get("ctx") == build_dialog_context(messages)
        assert "Python 3.13" in captured.get("ctx", "")

    @pytest.mark.asyncio
    async def test_handle_approval_propagates_source_and_mode(self, store, config, monkeypatch):
        """_handle_approval 把 error 的 source_context/mode 传给 ApprovalPending。

        接线守门：若漏传，ApprovalPending 字段为空。
        """
        from src.tools.memory_tools import MemorySaveNeedsApproval

        loop = _make_loop(store, config)

        fake_req = MagicMock()
        fake_req.id = "r1"
        fake_mgr = MagicMock()
        fake_mgr.request_approval.return_value = fake_req
        monkeypatch.setattr("src.tools.approval.get_approval_manager", lambda: fake_mgr)

        error = MemorySaveNeedsApproval(content="内容", source_context="来源原文", mode="manual")
        tc = _make_tc("memory", {"action": "save", "content": "内容"})
        state = AgentState(messages=[{"role": "user", "content": "x"}])

        with pytest.raises(ApprovalPending) as ei:
            await loop._handle_approval(error, tc, state, "t1")

        assert ei.value.source_context == "来源原文"
        assert ei.value.memory_save_mode == "manual"
