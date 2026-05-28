"""Tests for AgentLoop — core execution, tool calls, approval, resume, limits."""

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.client import ChatResponse
from src.agent.loop import AgentLoop, ApprovalPending
from src.agent.state import AgentState, MemoryStateStore
from src.agent.tooldef import ToolDef


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
        with patch("src.agent.loop.AgentLoop._handle_approval") as mock_handle, \
             patch("src.tools.approval.get_approval_manager", return_value=mock_mgr):
            mock_handle.side_effect = ApprovalPending(
                thread_id="t5", request_id="r1", tool_name="exec_command",
                command_preview="rm -rf /",
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
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "tc1", "type": "function", "function": {"name": "echo", "arguments": '{"text":"hi"}'}}
                ]},
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
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "tc2", "type": "function", "function": {"name": "echo", "arguments": '{}'}}
                ]},
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

        loaded = await store.aload("persist_test")
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


class TestSanitizeApiMessages:
    def test_inserts_stub_for_missing_result(self):
        from src.agent.loop import AgentLoop
        loop = AgentLoop.__new__(AgentLoop)
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc1", "type": "function", "function": {"name": "x", "arguments": "{}"}},
            ]},
        ]
        sanitized = loop._sanitize_api_messages(messages)
        tool_msgs = [m for m in sanitized if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "tc1"
        assert "tool result unavailable" in tool_msgs[0]["content"]

    def test_removes_orphan_result(self):
        from src.agent.loop import AgentLoop
        loop = AgentLoop.__new__(AgentLoop)
        messages = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc1", "type": "function", "function": {"name": "x", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "tc1", "content": "ok"},
            {"role": "tool", "tool_call_id": "tc2", "content": "orphan"},
        ]
        sanitized = loop._sanitize_api_messages(messages)
        tool_msgs = [m for m in sanitized if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "tc1"

    def test_no_change_when_balanced(self):
        from src.agent.loop import AgentLoop
        loop = AgentLoop.__new__(AgentLoop)
        messages = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc1", "type": "function", "function": {"name": "x", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "tc1", "content": "ok"},
        ]
        sanitized = loop._sanitize_api_messages(messages)
        assert len(sanitized) == len(messages)


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
        messages = [{"role": "assistant", "tool_calls": [
            {"id": "tc1", "function": {"name": "x", "arguments": '{"a": "\udc00"}'}},
        ]}]
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

        asyncio.create_task(run_and_steer())
        result = await loop.run(state, "steer_test")

        # Check that steer text was injected into a tool message
        tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
        assert any("User guidance" in m.get("content", "") for m in tool_msgs)

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
        asyncio.get_event_loop().call_later(0.1, lambda: flag.interrupt("stop model"))

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
        asyncio.get_event_loop().call_later(0.1, lambda: flag.interrupt("stop tool"))

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
    def test_small_content_not_truncated(self):
        from src.agent.loop import AgentLoop
        loop = AgentLoop.__new__(AgentLoop)
        content = "short text"
        messages = [{"role": "tool", "tool_call_id": "tc1", "content": content}]
        loop._truncate_large_outputs(messages, "test_thread")
        assert messages[0]["content"] == content

    def test_large_content_truncated(self):
        from src.agent.loop import AgentLoop
        loop = AgentLoop.__new__(AgentLoop)
        content = "x" * 10000
        messages = [{"role": "tool", "tool_call_id": "tc1", "content": content}]
        loop._truncate_large_outputs(messages, "test_thread")
        assert len(messages[0]["content"]) < len(content)
        assert "truncated" in messages[0]["content"]

    def test_non_tool_messages_untouched(self):
        from src.agent.loop import AgentLoop
        loop = AgentLoop.__new__(AgentLoop)
        content = "x" * 10000
        messages = [{"role": "user", "content": content}]
        loop._truncate_large_outputs(messages, "test_thread")
        assert messages[0]["content"] == content

    def test_sanitize_strips_truncated_flag(self):
        from src.agent.loop import AgentLoop
        loop = AgentLoop.__new__(AgentLoop)
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc1", "type": "function", "function": {"name": "x", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "tc1", "content": "ok", "_truncated": True},
        ]
        sanitized = loop._sanitize_api_messages(messages)
        tool_msg = [m for m in sanitized if m.get("role") == "tool"][0]
        assert "_truncated" not in tool_msg


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
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc1", "type": "function", "function": {"name": "x", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "tc1", "content": "result"},
            {"role": "assistant", "content": "done"},
        ]
        # tail_count=2 → cut=2 → group adjust to 1 → forward: no user after 1, stays at 1
        assert _find_safe_cut(msgs, 2) == 1

    def test_preserves_parallel_tool_group(self):
        from src.compressor.compressor import _find_safe_cut
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
                {"id": "tc2", "type": "function", "function": {"name": "b", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "tc1", "content": "r1"},
            {"role": "tool", "tool_call_id": "tc2", "content": "r2"},
            {"role": "assistant", "content": "done"},
        ]
        # tail_count=3 → cut=2 → group adjust to 1 → forward: no user after 1, stays at 1
        assert _find_safe_cut(msgs, 3) == 1

    def test_cut_before_group(self):
        from src.compressor.compressor import _find_safe_cut
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc1", "type": "function", "function": {"name": "x", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "tc1", "content": "result"},
        ]
        # tail_count=3 keeps all, cut=0
        assert _find_safe_cut(msgs, 3) == 0

    def test_cut_after_group(self):
        from src.compressor.compressor import _find_safe_cut
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc1", "type": "function", "function": {"name": "x", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "tc1", "content": "result"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "next"},
        ]
        # tail_count=2 → cut=3 (assistant) → forward to idx=4 (user "next")
        assert _find_safe_cut(msgs, 2) == 4


class TestApprovalPendingPartialResults:
    def test_partial_results_default_empty(self):
        exc = ApprovalPending("t1", "r1", "exec", "cmd")
        assert exc.partial_results == []

    def test_partial_results_stored(self):
        exc = ApprovalPending("t1", "r1", "exec", "cmd", partial_results=[("tc1", "ok")])
        assert exc.partial_results == [("tc1", "ok")]

    def test_partial_results_is_independent_copy(self):
        data = [("tc1", "ok")]
        exc = ApprovalPending("t1", "r1", "exec", "cmd", partial_results=data)
        data.append(("tc2", "extra"))
        assert len(exc.partial_results) == 1


class TestFindSafeCutEdgeCases:
    def test_adjacent_groups_cut_at_boundary(self):
        from src.compressor.compressor import _find_safe_cut
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "tc1", "content": "r1"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc2", "type": "function", "function": {"name": "b", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "tc2", "content": "r2"},
        ]
        # tail_count=2 → cut=3 (assistant+tc2) → forward: no user after 3, stays at 3
        assert _find_safe_cut(msgs, 2) == 3

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
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc_no_result", "type": "function", "function": {"name": "x", "arguments": "{}"}},
            ]},
            {"role": "assistant", "content": "follow-up"},
            {"role": "user", "content": "next"},
        ]
        # tail_count=2 → cut=2 (assistant) → forward to idx=3 (user "next")
        assert _find_safe_cut(msgs, 2) == 3

    def test_empty_tool_call_id(self):
        from src.compressor.compressor import _find_safe_cut
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "", "type": "function", "function": {"name": "x", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "", "content": "r"},
            {"role": "assistant", "content": "done"},
        ]
        # tail_count=2 → cut=2 (tool) → forward: no user after 2, stays at 2
        assert _find_safe_cut(msgs, 2) == 2

    def test_single_element_list(self):
        from src.compressor.compressor import _find_safe_cut
        msgs = [{"role": "user", "content": "hi"}]
        assert _find_safe_cut(msgs, 1) == 0

    def test_tail_count_equals_length(self):
        from src.compressor.compressor import _find_safe_cut
        msgs = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc1", "type": "function", "function": {"name": "x", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "tc1", "content": "r"},
        ]
        assert _find_safe_cut(msgs, 2) == 0

    def test_adjustment_only_once(self):
        from src.compressor.compressor import _find_safe_cut
        msgs = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "tc1", "content": "r1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc2", "type": "function", "function": {"name": "b", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "tc2", "content": "r2"},
        ]
        # tail_count=2 → cut=4 (assistant+tc2) → forward: no user after 4, stays at 4
        assert _find_safe_cut(msgs, 2) == 4

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
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc1", "type": "function", "function": {"name": "x", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "tc1", "content": "r1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc2", "type": "function", "function": {"name": "y", "arguments": "{}"}},
            ]},
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
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc1", "type": "function", "function": {"name": "x", "arguments": "{}"}},
            ]},
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
        # No user messages at all → cut stays at computed value
        assert _find_safe_cut(msgs, 2) == 1
