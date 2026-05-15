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

        with patch("src.agent.loop.AgentLoop._handle_approval") as mock_handle:
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


class TestAgentLoopMaxRounds:
    @pytest.mark.asyncio
    async def test_max_tool_rounds_disables_tools(self, store):
        config = _make_config()
        config.agents.max_tool_rounds = 1

        tool = _make_tool("always_call")
        client = AsyncMock()

        async def fake_chat(messages, tools=None, **kw):
            has_tools = tools is not None and len(tools) > 0
            if has_tools:
                return ChatResponse(content="", tool_calls=[_make_tc("always_call")])
            return ChatResponse(content="No more tools.", tool_calls=[])

        client.chat.side_effect = fake_chat
        loop = _make_loop(store, config, tools=[tool], client=client)

        state = AgentState(messages=[{"role": "user", "content": "go"}])
        result = await loop.run(state, "t8")

        last = result.messages[-1]
        assert last["role"] == "assistant"


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


class TestAgentLoopToolPolicy:
    @pytest.mark.asyncio
    async def test_channel_filter_removes_feishu_tools_on_qq(self, store, config):
        tool_feishu = _make_tool("feishu_send")
        tool_qq = _make_tool("qq_send")
        tool_other = _make_tool("other_tool")
        client = AsyncMock()
        client.chat.return_value = ChatResponse(content="ok", tool_calls=[])

        loop = _make_loop(store, config, tools=[tool_feishu, tool_qq, tool_other], client=client)

        state = AgentState(
            messages=[{"role": "user", "content": "test"}],
            channel="qq",
        )
        await loop.run(state, "t9")

        call_args = client.chat.call_args
        openai_tools = call_args.kwargs.get("tools") or call_args[1].get("tools") or []
        tool_names = [t["function"]["name"] for t in openai_tools]
        assert "feishu_send" not in tool_names
        assert "qq_send" in tool_names
        assert "other_tool" in tool_names
