"""Tests for LiteGraphAgentLoop — core execution, tool calls, approval, resume, limits."""

import asyncio
import json
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.client import ChatResponse
from src.agent.litegraph_loop import LiteGraphAgentLoop, _estimate_tokens_simple
from src.agent.loop import ApprovalPending
from src.agent.state import AgentState, MemoryStateStore
from src.agent.tooldef import ToolDef


# ── Helpers ──────────────────────────────────────────────────


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

    return TC(
        id=tc_id or f"call_{name}",
        function=Fn(name=name, arguments=json.dumps(args or {})),
    )


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
    config.compression = MagicMock()
    config.compression.enabled = False
    config.checkpointer = MagicMock()
    config.checkpointer.path = ":memory:"
    config.tools = MagicMock()
    config.tools.policy = MagicMock()
    config.tools.policy.allow = ["*"]
    config.tools.policy.deny = []
    config.tools.policy.owner_only = []
    config.model = MagicMock()
    config.model.provider = "test"
    config.model.name = "test-model"
    for k, v in overrides.items():
        setattr(config, k, v)
    return config


def _make_loop(store, config, tools=None, client=None):
    """Create LiteGraphAgentLoop bypassing __init__ to avoid heavy imports."""
    if tools is None:
        tools = []
    if client is None:
        client = AsyncMock()
    loop = LiteGraphAgentLoop.__new__(LiteGraphAgentLoop)
    loop._client = client
    loop._tools = tools
    loop._store = store
    loop._config = config
    loop._skills_prompt = ""
    loop._ctx_window_tokens = 100000
    loop._tool_map = {t.name: t for t in tools}
    loop._compiled = None
    loop._checkpointer = None
    loop._compressor = None
    loop._context_files = []
    return loop


@contextmanager
def _graph_patches():
    """Patch all dependencies needed for graph execution."""
    with (
        patch("src.events.emit_async", new_callable=AsyncMock),
        patch("src.prompt.build_system_prompt", return_value="Test"),
        patch("src.tools.policy.apply_tool_policy", side_effect=lambda tools, *a, **kw: tools),
        patch("src.security.redact.redact", side_effect=lambda x: x),
    ):
        yield


# ── Unit Tests: Helper Methods ─────────────────────────────


class TestEstimateTokens:
    def test_string_content(self):
        assert _estimate_tokens_simple([{"role": "user", "content": "Hello world"}]) == 12

    def test_list_content(self):
        tokens = _estimate_tokens_simple(
            [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]
        )
        assert tokens == 10

    def test_empty(self):
        assert _estimate_tokens_simple([]) == 0


class TestTruncateLargeOutputs:
    def test_truncates_large_tool_output(self):
        loop = _make_loop(MemoryStateStore(), _make_config())
        msgs = [{"role": "tool", "content": "x" * 3000}]
        result = loop._truncate_large_outputs(msgs)
        assert "truncated" in result[0]["content"]
        assert len(result[0]["content"]) < 3000

    def test_keeps_small_tool_output(self):
        loop = _make_loop(MemoryStateStore(), _make_config())
        result = loop._truncate_large_outputs([{"role": "tool", "content": "small"}])
        assert result[0]["content"] == "small"

    def test_keeps_non_tool_messages(self):
        loop = _make_loop(MemoryStateStore(), _make_config())
        result = loop._truncate_large_outputs([{"role": "user", "content": "x" * 3000}])
        assert result[0]["content"] == "x" * 3000


class TestFixToolCallsArgs:
    def test_valid_json_passes(self):
        loop = _make_loop(MemoryStateStore(), _make_config())
        msgs = [
            {"role": "assistant", "tool_calls": [{"function": {"name": "t", "arguments": '{"k":1}'}}]}
        ]
        assert loop._fix_tool_calls_args(msgs) == msgs

    def test_invalid_json_does_not_crash(self):
        loop = _make_loop(MemoryStateStore(), _make_config())
        msgs = [
            {"role": "assistant", "tool_calls": [{"function": {"name": "t", "arguments": "bad"}}]}
        ]
        result = loop._fix_tool_calls_args(msgs)
        assert result == msgs


class TestBuildAssistantMsg:
    def test_basic_message(self):
        loop = _make_loop(MemoryStateStore(), _make_config())
        msg = loop._build_assistant_msg(ChatResponse(content="Hello!", tool_calls=[]))
        assert msg["role"] == "assistant"
        assert msg["content"] == "Hello!"
        assert "tool_calls" not in msg

    def test_with_tool_calls(self):
        loop = _make_loop(MemoryStateStore(), _make_config())
        msg = loop._build_assistant_msg(
            ChatResponse(content="", tool_calls=[_make_tc("get_weather", {"city": "Tokyo"}, "tc1")])
        )
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["id"] == "tc1"
        assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"city": "Tokyo"}

    def test_fixes_truncated_json(self):
        loop = _make_loop(MemoryStateStore(), _make_config())
        tc = _make_tc("test", tc_id="tc1")
        tc.function.arguments = '{"key": "value"'
        msg = loop._build_assistant_msg(ChatResponse(content="", tool_calls=[tc]))
        args = json.loads(msg["tool_calls"][0]["function"]["arguments"])
        assert isinstance(args, dict)

    def test_invalid_json_fallback_empty(self):
        loop = _make_loop(MemoryStateStore(), _make_config())
        tc = _make_tc("test", tc_id="tc1")
        tc.function.arguments = "not json at all"
        msg = loop._build_assistant_msg(ChatResponse(content="", tool_calls=[tc]))
        assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {}


class TestShouldContinue:
    def test_with_tool_calls(self):
        loop = _make_loop(MemoryStateStore(), _make_config())
        assert loop._should_continue(
            {"messages": [{"role": "assistant", "tool_calls": [{"id": "c1"}]}]}
        ) == "tools"

    def test_without_tool_calls(self):
        from litegraph import END

        loop = _make_loop(MemoryStateStore(), _make_config())
        assert loop._should_continue(
            {"messages": [{"role": "assistant", "content": "Hi"}]}
        ) == END

    def test_empty_messages(self):
        from litegraph import END

        loop = _make_loop(MemoryStateStore(), _make_config())
        assert loop._should_continue({"messages": []}) == END


class TestStateConversion:
    def test_state_to_input(self):
        loop = _make_loop(MemoryStateStore(), _make_config())
        state = AgentState(
            messages=[{"role": "user", "content": "hi"}],
            system_prompt="sys",
            sender_id="u1",
        )
        result = loop._state_to_input(state)
        assert result["messages"] == [{"role": "user", "content": "hi"}]
        assert result["system_prompt"] == "sys"

    def test_result_to_state_with_base(self):
        loop = _make_loop(MemoryStateStore(), _make_config())
        base = AgentState(messages=[], system_prompt="sys", sender_id="u1")
        result = loop._result_to_state(
            {"messages": [{"role": "assistant", "content": "hi"}], "system_prompt": "new"},
            base,
        )
        assert result.system_prompt == "new"
        assert result.sender_id == "u1"

    def test_result_to_state_no_base(self):
        loop = _make_loop(MemoryStateStore(), _make_config())
        state = loop._result_to_state({"messages": [{"role": "assistant", "content": "hi"}]})
        assert state.system_prompt == ""
        assert state.sender_id == ""

    def test_result_to_state_none_result(self):
        loop = _make_loop(MemoryStateStore(), _make_config())
        state = loop._result_to_state(None)
        assert state.messages == []


class TestStoreAndClose:
    def test_get_store(self):
        store = MemoryStateStore()
        loop = _make_loop(store, _make_config())
        assert loop.get_store() is store

    def test_close_with_checkpointer(self):
        loop = _make_loop(MemoryStateStore(), _make_config())
        mock_cp = MagicMock()
        loop._checkpointer = mock_cp
        loop.close()
        mock_cp.close.assert_called_once()
        assert loop._checkpointer is None

    def test_close_without_checkpointer(self):
        loop = _make_loop(MemoryStateStore(), _make_config())
        loop.close()

    def test_close_handles_error(self):
        loop = _make_loop(MemoryStateStore(), _make_config())
        mock_cp = MagicMock()
        mock_cp.close.side_effect = RuntimeError("boom")
        loop._checkpointer = mock_cp
        loop.close()
        assert loop._checkpointer is None


class TestGetMaxToolRounds:
    def test_no_config(self):
        assert _make_loop(MemoryStateStore(), None)._get_max_tool_rounds() == 50

    def test_configured(self):
        config = _make_config()
        config.agents.max_tool_rounds = 20
        assert _make_loop(MemoryStateStore(), config)._get_max_tool_rounds() == 20

    def test_zero_fallback(self):
        config = _make_config()
        config.agents.max_tool_rounds = 0
        assert _make_loop(MemoryStateStore(), config)._get_max_tool_rounds() == 50


class TestFilterTools:
    def test_feishu_channel_removes_qq_tools(self):
        tools = [_make_tool("qq_send"), _make_tool("other")]
        loop = _make_loop(MemoryStateStore(), _make_config(), tools=tools)
        state = {"channel": "feishu", "sender_id": "", "messages": [{"role": "user", "content": "hi"}]}
        with patch("src.tools.policy.apply_tool_policy", side_effect=lambda t, *a, **kw: t):
            filtered = loop._filter_tools(state)
        names = [t.name for t in filtered]
        assert "qq_send" not in names
        assert "other" in names

    def test_qq_channel_removes_feishu_tools(self):
        tools = [_make_tool("feishu_send"), _make_tool("qq_send"), _make_tool("other")]
        loop = _make_loop(MemoryStateStore(), _make_config(), tools=tools)
        state = {"channel": "qq", "sender_id": "", "messages": [{"role": "user", "content": "hi"}]}
        with patch("src.tools.policy.apply_tool_policy", side_effect=lambda t, *a, **kw: t):
            filtered = loop._filter_tools(state)
        names = [t.name for t in filtered]
        assert "feishu_send" not in names
        assert "qq_send" in names

    def test_qq_channel_removes_extra_feishu_tools(self):
        tools = [_make_tool("send_image_to_chat"), _make_tool("send_file_to_chat"), _make_tool("ok")]
        loop = _make_loop(MemoryStateStore(), _make_config(), tools=tools)
        state = {"channel": "qq", "sender_id": "", "messages": [{"role": "user", "content": "hi"}]}
        with patch("src.tools.policy.apply_tool_policy", side_effect=lambda t, *a, **kw: t):
            filtered = loop._filter_tools(state)
        names = [t.name for t in filtered]
        assert "send_image_to_chat" not in names
        assert "send_file_to_chat" not in names
        assert "ok" in names

    def test_max_rounds_disables_tools(self):
        config = _make_config()
        config.agents.max_tool_rounds = 1
        tools = [_make_tool("tool_a")]
        loop = _make_loop(MemoryStateStore(), config, tools=tools)
        state = {
            "channel": "",
            "sender_id": "",
            "messages": [
                {"role": "user", "content": "go"},
                {"role": "tool", "content": "result", "tool_call_id": "c1"},
            ],
        }
        with patch("src.tools.policy.apply_tool_policy", side_effect=lambda t, *a, **kw: t):
            assert loop._filter_tools(state) == []

    def test_no_config_passes_all(self):
        tools = [_make_tool("a"), _make_tool("b")]
        loop = _make_loop(MemoryStateStore(), None, tools=tools)
        state = {"channel": "", "sender_id": "", "messages": []}
        assert loop._filter_tools(state) == tools


class TestStaticFallback:
    def test_short_messages_unchanged(self):
        loop = _make_loop(MemoryStateStore(), _make_config())
        msgs = [{"role": "user", "content": "short"}]
        assert loop._static_fallback(msgs) == msgs

    def test_long_messages_truncated(self):
        loop = _make_loop(MemoryStateStore(), _make_config())
        msgs = [{"role": "user", "content": "x" * 500000}] * 10
        result = loop._static_fallback(msgs)
        total = sum(len(m.get("content", "")) for m in result)
        assert total < sum(len(m.get("content", "")) for m in msgs)


class TestExecuteTool:
    @pytest.mark.asyncio
    async def test_known_tool(self):
        tool = _make_tool("add")
        loop = _make_loop(MemoryStateStore(), _make_config(), tools=[tool])
        tc = _make_tc("add", {"a": 1})
        result = await loop._execute_tool(tc)
        assert "add result" in result

    @pytest.mark.asyncio
    async def test_unknown_tool(self):
        loop = _make_loop(MemoryStateStore(), _make_config(), tools=[])
        tc = _make_tc("nonexistent")
        result = await loop._execute_tool(tc)
        assert "Unknown tool" in result


class TestRedactAssistantContent:
    def test_redacts_string_content(self):
        loop = _make_loop(MemoryStateStore(), _make_config())
        msg = {"content": "secret stuff"}
        with patch("src.security.redact.redact", side_effect=lambda x: "[REDACTED]"):
            loop._redact_assistant_content(msg)
        assert msg["content"] == "[REDACTED]"

    def test_skips_empty_content(self):
        loop = _make_loop(MemoryStateStore(), _make_config())
        msg = {"content": ""}
        loop._redact_assistant_content(msg)
        assert msg["content"] == ""


# ── Integration Tests: Graph Execution ─────────────────────


class TestLiteGraphSingleTurn:
    @pytest.mark.asyncio
    async def test_no_tool_call_returns_immediately(self):
        client = AsyncMock()
        client.chat.return_value = ChatResponse(content="Hello!", tool_calls=[])
        loop = _make_loop(MemoryStateStore(), _make_config(), client=client)

        state = AgentState(messages=[{"role": "user", "content": "hi"}])
        with _graph_patches():
            result = await loop.run(state, "t_single_1")

        assert result.messages[-1]["role"] == "assistant"
        assert result.messages[-1]["content"] == "Hello!"

    @pytest.mark.asyncio
    async def test_assistant_message_structure(self):
        client = AsyncMock()
        client.chat.return_value = ChatResponse(content="world", tool_calls=[])
        loop = _make_loop(MemoryStateStore(), _make_config(), client=client)

        state = AgentState(messages=[{"role": "user", "content": "hi"}])
        with _graph_patches():
            result = await loop.run(state, "t_single_2")

        last = result.messages[-1]
        assert last["role"] == "assistant"
        assert "content" in last


class TestLiteGraphToolCalls:
    @pytest.mark.asyncio
    async def test_single_tool_call(self):
        tool = _make_tool("get_weather")
        client = AsyncMock()
        call_count = 0

        async def fake_chat(msgs, tools=None, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ChatResponse(content="", tool_calls=[_make_tc("get_weather", {"city": "Tokyo"})])
            return ChatResponse(content="Tokyo is sunny.", tool_calls=[])

        client.chat.side_effect = fake_chat
        loop = _make_loop(MemoryStateStore(), _make_config(), tools=[tool], client=client)

        state = AgentState(messages=[{"role": "user", "content": "weather?"}])
        with _graph_patches():
            result = await loop.run(state, "t_tool_1")

        assert call_count == 2
        tool_msgs = [m for m in result.messages if m["role"] == "tool"]
        assert len(tool_msgs) == 1
        assert "get_weather result" in tool_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self):
        client = AsyncMock()
        call_count = 0

        async def fake_chat(msgs, tools=None, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ChatResponse(content="", tool_calls=[_make_tc("nonexistent")])
            return ChatResponse(content="Done", tool_calls=[])

        client.chat.side_effect = fake_chat
        loop = _make_loop(MemoryStateStore(), _make_config(), tools=[], client=client)

        state = AgentState(messages=[{"role": "user", "content": "test"}])
        with _graph_patches():
            result = await loop.run(state, "t_tool_2")

        tool_msgs = [m for m in result.messages if m["role"] == "tool"]
        assert len(tool_msgs) == 1
        assert "Unknown tool" in tool_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_multiple_tool_calls(self):
        tool_a = _make_tool("tool_a")
        tool_b = _make_tool("tool_b")
        client = AsyncMock()
        call_count = 0

        async def fake_chat(msgs, tools=None, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ChatResponse(
                    content="",
                    tool_calls=[_make_tc("tool_a", tc_id="ca"), _make_tc("tool_b", tc_id="cb")],
                )
            return ChatResponse(content="Both done.", tool_calls=[])

        client.chat.side_effect = fake_chat
        loop = _make_loop(MemoryStateStore(), _make_config(), tools=[tool_a, tool_b], client=client)

        state = AgentState(messages=[{"role": "user", "content": "go"}])
        with _graph_patches():
            result = await loop.run(state, "t_tool_3")

        tool_msgs = [m for m in result.messages if m["role"] == "tool"]
        assert len(tool_msgs) == 2
        assert tool_msgs[0]["tool_call_id"] == "ca"
        assert tool_msgs[1]["tool_call_id"] == "cb"

    @pytest.mark.asyncio
    async def test_tool_chain_three_rounds(self):
        tool = _make_tool("step")
        client = AsyncMock()
        call_count = 0

        async def fake_chat(msgs, tools=None, **kw):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                return ChatResponse(content="", tool_calls=[_make_tc("step", tc_id=f"s{call_count}")])
            return ChatResponse(content="All done.", tool_calls=[])

        client.chat.side_effect = fake_chat
        loop = _make_loop(MemoryStateStore(), _make_config(), tools=[tool], client=client)

        state = AgentState(messages=[{"role": "user", "content": "chain"}])
        with _graph_patches():
            result = await loop.run(state, "t_tool_4")

        assert call_count == 4
        tool_msgs = [m for m in result.messages if m["role"] == "tool"]
        assert len(tool_msgs) == 3


class TestLiteGraphStatePersistence:
    @pytest.mark.asyncio
    async def test_state_saved_after_completion(self):
        client = AsyncMock()
        client.chat.return_value = ChatResponse(content="Saved!", tool_calls=[])
        store = MemoryStateStore()
        loop = _make_loop(store, _make_config(), client=client)

        state = AgentState(messages=[{"role": "user", "content": "save me"}])
        with _graph_patches():
            await loop.run(state, "t_persist_1")

        loaded = await store.aload("t_persist_1")
        assert loaded is not None
        assert any(m.get("content") == "Saved!" for m in loaded.messages)


class TestLiteGraphMaxRounds:
    @pytest.mark.asyncio
    async def test_max_rounds_limits_tool_calls(self):
        config = _make_config()
        config.agents.max_tool_rounds = 1
        tool = _make_tool("always_call")
        client = AsyncMock()

        async def fake_chat(msgs, tools=None, **kw):
            if tools:
                return ChatResponse(content="", tool_calls=[_make_tc("always_call")])
            return ChatResponse(content="No more tools.", tool_calls=[])

        client.chat.side_effect = fake_chat
        loop = _make_loop(MemoryStateStore(), config, tools=[tool], client=client)

        state = AgentState(messages=[{"role": "user", "content": "go"}])
        with _graph_patches():
            result = await loop.run(state, "t_max_1")

        assert result.messages[-1]["role"] == "assistant"


class TestLiteGraphApproval:
    @pytest.mark.asyncio
    async def test_approval_pending_raised(self):
        from src.tools.exec import ApprovalNeededError

        async def approval_fn(command: str = ""):
            raise ApprovalNeededError(command=command, denylisted=False)

        tool = _make_tool("exec_command", fn=approval_fn)
        client = AsyncMock()
        client.chat.return_value = ChatResponse(
            content="",
            tool_calls=[_make_tc("exec_command", {"command": "rm -rf /"}, "tc_ap")],
        )
        loop = _make_loop(MemoryStateStore(), _make_config(), tools=[tool], client=client)

        mock_mgr = MagicMock()
        mock_mgr.request_approval.return_value = MagicMock(request_id="r1")
        mock_mgr._durable = {}
        mock_mgr._save_durable = MagicMock()

        state = AgentState(messages=[{"role": "user", "content": "run"}])
        with _graph_patches(), patch(
            "src.tools.approval.get_approval_manager", return_value=mock_mgr
        ):
            with pytest.raises(ApprovalPending):
                await loop.run(state, "t_approval_1")

    @pytest.mark.asyncio
    async def test_resume_allow_executes(self):
        from src.tools.exec import ApprovalNeededError

        call_count = 0

        async def conditional_approval(command: str = ""):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ApprovalNeededError(command=command, denylisted=False)
            return f"executed: {command}"

        tool = _make_tool("exec_command", fn=conditional_approval)
        client = AsyncMock()
        chat_count = 0

        async def fake_chat(msgs, tools=None, **kw):
            nonlocal chat_count
            chat_count += 1
            if chat_count == 1:
                return ChatResponse(
                    content="",
                    tool_calls=[_make_tc("exec_command", {"command": "ls -la"}, "tc_r1")],
                )
            return ChatResponse(content="Done!", tool_calls=[])

        client.chat.side_effect = fake_chat
        store = MemoryStateStore()
        loop = _make_loop(store, _make_config(), tools=[tool], client=client)

        mock_mgr = MagicMock()
        mock_mgr.request_approval.return_value = MagicMock(request_id="r1")
        mock_mgr._durable = {}
        mock_mgr._save_durable = MagicMock()

        state = AgentState(messages=[{"role": "user", "content": "run"}])
        with _graph_patches(), patch(
            "src.tools.approval.get_approval_manager", return_value=mock_mgr
        ):
            with pytest.raises(ApprovalPending):
                await loop.run(state, "t_resume_1")

        # Resume with approval
        with _graph_patches(), patch(
            "src.tools.approval.get_approval_manager", return_value=mock_mgr
        ):
            result = await loop.resume("t_resume_1", "allow_once")

        assert result is not None
        tool_msgs = [m for m in result.messages if m["role"] == "tool"]
        assert any("executed" in m.get("content", "") for m in tool_msgs)

    @pytest.mark.asyncio
    async def test_resume_deny_skips_tool(self):
        from src.tools.exec import ApprovalNeededError

        call_count = 0

        async def always_needs_approval(command: str = ""):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ApprovalNeededError(command=command, denylisted=False)
            return "should not reach"

        tool = _make_tool("exec_command", fn=always_needs_approval)
        client = AsyncMock()
        chat_count = 0

        async def fake_chat(msgs, tools=None, **kw):
            nonlocal chat_count
            chat_count += 1
            if chat_count == 1:
                return ChatResponse(
                    content="",
                    tool_calls=[_make_tc("exec_command", {"command": "rm -rf /"}, "tc_d1")],
                )
            return ChatResponse(content="Denied result.", tool_calls=[])

        client.chat.side_effect = fake_chat
        store = MemoryStateStore()
        loop = _make_loop(store, _make_config(), tools=[tool], client=client)

        mock_mgr = MagicMock()
        mock_mgr.request_approval.return_value = MagicMock(request_id="r1")
        mock_mgr._durable = {}
        mock_mgr._save_durable = MagicMock()

        state = AgentState(messages=[{"role": "user", "content": "run"}])
        with _graph_patches(), patch(
            "src.tools.approval.get_approval_manager", return_value=mock_mgr
        ):
            with pytest.raises(ApprovalPending):
                await loop.run(state, "t_deny_1")

        with _graph_patches(), patch(
            "src.tools.approval.get_approval_manager", return_value=mock_mgr
        ):
            result = await loop.resume("t_deny_1", "deny")

        tool_msgs = [m for m in result.messages if m["role"] == "tool"]
        assert any("denied" in m.get("content", "").lower() for m in tool_msgs)


class TestLiteGraphDenyCascade:
    @pytest.mark.asyncio
    async def test_deny_cascades_to_remaining_tools(self):
        from src.tools.exec import ApprovalNeededError

        call_count = 0

        async def needs_approval(command: str = ""):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ApprovalNeededError(command=command, denylisted=False)
            return "ok"

        tool = _make_tool("exec_command", fn=needs_approval)
        client = AsyncMock()
        chat_count = 0

        async def fake_chat(msgs, tools=None, **kw):
            nonlocal chat_count
            chat_count += 1
            if chat_count == 1:
                return ChatResponse(
                    content="",
                    tool_calls=[
                        _make_tc("exec_command", {"command": "cmd1"}, "tc_c1"),
                        _make_tc("exec_command", {"command": "cmd2"}, "tc_c2"),
                    ],
                )
            return ChatResponse(content="Cascade done.", tool_calls=[])

        client.chat.side_effect = fake_chat
        store = MemoryStateStore()
        loop = _make_loop(store, _make_config(), tools=[tool], client=client)

        mock_mgr = MagicMock()
        mock_mgr.request_approval.return_value = MagicMock(request_id="r1")
        mock_mgr._durable = {}
        mock_mgr._save_durable = MagicMock()

        state = AgentState(messages=[{"role": "user", "content": "run"}])
        with _graph_patches(), patch(
            "src.tools.approval.get_approval_manager", return_value=mock_mgr
        ):
            with pytest.raises(ApprovalPending):
                await loop.run(state, "t_cascade_1")

        with _graph_patches(), patch(
            "src.tools.approval.get_approval_manager", return_value=mock_mgr
        ):
            result = await loop.resume("t_cascade_1", "deny")

        tool_msgs = [m for m in result.messages if m["role"] == "tool"]
        assert len(tool_msgs) == 2
        assert all("denied" in m.get("content", "").lower() for m in tool_msgs)


class TestLiteGraphMetadata:
    @pytest.mark.asyncio
    async def test_metadata_preserved_through_graph(self):
        client = AsyncMock()
        client.chat.return_value = ChatResponse(content="ok", tool_calls=[])
        loop = _make_loop(MemoryStateStore(), _make_config(), client=client)

        state = AgentState(
            messages=[{"role": "user", "content": "hi"}],
            sender_id="user123",
            chat_id="chat456",
            channel="qq",
        )
        with _graph_patches():
            result = await loop.run(state, "t_meta_1")

        assert result.sender_id == "user123"
        assert result.chat_id == "chat456"
        assert result.channel == "qq"
