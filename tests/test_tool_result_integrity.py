"""Tests for tool result integrity across compression, interrupts, and approval flows.

Ensures that tool_call / tool_result pairs are never split or lost during:
- Context compression (compress / compact)
- Parallel tool execution with interrupts
- Parallel tool execution with ApprovalPending
- Simple truncation of large outputs
"""

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.client import ChatResponse
from src.agent.loop import AgentLoop, ApprovalPending
from src.agent.state import AgentState, MemoryStateStore
from src.agent.tooldef import ToolDef
from src.config import CompressionConfig


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
    config.agents.language = "zh"
    config.agents.tool_progress_notifications = False
    config.auth = None
    config.tools = MagicMock()
    config.tools.policy = MagicMock()
    config.tools.policy.allow = ["*"]
    config.tools.policy.deny = []
    config.tools.policy.owner_only = []
    config.tools.guardrails = MagicMock()
    config.tools.guardrails.enabled = False
    config.compression = CompressionConfig()
    config.skills = MagicMock()
    config.skills.creation_nudge_interval = 0
    for k, v in overrides.items():
        setattr(config, k, v)
    return config


def _build_messages_with_groups(n_groups: int, extra_middle: int = 0) -> list[dict]:
    """Build a message list with n_groups of tool_call/result pairs + extra filler."""
    msgs: list[dict] = [{"role": "user", "content": "start"}]
    for g in range(n_groups):
        msgs.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": f"tc_{g}", "type": "function", "function": {"name": "tool_x", "arguments": "{}"}},
                ],
            }
        )
        msgs.append({"role": "tool", "tool_call_id": f"tc_{g}", "content": f"result_{g}"})
        for _ in range(extra_middle):
            msgs.append({"role": "assistant", "content": "filler text " * 20})
    msgs.append({"role": "user", "content": "end"})
    return msgs


def _verify_no_orphan_pairs(messages: list[dict]) -> list[str]:
    """Verify all tool_calls have matching tool_results and vice versa. Returns errors."""
    errors: list[str] = []
    tc_ids: set[str] = set()
    result_ids: set[str] = set()
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                tid = tc.get("id", "")
                if tid:
                    tc_ids.add(tid)
        if m.get("role") == "tool" and m.get("tool_call_id"):
            result_ids.add(m["tool_call_id"])
    missing_results = tc_ids - result_ids
    orphan_results = result_ids - tc_ids
    if missing_results:
        errors.append(f"Missing tool_results for: {missing_results}")
    if orphan_results:
        errors.append(f"Orphan tool_results for: {orphan_results}")
    return errors


# ======================================================================
# B: Compressor integration tests
# ======================================================================


class TestCompressToolGroupIntegrity:
    @pytest.mark.asyncio
    async def test_compress_preserves_tool_groups(self):
        from src.compressor.compressor import ContextCompressor

        cfg = CompressionConfig(enabled=False, tail_messages=4)
        compressor = ContextCompressor(cfg)

        messages = _build_messages_with_groups(10, extra_middle=2)

        result = compressor._compact(messages, context_window_tokens=200)

        errors = _verify_no_orphan_pairs(result)
        assert errors == [], "Compress produced orphan pairs: " + "; ".join(errors)

    @pytest.mark.asyncio
    async def test_compress_enabled_preserves_tool_groups(self):
        from src.compressor.compressor import ContextCompressor

        cfg = CompressionConfig(enabled=True, tail_messages=4, threshold_percent=0.01)
        client = AsyncMock()
        client.chat.return_value = ChatResponse(content="Summary of conversation.", tool_calls=[])
        compressor = ContextCompressor(cfg, client=client)

        messages = _build_messages_with_groups(10, extra_middle=3)

        result = await compressor.compress(messages, context_window_tokens=200)

        errors = _verify_no_orphan_pairs(result)
        assert errors == [], "Compress (enabled) produced orphan pairs: " + "; ".join(errors)


# ======================================================================
# C: Parallel tool + ApprovalPending
# ======================================================================


class TestParallelApprovalPending:
    @pytest.fixture(autouse=True)
    def _mock_approval_mgr(self):
        """Mock get_approval_manager to avoid ServiceContainer dependency."""
        mgr = MagicMock()
        mgr.list_pending.return_value = []
        mgr.is_resolved.return_value = True
        with patch("src.tools.approval.get_approval_manager", return_value=mgr):
            yield mgr

    @pytest.mark.asyncio
    async def test_partial_results_contains_completed_tools(self):
        from src.tools.exec import ApprovalNeededError

        async def safe_fn(query: str = ""):
            return f"safe_result"

        async def approval_fn(query: str = ""):
            raise ApprovalNeededError(command="needs approval", denylisted=False)

        safe_tool = _make_tool("web_search", fn=safe_fn)
        approval_tool = _make_tool("glob", fn=approval_fn)
        client = AsyncMock()
        config = _make_config()

        loop = AgentLoop(
            client=client,
            tools=[safe_tool, approval_tool],
            state_store=MemoryStateStore(),
            config=config,
        )

        state = AgentState(messages=[])

        with patch("src.agent.loop.AgentLoop._handle_approval") as mock_handle:
            mock_handle.side_effect = ApprovalPending(
                thread_id="test_thread",
                request_id="r1",
                tool_name="glob",
                command_preview="needs approval",
                tc_id="",
            )
            with pytest.raises(ApprovalPending) as exc_info:
                await loop._execute_tools_parallel(
                    [
                        _make_tc("web_search", {"query": "test"}, tc_id="tc_safe"),
                        _make_tc("glob", {"pattern": "*"}, tc_id="tc_approval"),
                    ],
                    state,
                    "test_thread",
                )

        ap = exc_info.value
        pr_tc_ids = {tc_id for tc_id, _ in ap.partial_results}
        assert "tc_safe" in pr_tc_ids, f"tc_safe missing from partial_results, got: {pr_tc_ids}"
        safe_result = next(r for tid, r in ap.partial_results if tid == "tc_safe")
        assert "safe_result" in safe_result

    @pytest.mark.asyncio
    async def test_partial_results_attached_in_run_inner(self):
        from src.tools.exec import ApprovalNeededError

        async def safe_fn(query: str = ""):
            return "safe_result"

        async def approval_fn(query: str = ""):
            raise ApprovalNeededError(command="needs approval", denylisted=False)

        safe_tool = _make_tool("web_search", fn=safe_fn)
        approval_tool = _make_tool("glob", fn=approval_fn)
        client = AsyncMock()
        config = _make_config()

        loop = AgentLoop(
            client=client,
            tools=[safe_tool, approval_tool],
            state_store=MemoryStateStore(),
            config=config,
        )

        store = loop.get_store()
        await store.save(
            "par_ap",
            AgentState(
                messages=[
                    {"role": "user", "content": "go"},
                ]
            ),
        )

        call_count = 0

        async def fake_chat(messages, tools=None, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ChatResponse(
                    content="",
                    tool_calls=[
                        _make_tc("web_search", {"query": "a"}, tc_id="tc1"),
                        _make_tc("glob", {"pattern": "*"}, tc_id="tc2"),
                    ],
                )
            return ChatResponse(content="done", tool_calls=[])

        client.chat.side_effect = fake_chat

        with patch("src.agent.loop.AgentLoop._handle_approval") as mock_handle:
            mock_handle.side_effect = ApprovalPending(
                thread_id="par_ap",
                request_id="r1",
                tool_name="glob",
                command_preview="needs approval",
                tc_id="",
            )
            state = AgentState(messages=[{"role": "user", "content": "go"}])
            with pytest.raises(ApprovalPending):
                await loop.run(state, "par_ap")

        loaded = await store.aload("par_ap")
        assert loaded is not None
        tool_msgs = [m for m in loaded.messages if m.get("role") == "tool"]
        tool_ids = {m.get("tool_call_id") for m in tool_msgs}
        assert "tc1" in tool_ids, f"tc1 result missing from state, got: {tool_ids}"

    @pytest.mark.asyncio
    async def test_three_parallel_one_approval(self):
        from src.tools.exec import ApprovalNeededError

        async def safe_fn_a(query: str = ""):
            return "ok_a"

        async def safe_fn_b(query: str = ""):
            return "ok_b"

        async def approval_fn(query: str = ""):
            raise ApprovalNeededError(command="needs approval", denylisted=False)

        tool_a = _make_tool("web_search", fn=safe_fn_a)
        tool_b = _make_tool("read_file", fn=safe_fn_b)
        tool_c = _make_tool("glob", fn=approval_fn)

        config = _make_config()
        loop = AgentLoop(
            client=AsyncMock(),
            tools=[tool_a, tool_b, tool_c],
            state_store=MemoryStateStore(),
            config=config,
        )

        state = AgentState(messages=[])

        with patch("src.agent.loop.AgentLoop._handle_approval") as mock_handle:
            mock_handle.side_effect = ApprovalPending(
                thread_id="test_3par",
                request_id="r1",
                tool_name="glob",
                command_preview="needs approval",
                tc_id="",
            )
            with pytest.raises(ApprovalPending) as exc_info:
                await loop._execute_tools_parallel(
                    [
                        _make_tc("web_search", tc_id="tc_a"),
                        _make_tc("read_file", tc_id="tc_b"),
                        _make_tc("glob", {"pattern": "*"}, tc_id="tc_c"),
                    ],
                    state,
                    "test_3par",
                )

        ap = exc_info.value
        pr_map = {tc_id: content for tc_id, content in ap.partial_results}
        safe_results = {tid for tid in pr_map if tid in ("tc_a", "tc_b")}
        assert len(safe_results) >= 1, f"Expected at least 1 safe result, got: {list(pr_map.keys())}"


# ======================================================================
# D: Truncation edge cases (was E before parallel interrupt removal)
# ======================================================================


class TestTruncateEdgeCases:
    def test_exactly_8000_not_truncated(self):
        loop = AgentLoop.__new__(AgentLoop)
        content = "x" * 8000
        messages = [{"role": "tool", "tool_call_id": "tc1", "content": content}]
        loop._truncate_large_outputs(messages, "t")
        assert len(messages[0]["content"]) == 8000

    def test_8001_truncated(self):
        loop = AgentLoop.__new__(AgentLoop)
        content = "x" * 8001
        messages = [{"role": "tool", "tool_call_id": "tc1", "content": content}]
        loop._truncate_large_outputs(messages, "t")
        assert "truncated" in messages[0]["content"]
        assert messages[0]["content"].startswith("x" * 8000)

    def test_truncation_reduces_content(self):
        loop = AgentLoop.__new__(AgentLoop)
        content = "x" * 10000
        messages = [{"role": "tool", "tool_call_id": "tc1", "content": content}]
        loop._truncate_large_outputs(messages, "t")
        assert len(messages[0]["content"]) < len(content)

    def test_none_content_no_crash(self):
        loop = AgentLoop.__new__(AgentLoop)
        messages = [{"role": "tool", "tool_call_id": "tc1", "content": None}]
        loop._truncate_large_outputs(messages, "t")
        assert messages[0]["content"] is None

    def test_list_content_no_crash(self):
        loop = AgentLoop.__new__(AgentLoop)
        messages = [{"role": "tool", "tool_call_id": "tc1", "content": [{"text": "x" * 10000}]}]
        loop._truncate_large_outputs(messages, "t")
        assert isinstance(messages[0]["content"], list)


# ======================================================================
# F: End-to-end compress + sanitize
# ======================================================================


class TestEndToEndSanitize:
    @pytest.mark.asyncio
    async def test_compress_with_large_tail_count(self):
        from src.compressor.compressor import ContextCompressor

        cfg = CompressionConfig(enabled=False, tail_messages=100)
        compressor = ContextCompressor(cfg)

        messages = _build_messages_with_groups(5)
        result = compressor._compact(messages, context_window_tokens=200)

        errors = _verify_no_orphan_pairs(result)
        assert errors == [], "Large tail_count: " + "; ".join(errors)

    @pytest.mark.asyncio
    async def test_compact_summary_is_assistant_role(self):
        from src.compressor.compressor import ContextCompressor

        cfg = CompressionConfig(enabled=False, tail_messages=4)
        compressor = ContextCompressor(cfg)

        messages = _build_messages_with_groups(10, extra_middle=2)
        result = compressor._compact(messages, context_window_tokens=200)

        non_system = [m for m in result if m.get("role") != "system"]
        assert non_system[0]["role"] == "assistant"
        assert "tool_calls" not in non_system[0] or not non_system[0].get("tool_calls")

    @pytest.mark.asyncio
    async def test_no_consecutive_assistant_messages(self):
        from src.compressor.compressor import ContextCompressor

        cfg = CompressionConfig(enabled=False, tail_messages=4)
        compressor = ContextCompressor(cfg)

        messages = []
        for i in range(12):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": f"msg {i} " * 50})

        result = compressor._compact(messages, context_window_tokens=200)

        non_system = [m for m in result if m.get("role") != "system"]
        for i in range(1, len(non_system)):
            if non_system[i - 1]["role"] == "assistant" and non_system[i]["role"] == "assistant":
                pytest.fail(
                    f"Consecutive assistant messages at indices {i - 1} and {i}: "
                    f"{non_system[i - 1].get('content', '')[:60]} / {non_system[i].get('content', '')[:60]}"
                )

    @pytest.mark.asyncio
    async def test_summary_merged_when_tail_starts_with_assistant(self):
        from src.compressor.compressor import ContextCompressor

        cfg = CompressionConfig(enabled=False, tail_messages=4)
        compressor = ContextCompressor(cfg)

        messages = []
        for i in range(12):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": f"msg {i} " * 50})

        result = compressor._compact(messages, context_window_tokens=200)

        non_system = [m for m in result if m.get("role") != "system"]
        if non_system[0]["role"] == "assistant" and non_system[1]["role"] == "assistant":
            pytest.fail("Summary assistant message not merged into tail assistant")

    @pytest.mark.asyncio
    async def test_compress_enabled_summary_is_assistant_role(self):
        from src.compressor.compressor import ContextCompressor

        cfg = CompressionConfig(enabled=True, tail_messages=4, threshold_percent=0.01)
        client = AsyncMock()
        client.chat.return_value = ChatResponse(content="Summary of conversation.", tool_calls=[])
        compressor = ContextCompressor(cfg, client=client)

        messages = _build_messages_with_groups(10, extra_middle=3)
        result = await compressor.compress(messages, context_window_tokens=200)

        non_system = [m for m in result if m.get("role") != "system"]
        assert non_system[0]["role"] == "assistant"
