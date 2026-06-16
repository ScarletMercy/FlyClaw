"""Regression: unify the thread_id source used for session-approval lookups.

Background
----------
exec_command resolved its session-approval thread_id from ``_current_thread_id``
(set only at entry points: message inbound / resume / delegate / task — NOT the
gateway/OpenAI-API path), while memory_delete resolved it from
``_current_agent_context.parent_thread_id`` (set inside ``_execute_tool`` on
every path). The two therefore DISAGREED on the gateway path:

  - exec_command saw ``""`` → its session approval never stuck via the API
  - memory_delete saw the real thread_id → worked

Fix (two linked changes):
  1. ``_execute_tool`` also sets ``_current_thread_id`` (covers the gateway gap;
     makes the loop self-contained instead of relying on the caller).
  2. ``memory_delete`` reads ``_current_thread_id`` — single source of truth,
     matching exec_command.
"""

import pytest
from unittest.mock import AsyncMock, patch

from src.agent.loop import AgentLoop
from src.agent.state import AgentState, MemoryStateStore
from src.agent.tooldef import ToolDef
from src.tools.exec import _current_thread_id


class TestExecuteToolSetsCurrentThreadId:
    """Change 1: _execute_tool establishes _current_thread_id for the tool it runs."""

    @pytest.mark.asyncio
    async def test_execute_tool_sets_current_thread_id(self):
        from tests.test_agent_loop import _make_config, _make_tc

        seen = {}

        async def _spy():
            seen["tid"] = _current_thread_id.get()
            return "ok"

        spy = ToolDef.from_function(_spy, name="spy")
        client = AsyncMock()
        loop = AgentLoop(
            client=client,
            tools=[spy],
            state_store=MemoryStateStore(),
            config=_make_config(),
        )
        state = AgentState(messages=[])

        # Precondition: gateway path — no inbound handler set _current_thread_id.
        assert _current_thread_id.get() == ""

        await loop._execute_tool(_make_tc("spy"), state, "gateway-thread-1")

        assert seen["tid"] == "gateway-thread-1"


class TestMemoryDeleteUsesCurrentThreadId:
    """Change 2: memory(action=delete) resolves session approval from _current_thread_id."""

    @pytest.mark.asyncio
    async def test_session_approval_honored_via_current_thread_id(self, tmp_path):
        from src.tools.approval import ApprovalManager
        from src.tools.memory_tools import MemoryStore, memory

        store = MemoryStore(db_path=str(tmp_path / "mem.db"))
        await store.initialize()
        await store.remember(content="my secret value", key="K1", category="fact")

        mgr = ApprovalManager(data_dir=str(tmp_path / "appr"))
        # Grant a session approval for the exact args-preview memory() will compute.
        mgr.approve_session("T1", "memory_delete", "- [K1]: my secret value")

        token = _current_thread_id.set("T1")
        try:
            with (
                patch("src.tools.memory_tools.get_memory_store", return_value=store),
                patch("src.tools.approval.get_approval_manager", return_value=mgr),
            ):
                # Must NOT raise MemoryDeleteNeedsApproval — session approval matches.
                result = await memory(action="delete", keys=["K1"])
        finally:
            _current_thread_id.reset(token)

        assert '"ok": true' in result
