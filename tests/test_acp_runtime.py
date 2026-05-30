import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from src.acp.runtime import AgentLoopRuntime, AcpRuntimeEvent


class TestAgentLoopRuntime:
    @pytest.mark.asyncio
    async def test_run_turn_returns_events(self):
        runtime = AgentLoopRuntime()
        mock_result = MagicMock()
        mock_result.messages = [{"role": "assistant", "content": "Hello world"}]

        with patch("src.acp.runtime._build_agent_loop") as mock_build:
            mock_loop = AsyncMock()
            mock_loop.run.return_value = mock_result
            mock_build.return_value = mock_loop

            events = []
            async for event in runtime.run_turn(session_id="test", prompt="Say hello", agent_id="default"):
                events.append(event)

        assert any(e.type == "text_delta" and "Hello world" in e.text for e in events)
        assert any(e.type == "done" for e in events)

    @pytest.mark.asyncio
    async def test_run_turn_error_produces_done(self):
        runtime = AgentLoopRuntime()
        with patch("src.acp.runtime._build_agent_loop") as mock_build:
            mock_loop = AsyncMock()
            mock_loop.run.side_effect = RuntimeError("boom")
            mock_build.return_value = mock_loop

            events = []
            async for event in runtime.run_turn(session_id="test", prompt="test"):
                events.append(event)

        assert any(e.type == "error" for e in events)
        assert any(e.type == "done" for e in events)

    @pytest.mark.asyncio
    async def test_close_removes_session(self):
        from src.acp.session import AcpSessionManager

        mgr = AcpSessionManager()
        sid = mgr.create("default")
        runtime = AgentLoopRuntime(mgr)
        await runtime.close(sid)
        assert mgr.get(sid) is None
