"""Tests for message handling chain — the on_message callback in Application.

Tests the core message flow: receive message → dispatch/agent loop → reply.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.state import AgentState, MemoryStateStore
from src.config import AppConfig


def _make_app():
    from src.main import Application
    app = Application.__new__(Application)
    app.config = AppConfig()
    app.agent_loop = AsyncMock()
    app.state_store = MemoryStateStore()
    app.session_tracker = MagicMock()
    app.session_tracker.touch = MagicMock()
    app.session_tracker.active_count = 0
    app.session_registry = MagicMock()
    app.session_registry.get_current = MagicMock(return_value=None)
    app.typing = AsyncMock()
    app._memory_searcher = None
    app._background_tasks = set()
    app._rbac = None
    app.dispatcher = MagicMock()
    app.dispatcher.match = MagicMock(return_value=None)
    app.dispatcher.dispatch = AsyncMock(return_value="Available commands: ...")
    app.feishu = MagicMock()
    app.feishu.client = MagicMock()
    app.qq = MagicMock()
    app.cron_service = None
    return app


@pytest.fixture
def app():
    return _make_app()


@pytest.fixture
def reply_fn():
    fn = AsyncMock()
    return fn


class TestMessageCallbackBasicFlow:
    @pytest.mark.asyncio
    async def test_user_message_gets_agent_reply(self, app, reply_fn):
        result_state = AgentState(
            messages=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        )
        app.agent_loop.run.return_value = result_state

        callback = app._create_message_callback("per_sender")
        await callback(
            text="hello",
            sender_id="u1",
            chat_id="c1",
            chat_type="p2p",
            message_id="m1",
            reply_fn=reply_fn,
            stream_fn=AsyncMock(),
        )

        app.agent_loop.run.assert_called_once()
        reply_fn.assert_awaited()
        assert "Hi there!" in reply_fn.call_args[0][0]

    @pytest.mark.asyncio
    async def test_session_key_per_sender(self, app, reply_fn):
        app.agent_loop.run.return_value = AgentState(
            messages=[{"role": "assistant", "content": "ok"}]
        )

        callback = app._create_message_callback("per_sender")
        await callback(
            text="test",
            sender_id="user123",
            chat_id="chat456",
            chat_type="p2p",
            message_id="m1",
            reply_fn=reply_fn,
            stream_fn=AsyncMock(),
        )

        call_args = app.agent_loop.run.call_args
        thread_id = call_args[0][1] if len(call_args[0]) > 1 else call_args.kwargs.get("thread_id", "")
        assert "user123" in thread_id

    @pytest.mark.asyncio
    async def test_session_key_global(self, app, reply_fn):
        app.agent_loop.run.return_value = AgentState(
            messages=[{"role": "assistant", "content": "ok"}]
        )

        callback = app._create_message_callback("global")
        await callback(
            text="test",
            sender_id="u1",
            chat_id="c1",
            chat_type="p2p",
            message_id="m1",
            reply_fn=reply_fn,
            stream_fn=AsyncMock(),
        )

        call_args = app.agent_loop.run.call_args
        thread_id = call_args[0][1] if len(call_args[0]) > 1 else call_args.kwargs.get("thread_id", "")
        assert "global" in thread_id


class TestMessageCallbackSlashCommands:
    @pytest.mark.asyncio
    async def test_slash_command_dispatches_and_returns(self, app, reply_fn):
        app.dispatcher.match.return_value = ("help", "")
        app.dispatcher.dispatch.return_value = "Available commands: ..."

        callback = app._create_message_callback("per_sender")
        await callback(
            text="/help",
            sender_id="u1",
            chat_id="c1",
            chat_type="p2p",
            message_id="m1",
            reply_fn=reply_fn,
            stream_fn=AsyncMock(),
        )

        app.dispatcher.dispatch.assert_awaited_once_with(
            "help", "",
            context={
                "thread_id": "feishu:user:u1",
                "sender_id": "u1",
                "chat_id": "c1",
                "user_key": "feishu:user:u1",
                "channel_prefix": "feishu",
            },
        )
        app.agent_loop.run.assert_not_called()
        reply_fn.assert_awaited_with("Available commands: ...")


class TestMessageCallbackPendingApproval:
    @pytest.mark.asyncio
    async def test_pending_approval_blocks_new_messages(self, app, reply_fn):
        pending_state = AgentState(
            messages=[{"role": "user", "content": "cmd"}],
            pending_approval={"tool_call_id": "tc1"},
        )
        await app.state_store.save("feishu:user:u1", pending_state)

        callback = app._create_message_callback("per_sender")
        await callback(
            text="new message",
            sender_id="u1",
            chat_id="c1",
            chat_type="p2p",
            message_id="m2",
            reply_fn=reply_fn,
            stream_fn=AsyncMock(),
        )

        app.agent_loop.run.assert_not_called()
        reply_fn.assert_awaited()
        assert "审批" in reply_fn.call_args[0][0]


class TestMessageCallbackApprovalViaQQ:
    @pytest.mark.asyncio
    async def test_qq_yes_approves_pending(self, app, reply_fn):
        mock_mgr = MagicMock()
        mock_req = MagicMock()
        mock_req.chat_id = "c1"
        mock_mgr.list_pending.return_value = [mock_req]

        callback = app._create_message_callback("per_sender", channel_prefix="qq")

        with patch("src.tools.approval.get_approval_manager", return_value=mock_mgr):
            await callback(
                text="yes",
                sender_id="u1",
                chat_id="c1",
                chat_type="p2p",
                message_id="m1",
                reply_fn=reply_fn,
                stream_fn=AsyncMock(),
            )

        mock_mgr.resolve.assert_called_once()
        app.agent_loop.run.assert_not_called()


class TestMessageCallbackHistoryLoad:
    @pytest.mark.asyncio
    async def test_existing_history_prepended(self, app, reply_fn):
        old_state = AgentState(
            messages=[
                {"role": "user", "content": "previous question"},
                {"role": "assistant", "content": "previous answer"},
            ]
        )
        await app.state_store.save("feishu:user:u1", old_state)

        new_result = AgentState(
            messages=[
                {"role": "user", "content": "previous question"},
                {"role": "assistant", "content": "previous answer"},
                {"role": "user", "content": "new question"},
                {"role": "assistant", "content": "new answer"},
            ]
        )
        app.agent_loop.run.return_value = new_result

        callback = app._create_message_callback("per_sender")
        await callback(
            text="new question",
            sender_id="u1",
            chat_id="c1",
            chat_type="p2p",
            message_id="m2",
            reply_fn=reply_fn,
            stream_fn=AsyncMock(),
        )

        call_args = app.agent_loop.run.call_args
        input_state = call_args[0][0]
        assert len(input_state.messages) == 3
        assert input_state.messages[0]["content"] == "previous question"
        assert input_state.messages[2]["content"] == "new question"


class TestMessageCallbackErrorHandling:
    @pytest.mark.asyncio
    async def test_agent_error_returns_error_message(self, app, reply_fn):
        app.agent_loop.run.side_effect = RuntimeError("LLM timeout")

        callback = app._create_message_callback("per_sender")
        await callback(
            text="test",
            sender_id="u1",
            chat_id="c1",
            chat_type="p2p",
            message_id="m1",
            reply_fn=reply_fn,
            stream_fn=AsyncMock(),
        )

        reply_fn.assert_awaited()
        error_msg = reply_fn.call_args[0][0]
        assert "error" in error_msg.lower() or "RuntimeError" in error_msg
