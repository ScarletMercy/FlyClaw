"""Tests for message handling chain — the on_message callback via MessageHandler.

Tests the core message flow: receive message → dispatch/agent loop → reply.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.state import AgentState, MemoryStateStore
from src.config import AppConfig
from src.message import MessageHandler


def _make_container():
    container = MagicMock()
    container.config = AppConfig()
    container.agent_loop = AsyncMock()
    container.agent_loop.is_thread_busy = MagicMock(return_value=False)
    container.state_store = MemoryStateStore()
    container.session_tracker = MagicMock()
    container.session_tracker.touch = MagicMock()
    container.session_tracker.active_count = 0
    container.session_registry = MagicMock()
    container.session_registry.get_current = MagicMock(return_value=None)
    container.typing = AsyncMock()
    container.memory_searcher = None
    container.background_tasks = set()
    container.rbac = None
    container.dispatcher = MagicMock()
    container.dispatcher.match = MagicMock(return_value=None)
    container.dispatcher.dispatch = AsyncMock(return_value="Available commands: ...")
    container.qq = MagicMock()
    container.qq.client = MagicMock()
    container.cron_service = None
    return container


@pytest.fixture
def container():
    return _make_container()


@pytest.fixture
def reply_fn():
    fn = AsyncMock()
    return fn


@pytest.fixture
def mock_approval_mgr():
    """Mock get_approval_manager to avoid ServiceContainer dependency."""
    mgr = MagicMock()
    mgr.list_pending.return_value = []
    mgr.is_resolved.return_value = True
    with patch("src.tools.approval.get_approval_manager", return_value=mgr):
        yield mgr


class TestMessageCallbackBasicFlow:
    @pytest.mark.asyncio
    async def test_user_message_gets_agent_reply(self, container, reply_fn, mock_approval_mgr):
        result_state = AgentState(
            messages=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        )
        container.agent_loop.run.return_value = result_state

        handler = MessageHandler(container)
        callback = handler.create_callback("per_sender")
        await callback(
            text="hello",
            sender_id="u1",
            chat_id="c1",
            chat_type="p2p",
            message_id="m1",
            reply_fn=reply_fn,
        )

        container.agent_loop.run.assert_called_once()
        reply_fn.assert_awaited()
        assert "Hi there!" in reply_fn.call_args[0][0]

    @pytest.mark.asyncio
    async def test_session_key_per_sender(self, container, reply_fn, mock_approval_mgr):
        container.agent_loop.run.return_value = AgentState(messages=[{"role": "assistant", "content": "ok"}])

        handler = MessageHandler(container)
        callback = handler.create_callback("per_sender")
        await callback(
            text="test",
            sender_id="user123",
            chat_id="chat456",
            chat_type="p2p",
            message_id="m1",
            reply_fn=reply_fn,
        )

        call_args = container.agent_loop.run.call_args
        thread_id = call_args[0][1] if len(call_args[0]) > 1 else call_args.kwargs.get("thread_id", "")
        assert "user123" in thread_id

    @pytest.mark.asyncio
    async def test_session_key_global(self, container, reply_fn, mock_approval_mgr):
        container.agent_loop.run.return_value = AgentState(messages=[{"role": "assistant", "content": "ok"}])

        handler = MessageHandler(container)
        callback = handler.create_callback("global")
        await callback(
            text="test",
            sender_id="u1",
            chat_id="c1",
            chat_type="p2p",
            message_id="m1",
            reply_fn=reply_fn,
        )

        call_args = container.agent_loop.run.call_args
        thread_id = call_args[0][1] if len(call_args[0]) > 1 else call_args.kwargs.get("thread_id", "")
        assert "global" in thread_id


class TestMessageCallbackSlashCommands:
    @pytest.mark.asyncio
    async def test_slash_command_dispatches_and_returns(self, container, reply_fn, mock_approval_mgr):
        container.dispatcher.match.return_value = ("help", "")
        container.dispatcher.dispatch.return_value = "Available commands: ..."

        handler = MessageHandler(container)
        callback = handler.create_callback("per_sender")
        await callback(
            text="/help",
            sender_id="u1",
            chat_id="c1",
            chat_type="p2p",
            message_id="m1",
            reply_fn=reply_fn,
        )

        container.dispatcher.dispatch.assert_awaited_once_with(
            "help",
            "",
            context={
                "thread_id": "qq:user:u1",
                "sender_id": "u1",
                "chat_id": "c1",
                "user_key": "qq:user:u1",
                "channel_prefix": "qq",
            },
        )
        container.agent_loop.run.assert_not_called()
        reply_fn.assert_awaited_with("Available commands: ...")


class TestMessageCallbackPendingApproval:
    @pytest.mark.asyncio
    async def test_pending_approval_blocks_new_messages(self, container, reply_fn, mock_approval_mgr):
        pending_state = AgentState(
            messages=[{"role": "user", "content": "cmd"}],
            pending_approval={"tool_call_id": "tc1"},
        )
        await container.state_store.save("qq:user:u1", pending_state)

        handler = MessageHandler(container)
        callback = handler.create_callback("per_sender")
        await callback(
            text="new message",
            sender_id="u1",
            chat_id="c1",
            chat_type="p2p",
            message_id="m2",
            reply_fn=reply_fn,
        )

        container.agent_loop.run.assert_not_called()
        reply_fn.assert_awaited()
        assert "审批" in reply_fn.call_args[0][0]


class TestMessageCallbackApprovalViaQQ:
    @pytest.mark.asyncio
    async def test_qq_yes_approves_pending(self, container, reply_fn):
        mock_mgr = MagicMock()
        mock_req = MagicMock()
        mock_req.chat_id = "c1"
        mock_mgr.list_pending.return_value = [mock_req]
        mock_mgr.is_resolved.return_value = False

        handler = MessageHandler(container)
        callback = handler.create_callback("per_sender", channel_prefix="qq")

        with patch("src.tools.approval.get_approval_manager", return_value=mock_mgr):
            await callback(
                text="/y",
                sender_id="u1",
                chat_id="c1",
                chat_type="p2p",
                message_id="m1",
                reply_fn=reply_fn,
            )

        mock_mgr.resolve.assert_called_once()
        container.agent_loop.run.assert_not_called()


class TestMessageCallbackHistoryLoad:
    @pytest.mark.asyncio
    async def test_existing_history_prepended(self, container, reply_fn, mock_approval_mgr):
        old_state = AgentState(
            messages=[
                {"role": "user", "content": "previous question"},
                {"role": "assistant", "content": "previous answer"},
            ]
        )
        await container.state_store.save("qq:user:u1", old_state)

        new_result = AgentState(
            messages=[
                {"role": "user", "content": "previous question"},
                {"role": "assistant", "content": "previous answer"},
                {"role": "user", "content": "new question"},
                {"role": "assistant", "content": "new answer"},
            ]
        )
        container.agent_loop.run.return_value = new_result

        handler = MessageHandler(container)
        callback = handler.create_callback("per_sender")
        await callback(
            text="new question",
            sender_id="u1",
            chat_id="c1",
            chat_type="p2p",
            message_id="m2",
            reply_fn=reply_fn,
        )

        call_args = container.agent_loop.run.call_args
        input_state = call_args[0][0]
        assert len(input_state.messages) == 3
        assert input_state.messages[0]["content"] == "previous question"
        assert input_state.messages[2]["content"] == "new question"


class TestMessageCallbackErrorHandling:
    @pytest.mark.asyncio
    async def test_agent_error_returns_error_message(self, container, reply_fn, mock_approval_mgr):
        container.agent_loop.run.side_effect = RuntimeError("LLM timeout")

        handler = MessageHandler(container)
        callback = handler.create_callback("per_sender")
        await callback(
            text="test",
            sender_id="u1",
            chat_id="c1",
            chat_type="p2p",
            message_id="m1",
            reply_fn=reply_fn,
        )

        reply_fn.assert_awaited()
        error_msg = reply_fn.call_args[0][0]
        assert "error" in error_msg.lower() or "RuntimeError" in error_msg


class TestAssistantMessageBlankContent:
    @pytest.mark.asyncio
    async def test_whitespace_only_content_not_replied(self, container, reply_fn):
        from src.events import emit_async

        handler = MessageHandler(container)
        cleanup = handler._subscribe_agent_events("t1", reply_fn)
        try:
            await emit_async("agent_loop.assistant_message", thread_id="t1", content="\n")
        finally:
            cleanup()

        reply_fn.assert_not_called()


class TestMediaAcknowledgement:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ('请描述这张图片\n[image_url: "C:/tmp/a.jpg"]', "我来看看这张图片："),
            (
                '请描述这 2 张图片\n[image_url: "C:/tmp/a.jpg"]\n[image_url: "C:/tmp/b.jpg"]',
                "我来看看这些图片：",
            ),
            ('请描述这个视频\n[video_url: "C:/tmp/a.mp4"]', "我来看看这个视频："),
            (
                '图+视频\n[image_url: "C:/tmp/a.jpg"]\n[video_url: "C:/tmp/b.mp4"]',
                "我来看看这些图片和视频：",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_media_message_sends_acknowledgement_first(
        self, container, reply_fn, mock_approval_mgr, text, expected
    ):
        container.agent_loop.run.return_value = AgentState(
            messages=[
                {"role": "user", "content": text},
                {"role": "assistant", "content": "这是描述。"},
            ]
        )
        handler = MessageHandler(container)
        callback = handler.create_callback("per_sender")
        await callback(
            text=text,
            sender_id="u1",
            chat_id="c1",
            chat_type="p2p",
            message_id="m1",
            reply_fn=reply_fn,
        )

        first_reply = reply_fn.call_args_list[0][0][0]
        assert first_reply == expected

    @pytest.mark.asyncio
    async def test_acknowledgement_followed_by_final_reply(self, container, reply_fn, mock_approval_mgr):
        container.agent_loop.run.return_value = AgentState(
            messages=[
                {"role": "user", "content": "x"},
                {"role": "assistant", "content": "这是一张橘猫。"},
            ]
        )
        handler = MessageHandler(container)
        callback = handler.create_callback("per_sender")
        await callback(
            text='请描述这张图片\n[image_url: "C:/tmp/a.jpg"]',
            sender_id="u1",
            chat_id="c1",
            chat_type="p2p",
            message_id="m1",
            reply_fn=reply_fn,
        )

        replies = [c[0][0] for c in reply_fn.call_args_list]
        assert replies[0] == "我来看看这张图片："
        assert "橘猫" in replies[-1]

    @pytest.mark.asyncio
    async def test_acknowledgement_send_failure_does_not_block_agent(self, container, mock_approval_mgr):
        container.agent_loop.run.return_value = AgentState(
            messages=[
                {"role": "user", "content": "x"},
                {"role": "assistant", "content": "描述。"},
            ]
        )
        calls = []

        async def flaky_reply(text):
            calls.append(text)
            if len(calls) == 1:
                raise RuntimeError("ack send failed")

        handler = MessageHandler(container)
        callback = handler.create_callback("per_sender")
        try:
            await callback(
                text='请描述这张图片\n[image_url: "C:/tmp/a.jpg"]',
                sender_id="u1",
                chat_id="c1",
                chat_type="p2p",
                message_id="m1",
                reply_fn=flaky_reply,
            )
        except RuntimeError:
            pass

        container.agent_loop.run.assert_called_once()


class TestInterruptedNoDuplicateReply:
    """When agent loop is interrupted, the last assistant message has tool_calls
    and was already sent via the assistant_message event. The final reply
    extraction must skip it to avoid duplication."""

    @pytest.mark.asyncio
    async def test_interrupted_last_assistant_with_tool_calls_not_sent(self, container, reply_fn, mock_approval_mgr):
        result_state = AgentState(
            messages=[
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant",
                    "content": "Let me check that for you.",
                    "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "echo", "arguments": "{}"}}],
                },
                {"role": "tool", "tool_call_id": "tc1", "content": "result"},
            ]
        )
        container.agent_loop.run.return_value = result_state

        handler = MessageHandler(container)
        callback = handler.create_callback("per_sender")
        await callback(
            text="hello",
            sender_id="u1",
            chat_id="c1",
            chat_type="p2p",
            message_id="m1",
            reply_fn=reply_fn,
        )

        reply_fn.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_normal_completion_last_assistant_sent(self, container, reply_fn, mock_approval_mgr):
        result_state = AgentState(
            messages=[
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant",
                    "content": "Let me check that for you.",
                    "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "echo", "arguments": "{}"}}],
                },
                {"role": "tool", "tool_call_id": "tc1", "content": "result"},
                {"role": "assistant", "content": "Here is your answer."},
            ]
        )
        container.agent_loop.run.return_value = result_state

        handler = MessageHandler(container)
        callback = handler.create_callback("per_sender")
        await callback(
            text="hello",
            sender_id="u1",
            chat_id="c1",
            chat_type="p2p",
            message_id="m1",
            reply_fn=reply_fn,
        )

        reply_fn.assert_awaited()
        assert "Here is your answer." in reply_fn.call_args[0][0]
