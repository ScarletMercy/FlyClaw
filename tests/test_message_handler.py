"""Tests for message handling chain — the on_message callback via MessageHandler.

Tests the core message flow: receive message → dispatch/agent loop → reply.
"""

import asyncio
from types import SimpleNamespace
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
        callback = handler.create_callback()
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
    async def test_session_key_dm_collapses(self, container, reply_fn, mock_approval_mgr):
        container.agent_loop.run.return_value = AgentState(messages=[{"role": "assistant", "content": "ok"}])

        handler = MessageHandler(container)
        callback = handler.create_callback()
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
        # DM 塌缩：私聊固定落到 qq:dm，openid(user123) 不渗入 key → 漂移根除
        assert thread_id == "qq:dm"
        assert "user123" not in thread_id

    @pytest.mark.asyncio
    async def test_session_key_group_keyed_by_chat(self, container, reply_fn, mock_approval_mgr):
        container.agent_loop.run.return_value = AgentState(messages=[{"role": "assistant", "content": "ok"}])

        handler = MessageHandler(container)
        callback = handler.create_callback()
        await callback(
            text="test",
            sender_id="u1",
            chat_id="group_abc",
            chat_type="group",
            message_id="m1",
            reply_fn=reply_fn,
        )

        call_args = container.agent_loop.run.call_args
        thread_id = call_args[0][1] if len(call_args[0]) > 1 else call_args.kwargs.get("thread_id", "")
        # 群聊按群 id 分，与私聊(qq:dm)互不串
        assert thread_id == "qq:group:group_abc"


class TestMessageCallbackSlashCommands:
    @pytest.mark.asyncio
    async def test_slash_command_dispatches_and_returns(self, container, reply_fn, mock_approval_mgr):
        container.dispatcher.match.return_value = ("help", "")
        container.dispatcher.dispatch.return_value = "Available commands: ..."

        handler = MessageHandler(container)
        callback = handler.create_callback()
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
                "thread_id": "qq:dm",
                "sender_id": "u1",
                "chat_id": "c1",
                "user_key": "qq:dm",
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
        await container.state_store.save("qq:dm", pending_state)

        handler = MessageHandler(container)
        callback = handler.create_callback()
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
        callback = handler.create_callback(channel_prefix="qq")

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
        await container.state_store.save("qq:dm", old_state)

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
        callback = handler.create_callback()
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
        callback = handler.create_callback()
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
        callback = handler.create_callback()
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
        callback = handler.create_callback()
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
        callback = handler.create_callback()
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
        callback = handler.create_callback()
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
        callback = handler.create_callback()
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


class TestBusyInterruptDuringFinishWindow:
    """收尾窗口内的 busy 打断：最终响应已产生、loop 正在退出时注册的
    interrupt flag 属于已结束的回合；drain 重跑必须真正处理被打断的消息。"""

    @pytest.mark.asyncio
    async def test_interrupt_message_answered_after_turn_finishes(self, container, reply_fn, mock_approval_mgr):
        from src.agent.client import ChatResponse
        from src.agent.loop import AgentLoop

        container.config.voice.enabled = False
        container.config.link_understanding.enabled = False
        container.config.session_search.enabled = False
        container.config.agents.busy_input_mode = "interrupt"

        calls = 0

        class CountingClient:
            async def chat(self, messages, tools=None, **kw):
                nonlocal calls
                calls += 1
                return ChatResponse(content=f"回答#{calls}", tool_calls=[], usage=None)

        container.agent_loop = AgentLoop(
            client=CountingClient(), tools=[], state_store=container.state_store, config=container.config
        )

        handler = MessageHandler(container)
        callback = handler.create_callback()
        reply_fn2 = AsyncMock()

        orig_save = container.state_store.save
        save_count = 0

        async def hooked_save(thread_id, state):
            nonlocal save_count
            save_count += 1
            if save_count == 1:
                # 第一回合完成路径的 save：run 仍持锁，第二条消息走 busy 打断路径
                await callback(
                    text="等等，先别结束",
                    sender_id="u1",
                    chat_id="c1",
                    chat_type="p2p",
                    message_id="m2",
                    reply_fn=reply_fn2,
                )
            return await orig_save(thread_id, state)

        container.state_store.save = hooked_save

        await callback(
            text="第一条",
            sender_id="u1",
            chat_id="c1",
            chat_type="p2p",
            message_id="m1",
            reply_fn=reply_fn,
        )

        assert calls == 2
        texts1 = [c[0][0] for c in reply_fn.call_args_list]
        assert "回答#1" in texts1
        assert "回答#2" in texts1
        texts2 = [c[0][0] for c in reply_fn2.call_args_list]
        assert len(texts2) == 1 and "已中断" in texts2[0]

        state = await container.state_store.load("qq:dm")
        dup = sum(1 for m in state.messages if m.get("role") == "user" and m.get("content") == "等等，先别结束")
        assert dup == 1
        assert not container.state_store.get_interrupt_flag("qq:dm").check()[0]
        assert "qq:dm" not in handler._interrupted_threads
        assert "qq:dm" not in handler._pending_queue


class TestSteerFinishWindowRescue:
    """steer 模式收尾窗口：回合完成路径 save 期间到达的引导消息已无工具轮
    注入点，不得只活在 interrupt flag 里跨过回合边界——drain 兜底转排队消息补跑。"""

    @pytest.mark.asyncio
    async def test_finish_window_steer_requeued_and_answered(self, container, reply_fn, mock_approval_mgr):
        from src.agent.client import ChatResponse
        from src.agent.loop import AgentLoop

        container.config.voice.enabled = False
        container.config.link_understanding.enabled = False
        container.config.session_search.enabled = False
        container.config.agents.busy_input_mode = "steer"

        calls = 0

        class CountingClient:
            async def chat(self, messages, tools=None, **kw):
                nonlocal calls
                calls += 1
                return ChatResponse(content=f"回答#{calls}", tool_calls=[], usage=None)

        container.agent_loop = AgentLoop(
            client=CountingClient(), tools=[], state_store=container.state_store, config=container.config
        )

        handler = MessageHandler(container)
        callback = handler.create_callback()
        reply_fn2 = AsyncMock()

        orig_save = container.state_store.save
        save_count = 0

        async def hooked_save(thread_id, state):
            nonlocal save_count
            save_count += 1
            if save_count == 1:
                # 第一回合完成路径的 save：run 仍持锁，引导消息走 busy steer 分支
                await callback(
                    text="拐个弯",
                    sender_id="u1",
                    chat_id="c1",
                    chat_type="p2p",
                    message_id="m2",
                    reply_fn=reply_fn2,
                )
            return await orig_save(thread_id, state)

        container.state_store.save = hooked_save

        await callback(
            text="第一条",
            sender_id="u1",
            chat_id="c1",
            chat_type="p2p",
            message_id="m1",
            reply_fn=reply_fn,
        )

        # 引导文本经排队补跑到达模型，得到自己的回答
        assert calls == 2
        texts1 = [c[0][0] for c in reply_fn.call_args_list]
        assert "回答#1" in texts1
        assert "回答#2" in texts1
        # steer 分支被接受时不回复；兜底回答走原回合的 reply_fn
        assert len(reply_fn2.call_args_list) == 0

        state = await container.state_store.load("qq:dm")
        steer_as_user = sum(1 for m in state.messages if m.get("role") == "user" and m.get("content") == "拐个弯")
        assert steer_as_user == 1
        assert container.state_store.get_interrupt_flag("qq:dm").drain_steer() is None
        assert "qq:dm" not in handler._interrupted_threads
        assert "qq:dm" not in handler._pending_queue


class TestResumeFallbackDrain:
    """/y 回退路径（_resume_and_reply）受"回合结束必须 drain"不变量约束：
    回退回合的收尾窗口里注册的 busy 打断必须被 drain 消费重跑，不得让
    线程永久卡在 busy 排队。"""

    @pytest.mark.asyncio
    async def test_fallback_resume_drains_interrupt_table(self, container, reply_fn, mock_approval_mgr):
        import src.tools.approval as approval_mod
        from src.agent.client import ChatResponse
        from src.agent.loop import AgentLoop
        from src.agent.tooldef import ToolDef

        container.config.voice.enabled = False
        container.config.link_understanding.enabled = False
        container.config.session_search.enabled = False
        container.config.agents.busy_input_mode = "interrupt"
        container.weixin = None  # _get_channel_for_chat_id 回落到 qq
        container.qq.send_text = AsyncMock()

        calls = 0

        class CountingClient:
            async def chat(self, messages, tools=None, **kw):
                nonlocal calls
                calls += 1
                return ChatResponse(content=f"回答#{calls}", tool_calls=[], usage=None)

        async def echo(**kwargs):
            return "echo result"

        container.agent_loop = AgentLoop(
            client=CountingClient(),
            tools=[ToolDef.from_function(echo, name="echo")],
            state_store=container.state_store,
            config=container.config,
        )

        # 前提：pending_approval 落盘、审批请求未解决、_approval_handler_threads 为空
        state = AgentState(
            messages=[
                {"role": "user", "content": "跑一下"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "echo", "arguments": "{}"}}],
                },
            ],
            pending_approval={
                "request_id": "r1",
                "tool_call_id": "tc1",
                "tool_name": "echo",
                "command_preview": "",
            },
            sender_id="u1",
            chat_id="c1",
            chat_type="p2p",
            channel="qq",
        )
        await container.state_store.save("qq:dm", state)

        req = SimpleNamespace(id="r1", chat_id="c1", thread_id="qq:dm", tool_name="echo")
        resolved = {"v": False}
        mgr = MagicMock()
        mgr.list_pending.side_effect = lambda: [] if resolved["v"] else [req]
        mgr.is_resolved.side_effect = lambda rid: resolved["v"]
        mgr.resolve.side_effect = lambda rid, decision, **kw: resolved.__setitem__("v", True)

        with patch.object(approval_mod, "get_approval_manager", return_value=mgr):
            handler = MessageHandler(container)
            callback = handler.create_callback()

            orig_save = container.state_store.save
            save_count = 0

            async def hooked_save(thread_id, st):
                nonlocal save_count
                save_count += 1
                if save_count == 2:
                    # save#1 = _resume_inner 清 pending 后落盘；save#2 = 回退回合
                    # 完成路径落盘（模型已回答、锁仍持有）→ 收尾窗口注入 busy 打断
                    await callback(
                        text="等等，先停一下",
                        sender_id="u1",
                        chat_id="c1",
                        chat_type="p2p",
                        message_id="m_busy",
                        reply_fn=AsyncMock(),
                    )
                return await orig_save(thread_id, st)

            container.state_store.save = hooked_save

            await callback(
                text="/y",
                sender_id="u1",
                chat_id="c1",
                chat_type="p2p",
                message_id="m_y",
                reply_fn=AsyncMock(),
            )
            await asyncio.sleep(0.3)  # 等 create_task(_resume_and_reply) 全链路跑完

        # 回退回合(#1) + 打断消息 drain 重跑(#2)
        assert calls == 2
        assert "qq:dm" not in handler._interrupted_threads
        assert "qq:dm" not in handler._pending_queue
        sent = [c[0][1] for c in container.qq.send_text.call_args_list]
        assert any("回答#1" in s for s in sent)
        assert any("回答#2" in s for s in sent)
        saved = await container.state_store.load("qq:dm")
        dup = sum(1 for m in saved.messages if m.get("role") == "user" and m.get("content") == "等等，先停一下")
        assert dup == 1

        # 线程已恢复健康：后续消息正常处理，不再排队
        follow_up = AsyncMock()
        await callback(
            text="第三条",
            sender_id="u1",
            chat_id="c1",
            chat_type="p2p",
            message_id="m3",
            reply_fn=follow_up,
        )
        assert calls == 3
        assert "回答#3" in follow_up.call_args[0][0]


class TestApprovalChannelMissing:
    """渠道缺失时审批 handler 不得留幻影状态：撤销孤儿请求 + 清 pending_approval，
    否则每次 /y 都会触发 _resume_and_reply 回退路径。"""

    @pytest.mark.asyncio
    async def test_missing_channel_cancels_request_and_clears_pending(self, container, mock_approval_mgr):
        from src.agent.loop import ApprovalPending

        container.qq = None  # 渠道缺失

        state = AgentState(
            messages=[{"role": "user", "content": "hi"}],
            pending_approval={
                "request_id": "r1",
                "tool_call_id": "tc1",
                "tool_name": "exec_command",
                "command_preview": "rm x",
            },
        )
        await container.state_store.save("qq:dm", state)

        exc = ApprovalPending(
            thread_id="qq:dm",
            request_id="r1",
            tool_name="exec_command",
            command_preview="rm x",
            tc_id="tc1",
        )
        handler = MessageHandler(container)
        await handler._handle_approval_pending(exc, "c1", "qq")

        mock_approval_mgr.cancel_pending.assert_called_once_with("r1")
        saved = await container.state_store.load("qq:dm")
        assert saved.pending_approval is None
        assert "qq:dm" not in handler._approval_handler_threads
