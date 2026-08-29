"""Tests for QQ Bot approval keyboard — send_approval_keyboard and _handle_interaction."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestSendApprovalKeyboard:
    """Tests for QQChannel.send_approval_keyboard()."""

    def _make_channel(self, markdown=True, approval_keyboard=True):
        from src.channels.qq import QQChannel
        from src.config import AppConfig

        cfg = AppConfig()
        cfg.channels.qq.enabled = True
        cfg.channels.qq.app_id = "test_id"
        cfg.channels.qq.client_secret = "test_secret"
        cfg.channels.qq.markdown_support = markdown
        cfg.channels.qq.approval_keyboard = approval_keyboard
        ch = QQChannel(cfg.channels.qq)
        ch._token_manager = MagicMock()
        ch._token_manager.get_token = AsyncMock(return_value="fake_token")
        ch._http_client = AsyncMock()
        return ch

    @pytest.mark.asyncio
    async def test_c2c_sends_keyboard(self):
        """C2C chat with markdown+keyboard enabled should send keyboard message."""
        ch = self._make_channel()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"id": "msg_123"}
        ch._api_post = AsyncMock(return_value={"id": "msg_123"})

        result = await ch.send_approval_keyboard(
            chat_id="c2c:user_openid_abc",
            request_id="req001",
            command_preview="rm -rf /tmp/test",
            is_dangerous=True,
            timeout_seconds=60,
            zh=True,
        )

        assert result is not None
        ch._api_post.assert_called_once()
        call_args = ch._api_post.call_args
        body = call_args[0][1]  # second positional arg is body

        # msg_type should be 2 (markdown)
        assert body["msg_type"] == 2
        assert "markdown" in body
        assert "keyboard" in body

        # Verify 3 buttons with action sub-objects
        buttons = body["keyboard"]["content"]["rows"][0]["buttons"]
        assert len(buttons) == 3
        assert buttons[0]["action"]["data"] == "approve:req001"
        assert buttons[0]["action"]["type"] == 1
        assert buttons[1]["action"]["data"] == "always:req001"
        assert buttons[1]["action"]["type"] == 1
        assert buttons[2]["action"]["data"] == "deny:req001"
        assert buttons[2]["action"]["type"] == 1

        # Verify button styles (inside render_data)
        assert buttons[0]["render_data"]["style"] == 1  # primary (approve)
        assert buttons[1]["render_data"]["style"] == 1  # primary (always)
        assert buttons[2]["render_data"]["style"] == 0  # grey (deny)

        # Verify permission and group_id
        for btn in buttons:
            assert "permission" in btn["action"]
            assert btn["action"]["permission"]["type"] == 2
            assert btn["group_id"] == "approval"

    @pytest.mark.asyncio
    async def test_group_sends_keyboard(self):
        """Group chat should also support keyboard."""
        ch = self._make_channel()
        ch._api_post = AsyncMock(return_value={"id": "msg_456"})

        result = await ch.send_approval_keyboard(
            chat_id="group:group_openid_xyz",
            request_id="req002",
            command_preview="ls -la",
            is_dangerous=False,
        )

        assert result is not None
        body = ch._api_post.call_args[0][1]
        assert "keyboard" in body

    @pytest.mark.asyncio
    async def test_channel_returns_none(self):
        """Guild channel does not support keyboard, should return None."""
        ch = self._make_channel()
        ch._api_post = AsyncMock(return_value={"id": "msg_789"})

        result = await ch.send_approval_keyboard(
            chat_id="channel:12345",
            request_id="req003",
            command_preview="echo hello",
        )

        assert result is None
        ch._api_post.assert_not_called()

    @pytest.mark.asyncio
    async def test_dm_returns_none(self):
        """DM (guild direct message) does not support keyboard."""
        ch = self._make_channel()

        result = await ch.send_approval_keyboard(
            chat_id="dm:guild_123",
            request_id="req004",
            command_preview="cat file.txt",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_no_markdown_returns_none(self):
        """Without markdown_support, keyboard is not supported."""
        ch = self._make_channel(markdown=False)

        result = await ch.send_approval_keyboard(
            chat_id="c2c:user_abc",
            request_id="req005",
            command_preview="echo test",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_approval_keyboard_disabled(self):
        """approval_keyboard=False should return None."""
        ch = self._make_channel(markdown=True, approval_keyboard=False)

        result = await ch.send_approval_keyboard(
            chat_id="c2c:user_abc",
            request_id="req006",
            command_preview="echo test",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_zh_labels(self):
        """Verify Chinese labels when zh=True."""
        ch = self._make_channel()
        ch._api_post = AsyncMock(return_value={"id": "msg_zh"})

        await ch.send_approval_keyboard(
            chat_id="c2c:user_abc",
            request_id="req_zh",
            command_preview="test cmd",
            zh=True,
        )

        body = ch._api_post.call_args[0][1]
        buttons = body["keyboard"]["content"]["rows"][0]["buttons"]
        assert "允许一次" in buttons[0]["render_data"]["label"]
        assert "始终允许" in buttons[1]["render_data"]["label"]
        assert "拒绝" in buttons[2]["render_data"]["label"]

    @pytest.mark.asyncio
    async def test_en_labels(self):
        """Verify English labels when zh=False."""
        ch = self._make_channel()
        ch._api_post = AsyncMock(return_value={"id": "msg_en"})

        await ch.send_approval_keyboard(
            chat_id="c2c:user_abc",
            request_id="req_en",
            command_preview="test cmd",
            zh=False,
        )

        body = ch._api_post.call_args[0][1]
        buttons = body["keyboard"]["content"]["rows"][0]["buttons"]
        assert "Allow Once" in buttons[0]["render_data"]["label"]
        assert "Always Allow" in buttons[1]["render_data"]["label"]
        assert "Deny" in buttons[2]["render_data"]["label"]


class TestHandleInteraction:
    """Tests for QQChannel._handle_interaction()."""

    def _make_channel(self):
        from src.channels.qq import QQChannel
        from src.config import AppConfig

        cfg = AppConfig()
        cfg.channels.qq.enabled = True
        cfg.channels.qq.app_id = "test_id"
        ch = QQChannel(cfg.channels.qq)
        ch._token_manager = MagicMock()
        ch._token_manager.get_token = AsyncMock(return_value="fake_token")
        ch._http_client = AsyncMock()
        return ch

    @pytest.mark.asyncio
    async def test_approve_resolves_allow_once(self):
        """approve action should resolve as 'allow_once'."""
        ch = self._make_channel()

        with patch("src.tools.approval.get_approval_manager") as mock_get_mgr:
            mgr = MagicMock()
            mgr.resolve.return_value = True
            mock_get_mgr.return_value = mgr

            await ch._handle_interaction(
                {
                    "id": "evt_001",
                    "data": {"resolved": {"button_data": "approve:req123"}},
                }
            )

            mgr.resolve.assert_called_once_with("req123", "allow_once")

    @pytest.mark.asyncio
    async def test_deny_resolves_deny(self):
        """deny action should resolve as 'deny'."""
        ch = self._make_channel()

        with patch("src.tools.approval.get_approval_manager") as mock_get_mgr:
            mgr = MagicMock()
            mgr.resolve.return_value = True
            mock_get_mgr.return_value = mgr

            await ch._handle_interaction(
                {
                    "id": "evt_002",
                    "data": {"resolved": {"button_data": "deny:req456"}},
                }
            )

            mgr.resolve.assert_called_once_with("req456", "deny")

    @pytest.mark.asyncio
    async def test_always_approves_session_pattern(self):
        """always action with approval_key should call approve_session_pattern."""
        ch = self._make_channel()

        with patch("src.tools.approval.get_approval_manager") as mock_get_mgr:
            mgr = MagicMock()
            mgr.resolve.return_value = True
            pending = MagicMock()
            pending.thread_id = "thread_abc"
            pending.tool_name = "exec_command"
            pending.args_preview = "del C:\\test.txt"
            pending.approval_key = "del"
            mgr.get_pending.return_value = pending
            mock_get_mgr.return_value = mgr

            await ch._handle_interaction(
                {
                    "id": "evt_003",
                    "data": {"resolved": {"button_data": "always:req789"}},
                }
            )

            mgr.approve_session_pattern.assert_called_once_with(
                "thread_abc",
                "del",
            )
            mgr.approve_session.assert_not_called()
            mgr.resolve.assert_called_once_with("req789", "allow_once")

    @pytest.mark.asyncio
    async def test_always_without_approval_key_uses_session(self):
        """always action without approval_key should fall back to approve_session."""
        ch = self._make_channel()

        with patch("src.tools.approval.get_approval_manager") as mock_get_mgr:
            mgr = MagicMock()
            mgr.resolve.return_value = True
            pending = MagicMock()
            pending.thread_id = "thread_abc"
            pending.tool_name = "memory_delete"
            pending.args_preview = "key1,key2"
            pending.approval_key = ""
            mgr.get_pending.return_value = pending
            mock_get_mgr.return_value = mgr

            await ch._handle_interaction(
                {
                    "id": "evt_003b",
                    "data": {"resolved": {"button_data": "always:req789b"}},
                }
            )

            mgr.approve_session.assert_called_once_with(
                "thread_abc",
                "memory_delete",
                "key1,key2",
            )
            mgr.approve_session_pattern.assert_not_called()

    @pytest.mark.asyncio
    async def test_always_without_pending_still_resolves(self):
        """always action with no pending request should still resolve."""
        ch = self._make_channel()

        with patch("src.tools.approval.get_approval_manager") as mock_get_mgr:
            mgr = MagicMock()
            mgr.resolve.return_value = True
            mgr.get_pending.return_value = None
            mock_get_mgr.return_value = mgr

            await ch._handle_interaction(
                {
                    "id": "evt_004",
                    "data": {"resolved": {"button_data": "always:req_nopending"}},
                }
            )

            mgr.approve_session.assert_not_called()
            mgr.resolve.assert_called_once_with("req_nopending", "allow_once")

    @pytest.mark.asyncio
    async def test_invalid_action_ignored(self):
        """Unknown action should be silently ignored."""
        ch = self._make_channel()

        with patch("src.tools.approval.get_approval_manager") as mock_get_mgr:
            mgr = MagicMock()
            mock_get_mgr.return_value = mgr

            await ch._handle_interaction(
                {
                    "id": "evt_005",
                    "data": {"resolved": {"button_data": "unknown:req999"}},
                }
            )

            mgr.resolve.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_data_ignored(self):
        """Empty data field should be silently ignored."""
        ch = self._make_channel()

        with patch("src.tools.approval.get_approval_manager") as mock_get_mgr:
            mgr = MagicMock()
            mock_get_mgr.return_value = mgr

            await ch._handle_interaction(
                {
                    "id": "evt_006",
                    "data": {},
                }
            )

            mgr.resolve.assert_not_called()

    @pytest.mark.asyncio
    async def test_malformed_data_ignored(self):
        """Data without colon should be silently ignored."""
        ch = self._make_channel()

        with patch("src.tools.approval.get_approval_manager") as mock_get_mgr:
            mgr = MagicMock()
            mock_get_mgr.return_value = mgr

            await ch._handle_interaction(
                {
                    "id": "evt_007",
                    "data": {"resolved": {"button_data": "no_colon_here"}},
                }
            )

            mgr.resolve.assert_not_called()

    @pytest.mark.asyncio
    async def test_interaction_acknowledgment_sent(self):
        """After handling, should PUT /interactions/{id} with {"code": 0}."""
        ch = self._make_channel()

        mock_put_resp = MagicMock()
        mock_put_resp.status_code = 200
        ch._http_client.put = AsyncMock(return_value=mock_put_resp)

        with patch("src.tools.approval.get_approval_manager") as mock_get_mgr:
            mgr = MagicMock()
            mgr.resolve.return_value = True
            mock_get_mgr.return_value = mgr

            await ch._handle_interaction(
                {
                    "id": "evt_ack_001",
                    "data": {"resolved": {"button_data": "approve:req_ack"}},
                }
            )

            ch._http_client.put.assert_called_once()
            call_url = ch._http_client.put.call_args[0][0]
            call_json = ch._http_client.put.call_args[1].get("json") or ch._http_client.put.call_args.kwargs.get("json")
            assert "/interactions/evt_ack_001" in call_url
            assert call_json == {"code": 0}
