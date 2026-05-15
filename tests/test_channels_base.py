"""Tests for channel base — Feishu and QQ channel message parsing and callbacks."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestFeishuChannelInit:
    def test_feishu_channel_stores_config(self):
        from src.config import AppConfig

        config = AppConfig()
        config.channels.feishu.enabled = True
        config.channels.feishu.app_id = "test_id"
        config.channels.feishu.app_secret = "test_secret"

        from src.channels.feishu import FeishuChannel

        ch = FeishuChannel(config.channels.feishu)
        assert ch.config.app_id == "test_id"

    def test_feishu_set_message_callback(self):
        from src.channels.feishu import FeishuChannel
        from src.config import AppConfig

        config = AppConfig()
        config.channels.feishu.enabled = False

        ch = FeishuChannel(config.channels.feishu)
        callback = AsyncMock()
        ch.set_message_callback(callback)
        assert ch._on_message_callback == callback


class TestQQChannelInit:
    def test_qq_channel_stores_config(self):
        from src.channels.qq import QQChannel
        from src.config import AppConfig

        config = AppConfig()
        config.channels.qq.enabled = True
        config.channels.qq.app_id = "qq_test_id"

        ch = QQChannel(config.channels.qq)
        assert ch.config.app_id == "qq_test_id"

    def test_qq_set_message_callback(self):
        from src.channels.qq import QQChannel
        from src.config import AppConfig

        config = AppConfig()
        config.channels.qq.enabled = False

        ch = QQChannel(config.channels.qq)
        callback = AsyncMock()
        ch.set_message_callback(callback)
        assert ch._on_message_callback == callback


class TestSessionKeyResolution:
    def test_per_sender_p2p(self):
        from src.main import Application

        app = Application.__new__(Application)
        assert app._resolve_session_key("user1", "p2p", "chat1", "per_sender") == "user:user1"

    def test_per_sender_group(self):
        from src.main import Application

        app = Application.__new__(Application)
        assert app._resolve_session_key("user1", "group", "chat1", "per_sender") == "group:chat1"

    def test_global_scope(self):
        from src.main import Application

        app = Application.__new__(Application)
        assert app._resolve_session_key("user1", "p2p", "chat1", "global") == "global"


class TestTypingIndicator:
    @pytest.mark.asyncio
    async def test_typing_start_stop(self):
        from src.channels.typing import TypingIndicator

        mock_client = MagicMock()
        mock_client.im = MagicMock()
        mock_client.im.message = MagicMock()
        mock_client.im.message.create = AsyncMock()

        typing = TypingIndicator(mock_client, enabled=True)

        await typing.start("msg_1")
        await typing.stop("msg_1")

    @pytest.mark.asyncio
    async def test_typing_disabled_is_noop(self):
        from src.channels.typing import TypingIndicator

        mock_client = MagicMock()
        typing = TypingIndicator(mock_client, enabled=False)

        await typing.start("msg_1")
        mock_client.im.message.create.assert_not_called()
