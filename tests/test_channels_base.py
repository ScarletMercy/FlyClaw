"""Tests for channel base — QQ channel message parsing and callbacks."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


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
        from src.message import MessageHandler

        assert MessageHandler._resolve_session_key("user1", "p2p", "chat1", "per_sender") == "user:user1"

    def test_per_sender_group(self):
        from src.message import MessageHandler

        assert MessageHandler._resolve_session_key("user1", "group", "chat1", "per_sender") == "group:chat1"

    def test_global_scope(self):
        from src.message import MessageHandler

        assert MessageHandler._resolve_session_key("user1", "p2p", "chat1", "global") == "global"
