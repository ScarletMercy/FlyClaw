"""Tests for channel base — QQ channel message parsing and callbacks."""

from unittest.mock import AsyncMock


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
    def test_dm_collapses_regardless_of_sender(self):
        from src.message import MessageHandler

        # 私聊塌缩：不同 sender/chat 都落到同一 key
        assert MessageHandler._resolve_session_key("user1", "p2p", "chat1") == "dm"
        assert MessageHandler._resolve_session_key("user2", "p2p", "chat2") == "dm"

    def test_group_keyed_by_chat(self):
        from src.message import MessageHandler

        assert MessageHandler._resolve_session_key("user1", "group", "chat1") == "group:chat1"

    def test_dm_does_not_leak_sender(self):
        from src.message import MessageHandler

        # 核心不变量：openid 永不渗入 DM key → 身份漂移不再断历史
        key = MessageHandler._resolve_session_key("43792F51FB004A795033172075EE6ED4", "p2p", "c2c:ABC")
        assert "43792F51FB004A795033172075EE6ED4" not in key
