"""Regression: on_config_reload must not drop the auto-generated gateway token.

ensure_gateway_token persists the token to a standalone file and injects it into
config.gateway.auth_token IN MEMORY. But on_config_reload replaces self.config
with a freshly-loaded config from disk (where auth_token is empty). Without
re-running ensure on the new config, the token vanishes mid-run and auth flips.

This test drives the REAL on_config_reload (unbound, MagicMock container) and
asserts the new config carries the token after reload — same pattern as
TestPartialHandlerFailureDoesNotRaise.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config import AppConfig
from src.gateway import ensure_gateway_token


class TestReloadPreservesGatewayToken:
    @pytest.mark.asyncio
    async def test_reload_reinjects_token_into_new_config(self, tmp_path):
        token_file = tmp_path / "gateway_token"

        # 首启:生成 token 并持久化(磁盘 config.yaml 里 auth_token 仍是空)
        first = AppConfig()
        original = ensure_gateway_token(first, token_file=token_file)
        assert original

        # on_config_reload 里的 new_config = load_config() 从磁盘重读 → auth_token 为空
        new_config = AppConfig()
        assert new_config.gateway.auth_token == ""

        # 构造 container mock,模拟真实 on_config_reload 的运行环境
        app = MagicMock()
        app.agent_loop = MagicMock()
        app.dispatcher = MagicMock()
        app.qq = MagicMock()
        app._reload_executor = MagicMock()
        app._reload_executor.execute = AsyncMock(return_value={"succeeded": [], "failed": []})

        # 关键:让 ensure_gateway_token 用我们的独立文件(token_file 默认指向 data_dir,
        # 测试里 monkeypatch 让它指向 tmp_path)
        import src.gateway as gateway_mod

        original_default = gateway_mod._default_token_file
        gateway_mod._default_token_file = lambda: token_file
        try:
            from src.app import ServiceContainer

            await ServiceContainer.on_config_reload(app, AppConfig(), new_config, MagicMock())
        finally:
            gateway_mod._default_token_file = original_default

        # 核心断言:reload 后 new_config 必须重新带上 token(否则认证失效)
        assert new_config.gateway.auth_token == original, (
            "热重载后 token 丢失:on_config_reload 未对新 config 重新 ensure"
        )
        assert app.config is new_config
