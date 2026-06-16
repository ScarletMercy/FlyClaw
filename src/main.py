from __future__ import annotations

import asyncio
import logging
import sys

# ── 尽早配置日志，消灭重型依赖 import 期间的"漆黑期" ──
# 下面 import uvicorn / src.app（→ openai SDK ~4s）等重型依赖要数秒；
# 若 basicConfig 排在它们之后，敲下 flyclaw 会有数秒零输出（像卡死）。
# 把日志配置 + 启动提示挪到重型 import 之前，立刻给用户反馈。
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("flyclaw")
logger.info("flyclaw 启动中，正在加载依赖…")

import uvicorn

from src.app import ServiceContainer
from src.message import MessageHandler
from src.commands.register import register_builtin_commands


class Application:
    def __init__(self, config=None):
        self.container = ServiceContainer(config)

    async def setup(self):
        tools, skills = await self.container.setup()

        handler = MessageHandler(self.container)
        scope = self.container.config.session.scope

        if self.container.qq:
            self.container.qq.set_message_callback(handler.create_callback(scope, channel_prefix="qq"))

        if self.container.weixin:
            self.container.weixin.set_message_callback(handler.create_callback(scope, channel_prefix="weixin"))

        register_builtin_commands(self.container.dispatcher, self.container, tools, skills)

        from src.dashboard.routes import register_dashboard

        register_dashboard(self.container.api, self.container)

        self.container.api.on_event("startup")(self.container.on_startup)
        self.container.api.on_event("shutdown")(self.container.on_shutdown)

    async def run_async(self):
        from src.gateway import GATEWAY_HOST

        config = uvicorn.Config(
            self.container.api,
            host=GATEWAY_HOST,
            port=self.container.config.gateway.port,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        await server.serve()


def main():
    from src.instance import get_instance, parse_instance_from_argv, set_instance

    if get_instance() is None:
        n = parse_instance_from_argv()
        set_instance(n)

    import os

    if sys.platform == "win32":
        import ctypes

        _HANDLER = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_uint)

        import threading

        def _ctrl_handler(ctrl_type):
            if ctrl_type == 0:
                try:
                    main_thread = threading.main_thread()
                    if main_thread is not None and main_thread.is_alive():
                        ctypes.pythonapi.PyThreadState_SetAsyncExc(
                            ctypes.c_ulong(main_thread.ident),
                            ctypes.py_object(KeyboardInterrupt),
                        )
                    else:
                        os._exit(0)
                except Exception:
                    os._exit(0)
                return True
            if ctrl_type in (2, 5, 6):
                os._exit(0)
            return False

        ctypes.windll.kernel32.SetConsoleCtrlHandler(_HANDLER(_ctrl_handler), True)

    from src.config import load_config

    config = load_config()
    if not config.model.api_key:
        print("\n[错误] 模型 API 密钥未配置")
        print("请运行以下命令初始化配置:")
        print("  flyclaw setup")
        print("  或")
        print("  flyclaw-setup\n")
        sys.exit(1)

    # Mint a strong gateway token if none is configured, so the local API /
    # Dashboard are never unauthenticated. Idempotent (won't overwrite).
    from src.gateway import ensure_gateway_token

    ensure_gateway_token(config)

    app = Application(config)

    async def _run():
        await app.setup()
        await app.run_async()

    asyncio.run(_run())


if __name__ == "__main__":
    main()
