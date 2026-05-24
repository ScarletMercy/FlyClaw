from __future__ import annotations

import asyncio
import logging
import sys

from dotenv import load_dotenv

load_dotenv()

import uvicorn

from src.app import ServiceContainer
from src.message import MessageHandler
from src.commands.register import register_builtin_commands

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("myclaw")


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
        config = uvicorn.Config(
            self.container.api,
            host=self.container.config.gateway.host,
            port=self.container.config.gateway.port,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        await server.serve()


def main():
    import os

    if sys.platform == "win32":
        import ctypes

        _HANDLER = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_uint)

        def _ctrl_handler(ctrl_type):
            if ctrl_type == 0:
                try:
                    loop = asyncio.get_event_loop()
                    loop.call_soon_threadsafe(loop.stop)
                except Exception:
                    os._exit(0)
                return True
            if ctrl_type in (2, 5, 6):
                os._exit(0)
            return False

        ctypes.windll.kernel32.SetConsoleCtrlHandler(_HANDLER(_ctrl_handler), True)

    app = Application()

    async def _run():
        await app.setup()
        await app.run_async()

    asyncio.run(_run())


if __name__ == "__main__":
    main()
