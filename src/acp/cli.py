from __future__ import annotations

import asyncio
import logging

from src.acp.server import AcpServer
from src.acp.transport import NdjsonTransport


def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")
    transport = NdjsonTransport()
    server = AcpServer(transport=transport)
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
