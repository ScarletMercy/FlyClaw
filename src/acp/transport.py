from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import AsyncIterator
from typing import IO, Any


class NdjsonTransport:
    def __init__(self, writer: IO[bytes] | None = None) -> None:
        self._writer = writer

    def write(self, msg: dict[str, Any]) -> None:
        line = json.dumps(msg) + "\n"
        data = line.encode("utf-8")
        if self._writer is not None:
            self._writer.write(data)
        else:
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()

    async def read(self) -> dict[str, Any] | None:
        loop = asyncio.get_running_loop()
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            return None
        return json.loads(line)

    async def messages(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            msg = await self.read()
            if msg is None:
                break
            yield msg
