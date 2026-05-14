"""Stdio transport: launches a subprocess and communicates via stdin/stdout."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from typing import Any

from src.mcp.transport.base import MCPTransport
from src.mcp.transport.jsonrpc import JSONRPCError, JSONRPCProtocol

logger = logging.getLogger("myclaw.mcp.transport.stdio")

_HEALTH_CHECK_INTERVAL = 60
_RECONNECT_ATTEMPTS = 3


class StdioTransport(MCPTransport):
    """MCP transport over subprocess stdin/stdout."""

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        timeout: float = 30.0,
    ):
        self._command = command
        self._args = args or []
        self._env = env
        self._timeout = timeout
        self._process: asyncio.subprocess.Process | None = None
        self._protocol = JSONRPCProtocol()
        self._read_task: asyncio.Task | None = None
        self._connected = False
        self._stopped = False
        self._health_task: asyncio.Task | None = None

    async def connect(self) -> None:
        if self._connected:
            return

        env = {**os.environ, **(self._env or {})}

        resolved = shutil.which(self._command)
        if resolved:
            self._command = resolved
        else:
            logger.warning("Command '%s' not found in PATH", self._command)

        logger.info("Starting MCP server: %s %s", self._command, " ".join(self._args))
        self._process = await asyncio.create_subprocess_exec(
            self._command,
            *self._args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        self._read_task = asyncio.create_task(self._read_loop())
        self._connected = True
        self._stopped = False

        # MCP initialize handshake
        result = await self._protocol.send_request(
            self.send,
            "initialize",
            params={
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "myclaw", "version": "0.1.0"},
            },
            timeout=self._timeout,
        )
        logger.info("MCP server initialized: %s", result.get("serverInfo", {}).get("name", "unknown"))

        await self._protocol.send_notification(
            self.send,
            "notifications/initialized",
        )

        if self._health_task is None or self._health_task.done():
            self._health_task = asyncio.create_task(self._health_check_loop())

    async def _read_loop(self) -> None:
        assert self._process and self._process.stdout
        try:
            while True:
                line = await self._process.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8").strip()
                if text:
                    self._protocol.handle_message(text)
        except Exception as e:
            logger.warning("Stdio read loop error: %s", e)
        finally:
            self._connected = False
            self._protocol.cancel_all()

    async def _health_check_loop(self) -> None:
        try:
            while not self._stopped:
                await asyncio.sleep(_HEALTH_CHECK_INTERVAL)
                if self._stopped:
                    break
                if not self.is_connected:
                    logger.info("MCP stdio health check: disconnected, attempting reconnect")
                    await self._do_reconnect()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning("MCP stdio health check error: %s", e)

    async def _do_reconnect(self) -> None:
        for attempt in range(_RECONNECT_ATTEMPTS):
            if self._stopped:
                return
            try:
                await self.disconnect()
                await self.connect()
                logger.info("MCP stdio reconnected on attempt %d", attempt + 1)
                return
            except Exception as e:
                wait = min(2 ** attempt, 30)
                logger.warning("MCP stdio reconnect attempt %d/%d failed, wait %ds: %s", attempt + 1, _RECONNECT_ATTEMPTS, wait, e)
                await asyncio.sleep(wait)
        logger.error("MCP stdio reconnect failed after %d attempts, will retry next health check", _RECONNECT_ATTEMPTS)
        self._stopped = False
        if self._health_task is None or self._health_task.done():
            self._health_task = asyncio.create_task(self._health_check_loop())

    async def send(self, data: str) -> None:
        if not self._process or not self._process.stdin:
            raise ConnectionError("Stdio transport not connected")
        self._process.stdin.write((data + "\n").encode("utf-8"))
        await self._process.stdin.drain()

    async def disconnect(self) -> None:
        self._stopped = True
        self._connected = False
        if self._health_task and not self._health_task.done():
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
            self._health_task = None
        if self._read_task and not self._read_task.done():
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass
        if self._process:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except Exception:
                try:
                    self._process.kill()
                    await self._process.wait()
                except Exception:
                    pass
            self._process = None
        self._protocol.cancel_all()

    async def list_tools(self) -> list[dict]:
        result = await self._protocol.send_request(self.send, "tools/list", timeout=self._timeout)
        return result.get("tools", []) if isinstance(result, dict) else []

    async def call_tool(self, name: str, arguments: dict) -> Any:
        result = await self._protocol.send_request(
            self.send,
            "tools/call",
            params={"name": name, "arguments": arguments},
            timeout=self._timeout,
        )
        return result

    @property
    def is_connected(self) -> bool:
        return self._connected and self._process is not None and self._process.returncode is None
