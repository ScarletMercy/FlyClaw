"""MCPClient: manages a single MCP server connection using a persistent background task.

Architecture (hermes-agent pattern):
  - A persistent asyncio task runs the transport (stdio or HTTP).
  - The task starts the subprocess, establishes the session, then blocks
    waiting for lifecycle events (shutdown or reconnect).
  - Tools access the session directly; no per-call connection overhead.
  - Hot-reload triggers a reconnect event instead of killing the task.
  - Connection drops trigger automatic reconnection with exponential backoff.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from src.mcp.config_models import MCPServerConfig

logger = logging.getLogger("myclaw.mcp.client")

_MAX_INITIAL_CONNECT_RETRIES = 3
_MAX_BACKOFF_SECONDS = 30.0
_INITIAL_BACKOFF = 1.0


class MCPClient:
    """Manages connection to a single MCP server via a persistent background task."""

    def __init__(self, name: str, config: MCPServerConfig):
        self.name = name
        self.config = config
        self._session: ClientSession | None = None
        self._tools_cache: list[dict] = []
        self._connected_at: float | None = None
        self._task: asyncio.Task | None = None
        self._shutdown_event = asyncio.Event()
        self._ready_event = asyncio.Event()
        self._reconnect_event = asyncio.Event()
        self._error: Exception | None = None

    @property
    def is_connected(self) -> bool:
        return self._session is not None

    async def ensure_connected(self) -> None:
        """Ensure the background task is running and the session is ready."""
        if self.is_connected:
            return

        # If a background task is running but not connected (e.g. reconnecting),
        # wait for it. If it succeeds, we're done. If it fails or is stuck,
        # cancel it and start fresh.
        if self._task and not self._task.done():
            try:
                await asyncio.wait_for(self._ready_event.wait(), timeout=10.0)
                if self.is_connected:
                    return
            except asyncio.TimeoutError:
                pass
            # Task didn't become ready in time — cancel and restart
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

        self._shutdown_event.clear()
        self._reconnect_event.clear()
        self._ready_event.clear()
        self._error = None
        self._task = asyncio.create_task(self._run_loop())
        await self._ready_event.wait()
        if self._error is not None:
            raise self._error
        if not self.is_connected:
            raise ConnectionError(
                f"MCP server '{self.name}': failed to connect after "
                f"{_MAX_INITIAL_CONNECT_RETRIES} attempts. "
                f"Check that the server command '{self.config.command}' is installed and accessible."
            )

    async def _run_loop(self) -> None:
        """Persistent background task: connect, wait for events, reconnect on failure."""
        retries = 0
        initial_retries = 0
        backoff = _INITIAL_BACKOFF

        while True:
            try:
                if self.config.transport == "stdio":
                    await self._run_stdio()
                elif self.config.transport == "streamable_http":
                    await self._run_http()
                else:
                    raise ValueError(f"Unknown transport: {self.config.transport}")

                # Transport returned cleanly — check which event triggered
                if self._shutdown_event.is_set():
                    break
                # Reconnect requested (hot-reload or manual refresh)
                logger.info("MCP server '%s': reconnecting", self.name)
                self._session = None
                self._tools_cache = []
                self._connected_at = None
                self._ready_event.clear()
                # Reset retry state for fresh reconnection
                retries = 0
                initial_retries = 0
                backoff = _INITIAL_BACKOFF
                continue

            except asyncio.CancelledError:
                self._session = None
                raise

            except Exception as exc:
                self._session = None
                self._tools_cache = []
                self._connected_at = None

                # Initial connection retries with backoff
                if not self._ready_event.is_set():
                    initial_retries += 1
                    if initial_retries > _MAX_INITIAL_CONNECT_RETRIES:
                        logger.warning(
                            "MCP server '%s' failed initial connection after "
                            "%d attempts, giving up: %s",
                            self.name, _MAX_INITIAL_CONNECT_RETRIES, exc,
                        )
                        self._error = exc
                        self._ready_event.set()
                        return

                    logger.warning(
                        "MCP server '%s' initial connection failed "
                        "(attempt %d/%d), retrying in %.0fs: %s",
                        self.name, initial_retries,
                        _MAX_INITIAL_CONNECT_RETRIES, backoff, exc,
                    )
                    try:
                        await asyncio.wait_for(
                            self._shutdown_event.wait(), timeout=backoff,
                        )
                        self._error = exc
                        self._ready_event.set()
                        return
                    except asyncio.TimeoutError:
                        pass
                    backoff = min(backoff * 2, _MAX_BACKOFF_SECONDS)

                    if self._shutdown_event.is_set():
                        self._error = exc
                        self._ready_event.set()
                        return
                    continue

                # Already connected — check if shutdown requested
                if self._shutdown_event.is_set():
                    break

                # Connection dropped — retry with backoff
                retries += 1
                if retries > 5:
                    logger.error(
                        "MCP server '%s' disconnected after %d retries, giving up: %s",
                        self.name, retries, exc,
                    )
                    self._error = exc
                    self._ready_event.set()
                    return

                logger.warning(
                    "MCP server '%s' disconnected (retry %d/5), "
                    "reconnecting in %.0fs: %s",
                    self.name, retries, backoff, exc,
                )
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(), timeout=backoff,
                    )
                    break
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2, _MAX_BACKOFF_SECONDS)

    async def _run_stdio(self) -> None:
        """Run the stdio transport loop."""
        if not self.config.command:
            raise ValueError(f"MCP server '{self.name}': stdio transport requires 'command'")

        params = StdioServerParameters(
            command=self.config.command,
            args=self.config.args or [],
            env=self.config.env or None,
        )

        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                self._session = session
                self._connected_at = time.time()
                self._error = None
                self._ready_event.set()
                logger.info("MCP server '%s' connected (stdio)", self.name)

                # Block until shutdown or reconnect is requested
                await self._wait_for_event()

    async def _run_http(self) -> None:
        """Run the HTTP/StreamableHTTP transport loop."""
        if not self.config.url:
            raise ValueError(f"MCP server '{self.name}': http transport requires 'url'")

        try:
            from mcp.client.streamable_http import streamablehttp_client
        except ImportError:
            raise ImportError("MCP SDK streamable_http transport not available. Update mcp package.")

        async with streamablehttp_client(
            url=self.config.url,
            headers=self.config.headers or None,
            timeout=self.config.timeout,
        ) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                self._session = session
                self._connected_at = time.time()
                self._error = None
                self._ready_event.set()
                logger.info("MCP server '%s' connected (streamable_http)", self.name)

                await self._wait_for_event()

    async def _wait_for_event(self) -> None:
        """Block until either shutdown or reconnect event fires."""
        shutdown_task = asyncio.create_task(self._shutdown_event.wait())
        reconnect_task = asyncio.create_task(self._reconnect_event.wait())

        try:
            done, _ = await asyncio.wait(
                {shutdown_task, reconnect_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in done:
                t.cancel()
        finally:
            for t in (shutdown_task, reconnect_task):
                if not t.done():
                    t.cancel()
                    try:
                        await t
                    except asyncio.CancelledError:
                        pass

        self._reconnect_event.clear()

    async def disconnect(self) -> None:
        """Shut down the background task and disconnect."""
        self._shutdown_event.set()
        self._reconnect_event.set()  # Unblock _wait_for_event if waiting
        if self._task and not self._task.done():
            try:
                await asyncio.wait_for(self._task, timeout=10.0)
            except asyncio.TimeoutError:
                self._task.cancel()
                try:
                    await self._task
                except (asyncio.CancelledError, Exception):
                    pass
        self._session = None
        self._tools_cache = []
        self._connected_at = None
        self._task = None
        # Clear event state so the client can be safely reused
        self._shutdown_event.clear()
        self._reconnect_event.clear()
        self._ready_event.clear()
        self._error = None
        logger.info("MCP server '%s' disconnected", self.name)

    def trigger_reconnect(self) -> None:
        """Signal the background task to reconnect (for hot-reload)."""
        if self._task and not self._task.done():
            self.invalidate_tool_cache()
            self._reconnect_event.set()

    async def list_tools(self) -> list[dict]:
        """List tools from the server, using cache if available."""
        if not self._tools_cache:
            await self.ensure_connected()
            if self._session is None:
                raise ConnectionError(f"MCP server '{self.name}' is not connected")
            result = await self._session.list_tools()
            self._tools_cache = [
                {
                    "name": t.name,
                    "description": t.description or "",
                    "inputSchema": t.inputSchema or {"type": "object", "properties": {}},
                }
                for t in result.tools
            ]
            logger.info("MCP server '%s': %d tools discovered", self.name, len(self._tools_cache))
        return self._tools_cache

    async def call_tool(self, tool_name: str, arguments: dict) -> Any:
        """Call a tool on the MCP server."""
        await self.ensure_connected()
        if self._session is None:
            raise ConnectionError(f"MCP server '{self.name}' is not connected")
        logger.info("MCP tool call: %s -> %s", self.name, tool_name)
        result = await self._session.call_tool(tool_name, arguments)
        return {
            "content": [
                {"type": c.type, "text": c.text}
                if hasattr(c, "text")
                else {"type": c.type, "data": c.data, "mimeType": c.mimeType}
                if hasattr(c, "data")
                else {"type": c.type}
                for c in result.content
            ],
            "isError": result.isError if hasattr(result, "isError") else False,
        }

    def invalidate_tool_cache(self) -> None:
        self._tools_cache = []
