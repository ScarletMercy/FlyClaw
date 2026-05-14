"""MCPClient: manages a single MCP server connection and tool invocation."""

from __future__ import annotations

import logging
import time
from typing import Any

from src.mcp.config_models import MCPServerConfig
from src.mcp.transport.base import MCPTransport
from src.mcp.transport.stdio import StdioTransport
from src.mcp.transport.streamable_http import StreamableHTTPTransport

logger = logging.getLogger("myclaw.mcp.client")


class MCPClient:
    """Manages connection to a single MCP server."""

    def __init__(self, name: str, config: MCPServerConfig):
        self.name = name
        self.config = config
        self._transport: MCPTransport | None = None
        self._tools_cache: list[dict] = []
        self._connected_at: float | None = None

    @property
    def transport(self) -> MCPTransport:
        if self._transport is None:
            raise ConnectionError(f"MCP server '{self.name}' is not connected")
        return self._transport

    @property
    def is_connected(self) -> bool:
        return self._transport is not None and self._transport.is_connected

    async def connect(self) -> None:
        if self.is_connected:
            return

        timeout = self.config.timeout or 30.0

        if self.config.transport == "stdio":
            if not self.config.command:
                raise ValueError(f"MCP server '{self.name}': stdio transport requires 'command'")
            self._transport = StdioTransport(
                command=self.config.command,
                args=self.config.args,
                env=self.config.env,
                timeout=timeout,
            )
        elif self.config.transport == "streamable_http":
            if not self.config.url:
                raise ValueError(f"MCP server '{self.name}': http transport requires 'url'")
            self._transport = StreamableHTTPTransport(
                url=self.config.url,
                headers=self.config.headers,
                timeout=timeout,
                max_retries=self.config.max_retries,
            )
        else:
            raise ValueError(f"Unknown transport: {self.config.transport}")

        await self._transport.connect()
        self._connected_at = time.time()
        logger.info("MCP server '%s' connected (%s)", self.name, self.config.transport)

    async def disconnect(self) -> None:
        if self._transport:
            await self._transport.disconnect()
            self._transport = None
            self._tools_cache = []
            self._connected_at = None
            logger.info("MCP server '%s' disconnected", self.name)

    async def list_tools(self) -> list[dict]:
        """List tools from the server, using cache if available."""
        if not self._tools_cache:
            self._tools_cache = await self.transport.list_tools()
            logger.info("MCP server '%s': %d tools discovered", self.name, len(self._tools_cache))
        return self._tools_cache

    async def call_tool(self, tool_name: str, arguments: dict) -> Any:
        """Call a tool on the MCP server."""
        logger.info("MCP tool call: %s -> %s(%s)", self.name, tool_name, arguments)
        result = await self.transport.call_tool(tool_name, arguments)
        return result

    def invalidate_tool_cache(self) -> None:
        self._tools_cache = []
