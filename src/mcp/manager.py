"""MCPManager: singleton that manages all MCP server connections."""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from langchain_core.tools import BaseTool

from src.mcp.adapter import MCPToolAdapter
from src.mcp.client import MCPClient
from src.mcp.config_models import MCPServerConfig, ServerStatus

logger = logging.getLogger("myclaw.mcp.manager")

_manager: Optional[MCPManager] = None


class MCPManager:
    """Manages all MCP server connections, tool discovery, and dynamic lifecycle."""

    def __init__(self):
        self._configs: dict[str, MCPServerConfig] = {}
        self._clients: dict[str, MCPClient] = {}
        self._adapter = MCPToolAdapter(self.ensure_connected)
        self._tools: dict[str, BaseTool] = {}  # tool_name -> BaseTool
        self._connect_locks: dict[str, asyncio.Lock] = {}

    def load_config(self, configs: dict[str, MCPServerConfig]) -> None:
        """Load server configurations and eagerly connect to discover tools."""
        self._configs.update(configs)
        # Register lazy-connect tools so the LLM sees MCP servers are available
        for name in configs:
            self._create_server_tool(name)

    async def ensure_connected(self, server_name: str) -> MCPClient:
        """Ensure a server is connected, connecting lazily if needed."""
        if server_name in self._clients and self._clients[server_name].is_connected:
            return self._clients[server_name]

        if server_name not in self._connect_locks:
            self._connect_locks[server_name] = asyncio.Lock()

        async with self._connect_locks[server_name]:
            # Double-check after acquiring lock
            if server_name in self._clients and self._clients[server_name].is_connected:
                return self._clients[server_name]

            config = self._configs.get(server_name)
            if config is None:
                raise ValueError(f"MCP server '{server_name}' not configured")

            client = MCPClient(server_name, config)
            await client.connect()
            self._clients[server_name] = client

            # Discover and register tools
            await self._register_tools(client)
            return client

    async def add_server(self, name: str, config: MCPServerConfig) -> None:
        """Dynamically add a new MCP server at runtime."""
        self._configs[name] = config
        self._create_server_tool(name)
        logger.info("MCP server '%s' added (not yet connected)", name)

    async def remove_server(self, name: str) -> None:
        """Remove an MCP server, disconnecting if necessary."""
        self._unregister_tools(name)
        client = self._clients.pop(name, None)
        if client:
            await client.disconnect()
        self._configs.pop(name, None)
        logger.info("MCP server '%s' removed", name)

    def get_all_tools(self) -> list[BaseTool]:
        """Return all MCP tools (both connected and placeholders)."""
        return list(self._tools.values())

    async def list_servers(self) -> list[ServerStatus]:
        """Return status of all configured servers."""
        statuses = []
        for name, config in self._configs.items():
            client = self._clients.get(name)
            connected = client is not None and client.is_connected
            tool_count = len([t for t in self._tools if t.startswith(f"mcp__{name}__")])
            error = None
            if not connected and client is not None:
                error = "disconnected"
            statuses.append(
                ServerStatus(
                    name=name,
                    transport=config.transport,
                    connected=connected,
                    tool_count=tool_count,
                    error=error,
                )
            )
        return statuses

    async def disconnect_all(self) -> None:
        """Disconnect all servers. Called during shutdown."""
        for client in self._clients.values():
            try:
                await client.disconnect()
            except Exception as e:
                logger.warning("Error disconnecting MCP server '%s': %s", client.name, e)
        self._clients.clear()

    def _create_server_tool(self, server_name: str) -> None:
        """Create a lazy-connect tool that triggers connection on first use.

        This ensures the LLM sees MCP servers as available tools at graph
        build time, even before servers are actually connected.
        """
        tool_name = f"mcp__{server_name}__list_tools"

        # Avoid duplicate registration
        if tool_name in self._tools:
            return

        async def _list_and_connect(**kwargs):
            client = await self.ensure_connected(server_name)
            tools = await client.list_tools()
            return f"MCP server '{server_name}': {len(tools)} tools available: " + ", ".join(
                t.get("name", "?") for t in tools
            )

        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel

        class EmptyArgs(BaseModel):
            pass

        tool = StructuredTool(
            name=tool_name,
            description=f"List available tools from MCP server '{server_name}'. Connects lazily on first call.",
            args_schema=EmptyArgs,
            coroutine=_list_and_connect,
        )
        self._tools[tool_name] = tool
        logger.debug("Lazy-connect tool registered for MCP server '%s'", server_name)

    async def _register_tools(self, client: MCPClient) -> None:
        """Discover tools from a connected server and register them."""
        mcp_tools = await client.list_tools()
        new_tools = []

        for mcp_tool in mcp_tools:
            tool = self._adapter.create_tool(client.name, mcp_tool)
            self._tools[tool.name] = tool
            new_tools.append(tool.name)

        if new_tools:
            logger.info(
                "MCP server '%s': registered %d tools: %s",
                client.name,
                len(new_tools),
                ", ".join(new_tools[:5]) + ("..." if len(new_tools) > 5 else ""),
            )

    def _unregister_tools(self, server_name: str) -> None:
        """Remove all tools for a server."""
        prefix = f"mcp__{server_name}__"
        removed = [k for k in self._tools if k.startswith(prefix)]
        for k in removed:
            del self._tools[k]
        logger.info("MCP server '%s': unregistered %d tools", server_name, len(removed))


def get_mcp_manager() -> MCPManager:
    global _manager
    if _manager is None:
        _manager = MCPManager()
    return _manager
