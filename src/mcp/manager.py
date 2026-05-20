"""MCPManager: singleton that manages all MCP server connections."""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from src.agent.tooldef import ToolDef

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
        self._tools: dict[str, ToolDef] = {}  # tool_name -> ToolDef
        self._connect_locks: dict[str, asyncio.Lock] = {}

    def load_config(self, configs: dict[str, MCPServerConfig]) -> None:
        """Load server configurations and eagerly connect to discover tools."""
        self._configs.update(configs)
        # Register lazy-connect tools so the LLM sees MCP servers are available
        for name in configs:
            self._create_server_tool(name)

    async def ensure_connected(self, server_name: str) -> MCPClient:
        """Ensure a server is connected, connecting lazily if needed."""
        client = self._clients.get(server_name)
        if client and client.is_connected:
            # Fast path: already connected.  Re-register tools inside the
            # lock below to avoid a concurrent _register_tools race.
            pass
        else:
            if server_name not in self._connect_locks:
                self._connect_locks[server_name] = asyncio.Lock()

            async with self._connect_locks[server_name]:
                # Double-check after acquiring lock
                client = self._clients.get(server_name)
                if client and client.is_connected:
                    # Another coroutine just connected — fall through to
                    # the tool-cache check below.
                    pass
                else:
                    config = self._configs.get(server_name)
                    if config is None:
                        raise ValueError(f"MCP server '{server_name}' not configured")

                    # If old client exists but is dead, remove it
                    if client is not None:
                        self._unregister_tools(server_name)
                        try:
                            await client.disconnect()
                        except Exception:
                            pass
                        del self._clients[server_name]

                    client = MCPClient(server_name, config)
                    self._clients[server_name] = client
                    await client.ensure_connected()

                    # Discover and register tools
                    await self._register_tools(client)
                    return client

        # Tool-cache stale check (runs under the lock for new connections,
        # or lock-free for the fast path — safe because _register_tools is
        # idempotent and only mutates self._tools).
        if client and client.is_connected and not client._tools_cache:
            if server_name not in self._connect_locks:
                self._connect_locks[server_name] = asyncio.Lock()
            async with self._connect_locks[server_name]:
                # Re-check after acquiring lock
                client = self._clients.get(server_name)
                if client and client.is_connected and not client._tools_cache:
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

    async def reload(self, mcp_config) -> None:
        """Reload MCP configuration (called by config hot-reload).

        Triggers reconnect on changed servers instead of killing them,
        so in-flight tool calls are not interrupted.
        """
        new_servers = mcp_config.servers or {}
        new_names = set(new_servers.keys())
        old_names = set(self._configs.keys())

        for removed in old_names - new_names:
            await self.remove_server(removed)

        for name, config in new_servers.items():
            if name not in self._configs or self._configs[name] != config:
                self._configs[name] = config
                self._create_server_tool(name)
                client = self._clients.get(name)
                if client and client.is_connected:
                    client.trigger_reconnect()
                    logger.info("MCP server '%s' reconnect triggered", name)
                else:
                    logger.info("MCP server '%s' config updated (not connected)", name)
            else:
                # Config unchanged — force-refresh tool list in case server
                # added or removed tools dynamically.
                client = self._clients.get(name)
                if client and client.is_connected:
                    client.invalidate_tool_cache()
                    await self._register_tools(client)
                    logger.info("MCP server '%s' tools refreshed", name)

    def get_all_tools(self) -> list[ToolDef]:
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
            try:
                client = await self.ensure_connected(server_name)
                tools = await client.list_tools()
                return f"MCP server '{server_name}': {len(tools)} tools available: " + ", ".join(
                    t.get("name", "?") for t in tools
                )
            except ConnectionError as e:
                return f"[MCP connection error] {e}"
            except Exception as e:
                return f"[MCP error] {type(e).__name__}: {e}"

        tool = ToolDef.from_schema(
            name=tool_name,
            description=f"List available tools from MCP server '{server_name}'. Connects lazily on first call.",
            parameters={"type": "object", "properties": {}},
            fn=_list_and_connect,
        )
        self._tools[tool_name] = tool
        logger.debug("Lazy-connect tool registered for MCP server '%s'", server_name)

    async def _register_tools(self, client: MCPClient) -> None:
        """Discover tools from a connected server and register them.

        Removes stale tools for the server first, so deleted tools on the
        server side are cleaned up.  The ``mcp__{server}__list_tools``
        placeholder is always restored in the finally block so the server
        remains visible even if discovery fails.
        """
        # Remove stale tools for this server before re-registering
        self._unregister_tools(client.name)

        try:
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
        finally:
            # Ensure the lazy-connect placeholder always exists
            self._create_server_tool(client.name)

    def _unregister_tools(self, server_name: str) -> None:
        """Remove all tools for a server."""
        prefix = f"mcp__{server_name}__"
        removed = [k for k in self._tools if k.startswith(prefix)]
        for k in removed:
            del self._tools[k]
        logger.info("MCP server '%s': unregistered %d tools", server_name, len(removed))


# ── Module-level singleton — delegates to ServiceContainer ──

from src._container import get_container


def get_mcp_manager() -> MCPManager:
    return get_container().mcp_manager
