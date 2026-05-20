from __future__ import annotations

import logging

from src._container import get_container

logger = logging.getLogger("myclaw.mcp_tools")


def _get_manager():
    from src.mcp.manager import get_mcp_manager
    return get_mcp_manager()


def _clean_name(name: str) -> str:
    return name.strip().strip("[]()`").strip().lower().replace(" ", "_")


async def mcp_manage(
    action: str,
    name: str = "",
    transport: str = "",
    command: str = "",
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
    url: str = "",
    headers: dict[str, str] | None = None,
    timeout: int = 30,
) -> str:
    """Manage MCP (Model Context Protocol) servers — list, add, remove, test connectivity.

    Args:
        action: One of: list, add, remove, test, connect.
        name: Server name (required for add/remove/test/connect).
        transport: "stdio" or "streamable_http" (required for add). Auto-detected: if command is set, defaults to stdio; if url is set, defaults to streamable_http.
        command: Command to run for stdio transport (e.g. "npx", "uvx", "python").
        args: Command arguments (e.g. ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]).
        env: Environment variables for the subprocess.
        url: URL for streamable_http transport.
        headers: HTTP headers for streamable_http transport.
        timeout: Connection timeout in seconds. Default 30.
    """
    mgr = _get_manager()
    if mgr is None:
        return "MCP manager not initialized. No MCP servers configured."
    action = action.strip().lower()

    if action == "list":
        servers = await mgr.list_servers()
        if not servers:
            return "No MCP servers configured. Use mcp_manage(action=\"add\", ...) to add one."
        lines = []
        for s in servers:
            status = "connected" if s.connected else "not connected"
            lines.append(f"- {s.name}  ({s.transport}, {status}, {s.tool_count} tools)")
        return "\n".join(lines)

    if action == "add":
        if not name:
            return "Error: name is required for add."
        name = _clean_name(name)

        if not transport:
            if command:
                transport = "stdio"
            elif url:
                transport = "streamable_http"
            else:
                return "Error: transport is required. Set command for stdio or url for streamable_http."

        from src.mcp.config_models import MCPServerConfig

        config = MCPServerConfig(
            transport=transport,
            command=command or None,
            args=args or [],
            env=env or {},
            url=url or None,
            headers=headers or {},
            timeout=timeout,
        )

        await mgr.add_server(name, config)

        try:
            from src.config import load_config, save_config
            cfg = load_config()
            if cfg.mcp.servers is None:
                cfg.mcp.servers = {}
            cfg.mcp.servers[name] = config
            save_config(cfg)
        except Exception as e:
            logger.warning("Failed to persist MCP server config: %s", e)

        return f"MCP server '{name}' added ({transport}). Use mcp_manage(action=\"test\", name=\"{name}\") to verify."

    if action == "remove":
        if not name:
            return "Error: name is required for remove."
        name = _clean_name(name)

        await mgr.remove_server(name)

        try:
            from src.config import load_config, save_config
            cfg = load_config()
            if cfg.mcp.servers and name in cfg.mcp.servers:
                del cfg.mcp.servers[name]
                save_config(cfg)
        except Exception as e:
            logger.warning("Failed to remove MCP server from config: %s", e)

        return f"MCP server '{name}' removed."

    if action == "test":
        if not name:
            return "Error: name is required for test."
        name = _clean_name(name)

        try:
            client = await mgr.ensure_connected(name)
            tools = await client.list_tools()
            if not tools:
                return f"Server '{name}' connected but has no tools."
            lines = [f"Server '{name}' connected! {len(tools)} tools:"]
            for t in tools:
                desc = t.get("description", "")[:80]
                lines.append(f"  - {t['name']}: {desc}")
            return "\n".join(lines)
        except Exception as e:
            return f"Connection failed: {type(e).__name__}: {e}"

    if action == "connect":
        if not name:
            return "Error: name is required for connect."
        name = _clean_name(name)

        try:
            client = await mgr.ensure_connected(name)
            return f"Server '{name}' connected ({client.config.transport})."
        except Exception as e:
            return f"Connection failed: {type(e).__name__}: {e}"

    return f"Unknown action: '{action}'. Use: list, add, remove, test, connect."


def get_tools() -> list:
    from src.agent.tooldef import ToolDef
    return [ToolDef.from_function(mcp_manage)]
