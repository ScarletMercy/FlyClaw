"""Adapter that converts MCP tools into ToolDef instances."""
from __future__ import annotations
import logging
from typing import Any
from src.agent.tooldef import ToolDef

logger = logging.getLogger("myclaw.mcp.adapter")

_DANGEROUS_KEYWORDS = frozenset(["delete","remove","write","execute","send","drop","truncate","kill","shutdown","restart","purge"])

class MCPToolAdapter:
    def __init__(self, get_client_fn):
        self._get_client = get_client_fn

    def create_tool(self, server_name: str, mcp_tool: dict) -> ToolDef:
        raw_name = mcp_tool["name"]
        tool_name = f"mcp__{server_name}__{raw_name}"
        description = mcp_tool.get("description", f"MCP tool: {raw_name}")
        schema = mcp_tool.get("inputSchema", {})
        async def execute(**kwargs):
            from src.mcp.manager import get_mcp_manager
            manager = get_mcp_manager()
            client = await manager.ensure_connected(server_name)
            result = await client.call_tool(raw_name, kwargs)
            return self._format_result(result)
        tool = ToolDef.from_schema(name=tool_name, description=description, parameters=schema or {"type":"object","properties":{}}, fn=execute)
        return tool

    @staticmethod
    def _needs_approval(mcp_tool: dict) -> bool:
        name = mcp_tool.get("name", "").lower()
        return any(kw in name for kw in _DANGEROUS_KEYWORDS)

    @staticmethod
    def _format_result(result: Any) -> str:
        if isinstance(result, dict):
            content = result.get("content", [])
            parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text": parts.append(item.get("text", ""))
                    elif item.get("type") == "image": parts.append("[image]")
                    elif item.get("type") == "resource": parts.append(f"[resource: {item.get('resource', {}).get('uri', '')}]")
                    else: parts.append(str(item))
                else: parts.append(str(item))
            return "\n".join(parts) if parts else str(result)
        return str(result)

def get_mcp_tools() -> list[ToolDef]:
    try:
        from src.mcp.manager import get_mcp_manager
        return get_mcp_manager().get_all_tools()
    except Exception:
        return []
