"""Adapter that converts MCP tools into LangGraph BaseTool instances."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

logger = logging.getLogger("myclaw.mcp.adapter")

_DANGEROUS_KEYWORDS = frozenset(
    [
        "delete",
        "remove",
        "write",
        "execute",
        "send",
        "drop",
        "truncate",
        "kill",
        "shutdown",
        "restart",
        "purge",
    ]
)


def _json_schema_to_pydantic(schema: dict, tool_name: str) -> type[BaseModel]:
    """Convert a JSON Schema to a Pydantic model for tool args."""
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    fields: dict[str, Any] = {}
    for prop_name, prop_schema in properties.items():
        prop_type = str
        if prop_schema.get("type") == "integer":
            prop_type = int
        elif prop_schema.get("type") == "number":
            prop_type = float
        elif prop_schema.get("type") == "boolean":
            prop_type = bool
        elif prop_schema.get("type") == "array":
            prop_type = list
        elif prop_schema.get("type") == "object":
            prop_type = dict

        field_default = ... if prop_name in required else None
        field_desc = prop_schema.get("description", "")
        fields[prop_name] = (prop_type, Field(default=field_default, description=field_desc))

    model_name = f"{tool_name}_args"
    if not fields:
        fields["_placeholder"] = (str, Field(default="", description="No arguments required"))
    return create_model(model_name, **fields)


class MCPToolAdapter:
    """Converts MCP tool definitions into LangGraph BaseTool instances."""

    def __init__(self, get_client_fn):
        """
        Args:
            get_client_fn: async callable(server_name) -> MCPClient
        """
        self._get_client = get_client_fn

    def create_tool(self, server_name: str, mcp_tool: dict) -> StructuredTool:
        """Create a LangGraph BaseTool from an MCP tool definition."""
        raw_name = mcp_tool["name"]
        tool_name = f"mcp__{server_name}__{raw_name}"
        description = mcp_tool.get("description", f"MCP tool: {raw_name}")
        schema = mcp_tool.get("inputSchema", {})

        args_model = _json_schema_to_pydantic(schema, tool_name)
        requires_approval = self._needs_approval(mcp_tool)

        async def execute(**kwargs):
            from src.mcp.manager import get_mcp_manager

            manager = get_mcp_manager()
            client = await manager.ensure_connected(server_name)
            result = await client.call_tool(raw_name, kwargs)
            return self._format_result(result)

        tool = StructuredTool(
            name=tool_name,
            description=description,
            args_schema=args_model,
            func=execute,
            coroutine=execute,
        )

        if requires_approval:
            tool.metadata = {"requires_approval": True}

        return tool

    @staticmethod
    def _needs_approval(mcp_tool: dict) -> bool:
        """Heuristic: flag dangerous-looking tools for approval."""
        name = mcp_tool.get("name", "").lower()
        return any(kw in name for kw in _DANGEROUS_KEYWORDS)

    @staticmethod
    def _format_result(result: Any) -> str:
        """Format an MCP tool call result as a string for the LLM."""
        if isinstance(result, dict):
            # MCP tools/call returns {"content": [...]}
            content = result.get("content", [])
            parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(item.get("text", ""))
                    elif item.get("type") == "image":
                        parts.append("[image]")
                    elif item.get("type") == "resource":
                        parts.append(f"[resource: {item.get('resource', {}).get('uri', '')}]")
                    else:
                        parts.append(str(item))
                else:
                    parts.append(str(item))
            return "\n".join(parts) if parts else str(result)
        return str(result)
