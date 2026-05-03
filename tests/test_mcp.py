"""Tests for MCP subsystem."""

import asyncio
import json
import os
import sys
import pytest

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestJSONRPCProtocol:
    """Test JSON-RPC 2.0 message handling."""

    def test_handle_response(self):
        from src.mcp.transport.jsonrpc import JSONRPCProtocol

        proto = JSONRPCProtocol()

        # Simulate a response
        msg = json.dumps({"jsonrpc": "2.0", "id": "0", "result": {"tools": []}})
        result = proto.handle_message(msg)
        assert result == {"tools": []}

    def test_handle_error_response(self):
        from src.mcp.transport.jsonrpc import JSONRPCProtocol, JSONRPCError

        proto = JSONRPCProtocol()

        # Manually create a pending future to test error handling
        loop = asyncio.new_event_loop()
        try:
            future = loop.create_future()
            proto._pending["1"] = future

            msg = json.dumps({"jsonrpc": "2.0", "id": "1", "error": {"code": -32600, "message": "Invalid"}})
            proto.handle_message(msg)

            with pytest.raises(JSONRPCError):
                loop.run_until_complete(future)
        finally:
            loop.close()

    def test_cancel_all(self):
        from src.mcp.transport.jsonrpc import JSONRPCProtocol, JSONRPCError

        proto = JSONRPCProtocol()
        loop = asyncio.new_event_loop()
        try:
            future = loop.create_future()
            proto._pending["test"] = future
            proto.cancel_all()

            with pytest.raises(JSONRPCError):
                loop.run_until_complete(future)
        finally:
            loop.close()


class TestMCPConfigModels:
    """Test MCP configuration models."""

    def test_stdio_config(self):
        from src.mcp.config_models import MCPServerConfig

        config = MCPServerConfig(
            transport="stdio",
            command="uvx",
            args=["context7-mcp"],
        )
        assert config.transport == "stdio"
        assert config.command == "uvx"

    def test_http_config(self):
        from src.mcp.config_models import MCPServerConfig

        config = MCPServerConfig(
            transport="streamable_http",
            url="https://example.com/mcp",
            headers={"Authorization": "Bearer token"},
        )
        assert config.transport == "streamable_http"
        assert config.url == "https://example.com/mcp"

    def test_mcp_config_with_servers(self):
        from src.mcp.config_models import MCPConfig, MCPServerConfig

        config = MCPConfig(
            enabled=True,
            servers={
                "test": MCPServerConfig(transport="stdio", command="echo"),
            },
        )
        assert config.enabled is True
        assert len(config.servers) == 1

    def test_server_status_model(self):
        from src.mcp.config_models import ServerStatus

        status = ServerStatus(name="test", transport="stdio", connected=True, tool_count=5)
        dumped = status.model_dump()
        assert dumped["name"] == "test"
        assert dumped["connected"] is True


class TestMCPToolAdapter:
    """Test MCP tool adapter."""

    def test_needs_approval_dangerous(self):
        from src.mcp.adapter import MCPToolAdapter

        assert MCPToolAdapter._needs_approval({"name": "delete_file"}) is True
        assert MCPToolAdapter._needs_approval({"name": "remove_record"}) is True
        assert MCPToolAdapter._needs_approval({"name": "execute_query"}) is True
        assert MCPToolAdapter._needs_approval({"name": "send_message"}) is True

    def test_needs_approval_safe(self):
        from src.mcp.adapter import MCPToolAdapter

        assert MCPToolAdapter._needs_approval({"name": "list_files"}) is False
        assert MCPToolAdapter._needs_approval({"name": "get_user"}) is False
        assert MCPToolAdapter._needs_approval({"name": "search"}) is False

    def test_format_result_text(self):
        from src.mcp.adapter import MCPToolAdapter

        result = {
            "content": [
                {"type": "text", "text": "hello world"},
                {"type": "text", "text": "second line"},
            ]
        }
        formatted = MCPToolAdapter._format_result(result)
        assert "hello world" in formatted
        assert "second line" in formatted

    def test_format_result_image(self):
        from src.mcp.adapter import MCPToolAdapter

        result = {
            "content": [
                {"type": "image", "data": "abc123"},
            ]
        }
        formatted = MCPToolAdapter._format_result(result)
        assert "[image]" in formatted

    def test_json_schema_to_pydantic(self):
        from src.mcp.adapter import _json_schema_to_pydantic

        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "description": "Max results"},
            },
            "required": ["query"],
        }
        model = _json_schema_to_pydantic(schema, "test_tool")
        assert "query" in model.model_fields
        assert "limit" in model.model_fields


class TestMCPManager:
    """Test MCPManager lifecycle."""

    def test_load_config(self):
        from src.mcp.manager import MCPManager
        from src.mcp.config_models import MCPServerConfig

        manager = MCPManager()
        manager.load_config({
            "test": MCPServerConfig(transport="stdio", command="echo"),
        })
        assert "test" in manager._configs

    def test_load_config_creates_lazy_tool(self):
        from src.mcp.manager import MCPManager
        from src.mcp.config_models import MCPServerConfig

        manager = MCPManager()
        manager.load_config({
            "myserver": MCPServerConfig(transport="stdio", command="echo"),
        })
        tools = manager.get_all_tools()
        assert len(tools) == 1
        assert tools[0].name == "mcp__myserver__list_tools"

    def test_get_all_tools_empty(self):
        from src.mcp.manager import MCPManager

        manager = MCPManager()
        tools = manager.get_all_tools()
        assert tools == []

    def test_add_remove_server(self):
        from src.mcp.manager import MCPManager
        from src.mcp.config_models import MCPServerConfig

        manager = MCPManager()

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                manager.add_server("test", MCPServerConfig(transport="stdio", command="echo"))
            )
            assert "test" in manager._configs

            loop.run_until_complete(manager.remove_server("test"))
            assert "test" not in manager._configs
        finally:
            loop.close()
