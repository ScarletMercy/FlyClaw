import json
from io import BytesIO
from unittest.mock import AsyncMock, patch
import pytest
from src.acp.server import AcpServer
from src.acp.transport import NdjsonTransport


class TestAcpServer:
    @pytest.mark.asyncio
    async def test_initialize(self):
        outgoing = BytesIO()
        transport = NdjsonTransport(writer=outgoing)
        server = AcpServer(transport=transport)
        msg = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "0.2"}}
        await server.handle_message(msg)
        outgoing.seek(0)
        response = json.loads(outgoing.readline())
        assert response["id"] == 1
        assert "protocolVersion" in response.get("result", {})

    @pytest.mark.asyncio
    async def test_new_session(self):
        outgoing = BytesIO()
        transport = NdjsonTransport(writer=outgoing)
        server = AcpServer(transport=transport)
        msg = {"jsonrpc": "2.0", "id": 2, "method": "newSession", "params": {"cwd": "/tmp"}}
        await server.handle_message(msg)
        outgoing.seek(0)
        response = json.loads(outgoing.readline())
        assert response["id"] == 2
        assert "sessionId" in response.get("result", {})

    @pytest.mark.asyncio
    async def test_list_sessions(self):
        outgoing = BytesIO()
        transport = NdjsonTransport(writer=outgoing)
        server = AcpServer(transport=transport)
        await server.handle_message({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        outgoing.seek(0)
        outgoing.truncate()
        await server.handle_message({"jsonrpc": "2.0", "id": 3, "method": "listSessions", "params": {}})
        outgoing.seek(0)
        response = json.loads(outgoing.readline())
        assert response["id"] == 3
        assert "sessions" in response.get("result", {})

    @pytest.mark.asyncio
    async def test_unknown_method_returns_error(self):
        outgoing = BytesIO()
        transport = NdjsonTransport(writer=outgoing)
        server = AcpServer(transport=transport)
        await server.handle_message({"jsonrpc": "2.0", "id": 99, "method": "nonexistent", "params": {}})
        outgoing.seek(0)
        response = json.loads(outgoing.readline())
        assert response["id"] == 99
        assert "error" in response
