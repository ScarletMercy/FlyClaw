"""Streamable HTTP transport: communicates via POST to an MCP endpoint."""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

import httpx

from src.mcp.transport.base import MCPTransport
from src.mcp.transport.jsonrpc import JSONRPCError, JSONRPCProtocol

logger = logging.getLogger("myclaw.mcp.transport.http")


class StreamableHTTPTransport(MCPTransport):
    """MCP transport over Streamable HTTP (POST-based)."""

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        self._url = url
        self._headers = headers or {}
        self._timeout = timeout
        self._max_retries = max_retries
        self._session_id: str | None = None
        self._protocol = JSONRPCProtocol()
        self._connected = False
        self._client: httpx.AsyncClient | None = None

    async def connect(self) -> None:
        if self._connected:
            return

        self._client = httpx.AsyncClient(
            headers=self._headers,
            timeout=httpx.Timeout(self._timeout + 10.0),
        )

        # MCP initialize handshake
        result = await self._send_request_raw(
            "initialize",
            params={
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "myclaw", "version": "0.1.0"},
            },
        )
        logger.info("MCP server initialized: %s", result.get("serverInfo", {}).get("name", "unknown"))

        # Send initialized notification
        await self._send_notification_raw("notifications/initialized")
        self._connected = True

    async def _send_request_raw(self, method: str, params: dict | None = None) -> Any:
        """Send a single JSON-RPC request and get the response directly."""
        assert self._client is not None

        request_id = str(self._protocol._next_id)
        self._protocol._next_id += 1

        message: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params is not None:
            message["params"] = params

        headers = {**self._headers}
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id

        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.post(self._url, json=message, headers=headers)
                response.raise_for_status()

                # Capture session id
                sid = response.headers.get("Mcp-Session-Id")
                if sid:
                    self._session_id = sid

                body = response.json()

                if "error" in body:
                    error = body["error"]
                    raise JSONRPCError(
                        error.get("code", -32000),
                        error.get("message", "Unknown error"),
                    )

                return body.get("result")

            except httpx.HTTPStatusError as e:
                if e.response.status_code < 500:
                    raise
                last_exc = e
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_exc = e

            if attempt < self._max_retries:
                wait = min(0.5 * (2 ** attempt), 10.0) + random.uniform(0, 0.5)
                logger.warning(
                    "MCP HTTP %s retry %d/%d in %.1fs: %s",
                    method, attempt + 1, self._max_retries, wait, last_exc,
                )
                await asyncio.sleep(wait)

        raise last_exc  # type: ignore[misc]

    async def _send_notification_raw(self, method: str, params: dict | None = None) -> None:
        """Send a JSON-RPC notification."""
        assert self._client is not None

        message: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params is not None:
            message["params"] = params

        headers = {**self._headers}
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id

        try:
            await self._client.post(self._url, json=message, headers=headers)
        except Exception as e:
            logger.warning("Failed to send notification: %s", e)

    async def send(self, data: str) -> None:
        """Not used directly in HTTP transport — requests are sent inline."""
        raise NotImplementedError("HTTP transport sends requests inline, not via send()")

    async def disconnect(self) -> None:
        self._connected = False
        if self._client:
            await self._client.aclose()
            self._client = None
        self._session_id = None

    async def list_tools(self) -> list[dict]:
        result = await self._send_request_raw("tools/list")
        return result.get("tools", []) if isinstance(result, dict) else []

    async def call_tool(self, name: str, arguments: dict) -> Any:
        return await self._send_request_raw(
            "tools/call",
            params={"name": name, "arguments": arguments},
        )

    @property
    def is_connected(self) -> bool:
        return self._connected and self._client is not None
