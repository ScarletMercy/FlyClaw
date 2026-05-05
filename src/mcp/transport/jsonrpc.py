"""JSON-RPC 2.0 message handling for MCP protocol."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

logger = logging.getLogger("myclaw.mcp.jsonrpc")


class JSONRPCError(Exception):
    """Error returned by the MCP server."""

    def __init__(self, code: int, message: str, data: Any = None):
        self.code = code
        self.message = message
        self.data = data
        super().__init__(f"JSON-RPC error {code}: {message}")


class JSONRPCProtocol:
    """JSON-RPC 2.0 protocol handler.

    Handles request/response correlation and notification parsing.
    Callers implement send_message / read_message for their transport.
    """

    def __init__(self):
        self._pending: dict[str, asyncio.Future] = {}
        self._next_id = 0

    async def send_request(
        self,
        send_fn,
        method: str,
        params: dict | None = None,
        timeout: float = 30.0,
    ) -> Any:
        """Send a JSON-RPC request and wait for the response."""
        request_id = str(self._next_id)
        self._next_id += 1

        message = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params is not None:
            message["params"] = params

        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future

        try:
            await send_fn(json.dumps(message))
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(request_id, None)
            raise JSONRPCError(-32000, f"Request timeout: {method}")
        except Exception:
            self._pending.pop(request_id, None)
            raise

    async def send_notification(self, send_fn, method: str, params: dict | None = None) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        message = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params is not None:
            message["params"] = params
        await send_fn(json.dumps(message))

    def handle_message(self, raw: str) -> Any | None:
        """Parse an incoming JSON-RPC message.

        Returns the result if it's a response to a pending request.
        Returns None if it's a notification.
        Raises JSONRPCError if the server returned an error response.
        """
        try:
            message = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning("Invalid JSON from MCP server: %s", e)
            return None

        if "id" in message and "error" in message:
            # Error response
            error = message["error"]
            code = error.get("code", -32000)
            err_msg = error.get("message", "Unknown error")
            request_id = str(message["id"])
            future = self._pending.pop(request_id, None)
            if future and not future.done():
                future.set_exception(JSONRPCError(code, err_msg, error.get("data")))
            return None

        if "id" in message and "result" in message:
            # Successful response
            request_id = str(message["id"])
            future = self._pending.pop(request_id, None)
            if future and not future.done():
                future.set_result(message["result"])
            return message["result"]

        # Notification or other message — ignore for now
        return None

    def cancel_all(self):
        """Cancel all pending requests."""
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(JSONRPCError(-32000, "Connection closed"))
        self._pending.clear()
