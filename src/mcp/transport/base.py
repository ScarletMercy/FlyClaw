"""Abstract base class for MCP transports."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class MCPTransport(ABC):
    """Abstract transport for MCP communication."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the MCP server."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Close the connection."""
        ...

    @abstractmethod
    async def send(self, data: str) -> None:
        """Send raw data to the server."""
        ...

    @abstractmethod
    async def list_tools(self) -> list[dict]:
        """List available tools from the server."""
        ...

    @abstractmethod
    async def call_tool(self, name: str, arguments: dict) -> Any:
        """Call a tool on the server."""
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Whether the transport is currently connected."""
        ...
