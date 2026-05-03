"""MCP server configuration models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server."""

    transport: Literal["stdio", "streamable_http"]
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    timeout: int = 30
    max_retries: int = 3


class MCPDefaultsConfig(BaseModel):
    """Default settings for MCP servers."""

    timeout: int = 30
    max_retries: int = 3


class MCPConfig(BaseModel):
    """Top-level MCP configuration."""

    enabled: bool = True
    servers: dict[str, MCPServerConfig] | None = Field(default_factory=dict)
    defaults: MCPDefaultsConfig = Field(default_factory=MCPDefaultsConfig)


class ServerStatus(BaseModel):
    """Runtime status of an MCP server."""

    name: str
    transport: str
    connected: bool
    tool_count: int
    error: str | None = None
