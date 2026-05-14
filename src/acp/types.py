from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AcpContentBlock(BaseModel):
    type: Literal["text", "resource", "resource_link", "image"]
    text: str | None = None
    data: str | None = None
    mimeType: str | None = None
    uri: str | None = None
    title: str | None = None


class AcpInitializeRequest(BaseModel):
    protocolVersion: str = "0.2"
    clientCapabilities: dict = Field(default_factory=dict)


class AcpInitializeResponse(BaseModel):
    protocolVersion: str = "0.2"
    agentCapabilities: dict = Field(default_factory=dict)
    configOptions: list = Field(default_factory=list)
    modes: list = Field(default_factory=list)


class AcpNewSessionRequest(BaseModel):
    cwd: str | None = None
    mcpServers: dict | None = None
    sessionLabel: str | None = None


class AcpNewSessionResponse(BaseModel):
    sessionId: str
    configOptions: list = Field(default_factory=list)
    modes: list = Field(default_factory=list)


class AcpSessionUpdate(BaseModel):
    type: Literal[
        "agent_message_chunk",
        "agent_thought_chunk",
        "tool_call",
        "tool_call_update",
        "available_commands_update",
        "session_info_update",
        "usage_update",
    ]
    content: dict | None = None
    toolCallId: str | None = None
    title: str | None = None
    status: str | None = None
    kind: str | None = None
    locations: list | None = None
    rawOutput: str | None = None
    used: int | None = None
    size: int | None = None


class AcpPromptRequest(BaseModel):
    sessionId: str
    content: list[AcpContentBlock]
    mode: str | None = None
    configOptions: list | None = None


class AcpPromptResponse(BaseModel):
    stopReason: Literal["end_turn", "cancelled", "max_tokens"] = "end_turn"
    usage: dict = Field(default_factory=dict)


class AcpCancelRequest(BaseModel):
    sessionId: str


class AcpLoadSessionRequest(BaseModel):
    sessionId: str


class AcpListSessionsResponse(BaseModel):
    sessions: list[dict] = Field(default_factory=list)


class AcpError(BaseModel):
    code: str
    message: str
