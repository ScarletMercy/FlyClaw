from __future__ import annotations

from src.acp.types import (
    AcpContentBlock,
    AcpInitializeRequest,
    AcpNewSessionResponse,
    AcpSessionUpdate,
)


def test_initialize_request():
    req = AcpInitializeRequest()
    assert req.protocolVersion == "0.2"
    assert req.clientCapabilities == {}

    req2 = AcpInitializeRequest(clientCapabilities={"streaming": True})
    assert req2.clientCapabilities == {"streaming": True}


def test_new_session_response():
    resp = AcpNewSessionResponse(sessionId="abc123")
    assert resp.sessionId == "abc123"
    assert resp.configOptions == []
    assert resp.modes == []

    resp2 = AcpNewSessionResponse(
        sessionId="xyz", configOptions=[{"key": "val"}], modes=["code"]
    )
    assert resp2.configOptions == [{"key": "val"}]
    assert resp2.modes == ["code"]


def test_session_update_text_chunk():
    update = AcpSessionUpdate(
        type="agent_message_chunk",
        content={"text": "hello"},
    )
    assert update.type == "agent_message_chunk"
    assert update.content == {"text": "hello"}
    assert update.toolCallId is None

    tool_update = AcpSessionUpdate(
        type="tool_call",
        toolCallId="tc1",
        title="read_file",
    )
    assert tool_update.toolCallId == "tc1"
    assert tool_update.title == "read_file"


def test_content_block_text():
    block = AcpContentBlock(type="text", text="hello world")
    assert block.type == "text"
    assert block.text == "hello world"
    assert block.data is None
    assert block.uri is None

    img = AcpContentBlock(type="image", data="base64...", mimeType="image/png")
    assert img.type == "image"
    assert img.mimeType == "image/png"
