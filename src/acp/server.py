from __future__ import annotations

import logging
from typing import Any

from src.acp.session import AcpSessionManager
from src.acp.runtime import AgentLoopRuntime, AcpRuntimeEvent
from src.acp.transport import NdjsonTransport
from src.utils.content import content_to_text

logger = logging.getLogger("flyclaw.acp.server")


class AcpServer:
    def __init__(
        self,
        transport: NdjsonTransport | None = None,
        session_manager: AcpSessionManager | None = None,
    ):
        self._transport = transport or NdjsonTransport()
        self._sessions = session_manager or AcpSessionManager()
        self._runtime = AgentLoopRuntime(self._sessions)
        self._initialized = False

    async def handle_message(self, msg: dict[str, Any]) -> None:
        method = msg.get("method", "")
        msg_id = msg.get("id")
        params = msg.get("params", {})

        handler = {
            "initialize": self._handle_initialize,
            "newSession": self._handle_new_session,
            "prompt": self._handle_prompt,
            "cancel": self._handle_cancel,
            "listSessions": self._handle_list_sessions,
            "loadSession": self._handle_load_session,
        }.get(method)

        if not handler:
            self._send_error(msg_id, -32601, f"Method not found: {method}")
            return

        try:
            result = await handler(params)
            if msg_id is not None:
                self._send_result(msg_id, result)
        except Exception as e:
            logger.error("ACP handler '%s' failed: %s", method, e, exc_info=True)
            self._send_error(msg_id, -32603, str(e))

    async def _handle_initialize(self, params: dict) -> dict:
        self._initialized = True
        return {
            "protocolVersion": "0.2",
            "agentCapabilities": {"streaming": True, "tools": True},
            "configOptions": [],
            "modes": ["default"],
        }

    async def _handle_new_session(self, params: dict) -> dict:
        cwd = params.get("cwd", "")
        agent_id = params.get("agentId", "default")
        session_id = self._sessions.create(agent_id, cwd=cwd)
        return {
            "sessionId": session_id,
            "configOptions": [],
            "modes": ["default"],
        }

    async def _handle_prompt(self, params: dict) -> dict:
        session_id = params.get("sessionId", "")
        content = params.get("content", [])
        prompt_text = content_to_text(content, joiner=" ")

        stop_reason = "end_turn"
        async for event in self._runtime.run_turn(
            session_id=session_id,
            prompt=prompt_text,
        ):
            if event.type == "text_delta" and event.text:
                self._transport.write(
                    {
                        "jsonrpc": "2.0",
                        "method": "sessionUpdate",
                        "params": {
                            "sessionId": session_id,
                            "update": {
                                "type": "agent_message_chunk",
                                "content": {"type": "text", "text": event.text},
                            },
                        },
                    }
                )
            elif event.type == "done":
                stop_reason = event.stop_reason or "end_turn"

        return {"stopReason": stop_reason, "usage": {}}

    async def _handle_cancel(self, params: dict) -> dict:
        session_id = params.get("sessionId", "")
        await self._runtime.cancel(session_id)
        return {"cancelled": True}

    async def _handle_list_sessions(self, params: dict) -> dict:
        sessions = self._sessions.list_sessions()
        return {"sessions": [{"sessionId": s.session_id, "agentId": s.agent_id, "cwd": s.cwd} for s in sessions]}

    async def _handle_load_session(self, params: dict) -> dict:
        session_id = params.get("sessionId", "")
        session = self._sessions.get(session_id)
        if not session:
            return {"error": {"code": "not_found", "message": "Session not found"}}
        return {"sessionId": session_id, "cwd": session.cwd}

    async def run(self) -> None:
        logger.info("ACP server starting (stdio NDJSON)")
        async for msg in self._transport.messages():
            await self.handle_message(msg)
        logger.info("ACP server shutting down (stdin closed)")

    def _send_result(self, msg_id, result: Any) -> None:
        self._transport.write({"jsonrpc": "2.0", "id": msg_id, "result": result})

    def _send_error(self, msg_id, code: int, message: str) -> None:
        self._transport.write({"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}})
