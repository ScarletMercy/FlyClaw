from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import AsyncIterator

from src.acp.session import AcpSessionManager

logger = logging.getLogger("myclaw.acp.runtime")


@dataclass
class AcpRuntimeEvent:
    type: str
    text: str = ""
    tool_call_id: str = ""
    status: str = ""
    stop_reason: str = ""
    used: int = 0
    size: int = 0


class AgentLoopRuntime:
    def __init__(self, session_manager: AcpSessionManager | None = None):
        self._sessions = session_manager or AcpSessionManager()

    async def run_turn(
        self,
        session_id: str,
        prompt: str,
        agent_id: str = "default",
        cwd: str = "",
    ) -> AsyncIterator[AcpRuntimeEvent]:
        loop = _build_agent_loop(agent_id, cwd)
        from src.agent.state import AgentState

        session = self._sessions.get(session_id)
        existing_messages: list[dict] = []
        if session and session.thread_id:
            try:
                from src.agent.state import get_state_store
                store = get_state_store()
                existing = await store.aload(session.thread_id)
                if existing:
                    existing_messages = existing.messages
            except Exception:
                pass

        state = AgentState(
            messages=existing_messages + [{"role": "user", "content": prompt}],
            system_prompt="",
        )

        thread_id = session.thread_id if session else f"acp:{session_id}"

        try:
            result = await loop.run(state, thread_id)
            for msg in result.messages:
                if msg.get("role") == "assistant" and msg.get("content"):
                    yield AcpRuntimeEvent(type="text_delta", text=msg["content"])
            yield AcpRuntimeEvent(type="done", stop_reason="end_turn")
        except Exception as e:
            logger.error("ACP run_turn failed: %s", e, exc_info=True)
            yield AcpRuntimeEvent(type="error", text=str(e))
            yield AcpRuntimeEvent(type="done", stop_reason="end_turn")

    async def cancel(self, session_id: str) -> None:
        pass

    async def close(self, session_id: str) -> None:
        self._sessions.close(session_id)


def _build_agent_loop(agent_id: str, cwd: str):
    from src.agent.loop import AgentLoop
    from src.agent.client import create_chain
    from src.agent.state import MemoryStateStore
    from src.config import load_config
    from src.tools.registry import get_tool_registry

    config = load_config()
    client = create_chain(config)
    tools = list(get_tool_registry().collect())

    return AgentLoop(
        client=client,
        tools=tools,
        state_store=MemoryStateStore(),
        config=config,
    )
