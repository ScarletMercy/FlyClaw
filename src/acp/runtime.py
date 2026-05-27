from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import AsyncIterator

from src.acp.session import AcpSessionManager
from src.agent.loop import interruptible

logger = logging.getLogger("flyclaw.acp.runtime")


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
        self._state_store = None
        self._agent_loop = None
        self._cancel_events: dict[str, asyncio.Event] = {}

    def _ensure_store(self):
        from src.config import load_config
        from src.agent.state import StateStore

        if self._state_store is None:
            config = load_config()
            self._state_store = StateStore(config.checkpointer.path)
        return self._state_store

    def _ensure_loop(self):
        if self._agent_loop is None:
            self._agent_loop = _build_agent_loop(self._ensure_store())
        return self._agent_loop

    async def run_turn(
        self,
        session_id: str,
        prompt: str,
        agent_id: str = "default",
        cwd: str = "",
    ) -> AsyncIterator[AcpRuntimeEvent]:
        loop = self._ensure_loop()
        from src.agent.state import AgentState

        session = self._sessions.get(session_id)
        existing_messages: list[dict] = []
        thread_id = f"acp:{session_id}"

        if session:
            tid = session.thread_id or thread_id
            if session.thread_id:
                try:
                    store = self._ensure_store()
                    existing = await store.aload(tid)
                    if existing:
                        existing_messages = existing.messages
                except Exception:
                    pass
            thread_id = tid

        state = AgentState(
            messages=existing_messages + [{"role": "user", "content": prompt}],
            system_prompt="",
        )

        cancel_ev = asyncio.Event()
        self._cancel_events[session_id] = cancel_ev
        try:
            result = await interruptible(cancel_ev, loop.run(state, thread_id))
            if result is None:
                yield AcpRuntimeEvent(type="cancelled")
                yield AcpRuntimeEvent(type="done", stop_reason="cancelled")
                return
            for msg in result.messages:
                if msg.get("role") == "assistant" and msg.get("content"):
                    yield AcpRuntimeEvent(type="text_delta", text=msg["content"])
            yield AcpRuntimeEvent(type="done", stop_reason="end_turn")
        except Exception as e:
            logger.error("ACP run_turn failed: %s", e, exc_info=True)
            yield AcpRuntimeEvent(type="error", text=str(e))
            yield AcpRuntimeEvent(type="done", stop_reason="end_turn")
        finally:
            self._cancel_events.pop(session_id, None)

    async def cancel(self, session_id: str) -> None:
        ev = self._cancel_events.get(session_id)
        if ev:
            ev.set()

    async def close(self, session_id: str) -> None:
        self._sessions.close(session_id)


def _build_agent_loop(state_store):
    from src.agent.loop import AgentLoop
    from src.agent.client import create_chain
    from src.config import load_config
    from src.tools.registry import get_tool_registry

    config = load_config()
    client = create_chain(config)
    tools = list(get_tool_registry().collect())

    return AgentLoop(
        client=client,
        tools=tools,
        state_store=state_store,
        config=config,
    )
