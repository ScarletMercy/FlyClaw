"""Background process management: spawn, poll, kill, wait, log.

Also provides ``kill_process_tree`` for cross-platform process-tree cleanup.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("flyclaw.process")


async def kill_process_tree(proc: asyncio.subprocess.Process) -> None:
    """Kill a process and all its descendants (cross-platform).

    Does NOT call ``await proc.wait()`` — the caller is responsible for
    reaping the process afterwards.
    """
    if proc.pid is None:
        return
    pid = proc.pid
    if os.name == "nt":
        try:
            t = await asyncio.create_subprocess_exec(
                "taskkill",
                "/T",
                "/F",
                "/PID",
                str(pid),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(t.wait(), timeout=10)
        except (FileNotFoundError, OSError, asyncio.TimeoutError):
            try:
                proc.kill()
            except ProcessLookupError:
                pass
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            try:
                proc.kill()
            except ProcessLookupError:
                pass


# ── Background process management ──


@dataclass
class BackgroundSession:
    """Tracks a single background process."""

    id: str
    command: str
    pid: int
    proc: asyncio.subprocess.Process
    started_at: float
    workdir: str
    exit_code: Optional[int] = None
    output_buffer: str = ""
    timed_out: bool = False
    killed: bool = False
    finished_at: Optional[float] = None
    _reader_task: Optional[asyncio.Task] = field(default=None, repr=False)

    @property
    def status(self) -> str:
        if self.exit_code is not None:
            return "exited"
        if self.proc.returncode is not None:
            return "exited"
        return "running"

    @property
    def elapsed(self) -> float:
        end = self.finished_at or time.monotonic()
        return round(end - self.started_at, 1)


class ProcessRegistry:
    """Thread-safe registry for background processes."""

    def __init__(self, max_output_chars: int = 200_000, max_sessions: int = 20):
        self._sessions: dict[str, BackgroundSession] = {}
        self._max_output = max_output_chars
        self._max_sessions = max_sessions
        self._finished_ttl = 1800  # 30 minutes

    async def spawn(self, command: str, workdir: str | None = None, env: dict | None = None) -> str:
        """Start a background process, return session ID."""
        session_id = uuid.uuid4().hex[:8]

        proc_kwargs: dict = dict(
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workdir,
        )
        if env:
            full_env = dict(os.environ)
            full_env.update(env)
            proc_kwargs["env"] = full_env
        if os.name != "nt":
            proc_kwargs["start_new_session"] = True

        proc = await asyncio.create_subprocess_shell(command, **proc_kwargs)

        session = BackgroundSession(
            id=session_id,
            command=command,
            pid=proc.pid,
            proc=proc,
            started_at=time.monotonic(),
            workdir=workdir or os.getcwd(),
        )

        # Start background reader
        session._reader_task = asyncio.create_task(self._read_output(session))

        self._sessions[session_id] = session
        self._evict()

        logger.info(
            "[bg-process] spawned session=%s pid=%d cmd=%.200s",
            session_id,
            proc.pid,
            command,
        )
        return session_id

    async def _read_output(self, session: BackgroundSession) -> None:
        """Background task: read stdout+stderr into rolling buffer."""
        proc = session.proc

        async def _read_stream(stream: asyncio.streams.StreamReader | None, prefix: str):
            if stream is None:
                return
            try:
                async for chunk in stream:
                    text = chunk.decode("utf-8", errors="replace")
                    if prefix:
                        for line in text.splitlines(keepends=True):
                            session.output_buffer += f"{prefix}{line}"
                    else:
                        session.output_buffer += text
                    # Rolling buffer: keep last N chars
                    if len(session.output_buffer) > self._max_output:
                        excess = len(session.output_buffer) - self._max_output
                        session.output_buffer = session.output_buffer[excess:]
            except Exception as e:
                logger.debug("[bg-process] reader error session=%s: %s", session.id, e)

        await asyncio.gather(
            _read_stream(proc.stdout, ""),
            _read_stream(proc.stderr, "[stderr] "),
            return_exceptions=True,
        )

        # Process exited
        await proc.wait()
        session.exit_code = proc.returncode
        session.finished_at = time.monotonic()
        logger.info(
            "[bg-process] exited session=%s pid=%d exit=%d",
            session.id,
            session.pid,
            session.exit_code,
        )

    async def poll(self, session_id: str) -> dict:
        """Check status and get recent output tail."""
        session = self._sessions.get(session_id)
        if not session:
            return {"status": "not_found", "error": f"Session {session_id} not found"}

        # Check if process is done
        if session.exit_code is None and session.proc.returncode is not None:
            session.exit_code = session.proc.returncode
            session.finished_at = time.monotonic()

        tail = session.output_buffer[-1000:] if session.output_buffer else ""

        return {
            "session_id": session.id,
            "status": session.status,
            "exit_code": session.exit_code,
            "pid": session.pid,
            "elapsed": session.elapsed,
            "output_tail": tail,
            "command": session.command[:200],
        }

    async def wait(self, session_id: str, timeout: int = 300) -> dict:
        """Block until process completes. Returns full result."""
        session = self._sessions.get(session_id)
        if not session:
            return {"status": "not_found", "error": f"Session {session_id} not found"}

        if session._reader_task and not session._reader_task.done():
            try:
                await asyncio.wait_for(session._reader_task, timeout=timeout)
            except asyncio.TimeoutError:
                # Kill the process if wait times out
                await self.kill(session_id)
                return {
                    "session_id": session.id,
                    "status": "timeout",
                    "exit_code": session.exit_code,
                    "pid": session.pid,
                    "elapsed": session.elapsed,
                    "output_tail": session.output_buffer[-4000:],
                    "command": session.command[:200],
                }

        return {
            "session_id": session.id,
            "status": session.status,
            "exit_code": session.exit_code,
            "pid": session.pid,
            "elapsed": session.elapsed,
            "output": session.output_buffer[-8000:],
            "command": session.command[:200],
        }

    async def kill(self, session_id: str) -> str:
        """Terminate a background process."""
        session = self._sessions.get(session_id)
        if not session:
            return f"Session {session_id} not found."
        if session.exit_code is not None:
            return f"Process already exited with code {session.exit_code}."

        try:
            await kill_process_tree(session.proc)
            await session.proc.wait()
            session.killed = True
            logger.info("[bg-process] killed session=%s pid=%d", session_id, session.pid)
            return f"Killed process {session_id} (pid {session.pid})."
        except Exception as e:
            logger.warning("[bg-process] kill failed session=%s: %s", session_id, e)
            return f"Failed to kill: {e}"

    async def log(self, session_id: str, offset: int = 0, limit: int = 200) -> str:
        """Read output with line-based pagination."""
        session = self._sessions.get(session_id)
        if not session:
            return f"Session {session_id} not found."

        lines = session.output_buffer.splitlines()
        total = len(lines)
        selected = lines[offset : offset + limit]

        header = (
            f"Session {session_id} ({session.status}, lines {offset + 1}-{min(offset + limit, total)} of {total})\n"
        )
        return header + "\n".join(selected)

    def list_sessions(self) -> list[dict]:
        """List all sessions."""
        now = time.monotonic()
        result = []
        for s in self._sessions.values():
            status = s.status
            elapsed = round(now - s.started_at, 1)
            result.append(
                {
                    "id": s.id,
                    "command": s.command[:100],
                    "pid": s.pid,
                    "status": status,
                    "exit_code": s.exit_code,
                    "elapsed": elapsed,
                    "started_at": s.started_at,
                }
            )
        return sorted(result, key=lambda x: x["started_at"], reverse=True)

    def _evict(self):
        """Remove old finished sessions and cancel their reader tasks."""
        now = time.monotonic()
        # Remove sessions past TTL
        expired = [
            sid for sid, s in self._sessions.items() if s.finished_at and (now - s.finished_at) > self._finished_ttl
        ]
        for sid in expired:
            s = self._sessions.pop(sid, None)
            if s and s._reader_task and not s._reader_task.done():
                s._reader_task.cancel()

        # If still over limit, remove oldest finished
        if len(self._sessions) > self._max_sessions:
            finished = sorted(
                [(sid, s) for sid, s in self._sessions.items() if s.finished_at],
                key=lambda x: x[1].finished_at or 0,
            )
            to_remove = len(self._sessions) - self._max_sessions
            for sid, s in finished[:to_remove]:
                self._sessions.pop(sid, None)
                if s._reader_task and not s._reader_task.done():
                    s._reader_task.cancel()


# ── Singleton ──

_registry: ProcessRegistry | None = None


def get_process_registry() -> ProcessRegistry:
    global _registry
    if _registry is None:
        _registry = ProcessRegistry()
    return _registry
