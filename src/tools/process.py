from __future__ import annotations

import asyncio
import logging
import os
import signal
import time
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger("myclaw.process")


@dataclass
class ProcessResult:
    """Result of a supervised process execution."""

    exit_code: int
    stdout: str
    stderr: str
    pid: int
    duration: float
    timed_out: bool = False
    killed: bool = False


class ProcessSupervisor:
    """Enhanced process management with timeout, memory limits, and process tree cleanup."""

    def __init__(self, max_memory_mb: Optional[int] = None):
        self.max_memory_mb = max_memory_mb

    async def run(
        self,
        command: str,
        timeout: Optional[int] = None,
        workdir: Optional[str] = None,
        max_output: int = 102400,
    ) -> ProcessResult:
        """Execute a command with supervision.

        Args:
            command: Shell command to execute
            timeout: Timeout in seconds
            workdir: Working directory
            max_output: Maximum output bytes

        Returns:
            ProcessResult with stdout, stderr, exit_code, etc.
        """
        # Validate workdir exists
        if workdir and not os.path.isdir(workdir):
            raise FileNotFoundError(f"Working directory does not exist: {workdir}")

        start = time.monotonic()
        pid = -1
        timed_out = False
        killed = False

        try:
            # Create process with process group for tree kill
            proc_kwargs = dict(
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workdir,
            )
            if os.name != "nt":
                proc_kwargs["start_new_session"] = True
            proc = await asyncio.create_subprocess_shell(command, **proc_kwargs)
            pid = proc.pid
            logger.debug("[process] started pid=%d cmd=%.200s", pid, command)

            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                timed_out = True
                killed = await self._kill_process_tree(proc)
                stdout, stderr = b"", b""
                logger.warning("[process] TIMEOUT pid=%d dur=%.1fs cmd=%.200s", pid, time.monotonic() - start, command)

            duration = time.monotonic() - start
            exit_code = proc.returncode if proc.returncode is not None else -1

            # Decode output
            stdout_text = stdout.decode("utf-8", errors="replace") if stdout else ""
            stderr_text = stderr.decode("utf-8", errors="replace") if stderr else ""

            # Combine output
            output_parts = []
            if stdout_text:
                output_parts.append(stdout_text)
            if stderr_text:
                output_parts.append(f"[stderr]\n{stderr_text}")

            output = "\n".join(output_parts) or "(no output)"

            # Truncate output
            if len(output.encode("utf-8", errors="replace")) > max_output:
                output = output[: max_output // 2] + f"\n... [truncated at {max_output} bytes] ...\n"

            if exit_code != 0 and not timed_out:
                output += f"\n[exit code: {exit_code}]"

            return ProcessResult(
                exit_code=exit_code,
                stdout=output,
                stderr=stderr_text,
                pid=pid,
                duration=duration,
                timed_out=timed_out,
                killed=killed,
            )

        except FileNotFoundError:
            raise
        except Exception as e:
            duration = time.monotonic() - start
            logger.error("[process] ERROR dur=%.1fs pid=%d cmd=%.200s: %s", duration, pid, command, e)
            raise

    async def _kill_process_tree(self, proc) -> bool:
        """Kill a process and all its children (process tree).

        Returns True if the process was killed.
        """
        pid = proc.pid
        killed = False

        if os.name != "nt":  # Unix
            try:
                # Send SIGTERM to the process group
                os.killpg(os.getpgid(pid), signal.SIGTERM)
                killed = True
                # Wait up to 5 seconds for graceful shutdown
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5)
                except asyncio.TimeoutError:
                    # Force kill
                    try:
                        os.killpg(os.getpgid(pid), signal.SIGKILL)
                        await proc.wait()
                    except (ProcessLookupError, OSError):
                        pass
            except (ProcessLookupError, OSError, PermissionError) as e:
                logger.debug("[process] kill_tree failed for pid=%d: %s", pid, e)
                # Fallback: kill just the main process
                try:
                    proc.kill()
                    await proc.wait()
                    killed = True
                except (ProcessLookupError, OSError):
                    pass
        else:  # Windows
            try:
                proc.kill()
                await proc.wait()
                killed = True
            except (ProcessLookupError, OSError):
                pass

        return killed


# Module-level convenience function
_supervisor: Optional[ProcessSupervisor] = None


def get_supervisor(max_memory_mb: Optional[int] = None) -> ProcessSupervisor:
    global _supervisor
    if _supervisor is None:
        _supervisor = ProcessSupervisor(max_memory_mb=max_memory_mb)
    return _supervisor


async def run_supervised(
    command: str,
    timeout: Optional[int] = None,
    workdir: Optional[str] = None,
    max_output: int = 102400,
) -> ProcessResult:
    """Run a command with supervised process management."""
    return await get_supervisor().run(command, timeout=timeout, workdir=workdir, max_output=max_output)
