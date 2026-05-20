"""Tool Loop Guardrails — detect and break tool loops.

Three anomaly patterns are tracked:
1. **Repeat failure**  — same tool + same args fails repeatedly
2. **Tool storm**      — same tool keeps failing with different args
3. **Idempotent stall** — read-only tool returns identical results repeatedly

Thresholds:
  Pattern              | Warn | Block
  ---------------------|------|------
  Repeat failure       |   2  |   5
  Tool storm           |   3  |   8
  Idempotent stall     |   2  |   5
"""

from __future__ import annotations

import hashlib
import logging
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger("myclaw.agent.guardrails")

_IDEMPOTENT_TOOLS: frozenset[str] = frozenset({
    "read_file", "list_dir", "grep_files", "glob_files",
    "web_search", "web_fetch", "session_search",
    "memory_get", "memory_list", "memory_search",
    "qq_list_guilds", "qq_list_channels", "qq_list_members",
    "qq_get_member", "cron_list",
})


@dataclass
class _ToolAttempt:
    tool_name: str
    args_sig: str
    success: bool
    result_sig: str


@dataclass
class GuardrailResult:
    blocked: bool
    reason: str
    synthetic_result: str


def _args_signature(args: dict) -> str:
    import json
    raw = json.dumps(args, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _result_signature(result: str) -> str:
    return hashlib.sha256(result.encode()).hexdigest()[:16]


@dataclass
class ToolLoopGuardrails:
    """Per-thread guardrail tracker."""

    repeat_fail_warn: int = 2
    repeat_fail_block: int = 5
    storm_warn: int = 3
    storm_block: int = 8
    stall_warn: int = 2
    stall_block: int = 5

    _history: deque[_ToolAttempt] = field(default_factory=lambda: deque(maxlen=64), repr=False)

    def record(self, tool_name: str, args: dict, success: bool, result: str = "") -> None:
        self._history.append(_ToolAttempt(
            tool_name=tool_name,
            args_sig=_args_signature(args),
            success=success,
            result_sig=_result_signature(result) if result else "",
        ))

    def check(self, tool_name: str, args: dict) -> GuardrailResult | None:
        """Check whether to allow this tool call.

        Returns ``None`` if the call is allowed, or a ``GuardrailResult``
        with ``blocked=True`` and a synthetic result to return instead.
        """
        if not self._history:
            return None

        args_sig = _args_signature(args)

        # --- Pattern 1: Repeat failure (same tool + same args, all failing) ---
        repeat_fails = 0
        for a in reversed(self._history):
            if a.tool_name != tool_name or a.args_sig != args_sig:
                break
            if not a.success:
                repeat_fails += 1
            else:
                break

        if repeat_fails >= self.repeat_fail_block:
            reason = f"Tool '{tool_name}' repeated failure (same args, {repeat_fails}x)"
            logger.warning("BLOCKED: %s", reason)
            return GuardrailResult(
                blocked=True,
                reason=reason,
                synthetic_result=f"[blocked] {reason}. Stop retrying this action and try a different approach.",
            )
        if repeat_fails >= self.repeat_fail_warn:
            logger.warning("WARN: Tool '%s' repeated failure (%dx same args)", tool_name, repeat_fails)

        # --- Pattern 2: Tool storm (same tool, consecutive failures, any args) ---
        storm_fails = 0
        for a in reversed(self._history):
            if a.tool_name != tool_name:
                break
            if not a.success:
                storm_fails += 1
            else:
                break

        if storm_fails >= self.storm_block:
            reason = f"Tool '{tool_name}' failure storm ({storm_fails}x consecutive failures)"
            logger.warning("BLOCKED: %s", reason)
            return GuardrailResult(
                blocked=True,
                reason=reason,
                synthetic_result=f"[blocked] {reason}. This tool is not working. Stop using it and summarize what you know.",
            )
        if storm_fails >= self.storm_warn:
            logger.warning("WARN: Tool '%s' storm (%dx failures)", tool_name, storm_fails)

        # --- Pattern 3: Idempotent stall (read-only tool, same result) ---
        if tool_name in _IDEMPOTENT_TOOLS:
            identical = 0
            for a in reversed(self._history):
                if a.tool_name != tool_name:
                    break
                if a.success and a.result_sig and a.result_sig == _result_signature(""):
                    break
                if a.success and a.result_sig:
                    identical += 1
                else:
                    break

            if identical >= self.stall_block:
                reason = f"Tool '{tool_name}' idempotent stall ({identical}x identical results)"
                logger.warning("BLOCKED: %s", reason)
                return GuardrailResult(
                    blocked=True,
                    reason=reason,
                    synthetic_result=f"[blocked] {reason}. You already have this information. Stop repeating this query.",
                )
            if identical >= self.stall_warn:
                logger.warning("WARN: Tool '%s' stall (%dx identical)", tool_name, identical)

        return None

    def reset(self) -> None:
        self._history.clear()
