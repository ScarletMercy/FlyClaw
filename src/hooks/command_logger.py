from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("myclaw.audit")


def log_tool_call(
    tool_name: str,
    args: dict[str, Any],
    sender_id: str = "",
    success: bool = True,
    error: str = "",
    duration: float = 0.0,
    chat_id: str = "",
) -> None:
    """Log a tool call for audit purposes.

    Args:
        tool_name: Name of the tool that was called
        args: Tool arguments (will be truncated for logging)
        sender_id: ID of the user who triggered the call
        success: Whether the call succeeded
        error: Error message if failed
        duration: Execution duration in seconds
        chat_id: Chat/conversation ID
    """
    args_summary = _summarize_args(args)

    status = "ok" if success else "err"
    parts = [f"tool={tool_name}", f"sender={sender_id}", f"args=\"{args_summary}\"", f"{status}"]
    if not success and error:
        parts.append(f"error=\"{error[:100]}\"")
    parts.append(f"dur={duration:.2f}s")

    if chat_id:
        parts.insert(1, f"chat={chat_id}")

    log_msg = " ".join(parts)

    if success:
        logger.info("[command-audit] %s", log_msg)
    else:
        logger.warning("[command-audit] %s", log_msg)


def _summarize_args(args: dict[str, Any], max_len: int = 200) -> str:
    """Create a safe summary of tool arguments for logging."""
    if not args:
        return ""

    sensitive_keys = {"api_key", "apikey", "secret", "token", "password", "authorization"}

    parts = []
    for key, value in args.items():
        if key.lower() in sensitive_keys:
            parts.append(f"{key}=***REDACTED***")
        elif isinstance(value, str) and len(value) > 100:
            parts.append(f"{key}={value[:100]}...")
        elif isinstance(value, (list, dict)) and len(str(value)) > 150:
            parts.append(f"{key}={str(value)[:150]}...")
        else:
            parts.append(f"{key}={value}")

    result = ", ".join(parts)
    return result[:max_len]
