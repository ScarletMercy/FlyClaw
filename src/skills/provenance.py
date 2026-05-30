"""Provenance tracking for skill writes.

Distinguishes foreground (user-directed) writes from background
self-improvement review writes using a ContextVar.

Only background-review-originated skill creates are marked as
agent-created, making them eligible for curator lifecycle management.
"""
from __future__ import annotations

import contextvars

_write_origin: contextvars.ContextVar[str] = contextvars.ContextVar(
    "skill_write_origin", default="foreground"
)

BACKGROUND_REVIEW = "background_review"


def set_current_write_origin(origin: str) -> contextvars.Token[str]:
    """Bind write origin to current async context. Returns a reset token."""
    return _write_origin.set(origin)


def reset_current_write_origin(token: contextvars.Token[str]) -> None:
    """Restore prior write origin context."""
    _write_origin.reset(token)


def is_background_review() -> bool:
    """True if current write origin is from background self-improvement review."""
    return _write_origin.get() == BACKGROUND_REVIEW
