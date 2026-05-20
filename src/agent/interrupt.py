"""Interrupt and steer flags — hermes-agent pattern.

Interrupt: stops the agent loop immediately. Clears any pending steer.
Steer: injects user guidance into the last tool result without stopping.
       Multiple steers concatenate with newlines.
"""

from __future__ import annotations

import threading


class InterruptFlag:
    """Thread-safe interrupt + steer flag for a single agent loop session."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requested = False
        self._message: str | None = None
        self._pending_steer: str | None = None

    def interrupt(self, message: str | None = None) -> None:
        """Request interrupt. Clears any pending steer (interrupt supersedes steer)."""
        with self._lock:
            self._requested = True
            self._message = message
            self._pending_steer = None

    def steer(self, text: str) -> bool:
        """Queue steer text. Appends if already pending. Returns False if already interrupting."""
        if not text or not text.strip():
            return False
        with self._lock:
            if self._requested:
                return False
            cleaned = text.strip()
            if self._pending_steer:
                self._pending_steer = self._pending_steer + "\n" + cleaned
            else:
                self._pending_steer = cleaned
        return True

    def check(self) -> tuple[bool, str | None]:
        """Returns (is_interrupted, interrupt_message)."""
        with self._lock:
            return self._requested, self._message

    def drain_steer(self) -> str | None:
        """Return and clear pending steer text."""
        with self._lock:
            text = self._pending_steer
            self._pending_steer = None
            return text

    def clear(self) -> None:
        """Clear all flags."""
        with self._lock:
            self._requested = False
            self._message = None
            self._pending_steer = None
