from __future__ import annotations

from .types import MediaResult, MediaCapability

__all__ = ["MediaResult", "MediaCapability", "MediaUnderstandingRunner"]


def __getattr__(name):
    if name == "MediaUnderstandingRunner":
        from .runner import MediaUnderstandingRunner

        return MediaUnderstandingRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
