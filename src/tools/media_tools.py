from __future__ import annotations

import logging
from contextvars import ContextVar

logger = logging.getLogger("flyclaw.media_tools")

_current_channel: ContextVar[str] = ContextVar("_current_channel", default="")


def set_current_channel(channel: str):
    _current_channel.set(channel)
