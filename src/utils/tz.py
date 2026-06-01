"""Timezone utilities."""

from __future__ import annotations

import logging
import zoneinfo

_log = logging.getLogger("flyclaw.utils.tz")


def get_tz(name: str) -> zoneinfo.ZoneInfo:
    """Convert a timezone name to ZoneInfo, falling back to UTC on invalid input."""
    try:
        return zoneinfo.ZoneInfo(name)
    except (KeyError, ValueError):
        _log.warning("Invalid timezone '%s', falling back to UTC", name)
        return zoneinfo.ZoneInfo("UTC")
