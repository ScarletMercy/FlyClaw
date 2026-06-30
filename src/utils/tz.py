"""Timezone utilities."""

from __future__ import annotations

import logging
import zoneinfo
from datetime import datetime, tzinfo

_log = logging.getLogger("flyclaw.utils.tz")

_LOCAL_SENTINEL = "local"


def get_tz(name: str | None) -> tzinfo:
    """Resolve a timezone name to a :class:`tzinfo`.

    The sentinel ``"local"`` (and ``None``) resolves to the system's local
    timezone via ``datetime.now().astimezone().tzinfo`` — a ``zoneinfo.ZoneInfo``
    on Linux and a fixed-offset ``datetime.timezone`` on Windows, both usable
    with ``datetime.now(tz=...)`` and APScheduler's ``timezone=``. Any other
    value is parsed as an IANA name, falling back to UTC on invalid input.
    """
    if name == _LOCAL_SENTINEL or name is None:
        tz = datetime.now().astimezone().tzinfo
        assert tz is not None  # astimezone() always yields an aware datetime
        return tz
    try:
        return zoneinfo.ZoneInfo(name)
    except (KeyError, ValueError):
        _log.warning("Invalid timezone '%s', falling back to UTC", name)
        return zoneinfo.ZoneInfo("UTC")


def now_iso(tz_name: str | None = None) -> str:
    """Current time as an ISO 8601 string with the timezone offset, to the second.

    ``tz_name`` of ``None`` or ``"local"`` uses the system's local timezone
    (e.g. ``2026-06-28T11:00:00+08:00``); any other value is resolved via
    :func:`get_tz` (falls back to UTC on invalid input). Microseconds are stripped.
    """
    if tz_name is None or tz_name == _LOCAL_SENTINEL:
        dt = datetime.now().astimezone()
    else:
        dt = datetime.now(get_tz(tz_name))
    return dt.replace(microsecond=0).isoformat()
