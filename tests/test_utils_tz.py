"""Tests for src/utils/tz.py — timezone handling."""

import re
import zoneinfo
from datetime import datetime

import pytest

from src.utils.tz import get_tz, now_iso


class TestGetTz:
    def test_valid_timezone(self):
        result = get_tz("Asia/Shanghai")
        assert isinstance(result, zoneinfo.ZoneInfo)
        assert str(result) == "Asia/Shanghai"

    def test_utc(self):
        result = get_tz("UTC")
        assert isinstance(result, zoneinfo.ZoneInfo)
        assert str(result) == "UTC"

    def test_america_new_york(self):
        result = get_tz("America/New_York")
        assert isinstance(result, zoneinfo.ZoneInfo)

    def test_invalid_falls_back_to_utc(self):
        result = get_tz("Invalid/Timezone")
        assert isinstance(result, zoneinfo.ZoneInfo)
        assert str(result) == "UTC"

    def test_empty_string_falls_back(self):
        result = get_tz("")
        assert isinstance(result, zoneinfo.ZoneInfo)
        assert str(result) == "UTC"

    def test_none_handling(self):
        # get_tz expects str, so None would raise TypeError
        # But let's verify the fallback works for bad strings
        result = get_tz("not-a-tz")
        assert str(result) == "UTC"


class TestNowIso:
    _ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$")

    def test_default_aware_with_offset(self):
        s = now_iso()
        assert self._ISO_RE.match(s), f"unexpected format: {s!r}"
        dt = datetime.fromisoformat(s)
        assert dt.tzinfo is not None
        assert dt.utcoffset() is not None

    def test_default_strips_microseconds(self):
        s = now_iso()
        assert "." not in s, f"microseconds present: {s!r}"

    def test_default_round_trips(self):
        dt = datetime.fromisoformat(now_iso())
        assert isinstance(dt, datetime)

    def test_default_not_utc_when_local_not_utc(self):
        import time as _time

        if _time.localtime().tm_gmtoff == 0:
            pytest.skip("system timezone is UTC")
        assert "+00:00" not in now_iso()

    def test_explicit_tz_name(self):
        s = now_iso("America/New_York")
        assert self._ISO_RE.match(s), f"unexpected format: {s!r}"
        assert datetime.fromisoformat(s).tzinfo is not None

    def test_invalid_tz_falls_back_to_utc(self):
        s = now_iso("Invalid/Foo")
        assert self._ISO_RE.match(s), f"unexpected format: {s!r}"
        assert datetime.fromisoformat(s).tzinfo is not None
