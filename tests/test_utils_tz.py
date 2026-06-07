"""Tests for src/utils/tz.py — timezone handling."""

import zoneinfo

from src.utils.tz import get_tz


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
