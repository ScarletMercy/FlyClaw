"""Unit tests for memory_consolidation.py pure functions."""

from datetime import datetime, timezone

import pytest

from src.services.memory_consolidation import _extract_json, _friendly_age

UTC = timezone.utc


def _dt(iso: str) -> datetime:
    return datetime.fromisoformat(iso).replace(tzinfo=UTC)


# ─── _friendly_age ───────────────────────────────────────────────────────────


class TestFriendlyAge:
    def test_today(self):
        now = datetime(2026, 6, 11, 12, 0, tzinfo=UTC)
        assert _friendly_age("2026-06-11T10:00:00+00:00", now) == "今天"

    def test_today_same_hour(self):
        now = datetime(2026, 6, 11, 12, 0, tzinfo=UTC)
        assert _friendly_age("2026-06-11T12:00:00+00:00", now) == "今天"

    def test_future_returns_today(self):
        now = datetime(2026, 6, 11, 12, 0, tzinfo=UTC)
        assert _friendly_age("2026-06-12T12:00:00+00:00", now) == "今天"

    def test_1_day(self):
        now = datetime(2026, 6, 12, 12, 0, tzinfo=UTC)
        assert _friendly_age("2026-06-11T10:00:00+00:00", now) == "1天前"

    def test_2_days(self):
        now = datetime(2026, 6, 13, 12, 0, tzinfo=UTC)
        assert _friendly_age("2026-06-11T10:00:00+00:00", now) == "2天前"

    def test_15_days(self):
        now = datetime(2026, 6, 26, 12, 0, tzinfo=UTC)
        assert _friendly_age("2026-06-11T10:00:00+00:00", now) == "15天前"

    def test_29_days(self):
        now = datetime(2026, 7, 10, 12, 0, tzinfo=UTC)
        assert _friendly_age("2026-06-11T10:00:00+00:00", now) == "29天前"

    def test_30_days_is_1_month(self):
        now = datetime(2026, 7, 11, 12, 0, tzinfo=UTC)
        assert _friendly_age("2026-06-11T10:00:00+00:00", now) == "约1个月前"

    def test_90_days_is_3_months(self):
        now = datetime(2026, 9, 9, 12, 0, tzinfo=UTC)
        assert _friendly_age("2026-06-11T10:00:00+00:00", now) == "约3个月前"

    def test_329_days_is_11_months(self):
        now = datetime(2027, 5, 6, 12, 0, tzinfo=UTC)
        assert _friendly_age("2026-06-11T10:00:00+00:00", now) == "约11个月前"

    def test_330_days_is_1_year(self):
        now = datetime(2027, 5, 7, 12, 0, tzinfo=UTC)
        assert _friendly_age("2026-06-11T10:00:00+00:00", now) == "约1年前"

    def test_400_days_is_1_year(self):
        now = datetime(2027, 7, 16, 12, 0, tzinfo=UTC)
        assert _friendly_age("2026-06-11T10:00:00+00:00", now) == "约1年前"

    def test_730_days_is_2_years(self):
        now = datetime(2028, 6, 10, 12, 0, tzinfo=UTC)
        assert _friendly_age("2026-06-11T10:00:00+00:00", now) == "约2年前"

    def test_none(self):
        now = datetime(2026, 6, 11, 12, 0, tzinfo=UTC)
        assert _friendly_age(None, now) == "未知"

    def test_empty(self):
        now = datetime(2026, 6, 11, 12, 0, tzinfo=UTC)
        assert _friendly_age("", now) == "未知"

    def test_invalid_format(self):
        now = datetime(2026, 6, 11, 12, 0, tzinfo=UTC)
        assert _friendly_age("not-a-date", now) == "未知"

    def test_naive_datetime_treated_as_utc(self):
        now = datetime(2026, 6, 12, 12, 0, tzinfo=UTC)
        assert _friendly_age("2026-06-11T10:00:00", now) == "1天前"

    def test_non_utc_timezone(self):
        now = datetime(2026, 6, 12, 12, 0, tzinfo=UTC)
        # +09:00 = 2026-06-11T03:00:00 UTC
        assert _friendly_age("2026-06-11T12:00:00+09:00", now) == "1天前"


# ─── _extract_json ────────────────────────────────────────────────────────────


class TestExtractJson:
    def test_bare_json(self):
        assert _extract_json('{"merge": [], "delete": [], "keep": ["k1"]}') == {
            "merge": [],
            "delete": [],
            "keep": ["k1"],
        }

    def test_json_with_whitespace(self):
        assert _extract_json('  \n{"merge": []}\n  ') == {"merge": []}

    def test_code_fence_json(self):
        text = '```json\n{"merge": [], "delete": []}\n```'
        assert _extract_json(text) == {"merge": [], "delete": []}

    def test_code_fence_no_lang(self):
        text = '```\n{"merge": []}\n```'
        assert _extract_json(text) == {"merge": []}

    def test_prose_before_code_fence(self):
        text = 'Here is the plan:\n```json\n{"merge": [], "keep": []}\n```'
        assert _extract_json(text) == {"merge": [], "keep": []}

    def test_brace_extraction(self):
        text = 'Some text before {"merge": []} some text after'
        assert _extract_json(text) == {"merge": []}

    def test_multiline_json(self):
        text = '```json\n{\n  "merge": [\n    {"from_keys": ["a"], "to_content": "b"}\n  ]\n}\n```'
        result = _extract_json(text)
        assert result is not None
        assert len(result["merge"]) == 1

    def test_no_json(self):
        assert _extract_json("no json here") is None

    def test_empty_string(self):
        assert _extract_json("") is None

    def test_invalid_json_in_fence(self):
        text = "```json\nnot json\n```"
        assert _extract_json(text) is None

    def test_nested_braces_picks_outermost(self):
        text = 'text {"a": {"b": 1}} end'
        assert _extract_json(text) == {"a": {"b": 1}}
