"""Tests for SkillCurator timestamp handling — guards the aware/naive fix.

`days_since_last_review` and `_is_stale` subtract a parsed `last_review`/`last_used`
from `datetime.now().astimezone()`. Once those stored values became timezone-aware
(+08:00), a naive `datetime.now()` would raise TypeError. These tests feed all three
historical formats (old naive, old UTC `+00:00`, new local `+08:00`) and assert no
TypeError plus correct staleness classification — a behavior invariant, not a wording check.
"""

from datetime import datetime, timedelta, timezone

from src.skills.curator import SkillCurator


class TestCuratorTimestampHandling:
    def _curator(self, tmp_path):
        return SkillCurator(skills_dir=tmp_path, stale_after_days=30, archive_after_days=90)

    def test_days_since_last_review_naive(self, tmp_path):
        c = self._curator(tmp_path)
        c.state["last_review"] = (datetime.now() - timedelta(days=10)).isoformat()
        days = c.days_since_last_review()
        assert isinstance(days, int) and 9 <= days <= 11

    def test_days_since_last_review_utc_aware(self, tmp_path):
        c = self._curator(tmp_path)
        c.state["last_review"] = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        days = c.days_since_last_review()
        assert isinstance(days, int) and 9 <= days <= 11

    def test_days_since_last_review_local_aware(self, tmp_path):
        c = self._curator(tmp_path)
        c.state["last_review"] = (datetime.now().astimezone() - timedelta(days=10)).isoformat()
        days = c.days_since_last_review()
        assert isinstance(days, int) and 9 <= days <= 11

    def test_is_stale_naive_old_and_recent(self, tmp_path):
        c = self._curator(tmp_path)
        old = (datetime.now() - timedelta(days=40)).isoformat()
        recent = (datetime.now() - timedelta(days=5)).isoformat()
        assert c._is_stale(old, days=30) is True
        assert c._is_stale(recent, days=30) is False

    def test_is_stale_utc_aware_old(self, tmp_path):
        c = self._curator(tmp_path)
        old = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
        assert c._is_stale(old, days=30) is True
