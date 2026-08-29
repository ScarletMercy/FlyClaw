"""Tests for src/tools/cron_tools.py — _clean_job_id, validation logic."""

from src.tools.cron_tools import _clean_job_id


class TestCleanJobId:
    def test_plain_id(self):
        assert _clean_job_id("abc123") == "abc123"

    def test_strips_brackets(self):
        assert _clean_job_id("[abc123]") == "abc123"

    def test_strips_backticks(self):
        assert _clean_job_id("`abc123`") == "abc123"

    def test_strips_parens(self):
        assert _clean_job_id("(abc123)") == "abc123"

    def test_strips_mixed(self):
        assert _clean_job_id("  [abc]  ") == "abc"

    def test_strips_all(self):
        assert _clean_job_id("[]()`abc`") == "abc"

    def test_empty_after_strip(self):
        assert _clean_job_id("[]()``") == ""
