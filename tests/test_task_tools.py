"""Tests for src/tools/task_tools.py — _parse_plan_json, _parse_relative_time, task_manage."""

import json
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import pytest

from src.tools.task_tools import (
    _parse_plan_json,
    _parse_relative_time,
    set_task_context,
)


# ── _parse_plan_json ───────────────────────────────────────


class TestParsePlanJson:
    def test_valid_json(self):
        raw = '{"steps": ["step1", "step2"], "checkpoints": []}'
        result = _parse_plan_json(raw)
        assert result is not None
        assert result["steps"] == ["step1", "step2"]

    def test_json_in_code_block(self):
        raw = '```json\n{"steps": ["a"]}\n```'
        result = _parse_plan_json(raw)
        assert result is not None
        assert result["steps"] == ["a"]

    def test_json_in_plain_code_block(self):
        raw = '```\n{"steps": ["b"]}\n```'
        result = _parse_plan_json(raw)
        assert result is not None
        assert result["steps"] == ["b"]

    def test_invalid_json_extracts_braces(self):
        raw = 'Some text {"steps": ["x"]} more text'
        result = _parse_plan_json(raw)
        assert result is not None
        assert result["steps"] == ["x"]

    def test_completely_invalid(self):
        result = _parse_plan_json("not json at all")
        assert result is None

    def test_empty_string(self):
        result = _parse_plan_json("")
        assert result is None

    def test_whitespace_only(self):
        result = _parse_plan_json("   ")
        assert result is None

    def test_nested_json(self):
        raw = '{"steps": [{"description": "do thing"}], "checkpoints": [{"at": "30分钟后"}]}'
        result = _parse_plan_json(raw)
        assert result is not None
        assert len(result["steps"]) == 1


# ── _parse_relative_time ──────────────────────────────────


class TestParseRelativeTime:
    def test_iso_format_passthrough(self):
        iso = "2026-06-07 12:00:00"
        result = _parse_relative_time(iso)
        assert result == iso

    def test_minutes_chinese(self):
        result = _parse_relative_time("30分钟")
        assert result is not None
        # Should be ~30 min from now
        dt = datetime.fromisoformat(result)
        now = datetime.now()
        delta = dt - now
        assert 29 * 60 <= delta.total_seconds() <= 31 * 60

    def test_hours_chinese(self):
        result = _parse_relative_time("2小时")
        assert result is not None
        dt = datetime.fromisoformat(result)
        now = datetime.now()
        delta = dt - now
        assert 1.9 * 3600 <= delta.total_seconds() <= 2.1 * 3600

    def test_days_chinese(self):
        result = _parse_relative_time("3天")
        assert result is not None
        dt = datetime.fromisoformat(result)
        now = datetime.now()
        delta = dt - now
        assert 2.9 * 86400 <= delta.total_seconds() <= 3.1 * 86400

    def test_minutes_english(self):
        result = _parse_relative_time("30m")
        assert result is not None

    def test_hours_english(self):
        result = _parse_relative_time("2h")
        assert result is not None

    def test_plain_number_as_minutes(self):
        result = _parse_relative_time("5")
        assert result is not None
        dt = datetime.fromisoformat(result)
        now = datetime.now()
        delta = dt - now
        assert 4 * 60 <= delta.total_seconds() <= 6 * 60

    def test_invalid_input(self):
        result = _parse_relative_time("invalid text")
        assert result is None

    def test_empty_string(self):
        result = _parse_relative_time("")
        assert result is None


# ── set_task_context ───────────────────────────────────────


class TestSetTaskContext:
    def test_sets_values(self):
        from src.tools.task_tools import _current_sender_id, _current_thread_id
        from src.tools.chat_tools import _current_chat_id

        set_task_context(chat_id="chat1", sender_id="sender1", thread_id="thread1")
        assert _current_chat_id.get("") == "chat1"
        assert _current_sender_id.get("") == "sender1"
        assert _current_thread_id.get("") == "thread1"

        # Reset
        set_task_context(chat_id="", sender_id="", thread_id="")

    def test_empty_values_no_overwrite(self):
        from src.tools.task_tools import _current_sender_id

        _current_sender_id.set("keep_me")
        set_task_context()  # all empty — should not overwrite
        assert _current_sender_id.get("") == "keep_me"


# ── task_manage edge cases ─────────────────────────────────


class TestTaskManage:
    @pytest.mark.asyncio
    async def test_unknown_action(self):
        from src.tools.task_tools import task_manage

        result = await task_manage(action="invalid")
        parsed = json.loads(result)
        assert "error" in parsed
        assert "Unknown action" in parsed["error"]

    @pytest.mark.asyncio
    async def test_plan_missing_goal(self):
        from src.tools.task_tools import task_manage

        result = await task_manage(action="plan", plan_json='{"steps": ["a"]}')
        parsed = json.loads(result)
        assert "error" in parsed
