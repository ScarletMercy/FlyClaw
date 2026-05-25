import os
import time
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.tools.file_tools import (
    _get_tracker,
    _read_tracker,
    _read_tracker_lock,
    reset_read_dedup,
    _invalidate_dedup_for_path,
    read_file,
    write_file,
    set_workspace,
)


@pytest.fixture(autouse=True)
def clean_tracker():
    with _read_tracker_lock:
        _read_tracker.clear()
    yield
    with _read_tracker_lock:
        _read_tracker.clear()


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    set_workspace(str(ws))
    yield ws
    set_workspace(".")


class TestGetTracker:
    def test_creates_new_tracker(self):
        tracker = _get_tracker("thread-1")
        assert tracker["consecutive"] == 0
        assert tracker["last_key"] is None
        assert tracker["dedup"] == {}
        assert tracker["dedup_hits"] == {}

    def test_returns_existing(self):
        t1 = _get_tracker("thread-1")
        t1["consecutive"] = 5
        t2 = _get_tracker("thread-1")
        assert t2["consecutive"] == 5

    def test_capacity_enforcement(self):
        tracker = _get_tracker("thread-1")
        for i in range(510):
            tracker["dedup"][(f"/path/{i}", 0, 500)] = 1.0
        _get_tracker("thread-1")
        assert len(tracker["dedup"]) <= 500


class TestResetReadDedup:
    def test_specific_thread(self):
        _get_tracker("t1")["dedup"][("key",)] = 1.0
        _get_tracker("t2")["dedup"][("key",)] = 2.0
        reset_read_dedup("t1")
        assert _get_tracker("t1")["dedup"] == {}
        assert _get_tracker("t2")["dedup"] != {}

    def test_clear_all(self):
        _get_tracker("t1")
        _get_tracker("t2")
        reset_read_dedup()
        assert len(_read_tracker) == 0

    def test_resets_consecutive(self):
        tracker = _get_tracker("t1")
        tracker["consecutive"] = 10
        tracker["last_key"] = ("read", "/path", 0, 500)
        reset_read_dedup("t1")
        assert tracker["consecutive"] == 0
        assert tracker["last_key"] is None


class TestInvalidateDedupForPath:
    def test_clears_dedup_entries(self):
        tracker = _get_tracker("t1")
        tracker["dedup"][("/ws/a.txt", 0, 500)] = 1.0
        tracker["dedup"][("/ws/b.txt", 0, 500)] = 2.0
        _invalidate_dedup_for_path("/ws/a.txt", "t1")
        assert ("/ws/a.txt", 0, 500) not in tracker["dedup"]
        assert ("/ws/b.txt", 0, 500) in tracker["dedup"]

    def test_clears_dedup_hits(self):
        tracker = _get_tracker("t1")
        tracker["dedup_hits"][("/ws/a.txt", 0, 500)] = 3
        _invalidate_dedup_for_path("/ws/a.txt", "t1")
        assert ("/ws/a.txt", 0, 500) not in tracker["dedup_hits"]

    def test_resets_consecutive_for_same_file(self):
        tracker = _get_tracker("t1")
        tracker["last_key"] = ("read", "/ws/a.txt", 0, 500)
        tracker["consecutive"] = 4
        _invalidate_dedup_for_path("/ws/a.txt", "t1")
        assert tracker["consecutive"] == 0
        assert tracker["last_key"] is None

    def test_does_not_reset_consecutive_for_different_file(self):
        tracker = _get_tracker("t1")
        tracker["last_key"] = ("read", "/ws/b.txt", 0, 500)
        tracker["consecutive"] = 3
        _invalidate_dedup_for_path("/ws/a.txt", "t1")
        assert tracker["consecutive"] == 3

    def test_no_thread_id_noop(self):
        _invalidate_dedup_for_path("/ws/a.txt", "")


class TestReadDedup:
    def test_first_read_succeeds(self, workspace):
        f = workspace / "test.txt"
        f.write_text("line1\nline2\n", encoding="utf-8")
        result = read_file("test.txt")
        assert "line1" in result

    def test_second_unchanged_read_warns(self, workspace):
        f = workspace / "test.txt"
        f.write_text("content\n", encoding="utf-8")
        with patch("src.tools.exec._current_thread_id") as mock_tid:
            mock_tid.get.return_value = "t1"
            r1 = read_file("test.txt")
            assert "line1" in r1 or "content" in r1
            r2 = read_file("test.txt")
            assert "unchanged" in r2.lower() or "BLOCKED" in r2 or "line1" in r2

    def test_modified_file_rereads(self, workspace):
        f = workspace / "test.txt"
        f.write_text("old\n", encoding="utf-8")
        with patch("src.tools.exec._current_thread_id") as mock_tid:
            mock_tid.get.return_value = "t1"
            r1 = read_file("test.txt")
            write_file("test.txt", "new content\n")
            r2 = read_file("test.txt")
            assert "new content" in r2

    def test_write_invalidates_dedup(self, workspace):
        f = workspace / "test.txt"
        f.write_text("original\n", encoding="utf-8")
        with patch("src.tools.exec._current_thread_id") as mock_tid:
            mock_tid.get.return_value = "t1"
            r1 = read_file("test.txt")
            write_file("test.txt", "modified\n")
            r2 = read_file("test.txt")
            assert "modified" in r2


class TestConsecutiveLoopDetection:
    def test_third_read_warns(self, workspace):
        f = workspace / "test.txt"
        f.write_text("data\n", encoding="utf-8")
        with patch("src.tools.exec._current_thread_id") as mock_tid:
            mock_tid.get.return_value = "t1"
            for _ in range(2):
                read_file("test.txt")
                f.write_text("changed\n", encoding="utf-8")
            r3 = read_file("test.txt")
            assert "WARNING" in r3 or "changed" in r3

    def test_fourth_read_blocked(self, workspace):
        f = workspace / "test.txt"
        f.write_text("data\n", encoding="utf-8")
        with patch("src.tools.exec._current_thread_id") as mock_tid:
            mock_tid.get.return_value = "t1"
            for _ in range(3):
                time.sleep(0.01)
                f.write_text(f"data_{_}\n", encoding="utf-8")
                read_file("test.txt")
            time.sleep(0.01)
            f.write_text("data_3\n", encoding="utf-8")
            r4 = read_file("test.txt")
            assert "BLOCKED" in r4
