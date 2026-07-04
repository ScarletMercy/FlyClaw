"""Tests for KV → archive migration."""

from __future__ import annotations

import time

import pytest

from src.services.memory_archive_migration import _compute_retention, _path_for_kv


class TestRetention:
    def test_keeps_recent_within_7d(self):
        now = time.time()
        rows = [{"key": "k1", "updated_ts": now - 100}]  # 100 秒前
        migrate = _compute_retention(rows, now=now, keep_n=20, keep_days=7)
        assert migrate == []

    def test_migrates_old_beyond_7d_and_not_top20(self):
        now = time.time()
        rows = [{"key": f"k{i}", "updated_ts": now - 8 * 86400} for i in range(25)]
        migrate = _compute_retention(rows, now=now, keep_n=20, keep_days=7)
        assert len(migrate) == 5  # 25 - 20 (8天全超7d，只靠 top20 保 20)

    def test_keeps_top20_even_if_old(self):
        now = time.time()
        rows = [{"key": f"k{i}", "updated_ts": now - 100 * 86400} for i in range(25)]
        migrate = _compute_retention(rows, now=now, keep_n=20, keep_days=7)
        assert len(migrate) == 5  # 全超 7d，但 top20 保留

    def test_keeps_union_age_or_rank(self):
        """age<=7d OR idx<20 并集保留。"""
        now = time.time()
        rows = [
            {"key": "new_but_21st", "updated_ts": now - 100},  # 新但排第 21
        ] + [{"key": f"old{i}", "updated_ts": now - 100 * 86400} for i in range(20)]
        # 排序后 new 排第 1（最新），old0-19 排 2-21
        migrate = _compute_retention(rows, now=now, keep_n=20, keep_days=7)
        keys = {m["key"] for m in migrate}
        assert "new_but_21st" not in keys


class TestPathForKv:
    def test_dm_path(self):
        assert _path_for_kv("k1", group_id="") == "kv:k1"

    def test_group_path(self):
        assert _path_for_kv("k1", group_id="g123", is_group=True) == "kv:g:g123:k1"

    def test_empty_group_id_keeps_group_prefix(self):
        """群 scope 下 group_id="" 时路径仍应带 kv:g: 前缀，不能降级为 DM 样式。

        回归 bug #5：群记忆 group_id="" 经 _path_for_kv 返回 "kv:{key}"（DM 样式），
        写入群归档库后 _list_past 群前缀过滤 "kv:g:{gid}:" 匹配不上；
        紧接着 forget 删 KV → 记忆两边都丢。
        """
        path = _path_for_kv("k1", group_id="", is_group=True)
        assert path.startswith("kv:g:"), (
            f"群 scope 空 group_id 归档路径降级为 DM 样式: {path!r}，群检索前缀过滤匹配不上 → 归档后丢失"
        )
