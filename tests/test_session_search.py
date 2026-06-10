"""Tests for session search feature."""

import os
import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


def _create_checkpoints_db(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS sessions "
        "(thread_id TEXT PRIMARY KEY, messages TEXT, updated_at REAL, is_active INTEGER DEFAULT 0)"
    )
    conn.commit()
    conn.close()


def _insert_session(db_path: str, thread_id: str, updated_at: float) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO sessions (thread_id, messages, updated_at, is_active) VALUES (?, ?, ?, 0)",
        (thread_id, "test messages", updated_at),
    )
    conn.commit()
    conn.close()


def _count_sessions(db_path: str) -> int:
    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    conn.close()
    return count


class TestSessionSearchConfig:
    def test_default_enabled(self, tmp_path):
        from src.config import load_config

        cfg = load_config(tmp_path / "nonexistent.yaml")
        assert cfg.session_search.enabled is True
        assert cfg.session_search.index_path == str(Path("~/.flyclaw/data/session_index.db").expanduser().resolve())
        assert cfg.session_search.auto_sync is True
        assert cfg.session_search.max_results == 10
        assert cfg.session_search.tool_content_max_chars == 500

    def test_yaml_config(self, tmp_path):
        from src.config import load_config

        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
session_search:
  enabled: true
  index_path: "custom/index.db"
  max_results: 20
"""
        )
        cfg = load_config(config_file)
        assert cfg.session_search.enabled is True
        assert cfg.session_search.index_path == str(Path("custom/index.db").resolve())
        assert cfg.session_search.max_results == 20

    def test_parse_thread_id_legacy_p2p(self):
        from src.session_index.store import parse_thread_id

        result = parse_thread_id("qq:user:ou_abc123")
        assert result["channel"] == "qq"
        assert result["chat_type"] == "p2p"
        assert result["sender_id"] == "ou_abc123"

    def test_parse_thread_id_legacy_group(self):
        from src.session_index.store import parse_thread_id

        result = parse_thread_id("qq:group:groupXYZ")
        assert result["channel"] == "qq"
        assert result["chat_type"] == "group"
        assert result["sender_id"] == ""

    def test_parse_thread_id_multi_session(self):
        from src.session_index.store import parse_thread_id

        result = parse_thread_id("qq:s1:ABC123")
        assert result["channel"] == "qq"
        assert result["chat_type"] == "p2p"
        assert result["sender_id"] == "ABC123"

    def test_parse_thread_id_global(self):
        from src.session_index.store import parse_thread_id

        result = parse_thread_id("qq:global")
        assert result["channel"] == "qq"
        assert result["chat_type"] == "p2p"

    def test_parse_thread_id_ws(self):
        from src.session_index.store import parse_thread_id

        result = parse_thread_id("ws:user:xyz")
        assert result["channel"] == "ws"
        assert result["chat_type"] == "p2p"
        assert result["sender_id"] == "xyz"


class TestSessionIndexStore:
    async def test_init_creates_tables(self, tmp_path):
        from src.session_index.store import SessionIndexStore

        db_path = str(tmp_path / "index.db")
        store = await SessionIndexStore.create(db_path)
        cursor = await store._db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = await cursor.fetchall()
        table_names = {t[0] for t in tables}
        assert "sessions" in table_names
        assert "messages" in table_names
        assert "messages_fts" in table_names
        await store.close()

    async def test_upsert_and_get_session(self, tmp_path):
        from src.session_index.store import SessionIndexStore

        store = await SessionIndexStore.create(str(tmp_path / "index.db"))
        await store.upsert_session(
            thread_id="qq:user:ou_abc",
            channel="qq",
            sender_id="ou_abc",
            chat_id="c2c:ou_abc",
            chat_type="p2p",
        )
        session = await store.get_session("qq:user:ou_abc")
        assert session["channel"] == "qq"
        assert session["sender_id"] == "ou_abc"
        assert session["is_active"] == 1
        await store.close()

    async def test_add_messages_idempotent(self, tmp_path):
        from src.session_index.store import SessionIndexStore

        store = await SessionIndexStore.create(str(tmp_path / "index.db"))
        await store.upsert_session("t1", "qq", "s1", "c1", "p2p")
        msgs = [
            {
                "message_id": "msg-001",
                "role": "human",
                "content": "hello",
                "tool_name": None,
                "tool_calls": None,
                "timestamp": 1000.0,
            },
            {
                "message_id": "msg-002",
                "role": "ai",
                "content": "hi there",
                "tool_name": None,
                "tool_calls": None,
                "timestamp": 1001.0,
            },
        ]
        await store.add_messages("t1", msgs)
        await store.add_messages("t1", msgs)  # duplicate insert
        cursor = await store._db.execute("SELECT COUNT(*) FROM messages")
        count = (await cursor.fetchone())[0]
        assert count == 2
        await store.close()

    async def test_add_messages_updates_existing_content(self, tmp_path):
        """Re-syncing with modified content should update the indexed message (upsert)."""
        from src.session_index.store import SessionIndexStore

        store = await SessionIndexStore.create(str(tmp_path / "index.db"))
        await store.upsert_session("t1", "qq", "s1", "c1", "p2p")

        # v1: insert original
        await store.add_messages(
            "t1",
            [
                {
                    "message_id": "m1",
                    "role": "human",
                    "content": "original content",
                    "tool_name": None,
                    "tool_calls": None,
                    "timestamp": 1000.0,
                },
            ],
        )

        # v2: re-sync with updated content
        await store.add_messages(
            "t1",
            [
                {
                    "message_id": "m1",
                    "role": "human",
                    "content": "updated content",
                    "tool_name": None,
                    "tool_calls": None,
                    "timestamp": 2000.0,
                },
            ],
        )

        # row count stays 1
        cursor = await store._db.execute("SELECT COUNT(*) FROM messages")
        assert (await cursor.fetchone())[0] == 1

        # content was actually updated
        cursor = await store._db.execute("SELECT content, timestamp FROM messages WHERE message_id = 'm1'")
        row = await cursor.fetchone()
        assert row["content"] == "updated content"
        assert row["timestamp"] == 2000.0

        await store.close()

    async def test_add_messages_upsert_updates_fts(self, tmp_path):
        """Re-syncing modified content should refresh the FTS index."""
        from src.session_index.store import SessionIndexStore

        store = await SessionIndexStore.create(str(tmp_path / "index.db"))
        await store.upsert_session("t1", "qq", "s1", "c1", "p2p")

        await store.add_messages(
            "t1",
            [
                {
                    "message_id": "m1",
                    "role": "human",
                    "content": "deploy with kubernetes",
                    "tool_name": None,
                    "tool_calls": None,
                    "timestamp": 1000.0,
                },
            ],
        )
        # FTS can find "kubernetes"
        assert len(await store.search("kubernetes")) >= 1

        # re-sync: content changed from kubernetes → docker
        await store.add_messages(
            "t1",
            [
                {
                    "message_id": "m1",
                    "role": "human",
                    "content": "deploy with docker",
                    "tool_name": None,
                    "tool_calls": None,
                    "timestamp": 2000.0,
                },
            ],
        )

        # FTS now finds "docker"
        results = await store.search("docker")
        assert len(results) >= 1
        assert "docker" in results[0]["snippet"].lower()

        # FTS no longer finds "kubernetes"
        results_old = await store.search("kubernetes")
        assert all("kubernetes" not in r.get("snippet", "").lower() for r in results_old)

        await store.close()

    async def test_mark_inactive(self, tmp_path):
        from src.session_index.store import SessionIndexStore

        store = await SessionIndexStore.create(str(tmp_path / "index.db"))
        await store.upsert_session("t1", "qq", "s1", "c1", "p2p")
        await store.mark_inactive("t1")
        session = await store.get_session("t1")
        assert session["is_active"] == 0
        await store.close()

    async def test_search_basic(self, tmp_path):
        from src.session_index.store import SessionIndexStore

        store = await SessionIndexStore.create(str(tmp_path / "index.db"))
        await store.upsert_session("t1", "qq", "s1", "c1", "p2p")
        await store.add_messages(
            "t1",
            [
                {
                    "message_id": "m1",
                    "role": "human",
                    "content": "how to deploy docker",
                    "tool_name": None,
                    "tool_calls": None,
                    "timestamp": 1000.0,
                },
                {
                    "message_id": "m2",
                    "role": "ai",
                    "content": "use docker-compose",
                    "tool_name": None,
                    "tool_calls": None,
                    "timestamp": 1001.0,
                },
            ],
        )
        results = await store.search("docker")
        assert len(results) >= 1
        assert results[0]["thread_id"] == "t1"
        assert "docker" in results[0]["snippet"].lower()
        await store.close()

    async def test_search_empty_returns_recent(self, tmp_path):
        from src.session_index.store import SessionIndexStore

        store = await SessionIndexStore.create(str(tmp_path / "index.db"))
        await store.upsert_session("t1", "qq", "s1", "c1", "p2p")
        results = await store.search("")
        assert len(results) == 1
        assert results[0]["thread_id"] == "t1"
        await store.close()

    async def test_search_filters_inactive(self, tmp_path):
        from src.session_index.store import SessionIndexStore

        store = await SessionIndexStore.create(str(tmp_path / "index.db"))
        await store.upsert_session("t1", "qq", "s1", "c1", "p2p")
        await store.mark_inactive("t1")
        results = await store.search("")
        assert all(r["is_active"] for r in results) if results else True
        await store.close()


class TestSanitizeFts5Query:
    def test_basic_query(self):
        from src.utils.fts import sanitize_fts5_query

        result = sanitize_fts5_query("hello world")
        assert result == "hello OR world"

    def test_empty_query(self):
        from src.utils.fts import sanitize_fts5_query

        assert sanitize_fts5_query("") == '""'
        assert sanitize_fts5_query("   ") == '""'

    def test_unbalanced_double_quotes_stripped(self):
        from src.utils.fts import sanitize_fts5_query

        result = sanitize_fts5_query('test "unbalanced')
        assert '"' not in result
        assert "test" in result
        assert "unbalanced" in result

    def test_balanced_double_quotes_stripped(self):
        from src.utils.fts import sanitize_fts5_query

        result = sanitize_fts5_query('"hello world"')
        assert '"' not in result
        assert "hello" in result
        assert "world" in result

    def test_special_chars_only(self):
        from src.utils.fts import sanitize_fts5_query

        result = sanitize_fts5_query('"*()')
        assert result == '""'

    def test_chinese_query(self):
        from src.utils.fts import sanitize_fts5_query

        result = sanitize_fts5_query("部署 测试")
        assert "部署" in result
        assert "测试" in result

    def test_wildcards_and_parens_stripped(self):
        from src.utils.fts import sanitize_fts5_query

        result = sanitize_fts5_query("test*(other)")
        assert "*" not in result
        assert "(" not in result
        assert ")" not in result
        assert "test" in result
        assert "other" in result


class TestFts5UpdateTrigger:
    async def test_update_message_reflected_in_fts(self, tmp_path):
        from src.session_index.store import SessionIndexStore

        store = await SessionIndexStore.create(str(tmp_path / "index.db"))
        await store.upsert_session("t1", "qq", "s1", "c1", "p2p")
        await store.add_messages(
            "t1",
            [
                {
                    "message_id": "m1",
                    "role": "human",
                    "content": "original content about alpha",
                    "tool_name": None,
                    "tool_calls": None,
                    "timestamp": 1000.0,
                },
            ],
        )
        results_before = await store.search("alpha")
        assert len(results_before) >= 1

        await store._db.execute(
            "UPDATE messages SET content = ? WHERE message_id = ?",
            ("updated content about beta", "m1"),
        )
        await store._db.commit()

        results_after = await store.search("beta")
        assert len(results_after) >= 1
        assert "beta" in results_after[0]["snippet"].lower()

        results_old = await store.search("alpha")
        assert all("alpha" not in r.get("snippet", "").lower() for r in results_old)

        await store.close()


class TestSyncMessages:
    async def test_sync_extracts_messages(self, tmp_path):
        from src.session_index.store import SessionIndexStore
        from src.session_index.sync import sync_messages

        store = await SessionIndexStore.create(str(tmp_path / "index.db"))
        msgs = [
            {"role": "user", "content": "hello world", "id": "h1"},
            {"role": "assistant", "content": "hi there", "id": "a1"},
            {"role": "tool", "content": "/bin/ls output...", "tool_call_id": "tc1", "id": "t1", "name": "exec_command"},
        ]
        await sync_messages(
            store,
            thread_id="t1",
            messages=msgs,
            channel="qq",
            sender_id="ou_abc",
            chat_id="c1",
            chat_type="p2p",
            tool_max_chars=500,
        )
        cursor = await store._db.execute("SELECT COUNT(*) FROM messages WHERE thread_id='t1'")
        assert (await cursor.fetchone())[0] == 3
        await store.close()

    async def test_sync_truncates_tool_content(self, tmp_path):
        from src.session_index.store import SessionIndexStore
        from src.session_index.sync import sync_messages

        store = await SessionIndexStore.create(str(tmp_path / "index.db"))
        msgs = [{"role": "tool", "content": "x" * 2000, "tool_call_id": "tc1", "id": "t1", "name": "exec_command"}]
        await sync_messages(
            store, "t1", msgs, channel="qq", sender_id="s", chat_id="c", chat_type="p2p", tool_max_chars=500
        )
        cursor = await store._db.execute("SELECT content FROM messages WHERE thread_id='t1'")
        row = await cursor.fetchone()
        assert len(row["content"]) == 500
        await store.close()

    async def test_sync_idempotent(self, tmp_path):
        from src.session_index.store import SessionIndexStore
        from src.session_index.sync import sync_messages

        store = await SessionIndexStore.create(str(tmp_path / "index.db"))
        msgs = [{"role": "user", "content": "hello", "id": "h1"}]
        await sync_messages(store, "t1", msgs, "qq", "s", "c", "p2p", 500)
        await sync_messages(store, "t1", msgs, "qq", "s", "c", "p2p", 500)
        cursor = await store._db.execute("SELECT COUNT(*) FROM messages WHERE thread_id='t1'")
        assert (await cursor.fetchone())[0] == 1
        await store.close()


class TestPruneSessionsOrder:
    async def test_prune_deletes_from_checkpoints_even_when_index_fails(self, tmp_path):
        from src.session.pruner import prune_sessions

        cp_path = str(tmp_path / "checkpoints.db")
        idx_path = str(tmp_path / "session_index.db")
        _create_checkpoints_db(cp_path)
        old_time = 0.0
        _insert_session(cp_path, "t1", old_time)

        with patch("src.session.pruner.prune_session_index", side_effect=Exception("index db broken")):
            stats = await prune_sessions(
                cp_path,
                older_than_days=1,
                session_index_path=idx_path,
            )

        assert stats["sessions_removed"] == 1
        assert _count_sessions(cp_path) == 0

    async def test_prune_deletes_index_before_checkpoints(self, tmp_path):
        from src.session.pruner import prune_sessions, prune_session_index as original_prune_session_index

        cp_path = str(tmp_path / "checkpoints.db")
        idx_path = str(tmp_path / "session_index.db")
        _create_checkpoints_db(cp_path)
        _insert_session(cp_path, "t1", 0.0)

        call_order = []

        async def tracking_prune(path, tids):
            call_order.append(("index", tids))
            return await original_prune_session_index(path, tids)

        with patch("src.session.pruner.prune_session_index", side_effect=tracking_prune):
            await prune_sessions(cp_path, older_than_days=1, session_index_path=idx_path)

        assert len(call_order) == 1
        assert call_order[0][0] == "index"
        assert "t1" in call_order[0][1]
