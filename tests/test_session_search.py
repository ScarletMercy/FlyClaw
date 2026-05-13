"""Tests for session search feature."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestSessionSearchConfig:
    def test_default_disabled(self, tmp_path):
        from src.config import load_config

        cfg = load_config(tmp_path / "nonexistent.yaml")
        assert cfg.session_search.enabled is False
        assert cfg.session_search.index_path == "data/session_index.db"
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
        assert cfg.session_search.index_path == "custom/index.db"
        assert cfg.session_search.max_results == 20

    def test_parse_thread_id_legacy_p2p(self):
        from src.session_index.store import parse_thread_id

        result = parse_thread_id("feishu:user:ou_abc123")
        assert result["channel"] == "feishu"
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

        result = parse_thread_id("feishu:global")
        assert result["channel"] == "feishu"
        assert result["chat_type"] == "p2p"

    def test_parse_thread_id_ws(self):
        from src.session_index.store import parse_thread_id

        result = parse_thread_id("ws:user:xyz")
        assert result["channel"] == "ws"
        assert result["chat_type"] == "p2p"
        assert result["sender_id"] == "xyz"


class TestSessionIndexStore:
    def test_init_creates_tables(self, tmp_path):
        from src.session_index.store import SessionIndexStore

        db_path = str(tmp_path / "index.db")
        store = SessionIndexStore(db_path)
        tables = store._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {t[0] for t in tables}
        assert "sessions" in table_names
        assert "messages" in table_names
        assert "messages_fts" in table_names
        store.close()

    def test_upsert_and_get_session(self, tmp_path):
        from src.session_index.store import SessionIndexStore

        store = SessionIndexStore(str(tmp_path / "index.db"))
        store.upsert_session(
            thread_id="feishu:user:ou_abc",
            channel="feishu",
            sender_id="ou_abc",
            chat_id="c2c:ou_abc",
            chat_type="p2p",
        )
        session = store.get_session("feishu:user:ou_abc")
        assert session["channel"] == "feishu"
        assert session["sender_id"] == "ou_abc"
        assert session["is_active"] == 1
        store.close()

    def test_add_messages_idempotent(self, tmp_path):
        from src.session_index.store import SessionIndexStore

        store = SessionIndexStore(str(tmp_path / "index.db"))
        store.upsert_session("t1", "feishu", "s1", "c1", "p2p")
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
        store.add_messages("t1", msgs)
        store.add_messages("t1", msgs)  # duplicate insert
        count = store._db.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        assert count == 2
        store.close()

    def test_mark_inactive(self, tmp_path):
        from src.session_index.store import SessionIndexStore

        store = SessionIndexStore(str(tmp_path / "index.db"))
        store.upsert_session("t1", "qq", "s1", "c1", "p2p")
        store.mark_inactive("t1")
        session = store.get_session("t1")
        assert session["is_active"] == 0
        store.close()

    def test_search_basic(self, tmp_path):
        from src.session_index.store import SessionIndexStore

        store = SessionIndexStore(str(tmp_path / "index.db"))
        store.upsert_session("t1", "feishu", "s1", "c1", "p2p")
        store.add_messages(
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
        results = store.search("docker")
        assert len(results) >= 1
        assert results[0]["thread_id"] == "t1"
        assert "docker" in results[0]["snippet"].lower()
        store.close()

    def test_search_empty_returns_recent(self, tmp_path):
        from src.session_index.store import SessionIndexStore

        store = SessionIndexStore(str(tmp_path / "index.db"))
        store.upsert_session("t1", "qq", "s1", "c1", "p2p")
        results = store.search("")
        assert len(results) == 1
        assert results[0]["thread_id"] == "t1"
        store.close()

    def test_search_filters_inactive(self, tmp_path):
        from src.session_index.store import SessionIndexStore

        store = SessionIndexStore(str(tmp_path / "index.db"))
        store.upsert_session("t1", "qq", "s1", "c1", "p2p")
        store.mark_inactive("t1")
        results = store.search("")
        assert all(r["is_active"] for r in results) if results else True
        store.close()


from unittest.mock import MagicMock


class TestSyncMessages:
    def _make_msg(self, role, content, msg_id=None, name=None, tool_calls=None):
        msg = MagicMock()
        msg.content = content
        msg.id = msg_id or f"uuid-{role}"
        msg.name = name
        msg.tool_calls = tool_calls
        # Set __class__ so isinstance works
        if role == "human":
            msg.__class__ = type("HumanMessage", (), {})
        elif role == "ai":
            msg.__class__ = type("AIMessage", (), {})
        elif role == "tool":
            msg.__class__ = type("ToolMessage", (), {})
        else:
            msg.__class__ = type("SystemMessage", (), {})
        return msg

    def test_sync_extracts_messages(self, tmp_path):
        from src.session_index.store import SessionIndexStore
        from src.session_index.sync import sync_messages

        store = SessionIndexStore(str(tmp_path / "index.db"))
        # Patch isinstance checks by using real LangChain messages
        from langchain_core.messages import HumanMessage as HM, AIMessage as AM, ToolMessage as TM

        msgs = [
            HM(content="hello world", id="h1"),
            AM(content="hi there", id="a1"),
            TM(content="/bin/ls output...", tool_call_id="tc1", id="t1", name="exec_command"),
        ]
        sync_messages(
            store,
            thread_id="t1",
            messages=msgs,
            channel="feishu",
            sender_id="ou_abc",
            chat_id="c1",
            chat_type="p2p",
            tool_max_chars=500,
        )
        count = store._db.execute("SELECT COUNT(*) FROM messages WHERE thread_id='t1'").fetchone()[0]
        assert count == 3
        store.close()

    def test_sync_truncates_tool_content(self, tmp_path):
        from src.session_index.store import SessionIndexStore
        from src.session_index.sync import sync_messages
        from langchain_core.messages import ToolMessage

        store = SessionIndexStore(str(tmp_path / "index.db"))
        long_content = "x" * 2000
        msgs = [ToolMessage(content=long_content, tool_call_id="tc1", id="t1", name="exec_command")]
        sync_messages(
            store, "t1", msgs,
            channel="qq", sender_id="s", chat_id="c", chat_type="p2p",
            tool_max_chars=500,
        )
        row = store._db.execute(
            "SELECT content FROM messages WHERE thread_id='t1'"
        ).fetchone()
        assert len(row["content"]) == 500
        store.close()

    def test_sync_idempotent(self, tmp_path):
        from src.session_index.store import SessionIndexStore
        from src.session_index.sync import sync_messages
        from langchain_core.messages import HumanMessage

        store = SessionIndexStore(str(tmp_path / "index.db"))
        msgs = [HumanMessage(content="hello", id="h1")]
        sync_messages(store, "t1", msgs, "qq", "s", "c", "p2p", 500)
        sync_messages(store, "t1", msgs, "qq", "s", "c", "p2p", 500)
        count = store._db.execute("SELECT COUNT(*) FROM messages WHERE thread_id='t1'").fetchone()[0]
        assert count == 1
        store.close()
