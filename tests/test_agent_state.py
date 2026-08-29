"""Tests for hardened AgentState (Pydantic), fine-grained checkpoints, and concurrency locks."""

from __future__ import annotations

import asyncio
import pytest

from src.agent.state import AgentState, StateStore, MemoryStateStore, _VALID_ROLES


# ---------------------------------------------------------------------------
# AgentState — Pydantic validation
# ---------------------------------------------------------------------------


class TestAgentStateValidation:
    def test_valid_messages(self):
        state = AgentState(
            messages=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ]
        )
        assert len(state.messages) == 2

    def test_invalid_role_auto_fixed(self):
        """Old data with bad roles should be auto-fixed, not rejected."""
        state = AgentState(messages=[{"role": "bad", "content": "x"}])
        assert state.messages[0]["role"] == "user"

    def test_tool_message_without_tool_call_id_auto_fixed(self):
        """Old data with missing tool_call_id should be auto-fixed."""
        state = AgentState(messages=[{"role": "tool", "content": "result"}])
        assert state.messages[0]["tool_call_id"] == "unknown"

    def test_tool_message_with_tool_call_id_accepted(self):
        state = AgentState(
            messages=[
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "test", "arguments": "{}"}}],
                },
                {"role": "tool", "tool_call_id": "tc1", "content": "ok"},
            ]
        )
        assert len(state.messages) == 2

    def test_assistant_tool_calls_missing_id_auto_fixed(self):
        """Old data with malformed tool_calls should be auto-fixed."""
        state = AgentState(
            messages=[
                {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "test", "arguments": "{}"}}]}
            ]
        )
        assert state.messages[0]["tool_calls"][0]["id"] == "unknown"

    def test_all_valid_roles(self):
        for role in _VALID_ROLES:
            if role == "tool":
                msgs = [{"role": "tool", "tool_call_id": "x", "content": ""}]
            elif role == "assistant":
                msgs = [{"role": "assistant", "content": ""}]
            else:
                msgs = [{"role": role, "content": ""}]
            state = AgentState(messages=msgs)
            assert state.messages[0]["role"] == role

    def test_default_values(self):
        state = AgentState()
        assert state.messages == []
        assert state.system_prompt == ""
        assert state.chat_type == "p2p"
        assert state.pending_approval is None


# ---------------------------------------------------------------------------
# AgentState — append_message
# ---------------------------------------------------------------------------


class TestAppendMessage:
    def test_append_valid(self):
        state = AgentState()
        state.append_message({"role": "user", "content": "hi"})
        assert len(state.messages) == 1

    def test_append_invalid_role(self):
        state = AgentState()
        with pytest.raises(ValueError, match="Invalid message role"):
            state.append_message({"role": "bad", "content": ""})

    def test_append_tool_without_id(self):
        state = AgentState()
        with pytest.raises(ValueError, match="tool_call_id"):
            state.append_message({"role": "tool", "content": "x"})

    def test_append_tool_with_id(self):
        state = AgentState()
        state.append_message({"role": "tool", "tool_call_id": "tc1", "content": "ok"})
        assert state.messages[0]["tool_call_id"] == "tc1"


# ---------------------------------------------------------------------------
# AgentState — meta_dict
# ---------------------------------------------------------------------------


class TestMetaDict:
    def test_meta_dict_roundtrip(self):
        import json

        state = AgentState(
            system_prompt="sys",
            sender_id="u1",
            chat_id="c1",
            chat_type="group",
            channel="qq",
            pending_approval={"request_id": "r1"},
        )
        meta = state.meta_dict()
        serialized = json.dumps(meta)
        restored = json.loads(serialized)
        assert restored["sender_id"] == "u1"
        assert restored["pending_approval"]["request_id"] == "r1"


# ---------------------------------------------------------------------------
# StateStore — save/load round-trip
# ---------------------------------------------------------------------------


class TestStateStore:
    def test_save_load_roundtrip(self, tmp_path):
        async def _test():
            store = StateStore(str(tmp_path / "test.db"))
            state = AgentState(
                messages=[{"role": "user", "content": "hello"}],
                sender_id="u1",
                channel="qq",
            )
            await store.save("t1", state)
            loaded = await store.load("t1")
            assert loaded is not None
            assert loaded.messages[0]["content"] == "hello"
            assert loaded.sender_id == "u1"
            assert loaded.channel == "qq"
            await store.close()

        asyncio.run(_test())

    def test_frozen_system_prompt_roundtrip(self, tmp_path):
        async def _test():
            store = StateStore(str(tmp_path / "test.db"))
            state = AgentState(
                messages=[{"role": "user", "content": "hello"}],
                frozen_system_prompt="You are a helpful assistant with skills: ...",
            )
            await store.save("t1", state)
            loaded = await store.load("t1")
            assert loaded is not None
            assert loaded.frozen_system_prompt == "You are a helpful assistant with skills: ..."
            await store.close()

        asyncio.run(_test())

    def test_frozen_system_prompt_migration_from_old_metadata(self, tmp_path):
        """Simulate loading a row written by the old schema (frozen in JSON, new column empty)."""

        async def _test():
            import json

            store = StateStore(str(tmp_path / "test.db"))
            conn = await store._get_conn()
            # Old-style row: frozen_system_prompt inside metadata JSON, new column is ''
            await conn.execute(
                "INSERT OR REPLACE INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    "t1",
                    json.dumps([{"role": "user", "content": "hi"}]),
                    json.dumps(
                        {
                            "system_prompt": "",
                            "sender_id": "u1",
                            "chat_id": "",
                            "chat_type": "p2p",
                            "message_id": "",
                            "user_role": "",
                            "channel": "",
                            "pending_approval": None,
                            "frozen_system_prompt": "old frozen prompt from JSON",
                        }
                    ),
                    "",  # frozen_system_prompt column empty
                    0.0,  # created_at
                    0.0,  # updated_at
                    0,  # organized
                ),
            )
            await conn.commit()
            loaded = await store.load("t1")
            assert loaded is not None
            assert loaded.frozen_system_prompt == "old frozen prompt from JSON"
            assert loaded.organized is False
            await store.close()

        asyncio.run(_test())

    def test_load_unknown_returns_none(self, tmp_path):
        async def _test():
            store = StateStore(str(tmp_path / "test.db"))
            assert await store.load("nonexistent") is None
            await store.close()

        asyncio.run(_test())

    def test_delete(self, tmp_path):
        async def _test():
            store = StateStore(str(tmp_path / "test.db"))
            state = AgentState(messages=[{"role": "user", "content": "x"}])
            await store.save("t1", state)
            assert await store.delete("t1") is True
            assert await store.load("t1") is None
            await store.close()

        asyncio.run(_test())

    def test_list_threads(self, tmp_path):
        async def _test():
            store = StateStore(str(tmp_path / "test.db"))
            for i in range(3):
                state = AgentState(messages=[{"role": "user", "content": str(i)}])
                await store.save(f"t{i}", state)
            threads = await store.list_threads()
            assert len(threads) == 3
            await store.close()

        asyncio.run(_test())

    def test_model_validate_tolerates_extra_keys(self, tmp_path):
        async def _test():
            import json

            store = StateStore(str(tmp_path / "test.db"))
            conn = await store._get_conn()
            # Manually insert metadata with an extra key
            await conn.execute(
                "INSERT OR REPLACE INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    "t1",
                    json.dumps([{"role": "user", "content": "hi"}]),
                    json.dumps(
                        {
                            "system_prompt": "",
                            "sender_id": "",
                            "chat_id": "",
                            "chat_type": "p2p",
                            "message_id": "",
                            "user_role": "",
                            "channel": "",
                            "pending_approval": None,
                            "future_field": "should_be_ignored",
                        }
                    ),
                    "",
                    0.0,  # created_at
                    0.0,  # updated_at
                    0,  # organized
                ),
            )
            await conn.commit()
            loaded = await store.load("t1")
            assert loaded is not None
            assert loaded.messages[0]["content"] == "hi"
            await store.close()

        asyncio.run(_test())


# ---------------------------------------------------------------------------
# MemoryStateStore
# ---------------------------------------------------------------------------


class TestMemoryStateStore:
    def test_memory_store_works(self):
        async def _test():
            store = MemoryStateStore()
            state = AgentState(messages=[{"role": "user", "content": "test"}])
            await store.save("t1", state)
            loaded = await store.load("t1")
            assert loaded is not None
            assert loaded.messages[0]["content"] == "test"
            await store.close()

        asyncio.run(_test())

    def test_memory_store_has_locks(self):
        store = MemoryStateStore()
        assert hasattr(store, "_locks")
        assert hasattr(store, "_locks_lock")


# ---------------------------------------------------------------------------
# Concurrency locks
# ---------------------------------------------------------------------------


class TestConcurrencyLocks:
    def test_acquire_thread_returns_lock(self, tmp_path):
        async def _test():
            store = StateStore(str(tmp_path / "test.db"))
            lock = await store.acquire_thread("t1")
            assert isinstance(lock, asyncio.Lock)
            await store.close()

        asyncio.run(_test())

    def test_same_thread_returns_same_lock(self, tmp_path):
        async def _test():
            store = StateStore(str(tmp_path / "test.db"))

            l1 = await store.acquire_thread("t1")
            l2 = await store.acquire_thread("t1")
            assert l1 is l2
            await store.close()

        asyncio.run(_test())

    def test_different_threads_different_locks(self, tmp_path):
        async def _test():
            store = StateStore(str(tmp_path / "test.db"))

            l1 = await store.acquire_thread("t1")
            l2 = await store.acquire_thread("t2")
            assert l1 is not l2
            await store.close()

        asyncio.run(_test())

    def test_same_thread_serialized(self):
        """Two coroutines hitting the same thread_id must run sequentially."""
        store = MemoryStateStore()
        order = []

        async def task(name, thread_id, delay):
            lock = await store.acquire_thread(thread_id)
            async with lock:
                order.append(f"{name}-start")
                await asyncio.sleep(delay)
                order.append(f"{name}-end")

        async def run():
            await asyncio.gather(task("A", "t1", 0.05), task("B", "t1", 0.05))

        asyncio.run(run())
        # A and B must not overlap
        assert order == ["A-start", "A-end", "B-start", "B-end"] or order == ["B-start", "B-end", "A-start", "A-end"]


# ---------------------------------------------------------------------------
# organized column + mark_organized
# ---------------------------------------------------------------------------


class TestOrganizedColumn:
    def test_save_does_not_write_organized_column(self, tmp_path):
        """save() 不写 organized 列（仅 mark_organized 可置 1）；新行 organized 默认 0。
        即便 AgentState.organized=True，save 后 load 仍为 False——organized 由 mark_organized 独占管理。"""

        async def _test():
            store = StateStore(str(tmp_path / "test.db"))
            state = AgentState(
                messages=[{"role": "user", "content": "hi"}],
                organized=True,
            )
            await store.save("t1", state)
            loaded = await store.load("t1")
            assert loaded is not None
            assert loaded.organized is False  # save() 不写 organized 列
            await store.close()

        asyncio.run(_test())

    def test_mark_organized_sets_flag(self, tmp_path):
        """mark_organized sets organized=1; load reflects it."""

        async def _test():
            store = StateStore(str(tmp_path / "test.db"))
            state = AgentState(messages=[{"role": "user", "content": "hi"}])
            await store.save("t1", state)
            loaded = await store.load("t1")
            assert loaded.organized is False

            changed = await store.mark_organized("t1")
            assert changed is True

            loaded2 = await store.load("t1")
            assert loaded2.organized is True
            await store.close()

        asyncio.run(_test())

    def test_mark_organized_unknown_thread(self, tmp_path):
        """mark_organized on non-existent thread returns False."""

        async def _test():
            store = StateStore(str(tmp_path / "test.db"))
            changed = await store.mark_organized("nonexistent")
            assert changed is False
            await store.close()

        asyncio.run(_test())

    def test_save_preserves_organized(self, tmp_path):
        """ON CONFLICT save() preserves organized column (doesn't reset to 0)."""

        async def _test():
            store = StateStore(str(tmp_path / "test.db"))
            state = AgentState(messages=[{"role": "user", "content": "hi"}])
            await store.save("t1", state)
            await store.mark_organized("t1")

            # Simulate handler save (fresh AgentState, organized=False)
            state2 = AgentState(messages=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}])
            await store.save("t1", state2)

            # organized should survive the save
            loaded = await store.load("t1")
            assert loaded.organized is True
            assert len(loaded.messages) == 2
            await store.close()

        asyncio.run(_test())

    def test_save_preserves_created_at(self, tmp_path):
        """ON CONFLICT save() preserves created_at (replaces old COALESCE)."""

        async def _test():
            store = StateStore(str(tmp_path / "test.db"))
            state = AgentState(messages=[{"role": "user", "content": "first"}])
            await store.save("t1", state)
            original = await store.load("t1")
            original_ts = original.created_at

            await asyncio.sleep(0.05)

            state2 = AgentState(messages=[{"role": "user", "content": "second"}])
            await store.save("t1", state2)
            loaded = await store.load("t1")
            assert loaded.created_at == original_ts  # preserved, not overwritten
            await store.close()

        asyncio.run(_test())

    def test_list_threads_since(self, tmp_path):
        """list_threads_since returns only threads with created_at >= since_ts."""

        async def _test():
            import time as _time

            store = StateStore(str(tmp_path / "test.db"))

            old_ts = _time.time() - 100000
            new_ts = _time.time()

            old_state = AgentState(messages=[{"role": "user", "content": "old"}])
            new_state = AgentState(messages=[{"role": "user", "content": "new"}])

            # Manually set created_at via raw insert for old thread
            conn = await store._get_conn()
            await conn.execute(
                "INSERT INTO sessions (thread_id, messages, metadata, frozen_system_prompt, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("old_thread", "[]", "{}", "", old_ts, old_ts),
            )
            await conn.commit()
            await store.save("new_thread", new_state)

            cutoff = _time.time() - 3600
            threads = await store.list_threads_since(cutoff)
            assert "new_thread" in threads
            assert "old_thread" not in threads
            await store.close()

        asyncio.run(_test())

    def test_list_threads_since_excludes_organized(self, tmp_path):
        """list_threads_since 在 SQL 层排除 organized=1 的会话（避免失败重扫时 O(n) 退化）。"""

        async def _test():
            import time as _time

            store = StateStore(str(tmp_path / "test.db"))

            done_state = AgentState(messages=[{"role": "user", "content": "done"}])
            pending_state = AgentState(messages=[{"role": "user", "content": "pending"}])
            await store.save("done_thread", done_state)
            await store.save("pending_thread", pending_state)
            # 标记 done_thread 已整理
            await store.mark_organized("done_thread")

            cutoff = _time.time() - 3600
            threads = await store.list_threads_since(cutoff)
            assert "pending_thread" in threads
            assert "done_thread" not in threads  # SQL 层已过滤

            # 整理 pending_thread 后，它也从结果中消失
            await store.mark_organized("pending_thread")
            threads2 = await store.list_threads_since(cutoff)
            assert threads2 == []
            await store.close()

        asyncio.run(_test())

    def test_migration_from_pre_created_at_schema(self, tmp_path):
        """最老 schema（无 frozen_system_prompt/created_at/organized）经 _get_conn 迁移成功。

        回归：idx_sessions_created 若放在 _SCHEMA（executescript 先于 ALTER 执行），
        老库会抛 no such column: created_at 导致 StateStore 初始化失败。
        """
        import json

        import aiosqlite

        db_path = str(tmp_path / "old.db")
        row_updated = 1700000000.0

        async def _test():
            async with aiosqlite.connect(db_path) as conn:
                await conn.execute(
                    "CREATE TABLE sessions ("
                    "thread_id TEXT PRIMARY KEY, messages TEXT NOT NULL, "
                    "metadata TEXT NOT NULL, updated_at REAL NOT NULL)"
                )
                await conn.execute(
                    "INSERT INTO sessions VALUES (?, ?, ?, ?)",
                    ("old_thread", json.dumps([{"role": "user", "content": "hi"}]), "{}", row_updated),
                )
                await conn.commit()

            store = StateStore(db_path)
            # 老库迁移不抛错（no such column 回归）
            conn = await store._get_conn()
            async with conn.execute("PRAGMA table_info(sessions)") as cur:
                cols = {r[1] for r in await cur.fetchall()}
            assert {"frozen_system_prompt", "created_at", "organized"} <= cols
            async with conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_sessions_created'"
            ) as cur:
                assert await cur.fetchone() is not None

            # backfill: created_at 从 updated_at 补齐；save/load 正常
            loaded = await store.load("old_thread")
            assert loaded is not None
            assert loaded.created_at == row_updated
            assert loaded.organized is False
            await store.close()

        asyncio.run(_test())
