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
                "INSERT OR REPLACE INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
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
                    0,
                    0,
                ),
            )
            await conn.commit()
            loaded = await store.load("t1")
            assert loaded is not None
            assert loaded.frozen_system_prompt == "old frozen prompt from JSON"
            assert loaded.status == 0
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
                "INSERT OR REPLACE INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
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
                    0,
                    0,
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
# status 状态机列（0=待处理 1=已完成 2=永久跳过 3=死信）+ attempts
# ---------------------------------------------------------------------------


class TestSessionStatus:
    def test_save_does_not_write_status_column(self, tmp_path):
        """save() 插入新行不写 status 列（默认 0）。
        即便 AgentState.status=1，save 后 load 仍为 0——status 由 DB 列独占管理。"""

        async def _test():
            store = StateStore(str(tmp_path / "test.db"))
            state = AgentState(
                messages=[{"role": "user", "content": "hi"}],
                status=1,
            )
            await store.save("t1", state)
            loaded = await store.load("t1")
            assert loaded is not None
            assert loaded.status == 0
            await store.close()

        asyncio.run(_test())

    def test_mark_organized_sets_flag(self, tmp_path):
        """mark_organized sets status=1; load reflects it."""

        async def _test():
            store = StateStore(str(tmp_path / "test.db"))
            state = AgentState(messages=[{"role": "user", "content": "hi"}])
            await store.save("t1", state)
            loaded = await store.load("t1")
            assert loaded.status == 0

            changed = await store.mark_organized("t1")
            assert changed is True

            loaded2 = await store.load("t1")
            assert loaded2.status == 1
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

    def test_save_preserves_status(self, tmp_path):
        """ON CONFLICT save() preserves status column (doesn't reset to 0)."""

        async def _test():
            store = StateStore(str(tmp_path / "test.db"))
            state = AgentState(messages=[{"role": "user", "content": "hi"}])
            await store.save("t1", state)
            await store.mark_organized("t1")

            # Simulate handler save (fresh AgentState, status=0)
            state2 = AgentState(messages=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}])
            await store.save("t1", state2)

            loaded = await store.load("t1")
            assert loaded.status == 1
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

    def test_list_pending_threads_includes_old_created(self, tmp_path):
        """list_pending_threads 不看 created_at：老会话照样返回。"""

        async def _test():
            import time as _time

            store = StateStore(str(tmp_path / "test.db"))

            old_ts = _time.time() - 100000
            new_state = AgentState(messages=[{"role": "user", "content": "new"}])

            conn = await store._get_conn()
            await conn.execute(
                "INSERT INTO sessions (thread_id, messages, metadata, frozen_system_prompt, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("old_thread", "[]", "{}", "", old_ts, old_ts),
            )
            await conn.commit()
            await store.save("new_thread", new_state)

            threads = await store.list_pending_threads()
            assert "new_thread" in threads
            assert "old_thread" in threads
            await store.close()

        asyncio.run(_test())

    def test_list_pending_threads_excludes_terminal_states(self, tmp_path):
        """SQL 层排除全部终态：完成(1)、永久跳过(2)、死信(3) 均不进入扫描。"""

        async def _test():
            store = StateStore(str(tmp_path / "test.db"))

            for tid in ("done", "skipped", "dead"):
                await store.save(tid, AgentState(messages=[{"role": "user", "content": tid}]))
            await store.save("pending", AgentState(messages=[{"role": "user", "content": "p"}]))

            conn = await store._get_conn()
            await conn.execute("UPDATE sessions SET status = 1 WHERE thread_id = 'done'")
            await conn.execute("UPDATE sessions SET status = 2 WHERE thread_id = 'skipped'")
            await conn.execute("UPDATE sessions SET status = 3 WHERE thread_id = 'dead'")
            await conn.commit()

            assert await store.list_pending_threads() == ["pending"]
            await store.close()

        asyncio.run(_test())

    def test_list_pending_threads_orders_oldest_first(self, tmp_path):
        """list_pending_threads 按 created_at 正序：最老的待整理会话先处理。"""

        async def _test():
            import time as _time

            store = StateStore(str(tmp_path / "test.db"))

            now = _time.time()
            conn = await store._get_conn()
            for tid, ts in (("older", now - 200), ("newer", now - 100)):
                await conn.execute(
                    "INSERT INTO sessions (thread_id, messages, metadata, frozen_system_prompt, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (tid, "[]", "{}", "", ts, ts),
                )
            await conn.commit()

            threads = await store.list_pending_threads()
            assert threads == ["older", "newer"]
            await store.close()

        asyncio.run(_test())

    def test_migration_from_pre_created_at_schema(self, tmp_path):
        """最老 schema（无 frozen_system_prompt/created_at/status）经 _get_conn 迁移成功。

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
            assert {"frozen_system_prompt", "created_at", "status", "attempts"} <= cols
            async with conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_sessions_created'"
            ) as cur:
                assert await cur.fetchone() is not None

            # backfill: created_at 从 updated_at 补齐；save/load 正常
            loaded = await store.load("old_thread")
            assert loaded is not None
            assert loaded.created_at == row_updated
            assert loaded.status == 0
            await store.close()

        asyncio.run(_test())

    def test_migration_from_organized_schema_preserves_values(self, tmp_path):
        """organized-era 老库：organized 列 rename 成 status，0/1 值原样保留。"""
        import aiosqlite

        db_path = str(tmp_path / "organized_era.db")

        async def _test():
            async with aiosqlite.connect(db_path) as conn:
                await conn.execute(
                    "CREATE TABLE sessions ("
                    "thread_id TEXT PRIMARY KEY, messages TEXT NOT NULL, metadata TEXT NOT NULL, "
                    "frozen_system_prompt TEXT NOT NULL DEFAULT '', created_at REAL NOT NULL DEFAULT 0, "
                    "updated_at REAL NOT NULL, organized INTEGER NOT NULL DEFAULT 0)"
                )
                await conn.execute(
                    "INSERT INTO sessions (thread_id, messages, metadata, updated_at, organized) "
                    "VALUES ('done_thread', '[]', '{}', 1.0, 1)"
                )
                await conn.execute(
                    "INSERT INTO sessions (thread_id, messages, metadata, updated_at, organized) "
                    "VALUES ('pending_thread', '[]', '{}', 1.0, 0)"
                )
                await conn.commit()

            store = StateStore(db_path)
            conn = await store._get_conn()
            async with conn.execute("PRAGMA table_info(sessions)") as cur:
                cols = {r[1] for r in await cur.fetchall()}
            assert "status" in cols and "attempts" in cols
            assert "organized" not in cols  # 列已 rename，不残留双列

            assert (await store.load("done_thread")).status == 1
            assert (await store.load("pending_thread")).status == 0
            assert await store.list_pending_threads() == ["pending_thread"]
            await store.close()

        asyncio.run(_test())


# ---------------------------------------------------------------------------
# age_stale_sessions / record_failed_attempt（终态入口）
# ---------------------------------------------------------------------------

_MSGS_GATE_OK = [{"role": "user", "content": f"u{i}"} for i in range(5)] + [
    {"role": "assistant", "content": f"a{i}"} for i in range(5)
]
_MSGS_BELOW = [{"role": "user", "content": "u0"}, {"role": "assistant", "content": "a0"}]


async def _raw_insert(store, thread_id, messages_json, updated_at):
    """绕过 save()（恒写 now）指定 updated_at，供老化场景造数据。"""
    conn = await store._get_conn()
    await conn.execute(
        "INSERT INTO sessions (thread_id, messages, metadata, frozen_system_prompt, created_at, updated_at) "
        "VALUES (?, ?, '{}', '', ?, ?)",
        (thread_id, messages_json, updated_at, updated_at),
    )
    await conn.commit()


class TestAgingAndDeadLetter:
    def test_age_skips_below_gate_and_idle(self, tmp_path):
        """低于门槛且超期无活动 → 老化进永久跳过(2)，不再进入扫描。"""
        import json as _json
        import time as _time

        async def _test():
            store = StateStore(str(tmp_path / "test.db"))
            await _raw_insert(store, "tiny", _json.dumps(_MSGS_BELOW), _time.time() - 40 * 86400)

            aged, revived = await store.age_stale_sessions(min_messages=10, idle_days=30)
            assert (aged, revived) == (1, 0)
            assert await store.list_pending_threads() == []
            assert (await store.load("tiny")).status == 2
            await store.close()

        asyncio.run(_test())

    def test_age_keeps_below_gate_but_active(self, tmp_path):
        """低于门槛但近期活跃（可能攒够）→ 不老化。"""
        import json as _json
        import time as _time

        async def _test():
            store = StateStore(str(tmp_path / "test.db"))
            await _raw_insert(store, "tiny", _json.dumps(_MSGS_BELOW), _time.time() - 3600)

            aged, _ = await store.age_stale_sessions(min_messages=10, idle_days=30)
            assert aged == 0
            assert await store.list_pending_threads() == ["tiny"]
            await store.close()

        asyncio.run(_test())

    def test_age_keeps_eligible_even_if_idle(self, tmp_path):
        """够门槛的老会话永不老化——饿死修复不被老化回退。"""
        import json as _json
        import time as _time

        async def _test():
            store = StateStore(str(tmp_path / "test.db"))
            await _raw_insert(store, "big", _json.dumps(_MSGS_GATE_OK), _time.time() - 40 * 86400)

            aged, _ = await store.age_stale_sessions(min_messages=10, idle_days=30)
            assert aged == 0
            assert await store.list_pending_threads() == ["big"]
            await store.close()

        asyncio.run(_test())

    def test_age_uses_user_count_gate_too(self, tmp_path):
        """总条数够但 user < 3 的会话同样老化（门槛谓词与 Python 层一致）。"""
        import json as _json
        import time as _time

        async def _test():
            msgs = [{"role": "user", "content": "u0"}] + [{"role": "assistant", "content": f"a{i}"} for i in range(9)]
            store = StateStore(str(tmp_path / "test.db"))
            await _raw_insert(store, "one_user", _json.dumps(msgs), _time.time() - 40 * 86400)

            aged, _ = await store.age_stale_sessions(min_messages=10, idle_days=30)
            assert aged == 1
            await store.close()

        asyncio.run(_test())

    def test_revived_when_later_crosses_gate(self, tmp_path):
        """已跳过(2)的会话攒够门槛后复活回待处理(0)。"""
        import json as _json
        import time as _time

        async def _test():
            store = StateStore(str(tmp_path / "test.db"))
            await _raw_insert(store, "t1", _json.dumps(_MSGS_BELOW), _time.time() - 40 * 86400)
            await store.age_stale_sessions(min_messages=10, idle_days=30)
            assert (await store.load("t1")).status == 2

            state = await store.load("t1")
            state.messages = list(_MSGS_GATE_OK)
            await store.save("t1", state)

            aged, revived = await store.age_stale_sessions(min_messages=10, idle_days=30)
            assert (aged, revived) == (0, 1)
            assert await store.list_pending_threads() == ["t1"]
            assert (await store.load("t1")).status == 0
            await store.close()

        asyncio.run(_test())

    def test_dead_never_revives(self, tmp_path):
        """死信(3) 不参与复活：复活会造成 nightly 失败死循环。"""
        import json as _json
        import time as _time

        async def _test():
            store = StateStore(str(tmp_path / "test.db"))
            await _raw_insert(store, "t1", _json.dumps(_MSGS_GATE_OK), _time.time())
            conn = await store._get_conn()
            await conn.execute("UPDATE sessions SET status = 3 WHERE thread_id = 't1'")
            await conn.commit()

            aged, revived = await store.age_stale_sessions(min_messages=10, idle_days=30)
            assert (aged, revived) == (0, 0)
            assert (await store.load("t1")).status == 3
            await store.close()

        asyncio.run(_test())

    def test_malformed_json_ages_out(self, tmp_path):
        """JSON 损坏的行 → json_valid=0 → 老化出局。"""
        import time as _time

        async def _test():
            store = StateStore(str(tmp_path / "test.db"))
            await _raw_insert(store, "corrupt", "not-json", _time.time() - 40 * 86400)

            aged, _ = await store.age_stale_sessions(min_messages=10, idle_days=30)
            assert aged == 1
            await store.close()

        asyncio.run(_test())

    def test_record_failed_attempt_deads_at_cap(self, tmp_path):
        """失败计数逐次 +1，达到 max_attempts 置死信(3)。"""

        async def _test():
            store = StateStore(str(tmp_path / "test.db"))
            await store.save("t1", AgentState(messages=[{"role": "user", "content": "hi"}]))

            assert await store.record_failed_attempt("t1", max_attempts=3) is False
            assert await store.record_failed_attempt("t1", max_attempts=3) is False
            assert (await store.load("t1")).status == 0
            assert await store.record_failed_attempt("t1", max_attempts=3) is True

            loaded = await store.load("t1")
            assert loaded.status == 3
            assert loaded.attempts == 3
            await store.close()

        asyncio.run(_test())

    def test_mark_organized_resets_attempts(self, tmp_path):
        """成功整理后 status=1 且失败计数清零。"""

        async def _test():
            store = StateStore(str(tmp_path / "test.db"))
            await store.save("t1", AgentState(messages=[{"role": "user", "content": "hi"}]))
            await store.record_failed_attempt("t1", max_attempts=3)
            await store.record_failed_attempt("t1", max_attempts=3)

            await store.mark_organized("t1")
            loaded = await store.load("t1")
            assert loaded.status == 1
            assert loaded.attempts == 0
            await store.close()

        asyncio.run(_test())

    def test_save_revives_dead_session(self, tmp_path):
        """死信会话有新活动（save）→ 重开为待处理并清空失败预算；完成/跳过不受影响。"""

        async def _test():
            store = StateStore(str(tmp_path / "test.db"))
            for tid in ("dead", "done", "skipped"):
                await store.save(tid, AgentState(messages=[{"role": "user", "content": "hi"}]))
            conn = await store._get_conn()
            await conn.execute("UPDATE sessions SET status = 3, attempts = 3 WHERE thread_id = 'dead'")
            await conn.execute("UPDATE sessions SET status = 1 WHERE thread_id = 'done'")
            await conn.execute("UPDATE sessions SET status = 2 WHERE thread_id = 'skipped'")
            await conn.commit()

            for tid in ("dead", "done", "skipped"):
                st = await store.load(tid)
                st.messages.append({"role": "user", "content": "new"})
                await store.save(tid, st)

            revived = await store.load("dead")
            assert revived.status == 0
            assert revived.attempts == 0
            assert await store.list_pending_threads() == ["dead"]
            assert (await store.load("done")).status == 1
            assert (await store.load("skipped")).status == 2
            await store.close()

        asyncio.run(_test())
