"""Tests for hardened AgentState (Pydantic), fine-grained checkpoints, and concurrency locks."""

from __future__ import annotations

import asyncio
import pytest
from pydantic import ValidationError

from src.agent.state import AgentState, StateStore, MemoryStateStore, _VALID_ROLES


# ---------------------------------------------------------------------------
# AgentState — Pydantic validation
# ---------------------------------------------------------------------------

class TestAgentStateValidation:
    def test_valid_messages(self):
        state = AgentState(messages=[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ])
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
        state = AgentState(messages=[
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc1", "type": "function", "function": {"name": "test", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "tc1", "content": "ok"},
        ])
        assert len(state.messages) == 2

    def test_assistant_tool_calls_missing_id_auto_fixed(self):
        """Old data with malformed tool_calls should be auto-fixed."""
        state = AgentState(messages=[{"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "test", "arguments": "{}"}}
        ]}])
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
            system_prompt="sys", sender_id="u1", chat_id="c1",
            chat_type="group", channel="qq",
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
        store = StateStore(str(tmp_path / "test.db"))
        state = AgentState(
            messages=[{"role": "user", "content": "hello"}],
            sender_id="u1", channel="qq",
        )
        asyncio.run(store.save("t1", state))
        loaded = store.load("t1")
        assert loaded is not None
        assert loaded.messages[0]["content"] == "hello"
        assert loaded.sender_id == "u1"
        assert loaded.channel == "qq"

    def test_load_unknown_returns_none(self, tmp_path):
        store = StateStore(str(tmp_path / "test.db"))
        assert store.load("nonexistent") is None

    def test_load_messages(self, tmp_path):
        store = StateStore(str(tmp_path / "test.db"))
        state = AgentState(messages=[
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ])
        asyncio.run(store.save("t1", state))
        msgs = store.load_messages("t1")
        assert len(msgs) == 2

    def test_delete(self, tmp_path):
        store = StateStore(str(tmp_path / "test.db"))
        state = AgentState(messages=[{"role": "user", "content": "x"}])
        asyncio.run(store.save("t1", state))
        assert store.delete("t1") is True
        assert store.load("t1") is None

    def test_list_threads(self, tmp_path):
        store = StateStore(str(tmp_path / "test.db"))
        for i in range(3):
            state = AgentState(messages=[{"role": "user", "content": str(i)}])
            asyncio.run(store.save(f"t{i}", state))
        threads = store.list_threads()
        assert len(threads) == 3

    def test_model_validate_tolerates_extra_keys(self, tmp_path):
        store = StateStore(str(tmp_path / "test.db"))
        import json
        # Manually insert metadata with an extra key
        store._db.execute(
            "INSERT OR REPLACE INTO sessions VALUES (?, ?, ?, ?)",
            ("t1", json.dumps([{"role": "user", "content": "hi"}]),
             json.dumps({"system_prompt": "", "sender_id": "", "chat_id": "",
                         "chat_type": "p2p", "message_id": "", "user_role": "",
                         "channel": "", "pending_approval": None,
                         "future_field": "should_be_ignored"}),
             0.0),
        )
        store._db.commit()
        loaded = store.load("t1")
        assert loaded is not None
        assert loaded.messages[0]["content"] == "hi"


# ---------------------------------------------------------------------------
# MemoryStateStore
# ---------------------------------------------------------------------------

class TestMemoryStateStore:
    def test_memory_store_works(self):
        store = MemoryStateStore()
        state = AgentState(messages=[{"role": "user", "content": "test"}])
        asyncio.run(store.save("t1", state))
        loaded = store.load("t1")
        assert loaded is not None
        assert loaded.messages[0]["content"] == "test"

    def test_memory_store_has_locks(self):
        store = MemoryStateStore()
        assert hasattr(store, '_locks')
        assert hasattr(store, '_locks_lock')


# ---------------------------------------------------------------------------
# Concurrency locks
# ---------------------------------------------------------------------------

class TestConcurrencyLocks:
    def test_acquire_thread_returns_lock(self, tmp_path):
        store = StateStore(str(tmp_path / "test.db"))
        lock = asyncio.run(store.acquire_thread("t1"))
        assert isinstance(lock, asyncio.Lock)

    def test_same_thread_returns_same_lock(self, tmp_path):
        store = StateStore(str(tmp_path / "test.db"))

        async def _get_two():
            l1 = await store.acquire_thread("t1")
            l2 = await store.acquire_thread("t1")
            return l1, l2

        lock1, lock2 = asyncio.run(_get_two())
        assert lock1 is lock2

    def test_different_threads_different_locks(self, tmp_path):
        store = StateStore(str(tmp_path / "test.db"))

        async def _get_two():
            l1 = await store.acquire_thread("t1")
            l2 = await store.acquire_thread("t2")
            return l1, l2

        lock1, lock2 = asyncio.run(_get_two())
        assert lock1 is not lock2

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
