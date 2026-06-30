"""Tests for MemoryStore schema migration: adding updated_ts for correct sorting."""

import aiosqlite
import pytest

from src.tools.memory_tools import MemoryStore


@pytest.mark.asyncio
async def test_migrate_add_updated_ts_backfills_from_mixed_formats(tmp_path):
    """老 schema(无 updated_ts)经 initialize 后,updated_ts 列被回填。

    核心:同一 UTC 时刻的 +00:00 与 +08:00 两种写法,回填出相同 epoch
    ——这正是字符串排序会错、改用 epoch 秒根治的关键。aware 串的比较
    不依赖系统时区,故此断言在 UTC 的 CI 上也成立。
    """
    db = tmp_path / "m.db"
    async with aiosqlite.connect(str(db)) as conn:
        await conn.execute(
            "CREATE TABLE memories ("
            "key TEXT PRIMARY KEY, content TEXT, category TEXT, "
            "created_at TEXT, updated_at TEXT)"
        )
        await conn.execute(
            "INSERT INTO memories (key, content, category, created_at, updated_at) "
            "VALUES ('utc', 'c', 'fact', 'x', '2026-06-28T03:00:00+00:00')"
        )
        await conn.execute(
            "INSERT INTO memories (key, content, category, created_at, updated_at) "
            "VALUES ('local', 'c', 'fact', 'x', '2026-06-28T11:00:00+08:00')"
        )
        await conn.execute(
            "INSERT INTO memories (key, content, category, created_at, updated_at) "
            "VALUES ('naive', 'c', 'fact', 'x', '2026-06-28T11:00:00')"
        )
        await conn.commit()

    store = MemoryStore(str(db))
    await store.initialize()
    try:
        async with store._conn.execute("SELECT key, updated_ts FROM memories") as cur:
            rows = {r[0]: r[1] for r in await cur.fetchall()}

        # aware 串:同一 UTC 时刻的两种偏移写法 → 相同 epoch
        assert rows["utc"] == rows["local"]
        assert rows["utc"] > 0
        # naive 串:按本地解释回填,成功即可(值依赖系统时区,不断言具体值)
        assert rows["naive"] > 0
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_migrate_add_updated_ts_is_idempotent(tmp_path):
    """重复 initialize 不重复加列、不报错。"""
    db_path = str(tmp_path / "m.db")
    store = MemoryStore(db_path)
    await store.initialize()
    await store.close()

    store2 = MemoryStore(db_path)
    await store2.initialize()
    try:
        async with store2._conn.execute("PRAGMA table_info(memories)") as cur:
            cols = [r[1] for r in await cur.fetchall()]
        assert cols.count("updated_ts") == 1
    finally:
        await store2.close()


@pytest.mark.asyncio
async def test_migrate_add_updated_ts_for_group_store(tmp_path):
    """GroupMemoryStore(群聊记忆,独立 db 文件)也触发 updated_ts 迁移。"""
    from src.tools.memory_tools import GroupMemoryStore

    db = tmp_path / "g.db"
    async with aiosqlite.connect(str(db)) as conn:
        await conn.execute(
            "CREATE TABLE memories ("
            "key TEXT NOT NULL, content TEXT, category TEXT, "
            "group_id TEXT NOT NULL DEFAULT '', "
            "created_at TEXT, updated_at TEXT, "
            "PRIMARY KEY (key, group_id))"
        )
        await conn.execute(
            "INSERT INTO memories (key, content, category, group_id, created_at, updated_at) "
            "VALUES ('k1', 'c', 'fact', '', 'x', '2026-06-28T03:00:00+00:00')"
        )
        await conn.commit()

    store = GroupMemoryStore(str(db))
    await store.initialize()
    try:
        async with store._conn.execute("SELECT updated_ts FROM memories WHERE key = 'k1'") as cur:
            (ts,) = await cur.fetchone()
        assert ts > 0
    finally:
        await store.close()
