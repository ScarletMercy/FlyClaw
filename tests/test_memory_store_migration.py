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


# ─── organized 列 ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_migrate_add_organized_backfills_zero(tmp_path):
    """老 schema(无 organized)经 initialize 后 organized 列被加，默认 0。"""
    db = tmp_path / "m.db"
    async with aiosqlite.connect(str(db)) as conn:
        await conn.execute(
            "CREATE TABLE memories ("
            "key TEXT PRIMARY KEY, content TEXT, category TEXT, "
            "created_at TEXT, updated_at TEXT, updated_ts REAL)"
        )
        await conn.execute(
            "INSERT INTO memories (key, content, category, created_at, updated_at, updated_ts) "
            "VALUES ('k1', 'c', 'fact', 'x', 'x', 1.0)"
        )
        await conn.commit()

    store = MemoryStore(str(db))
    await store.initialize()
    try:
        async with store._conn.execute("SELECT organized FROM memories WHERE key = 'k1'") as cur:
            (org,) = await cur.fetchone()
        assert org == 0
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_migrate_add_organized_idempotent(tmp_path):
    """重复 initialize 不重复加列。"""
    db_path = str(tmp_path / "m.db")
    store = MemoryStore(db_path)
    await store.initialize()
    await store.close()
    store2 = MemoryStore(db_path)
    await store2.initialize()
    try:
        async with store2._conn.execute("PRAGMA table_info(memories)") as cur:
            cols = [r[1] for r in await cur.fetchall()]
        assert cols.count("organized") == 1
    finally:
        await store2.close()


@pytest.mark.asyncio
async def test_mark_organized_sets_flag(tmp_path):
    """mark_organized 把指定 key 置 1，未指定的不动。"""
    store = MemoryStore(str(tmp_path / "m.db"))
    await store.initialize()
    try:
        await store.remember("内容1", key="k1")
        await store.remember("内容2", key="k2")
        n = await store.mark_organized(["k1"])
        assert n == 1
        items = {i["key"]: i["organized"] for i in await store.list_all()}
        assert items["k1"] == 1
        assert items["k2"] == 0
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_remember_reset_organized_on_content_change(tmp_path):
    """非 dedup 更新(显式 key + 内容变化)→ organized 重置 0；dedup(同内容)不重置。"""
    store = MemoryStore(str(tmp_path / "m.db"))
    await store.initialize()
    try:
        await store.remember("原文", key="k1")
        await store.mark_organized(["k1"])
        # 内容变化(显式 key)→ 重置 organized=0
        await store.remember("新内容", key="k1")
        async with store._conn.execute("SELECT organized FROM memories WHERE key = 'k1'") as cur:
            (org,) = await cur.fetchone()
        assert org == 0
        # dedup(同内容无 key)→ 不重置
        await store.mark_organized(["k1"])
        await store.remember("新内容")  # 命中 k1 的 dedup
        async with store._conn.execute("SELECT organized FROM memories WHERE key = 'k1'") as cur:
            (org2,) = await cur.fetchone()
        assert org2 == 1
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_group_store_mark_organized_scoped_by_group(tmp_path):
    """GroupMemoryStore.mark_organized 按 group_id 隔离：只标记指定群的 key。"""
    from src.tools.memory_tools import GroupMemoryStore

    store = GroupMemoryStore(str(tmp_path / "g.db"))
    await store.initialize()
    try:
        await store.remember("内容", key="k1", group_id="G1")
        await store.remember("内容", key="k1", group_id="G2")
        await store.mark_organized(["k1"], group_id="G1")
        g1 = await store.list_all(group_id="G1")
        g2 = await store.list_all(group_id="G2")
        assert g1[0]["organized"] == 1
        assert g2[0]["organized"] == 0  # G2 不受影响
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_remember_explicit_key_same_content_no_reset(tmp_path):
    """显式 key + 内容未变 → organized 不重置（content_changed=False）。"""
    store = MemoryStore(str(tmp_path / "m.db"))
    await store.initialize()
    try:
        await store.remember("原文", key="k1")
        await store.mark_organized(["k1"])
        # 显式 key + 相同内容 → organized 不应被重置
        await store.remember("原文", key="k1")
        async with store._conn.execute("SELECT organized FROM memories WHERE key = 'k1'") as cur:
            (org,) = await cur.fetchone()
        assert org == 1
    finally:
        await store.close()
