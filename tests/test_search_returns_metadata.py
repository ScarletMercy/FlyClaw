"""Tests that search results surface metadata column."""

from __future__ import annotations

import pytest

from src.memory.store import MemoryStore


@pytest.fixture
async def store(tmp_path):
    s = MemoryStore(
        db_path=str(tmp_path / "t.db"),
        dimensions=4,
    )
    await s.initialize()
    yield s
    await s.close()


class TestSearchMetadata:
    @pytest.mark.asyncio
    async def test_fts_search_returns_metadata(self, store):
        meta = {"category": "preference", "updated_ts": 1700000000.0, "group_id": ""}
        # Note: FTS 索引文本经单字切分（与查询侧 sanitize 同粒度）。
        await store.add_document("kv:pref1", "我喜欢 深色主题", metadata=meta, chunk=False)
        results = await store._fts_search("深色主题", limit=5)
        assert len(results) == 1
        assert results[0]["metadata"] == meta

    @pytest.mark.asyncio
    async def test_fts_search_metadata_none_when_absent(self, store):
        await store.add_document("doc:no_meta", "plain content", chunk=False)
        results = await store._fts_search("plain content", limit=5)
        assert results[0]["metadata"] is None


class TestLegacyFtsMigration:
    """旧库（chunks_fts 存原始文本）打开时自动按单字切分重建。"""

    @pytest.mark.asyncio
    async def test_legacy_rows_reindexed_on_initialize(self, tmp_path):
        import sqlite3

        from src.memory.store import MemoryStore

        db_path = str(tmp_path / "legacy.db")

        # 模拟旧版写入：chunks 表 + chunks_fts 直接存原始中文文本
        conn = sqlite3.connect(db_path)
        conn.executescript(
            """
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL,
                chunk_index INTEGER NOT NULL DEFAULT 0,
                content TEXT NOT NULL,
                metadata TEXT,
                group_id TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE chunks_fts USING fts5(path, content, tokenize='unicode61');
            """
        )
        conn.execute(
            "INSERT INTO chunks (path, chunk_index, content, created_at) VALUES ('kv:old', 0, '我喜欢深色主题', '2026-01-01')"
        )
        conn.execute("INSERT INTO chunks_fts(rowid, path, content) VALUES (1, 'kv:old', '我喜欢深色主题')")
        conn.commit()
        conn.close()

        store = MemoryStore(db_path=db_path, dimensions=4)
        await store.initialize()

        # 句中词命中——旧索引下这必然搜不到（unicode61 整串一个 token）
        results = await store._fts_search("深色主题", limit=5)
        assert len(results) == 1
        assert results[0]["path"] == "kv:old"
        await store.close()

        # 二次打开不重复重建（marker 已打标）
        store2 = MemoryStore(db_path=db_path, dimensions=4)
        await store2.initialize()
        assert len(await store2._fts_search("主题", limit=5)) == 1
        await store2.close()

    @pytest.mark.asyncio
    async def test_initialize_survives_missing_fts_table(self, tmp_path):
        """存量库 + chunks_fts 缺失（FTS5 不可用）：initialize 降级不崩。

        回归守卫：旧版对 FTS 建表失败是 warning 后继续；迁移不得把这一
        降级场景变成启动硬崩。
        """
        import sqlite3

        from src.memory.store import MemoryStore

        db_path = str(tmp_path / "degraded.db")
        store = MemoryStore(db_path=db_path, dimensions=4)
        await store.initialize()
        await store.add_document("kv:a", "我喜欢深色主题", metadata={}, chunk=False)
        await store.close()

        conn = sqlite3.connect(db_path)
        conn.execute("DROP TABLE chunks_fts")
        conn.execute("DROP TABLE memory_meta")  # 抹掉完成标记，等价于从未跑过迁移
        conn.commit()
        conn.close()

        # 非法 tokenizer 模拟"FTS 建表必失败"的环境（建表失败仅 warning 后继续）
        degraded = MemoryStore(db_path=db_path, dimensions=4, fts_tokenizer="no_such_tokenizer")
        await degraded.initialize()  # 不应抛异常
        assert await degraded._fts_search("深色主题", limit=5) == []  # FTS 降级为空结果
        await degraded.close()

    @pytest.mark.asyncio
    async def test_initialize_survives_fts_write_failure(self, tmp_path):
        """存量库 + chunks_fts 表在但写入失败（半坏 FTS）：迁移降级不崩、不打标记。

        用列不匹配的 FTS 表模拟"建表成功、INSERT 失败"。
        """
        import sqlite3

        from src.memory.store import MemoryStore

        db_path = str(tmp_path / "halfbroken.db")
        conn = sqlite3.connect(db_path)
        conn.executescript(
            """
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL,
                chunk_index INTEGER NOT NULL DEFAULT 0,
                content TEXT NOT NULL,
                metadata TEXT,
                group_id TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE chunks_fts USING fts5(wrong_col);
            """
        )
        conn.execute(
            "INSERT INTO chunks (path, chunk_index, content, created_at) VALUES ('kv:old', 0, '我喜欢深色主题', '2026-01-01')"
        )
        conn.commit()
        conn.close()

        store = MemoryStore(db_path=db_path, dimensions=4)
        await store.initialize()  # 不应抛异常

        # 未打标记：下次启动自动重试迁移
        cursor = await store._conn.execute("SELECT value FROM memory_meta WHERE key = 'fts_unigram'")
        assert await cursor.fetchone() is None
        await store.close()
