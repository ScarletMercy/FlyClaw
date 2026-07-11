"""E2E test: KV → vector migration + mode=past retrieval.

完全隔离：temp dir、fake embeddings、monkeypatch get_memory_store。
不触达 prod ~/.flyclaw/data/。

验证完整链路：
  1. KV 插入混合记忆（近期 + 旧）
  2. migrate_kv_to_archive 跑迁移
  3. KV 只留保留区（近期 + top20 并集），旧记忆进 vec archive
  4. memory(action=list, mode=past) 能召回已归档旧记忆，回填 key/category/updated_at

vec 后端用 src.memory.sqlitevec_store.SqliteVecMemoryStore（生产真实后端，sqlite-vec + FTS5 hybrid）。
fake embeddings 的向量会真实写入 LanceDB，search 走 hybrid（FTS5 BM25 + 向量相似）。
本测试验证的是流程正确性，不是向量相似质量——embeddings 用固定向量，FTS 主导召回。
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from src.config import MemoryConfig, MemoryStoreConfig
from src.memory.search import MemorySearcher
from src.memory.sqlitevec_store import SqliteVecMemoryStore
from src.memory.store import MemoryStore as VecMemoryStore
from src.services.memory_archive_migration import migrate_kv_to_archive
from src.tools import memory_tools
from src.tools.memory_tools import (
    GroupMemoryStore,
    MemoryStore as KVMemoryStore,
    memory,
    reset_memory_archive_searcher,
    set_memory_session,
    set_memory_archive_searcher,
)


class FakeEmbeddings:
    """固定向量嵌入，不打真实 API。"""

    async def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    async def close(self):
        pass


@pytest.fixture
async def isolated_env(tmp_path, monkeypatch):
    """搭建隔离的 KV + vec archive 环境，yield (container, kv_store, vec_store)。"""
    # KV store (key-value memories) — 跟 prod memories.db 完全无关
    kv_store = KVMemoryStore(db_path=str(tmp_path / "kv.db"))
    await kv_store.initialize()

    # Vec archive store (chunk + FTS5 + sqlite-vec 向量,默认后端)
    vec_store = SqliteVecMemoryStore(
        db_path=str(tmp_path / "vec.db"),
        dimensions=4,
    )
    await vec_store.initialize()

    cfg = SimpleNamespace(
        memory_store=MemoryStoreConfig(
            enabled=True,
            vector_enabled=True,
            vector_model="fake-embed",
            vector_base_url="https://fake.example.com",
            vector_api_key="sk-fake",
            vector_dimensions=4,
            vector_keep_recent_n=20,
            vector_keep_recent_days=7,
        ),
        memory=MemoryConfig(),
    )

    searcher = MemorySearcher(vec_store, FakeEmbeddings(), cfg.memory)
    await set_memory_archive_searcher(searcher, "p2p")
    set_memory_session("p2p", "")

    # 关键隔离：monkeypatch get_memory_store 避免触达 prod 单例 (~/.flyclaw/data/memories.db)
    async def _fake_get_memory_store(chat_type: str = "p2p", db_path: str | None = None):
        return kv_store

    monkeypatch.setattr(memory_tools, "get_memory_store", _fake_get_memory_store)

    container = SimpleNamespace(
        config=cfg,
        memory_archive_searchers=(searcher, None),
        agent_loop=None,
    )

    yield container, kv_store, vec_store

    # 清理：reset 关 searcher + vec_store；kv_store 单独关
    await reset_memory_archive_searcher()
    try:
        await kv_store.close()
    except Exception:
        pass


async def _set_updated_ts_async(kv_store, key: str, ts: float) -> None:
    await kv_store._conn.execute("UPDATE memories SET updated_ts = ? WHERE key = ?", (ts, key))
    await kv_store._conn.commit()


class TestMigrationE2E:
    @pytest.mark.asyncio
    async def test_full_migration_flow(self, isolated_env):
        container, kv_store, vec_store = isolated_env
        now = time.time()

        # 3 条近期记忆 (updated_ts ≈ now，age ≤ 7d → 保留)
        for i in range(3):
            await kv_store.remember(
                f"近期偏好 {i}: 用 vim 编辑器写代码",
                key=f"recent_pref_{i}",
                category="preference",
            )

        # 25 条旧记忆 (updated_ts = 100 天前，age > 7d)
        # 内容含 "postgres" 关键词，供 mode=past FTS 召回
        for i in range(25):
            await kv_store.remember(
                f"旧项目笔记 {i}: 用 postgres 部署在 aws",
                key=f"old_proj_{i}",
                category="project",
            )
            await _set_updated_ts_async(kv_store, f"old_proj_{i}", now - 100 * 86400)

        # 迁移前：28 条
        all_before = await kv_store.list_all(limit=2000)
        assert len(all_before) == 28

        # 跑迁移
        result = await migrate_kv_to_archive(container)

        # 保留区：3 近期 (age≤7d) + 17 旧 (idx 3-19, top20 并集) = 20
        # 迁移：8 旧 (idx 20-27)
        assert result["dm_migrated"] == 8, f"expected 8 migrated, got {result['dm_migrated']}"
        assert result["dm_failed"] == 0
        assert result["skipped_reason"] == ""

        # KV 只剩 20 条
        all_after = await kv_store.list_all(limit=2000)
        assert len(all_after) == 20, f"expected 20 kept, got {len(all_after)}"
        # 近期 3 条全在
        kept_keys = {m["key"] for m in all_after}
        for i in range(3):
            assert f"recent_pref_{i}" in kept_keys

        # vec archive 有 8 条归档（path 前缀 kv:）
        docs = await vec_store.list_documents()
        migrated_paths = [d["path"] for d in docs if d["path"].startswith("kv:")]
        assert len(migrated_paths) == 8, f"expected 8 archived, got {len(migrated_paths)}"
        # path 命名 kv:{key}
        for p in migrated_paths:
            assert p.startswith("kv:old_proj_")

    @pytest.mark.asyncio
    async def test_mode_past_retrieves_archived(self, isolated_env):
        """迁移后 mode=past 能召回已归档旧记忆，回填 key/category/updated_at。"""
        container, kv_store, vec_store = isolated_env
        now = time.time()

        # 1 条旧记忆，内容独特
        await kv_store.remember(
            "用 redis 做缓存层部署在 aliyun",
            key="old_redis_setup",
            category="service",
        )
        await _set_updated_ts_async(kv_store, "old_redis_setup", now - 100 * 86400)
        # 再插 20 条占位旧记忆把上面那条挤出 top20 保留区
        for i in range(20):
            await kv_store.remember(
                f"占位旧记忆 {i} xxx",
                key=f"placeholder_{i}",
                category="fact",
            )
            await _set_updated_ts_async(kv_store, f"placeholder_{i}", now - 200 * 86400)
        # redis 那条 updated_ts 更新 (100天前 > 200天前)，排在 placeholder 前面
        # sorted DESC: redis (100d ago, idx 0) + 20 placeholder (200d ago, idx 1-20)
        # redis: age=100d>7d, idx=0<20 → keep? idx<20 保留 → redis 保留!
        # 这不是我们想要的——要让 redis 迁移
        # 解决：把 redis 的 updated_ts 设得更旧，让它在排序中靠后
        await _set_updated_ts_async(kv_store, "old_redis_setup", now - 300 * 86400)
        # 现在 sorted DESC: 20 placeholder (200d ago, idx 0-19) + redis (300d ago, idx 20)
        # placeholder: age=200d>7d, idx<20 → keep (20 条)
        # redis: age=300d>7d, idx=20 → migrate ✓

        await migrate_kv_to_archive(container)

        # KV 还剩 20 条 placeholder，redis 已迁出
        kv_keys = {m["key"] for m in await kv_store.list_all(limit=2000)}
        assert "old_redis_setup" not in kv_keys

        # mode=past 召回 redis（返回 [{key,source,content,category,updated_at,score}]）
        result_json = await memory(action="list", query="redis", mode="past")
        data = json.loads(result_json)
        assert isinstance(data, list)
        assert any(d["key"] == "old_redis_setup" for d in data), f"expected mode=past to find redis, got {data}"
        hit = next(d for d in data if d["key"] == "old_redis_setup")
        assert hit["category"] == "service"
        assert "redis" in hit["content"]
        assert hit["updated_at"]  # updated_ts 回填（300 天前 epoch）

        # get 直接读归档（KV 已无此键，回退 archive 取回，shape 同 KV recall）
        got = json.loads(await memory(action="get", key="old_redis_setup"))
        assert "error" not in got, f"get 应回退归档取回，got {got}"
        assert "redis" in got["content"]
        assert got["category"] == "service"
        assert got["updated_at"]  # updated_ts 回填
        assert got["key"] == "old_redis_setup"

    @pytest.mark.asyncio
    async def test_idempotent_rerun(self, isolated_env):
        """迁移幂等：重跑不重 embed、不重复删。"""
        container, kv_store, vec_store = isolated_env
        now = time.time()

        # 25 条旧记忆（全超 7d，top20 保 20，迁 5）
        for i in range(25):
            await kv_store.remember(
                f"旧事实 {i} yyy",
                key=f"old_fact_{i}",
                category="fact",
            )
            await _set_updated_ts_async(kv_store, f"old_fact_{i}", now - 100 * 86400)

        # 第一次迁移
        result1 = await migrate_kv_to_archive(container)
        assert result1["dm_migrated"] == 5

        # 第二次迁移：KV 已剩 20（全在保留区），无新可迁
        result2 = await migrate_kv_to_archive(container)
        assert result2["dm_migrated"] == 0, f"second run should migrate 0, got {result2['dm_migrated']}"

        # vec archive 仍只有 5 条（没重复 embed）
        docs = await vec_store.list_documents()
        archived = [d for d in docs if d["path"].startswith("kv:")]
        assert len(archived) == 5

    @pytest.mark.asyncio
    async def test_forget_failure_heals_on_rerun(self, isolated_env):
        """add_document 成功后 forget 抛异常 → archive 已有 doc；重跑应清理 KV 残留。

        回归 bug #7：idempotency 检查 `if path not in existing` 跳过已归档 path，
        导致 forget 失败的 KV 行永久孤立、failed 计数每周递增、永不自愈。
        """
        container, kv_store, vec_store = isolated_env
        now = time.time()
        # 24 条旧记忆（递减 ts）+ old_redis 设最老（200d）→ 排序 DESC 后 old_redis 在 idx 24，必迁移
        for i in range(24):
            await kv_store.remember(f"旧事实 {i} yyy", key=f"old_fact_{i}", category="fact")
            await _set_updated_ts_async(kv_store, f"old_fact_{i}", now - 100 * 86400 - i)
        await kv_store.remember("旧 redis 部署 aliyun", key="old_redis", category="service")
        await _set_updated_ts_async(kv_store, "old_redis", now - 200 * 86400)

        # 只让 old_redis 的 forget 抛异常（模拟 transient "database is locked"）
        async def flaky_forget(key):
            if key == "old_redis":
                raise RuntimeError("simulated database is locked")

        kv_store.forget = flaky_forget  # 实例属性遮蔽类方法

        result1 = await migrate_kv_to_archive(container)
        # 5 条进迁移集（idx 20-24），其中 old_redis 的 forget 失败
        assert result1["dm_failed"] == 1, f"expected 1 failed (old_redis forget raised), got {result1['dm_failed']}"
        assert result1["dm_migrated"] == 4, f"expected 4 migrated, got {result1['dm_migrated']}"
        # add 都成功 → archive 有 5 条（含 old_redis）
        docs1 = await vec_store.list_documents()
        archived1 = [d["path"] for d in docs1 if d["path"].startswith("kv:")]
        assert "kv:old_redis" in archived1, "add_document 应已写入 archive"
        assert len(archived1) == 5
        # KV 剩 20 保留 + old_redis 残留（forget 失败）= 21
        kv_keys = {m["key"] for m in await kv_store.list_all(limit=2000)}
        assert "old_redis" in kv_keys, "forget 失败 → KV 行应残留"

        # 恢复 forget，重跑 —— 期望清理 old_redis 残留
        del kv_store.forget  # 删实例属性，回到类方法
        result2 = await migrate_kv_to_archive(container)
        kv_keys_after = {m["key"] for m in await kv_store.list_all(limit=2000)}
        assert "old_redis" not in kv_keys_after, f"forget 失败后重跑应清理 KV 残留，但行仍孤立: {kv_keys_after}"
        assert result2["dm_failed"] == 0, f"重跑不应再计 failed，got {result2['dm_failed']}"

    @pytest.mark.asyncio
    async def test_list_all_cap_does_not_strand_oldest(self, isolated_env):
        """KV >2000 条时，最老的记忆不能因 list_all(limit=2000) 截断而永不归档。

        回归 bug #9：migration 调 list_all(limit=2000) 无分页，ORDER BY updated_ts DESC
        返回最新 2000 条，第 2001+ 的最老记忆永不被评估/迁移——正是归档要保留的。
        """
        container, kv_store, vec_store = isolated_env
        from src.utils.tz import now_iso

        now = time.time()
        iso = now_iso()
        ts_old = now - 300 * 86400
        # 2000 条近期（age=0，保留区）+ 1 条最老（age=300d，应迁移）
        rows = [(f"recent_{i}", f"近期 {i}", "fact", iso, iso, now) for i in range(2000)]
        rows.append(("oldest_stranded", "最老的旧记忆 postgres 部署", "project", iso, iso, ts_old))
        await kv_store._conn.executemany(
            "INSERT INTO memories (key, content, category, created_at, updated_at, updated_ts) VALUES (?,?,?,?,?,?)",
            rows,
        )
        await kv_store._conn.commit()
        # 确认 list_all(limit=2000) 截断了第 2001 条
        assert len(await kv_store.list_all(limit=2000)) == 2000

        await migrate_kv_to_archive(container)

        docs = await vec_store.list_documents()
        archived = [d["path"] for d in docs if d["path"].startswith("kv:")]
        assert "kv:oldest_stranded" in archived, (
            f"最老记忆应被迁移到 archive，但 list_all(limit=2000) 截断致其未被评估。archived 共 {len(archived)} 条"
        )

    @pytest.mark.asyncio
    async def test_embed_failure_does_not_abandon_whole_batch(self, isolated_env):
        """embed_texts 批量失败时不应放弃整批 —— archive 仍应有 FTS5 文档可检索。

        回归 bug #6：embed_texts 抛错（如长内容触发 400）→ return {migrated:0, failed:len(pending)}，
        整批放弃且下周重跑再失败，单条超长 KV 记忆毒翻整个 store 的归档排水。
        """
        container, kv_store, vec_store = isolated_env
        from src.config import MemoryConfig
        from src.memory.search import MemorySearcher
        from src.services.memory_archive_migration import _migrate_one_store

        class FailingEmbeddings:
            async def embed_texts(self, texts):
                raise RuntimeError("simulated 400: token limit exceeded")

            async def embed_query(self, t):
                raise RuntimeError("fail")

            async def close(self):
                pass

        failing_searcher = MemorySearcher(vec_store, FailingEmbeddings(), MemoryConfig())

        now = time.time()
        for i in range(25):
            await kv_store.remember(f"旧记忆 {i} postgres 部署", key=f"old_{i}", category="project")
            await _set_updated_ts_async(kv_store, f"old_{i}", now - 100 * 86400 - i)

        result = await _migrate_one_store(
            kv_store=kv_store,
            archive_searcher=failing_searcher,
            now=now,
            keep_n=20,
            keep_days=7,
            group_id="",
        )
        # 当前 bug：embed 失败 → return {migrated:0, failed:5}，archive 空
        # 期望：FTS5 文档仍应存入 archive（不依赖向量），可被 FTS 检索
        docs = await vec_store.list_documents()
        archived = [d for d in docs if d["path"].startswith("kv:")]
        assert len(archived) > 0, (
            f"embed 失败不应放弃整批 —— FTS5 文档仍应存入 archive，got archived={len(archived)}, result={result}"
        )

    @pytest.mark.asyncio
    async def test_migration_with_group_searcher_none_does_not_crash(self, tmp_path, monkeypatch):
        """group_searcher=None（_setup_memory_archive group 失败）时 migration 不崩、DM 仍迁移、群保留。

        回归 #3 副作用：_setup_memory_archive group 失败后 memory_archive_searchers=(dm, None)，
        migrate_kv_to_archive 群部分若不检查 group_searcher 直接用，会 None.store 崩。
        """
        dm_kv = KVMemoryStore(db_path=str(tmp_path / "kv_dm.db"))
        await dm_kv.initialize()
        group_kv = GroupMemoryStore(db_path=str(tmp_path / "kv_group.db"))
        await group_kv.initialize()

        dm_vec = SqliteVecMemoryStore(db_path=str(tmp_path / "dm_vec.db"), dimensions=4)
        await dm_vec.initialize()

        cfg = SimpleNamespace(
            memory_store=MemoryStoreConfig(
                enabled=True,
                vector_enabled=True,
                vector_model="fake",
                vector_base_url="x",
                vector_api_key="x",
                vector_dimensions=4,
                vector_keep_recent_n=20,
                vector_keep_recent_days=7,
            ),
            memory=MemoryConfig(),
        )
        dm_searcher = MemorySearcher(dm_vec, FakeEmbeddings(), cfg.memory)
        await reset_memory_archive_searcher()
        await set_memory_archive_searcher(dm_searcher, "p2p")  # 只注册 DM，group=None

        async def _fake_get(chat_type="p2p", db_path=None):
            return group_kv if chat_type == "group" else dm_kv

        monkeypatch.setattr(memory_tools, "get_memory_store", _fake_get)

        container = SimpleNamespace(
            config=cfg,
            memory_archive_searchers=(dm_searcher, None),  # group_searcher=None
            agent_loop=None,
        )

        now = time.time()
        for i in range(25):
            await dm_kv.remember(f"旧DM {i} postgres", key=f"dm_old_{i}", category="project")
            await _set_updated_ts_async(dm_kv, f"dm_old_{i}", now - 100 * 86400 - i)
            await group_kv.remember(f"旧群 {i} postgres", key=f"grp_old_{i}", category="project", group_id="groupA")
            await group_kv._conn.execute(
                "UPDATE memories SET updated_ts = ? WHERE key = ? AND group_id = ?",
                (now - 100 * 86400 - i, f"grp_old_{i}", "groupA"),
            )
        await group_kv._conn.commit()

        # 不应崩
        result = await migrate_kv_to_archive(container)
        # DM 迁移正常（25 旧 - 20 top20 = 5）
        assert result["dm_migrated"] == 5, f"DM 应迁移 5 条, got {result['dm_migrated']}"
        # 群记忆全部保留（group_searcher=None，群迁移 skip）
        group_remaining = await group_kv.list_all(limit=2000, group_id="groupA")
        assert len(group_remaining) == 25, f"群记忆应全部保留, got {len(group_remaining)}"

        await reset_memory_archive_searcher()

    @pytest.mark.asyncio
    async def test_add_embeddings_failure_cleans_half_baked(self, isolated_env):
        """add_embeddings 失败时清半成品 chunk，避免下轮 pending_forget 删 KV 留孤儿。

        回归 bug #1：add_document 已 commit，add_embeddings 抛异常 → chunk 留库无向量。
        下轮 list_documents（只查 SQLite）把该 path 当完整归档 → pending_forget 删 KV → 孤儿永久留库。
        修法：except 里 delete_document(path) 清半成品，下轮重新 pending_add 重试。
        """
        container, kv_store, vec_store = isolated_env
        now = time.time()

        # 25 条旧记忆（25 旧 - 20 top20 = 5 迁移：old_20..old_24）
        for i in range(25):
            await kv_store.remember(f"旧事实 {i} postgres 部署", key=f"old_{i}", category="project")
            await _set_updated_ts_async(kv_store, f"old_{i}", now - 100 * 86400 - i)

        original_add_embeddings = vec_store.add_embeddings

        async def flaky_add_embeddings(chunk_ids, embeddings):
            for cid in chunk_ids:
                cursor = await vec_store._conn.execute("SELECT path FROM chunks WHERE id = ?", (cid,))
                row = await cursor.fetchone()
                if row and row["path"] == "kv:old_20":
                    raise RuntimeError("simulated lancedb write failure")
            return await original_add_embeddings(chunk_ids, embeddings)

        vec_store.add_embeddings = flaky_add_embeddings

        result = await migrate_kv_to_archive(container)
        assert result["dm_failed"] == 1, f"expected 1 failed, got {result['dm_failed']}"

        # 关键：old_20 半成品已清（list_documents 不含）
        docs = await vec_store.list_documents()
        archived_paths = [d["path"] for d in docs if d["path"].startswith("kv:")]
        assert "kv:old_20" not in archived_paths, (
            f"add_embeddings 失败应清半成品 chunk，但 kv:old_20 仍在 archive: {archived_paths}"
        )
        assert len(archived_paths) == 4, f"其余 4 条应正常归档, got {len(archived_paths)}"

        # KV 保留 old_20（forget 未执行）
        kv_keys = {m["key"] for m in await kv_store.list_all(limit=2000)}
        assert "old_20" in kv_keys, "add_embeddings 失败 → KV 应保留 old_20"

        # 恢复 add_embeddings，重跑 → old_20 成功迁移
        del vec_store.add_embeddings
        result2 = await migrate_kv_to_archive(container)
        docs2 = await vec_store.list_documents()
        archived2 = [d["path"] for d in docs2 if d["path"].startswith("kv:")]
        assert "kv:old_20" in archived2, f"重跑应成功迁移 old_20, got {archived2}"
        kv_keys2 = {m["key"] for m in await kv_store.list_all(limit=2000)}
        assert "old_20" not in kv_keys2, "重跑后 old_20 应从 KV 删除"


class TestMigrationE2EGroup:
    """Group 迁移：GroupMemoryStore 多群，每群独立算保留集，迁到 group vec store。"""

    @pytest.mark.asyncio
    async def test_group_migration_per_group(self, tmp_path, monkeypatch):
        group_kv = GroupMemoryStore(db_path=str(tmp_path / "kv_group.db"))
        await group_kv.initialize()

        group_vec_store = SqliteVecMemoryStore(
            db_path=str(tmp_path / "vec_group.db"),
            dimensions=4,
        )
        await group_vec_store.initialize()

        # 空 DM store（migration 先跑 DM，用空 DM 避免干扰）
        dm_kv = KVMemoryStore(db_path=str(tmp_path / "kv_dm.db"))
        await dm_kv.initialize()

        cfg = SimpleNamespace(
            memory_store=MemoryStoreConfig(
                enabled=True,
                vector_enabled=True,
                vector_model="fake-embed",
                vector_base_url="https://fake.example.com",
                vector_api_key="sk-fake",
                vector_dimensions=4,
                vector_keep_recent_n=20,
                vector_keep_recent_days=7,
            ),
            memory=MemoryConfig(),
        )

        group_searcher = MemorySearcher(group_vec_store, FakeEmbeddings(), cfg.memory)
        dm_searcher = MemorySearcher(
            SqliteVecMemoryStore(
                db_path=str(tmp_path / "dm_vec.db"),
                dimensions=4,
            ),
            FakeEmbeddings(),
            cfg.memory,
        )
        await dm_searcher.store.initialize()

        await set_memory_archive_searcher(dm_searcher, "p2p")
        await set_memory_archive_searcher(group_searcher, "group")
        set_memory_session("group", "groupA")

        async def _fake_get_memory_store(chat_type: str = "p2p", db_path: str | None = None):
            return group_kv if chat_type == "group" else dm_kv

        monkeypatch.setattr(memory_tools, "get_memory_store", _fake_get_memory_store)

        container = SimpleNamespace(
            config=cfg,
            memory_archive_searchers=(dm_searcher, group_searcher),
            agent_loop=None,
        )

        now = time.time()
        # 2 个 group，各 3 近期 + 25 旧
        for gid in ("groupA", "groupB"):
            for i in range(3):
                await group_kv.remember(f"近期 {i}", key=f"r{i}", category="fact", group_id=gid)
            for i in range(25):
                await group_kv.remember(f"旧 {i} postgres", key=f"o{i}", category="project", group_id=gid)
                await group_kv._conn.execute(
                    "UPDATE memories SET updated_ts = ? WHERE key = ? AND group_id = ?",
                    (now - 100 * 86400, f"o{i}", gid),
                )
        await group_kv._conn.commit()

        result = await migrate_kv_to_archive(container)

        # 每群迁 8 条（25 旧 - 17 top20 = 8），共 16
        total_group_migrated = sum(g["migrated"] for g in result["groups"])
        assert total_group_migrated == 16, f"expected 16, got {total_group_migrated}"
        assert len(result["groups"]) == 2

        # group vec store 有 16 条归档（path kv:g:）
        docs = await group_vec_store.list_documents()
        archived = [d for d in docs if d["path"].startswith("kv:g:")]
        assert len(archived) == 16

        # 每群 KV 剩 20
        for gid in ("groupA", "groupB"):
            remaining = await group_kv.list_all(limit=2000, group_id=gid)
            assert len(remaining) == 20, f"group {gid}: expected 20, got {len(remaining)}"

        await reset_memory_archive_searcher()


# ── FTS5-only archive（vector_enabled=False）────────────────────────────


@pytest.fixture
async def isolated_env_fts5_only(tmp_path, monkeypatch):
    """vector_enabled=False 的隔离环境：archive 用 MemoryStore（FTS5-only），无 embeddings。

    验证 archive 不依赖 vector——FTS5-only 也能迁移 + 检索。
    """
    kv_store = KVMemoryStore(db_path=str(tmp_path / "kv_fts5.db"))
    await kv_store.initialize()

    # archive store = MemoryStore（FTS5-only，无向量后端），不用 LanceMemoryStore
    archive_store = VecMemoryStore(
        db_path=str(tmp_path / "archive_fts5.db"),
        dimensions=4,
    )
    await archive_store.initialize()

    cfg = SimpleNamespace(
        memory_store=MemoryStoreConfig(
            enabled=True,
            vector_enabled=False,  # 关键：vector off
            vector_keep_recent_n=20,
            vector_keep_recent_days=7,
        ),
        memory=MemoryConfig(),
    )

    # embeddings=None —— vector off 不嵌入
    searcher = MemorySearcher(archive_store, None, cfg.memory)
    await set_memory_archive_searcher(searcher, "p2p")
    set_memory_session("p2p", "")

    async def _fake_get_memory_store(chat_type: str = "p2p", db_path: str | None = None):
        return kv_store

    monkeypatch.setattr(memory_tools, "get_memory_store", _fake_get_memory_store)

    container = SimpleNamespace(
        config=cfg,
        memory_archive_searchers=(searcher, None),
        agent_loop=None,
    )

    yield container, kv_store, archive_store

    await reset_memory_archive_searcher()
    try:
        await kv_store.close()
    except Exception:
        pass


class TestMigrationE2EFTS5Only:
    """vector_enabled=False：archive 走 FTS5-only，不 embed，仍能迁移 + 检索。"""

    @pytest.mark.asyncio
    async def test_fts5_only_migration_no_embed(self, isolated_env_fts5_only):
        """vector off → 迁移不 embed，旧记忆从 KV 移到 archive（FTS5 索引）。"""
        container, kv_store, archive_store = isolated_env_fts5_only
        now = time.time()

        # 3 近期 + 25 旧
        for i in range(3):
            await kv_store.remember(f"近期 {i} vim 编辑器", key=f"r{i}", category="preference")
        for i in range(25):
            await kv_store.remember(f"旧项目 {i} postgres 部署", key=f"o{i}", category="project")
            await _set_updated_ts_async(kv_store, f"o{i}", now - 100 * 86400)

        assert len(await kv_store.list_all(limit=2000)) == 28

        result = await migrate_kv_to_archive(container)

        # 25 旧 - 17 top20 = 8 迁移，无 embed 失败
        assert result["dm_migrated"] == 8, f"expected 8, got {result['dm_migrated']}"
        assert result["dm_failed"] == 0

        # KV 剩 20
        assert len(await kv_store.list_all(limit=2000)) == 20

        # archive 有 8 条（FTS5 索引，无向量）
        docs = await archive_store.list_documents()
        archived = [d for d in docs if d["path"].startswith("kv:")]
        assert len(archived) == 8

        # 确认 archive store 没有 vector 支持（FTS5-only）
        assert archive_store._has_vector_support() is False

    @pytest.mark.asyncio
    async def test_fts5_only_mode_past_retrieves(self, isolated_env_fts5_only):
        """vector off → mode=past 走 FTS5 召回已归档记忆。"""
        container, kv_store, archive_store = isolated_env_fts5_only
        now = time.time()

        # 1 条旧记忆 + 20 占位旧记忆（把目标挤出 top20）
        await kv_store.remember("用 redis 做缓存层 aliyun", key="old_redis", category="service")
        await _set_updated_ts_async(kv_store, "old_redis", now - 300 * 86400)
        for i in range(20):
            await kv_store.remember(f"占位 {i} xxx", key=f"p{i}", category="fact")
            await _set_updated_ts_async(kv_store, f"p{i}", now - 200 * 86400)

        await migrate_kv_to_archive(container)

        # redis 已迁出 KV
        kv_keys = {m["key"] for m in await kv_store.list_all(limit=2000)}
        assert "old_redis" not in kv_keys

        # mode=past FTS5 召回 redis（返回 [{key,source,content,category,updated_at,score}]）
        result_json = await memory(action="list", query="redis", mode="past")
        data = json.loads(result_json)
        assert isinstance(data, list)
        assert any(d["key"] == "old_redis" for d in data), f"FTS5 should find redis, got {data}"
        hit = next(d for d in data if d["key"] == "old_redis")
        assert hit["category"] == "service"
        assert "redis" in hit["content"]

        # get 直接读 FTS5-only 归档（KV 已无此键）
        got = json.loads(await memory(action="get", key="old_redis"))
        assert "error" not in got, f"get 应回退 FTS5 归档取回，got {got}"
        assert "redis" in got["content"]
        assert got["category"] == "service"
