"""Tests for memory(action=list, mode=) dispatch."""

from __future__ import annotations

import json

import pytest

from src.tools import memory_tools
from src.tools.memory_tools import memory, set_memory_session, set_memory_archive_searcher


class FakeKVStore:
    async def list_all(self, query="", limit=200, group_id=None):
        # "部署" 这个 query KV 里没有 → 用于测 auto-fallback
        if "部署" in query:
            return []
        return [{"key": "k1", "content": "邮箱是 a@b.com", "category": "contact", "updated_at": "2026-01-01"}]

    async def recall(self, key, group_id=""):
        return json.dumps({"error": "not found"})

    async def forget(self, key, group_id=""):
        return json.dumps({"ok": True})

    async def remember(self, content, key="", category="fact", group_id=""):
        return json.dumps({"ok": True, "key": key})


class FakeVecSearcher:
    async def search(self, query, max_results=6, min_score=None, group_id=None):
        return [
            {
                "content": "旧记忆：用 postgres 部署",
                "path": "kv:deploy_pg",
                "score": 0.8,
                "chunk_index": 0,
                "metadata": {"category": "project", "updated_ts": 1600000000.0, "group_id": ""},
            }
        ]


class FakeVecSearcherMultiGroup:
    """返回 DM + 多 group 混合结果，验证 group 前缀过滤。"""

    async def search(self, query, max_results=6, min_score=None, group_id=None):
        all_docs = [
            {
                "content": "群A 旧记忆 redis",
                "path": "kv:g:groupA:k1",
                "score": 0.9,
                "metadata": {"category": "service", "updated_ts": 1.0, "group_id": "groupA"},
            },
            {
                "content": "群B 旧记忆 mysql",
                "path": "kv:g:groupB:k2",
                "score": 0.85,
                "metadata": {"category": "service", "updated_ts": 2.0, "group_id": "groupB"},
            },
            {
                "content": "群A 另一条",
                "path": "kv:g:groupA:k3",
                "score": 0.8,
                "metadata": {"category": "fact", "updated_ts": 3.0, "group_id": "groupA"},
            },
            {
                "content": "DM 旧记忆",
                "path": "kv:dm_k1",
                "score": 0.7,
                "metadata": {"category": "fact", "updated_ts": 4.0, "group_id": ""},
            },
        ]
        if group_id is not None:
            all_docs = [d for d in all_docs if d["metadata"].get("group_id") == group_id]
        return all_docs


async def _async_return(val):
    return val


@pytest.fixture
async def patch_stores(monkeypatch):
    monkeypatch.setattr(memory_tools, "get_memory_store", lambda chat_type="p2p": _async_return(FakeKVStore()))
    set_memory_session("p2p", "")
    yield
    await memory_tools.reset_memory_archive_searcher()


class TestMemoryMode:
    @pytest.mark.asyncio
    async def test_recent_mode_hits_kv(self, patch_stores):
        result = await memory(action="list", query="邮箱", mode="recent", verbose=True)
        data = json.loads(result)
        assert isinstance(data, list)
        assert data[0]["key"] == "k1"

    @pytest.mark.asyncio
    async def test_recent_auto_fallback_to_past(self, patch_stores):
        """mode=recent 查 KV 空 → 自动 fallback 到 archive（past），返回 archive 结果。"""
        await set_memory_archive_searcher(FakeVecSearcher(), "p2p")
        # query="部署" → FakeKVStore.list_all 返回 [] → fallback 到 archive
        result = await memory(action="list", query="部署", mode="recent")
        data = json.loads(result)
        # verbose=False（默认）fallback 也应返回键名字符串数组（与 KV 命中一致）
        assert data == ["deploy_pg"], f"expected str list, got {data}"

    @pytest.mark.asyncio
    async def test_recent_fallback_verbose_false_returns_str_list(self, patch_stores):
        """verbose=False 时 fallback 到 archive 的返回类型必须与 KV 命中一致（list[str]）。

        回归 bug #4：KV 空命中 fallback 到 _list_past 返回对象数组 [{key,content,...}]，
        而 KV 命中时 verbose=False 返回字符串数组 ["k1"]。同一调用、同一 verbose=False，
        返回类型不应随 KV 是否命中而变（docstring 承诺"默认只返回键名"）。
        """
        await set_memory_archive_searcher(FakeVecSearcher(), "p2p")
        # query="部署" → FakeKVStore.list_all 返回 [] → fallback 到 archive
        result = await memory(action="list", query="部署", mode="recent", verbose=False)
        data = json.loads(result)
        assert isinstance(data, list)
        assert all(isinstance(x, str) for x in data), (
            f"verbose=False 应返回键名字符串数组（与 KV 命中一致），got: {data}"
        )
        assert "deploy_pg" in data

    @pytest.mark.asyncio
    async def test_recent_no_fallback_when_kv_has_match(self, patch_stores):
        """KV 有命中时不 fallback 到 archive。"""
        await set_memory_archive_searcher(FakeVecSearcher(), "p2p")
        # query="邮箱" → KV 有 k1 → 不 fallback
        result = await memory(action="list", query="邮箱", mode="recent", verbose=True)
        data = json.loads(result)
        assert data[0]["key"] == "k1"  # KV 结果，不是 archive 的 deploy_pg

    @pytest.mark.asyncio
    async def test_recent_kv_empty_archive_unenabled_returns_empty(self, patch_stores):
        """KV 空 + archive 未启用 → 返回 []，不返回 error（保持 recent 行为）。

        回归门：archive 未启用时 auto-fallback 不该报错，否则破坏无 archive 用户。
        """
        # 不注册 archive searcher → archive 未启用
        # query="部署" → FakeKVStore 返回 [] → fallback 检查 archive → None → 返回 []
        result = await memory(action="list", query="部署", mode="recent")
        data = json.loads(result)
        assert data == [], f"expected [] when archive unenabled, got {data}"

    @pytest.mark.asyncio
    async def test_recent_empty_query_no_fallback(self, patch_stores):
        """空 query + KV 空 → 不 fallback（recent 空查询返回空列表，不查 archive）。"""
        await set_memory_archive_searcher(FakeVecSearcher(), "p2p")
        # FakeKVStore.list_all("") 返回 k1（非空）→ 不触发 fallback 条件
        # 但若 KV 真空 + 空查询 → 不该 fallback
        result = await memory(action="list", query="", mode="recent")
        data = json.loads(result)
        # KV 有 k1 → 返回键名列表
        assert data == ["k1"]

    @pytest.mark.asyncio
    async def test_past_mode_hits_vector(self, patch_stores):
        await set_memory_archive_searcher(FakeVecSearcher(), "p2p")
        result = await memory(action="list", query="部署", mode="past", verbose=True)
        data = json.loads(result)
        assert len(data) == 1
        assert data[0]["key"] == "deploy_pg"
        assert data[0]["category"] == "project"

    @pytest.mark.asyncio
    async def test_past_mode_unenabled_returns_error(self, patch_stores):
        result = await memory(action="list", query="部署", mode="past")
        data = json.loads(result)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_group_scope_empty_group_id_blocks(self, patch_stores):
        await set_memory_archive_searcher(FakeVecSearcher(), "group")
        set_memory_session("group", "")
        result = await memory(action="list", query="部署", mode="past")
        data = json.loads(result)
        assert "error" in data
        assert "group_id" in data["error"]


class TestMemoryModeGroupScope:
    @pytest.mark.asyncio
    async def test_group_scope_filters_by_group_id(self, patch_stores):
        """group scope + 有效 group_id → 按 kv:g:{group_id}: 前缀过滤，只返回该群归档。"""
        await set_memory_archive_searcher(FakeVecSearcherMultiGroup(), "group")
        set_memory_session("group", "groupA")
        result = await memory(action="list", query="redis", mode="past", verbose=True)
        data = json.loads(result)
        # 只返回 groupA 的 2 条（k1, k3），过滤掉 groupB 和 DM
        assert len(data) == 2
        keys = {d["key"] for d in data}
        assert keys == {"k1", "k3"}
        # category 回填
        cats = {d["category"] for d in data}
        assert cats == {"service", "fact"}

    @pytest.mark.asyncio
    async def test_group_scope_colon_in_group_id_key_recovery(self, patch_stores):
        """group_id 带冒号（QQ 实际格式 f"group:{openid}"，见 channels/qq.py:786）时，
        _list_past 反解 path 必须还原原 key。

        回归 bug #1：path="kv:g:group:ABC:note" 经 split(":",3) 得
        ["kv","g","group","ABC:note"] → rkey="ABC:note"（应为 "note"）。
        每个群/频道归档查询都触发。前缀过滤（startswith）没问题，坏的是 key 反解。
        """

        class ColonGidSearcher:
            async def search(self, query, max_results=6, min_score=None, group_id=None):
                return [
                    {
                        "content": "群归档记忆 redis 部署",
                        "path": "kv:g:group:ABC:note",
                        "score": 0.9,
                        "metadata": {"category": "fact", "updated_ts": 1.0, "group_id": "group:ABC"},
                    }
                ]

        await set_memory_archive_searcher(ColonGidSearcher(), "group")
        set_memory_session("group", "group:ABC")  # 真实 QQ 格式：带冒号
        result = await memory(action="list", query="redis", mode="past", verbose=True)
        data = json.loads(result)
        assert len(data) == 1, f"前缀过滤应命中，got {data}"
        assert data[0]["key"] == "note", f"group_id 带冒号时 key 反解错误: got {data[0]['key']!r}, expected 'note'"

    @pytest.mark.asyncio
    async def test_group_scope_overfetch_truncates_to_6(self, patch_stores):
        """超过 6 条结果时截断到 6。"""

        class ManyResultsSearcher:
            async def search(self, query, max_results=6, min_score=None, group_id=None):
                return [
                    {
                        "content": f"群A 记忆 {i}",
                        "path": f"kv:g:groupA:k{i}",
                        "score": 0.9 - i * 0.01,
                        "metadata": {"category": "fact", "updated_ts": float(i), "group_id": "groupA"},
                    }
                    for i in range(10)
                ]

        await set_memory_archive_searcher(ManyResultsSearcher(), "group")
        set_memory_session("group", "groupA")
        result = await memory(action="list", query="记忆", mode="past")
        data = json.loads(result)
        assert len(data) == 6  # 截断到 6

    @pytest.mark.asyncio
    async def test_group_past_recall_when_target_below_global_top(self, patch_stores):
        """群归档是共享 store；目标群匹配掉出全局 top-20 时不应返回 0。

        回归 bug #8：_list_past 用 max_results=20 全局 cap，再按 kv:g:{gid}: 前缀过滤。
        其他群分数高占满 top-20 → 目标群 0 结果，尽管归档里有匹配。
        """

        class CappedSearcher:
            """模拟真实 store 的 max_results cap：按分数降序取前 max_results 条。"""

            def __init__(self):
                # 24 条其他群（高分）+ 1 条目标群（低分，掉出 top-20）
                self.all = (
                    [
                        {
                            "content": f"其他A{i}",
                            "path": f"kv:g:otherA:k{i}",
                            "score": 0.95 - i * 0.001,
                            "metadata": {"category": "fact", "updated_ts": float(i), "group_id": "otherA"},
                        }
                        for i in range(12)
                    ]
                    + [
                        {
                            "content": f"其他B{i}",
                            "path": f"kv:g:otherB:k{i}",
                            "score": 0.90 - i * 0.001,
                            "metadata": {"category": "fact", "updated_ts": float(i), "group_id": "otherB"},
                        }
                        for i in range(12)
                    ]
                    + [
                        {
                            "content": "目标群匹配",
                            "path": "kv:g:targetA:want",
                            "score": 0.5,
                            "metadata": {"category": "fact", "updated_ts": 1.0, "group_id": "targetA"},
                        }
                    ]
                )

            async def search(self, query, max_results=6, min_score=None, group_id=None):
                docs = self.all
                if group_id is not None:
                    docs = [d for d in docs if d["metadata"].get("group_id") == group_id]
                ranked = sorted(docs, key=lambda x: -x["score"])
                return ranked[:max_results]

        await set_memory_archive_searcher(CappedSearcher(), "group")
        set_memory_session("group", "targetA")
        result = await memory(action="list", query="匹配", mode="past", verbose=True)
        data = json.loads(result)
        target_hits = [d for d in data if d["key"] == "want"]
        assert len(target_hits) == 1, f"目标群匹配掉出全局 top-20 → 0 结果，但 archive 里有匹配: {data}"

    @pytest.mark.asyncio
    async def test_dm_scope_isolated_from_group_data(self, tmp_path):
        """DM scope 用独立 DM vec store，搜不到 group 数据（真实文件级隔离，非 mock）。

        之前用 mock 返回混合数据是循环论证。这里用真 LanceMemoryStore：DM store
        只放 DM 数据，group store 只放 group 数据，验证 DM scope 搜 group-only
        关键词返回空，反之亦然。
        """
        from src.config import MemoryConfig
        from src.memory.lance_store import LanceMemoryStore
        from src.memory.search import MemorySearcher

        class _FakeEmb:
            async def embed_query(self, t):
                return [0.1, 0.2, 0.3, 0.4]

            async def embed_texts(self, ts):
                return [[0.1, 0.2, 0.3, 0.4] for _ in ts]

        # DM vec store —— 只有 DM 数据（关键词 redis）
        dm_store = LanceMemoryStore(
            db_path=str(tmp_path / "dm.db"), dimensions=4, lancedb_uri=str(tmp_path / "dm.lance")
        )
        await dm_store.initialize()
        await dm_store.add_document(
            "kv:dm_redis",
            "DM 记忆 用 redis 做缓存",
            metadata={"category": "service", "updated_ts": 1.0, "group_id": ""},
            chunk=False,
        )

        # Group vec store —— 只有 group 数据（关键词 mysql）
        group_store = LanceMemoryStore(
            db_path=str(tmp_path / "grp.db"), dimensions=4, lancedb_uri=str(tmp_path / "grp.lance")
        )
        await group_store.initialize()
        await group_store.add_document(
            "kv:g:groupA:mysql_secret",
            "group 私密 用 mysql 存用户",
            metadata={"category": "service", "updated_ts": 2.0, "group_id": "groupA"},
            chunk=False,
        )

        dm_searcher = MemorySearcher(dm_store, _FakeEmb(), MemoryConfig())
        group_searcher = MemorySearcher(group_store, _FakeEmb(), MemoryConfig())
        await set_memory_archive_searcher(dm_searcher, "p2p")
        await set_memory_archive_searcher(group_searcher, "group")

        try:
            # DM scope 搜 "mysql"（只存在于 group store）→ 应返回空
            set_memory_session("p2p", "")
            result_dm_mysql = json.loads(await memory(action="list", query="mysql", mode="past", verbose=True))
            assert result_dm_mysql == [], f"DM scope 不应返回 group 数据，got {result_dm_mysql}"

            # DM scope 搜 "redis"（DM store 有）→ 返回 DM 数据
            result_dm_redis = json.loads(await memory(action="list", query="redis", mode="past", verbose=True))
            assert len(result_dm_redis) == 1
            assert result_dm_redis[0]["key"] == "dm_redis"

            # Group scope 搜 "mysql" → 返回 group 数据（走 group searcher）
            set_memory_session("group", "groupA")
            result_grp_mysql = json.loads(await memory(action="list", query="mysql", mode="past", verbose=True))
            assert len(result_grp_mysql) == 1
            assert result_grp_mysql[0]["key"] == "mysql_secret"

            # Group scope 搜 "redis"（只存在于 DM store）→ 应返回空
            result_grp_redis = json.loads(await memory(action="list", query="redis", mode="past", verbose=True))
            assert result_grp_redis == [], f"group scope 不应返回 DM 数据，got {result_grp_redis}"
        finally:
            await memory_tools.reset_memory_archive_searcher()
