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
    async def search(self, query, max_results=6, group_id=None):
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

    async def search(self, query, max_results=6, group_id=None):
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


class FakeArchiveStore:
    """归档 store mock：只实现 get_document_by_path（按 path 返 doc 或 None），记录查询。"""

    def __init__(self, by_path: dict):
        self._by_path = by_path
        self.calls: list[str] = []

    async def get_document_by_path(self, path: str):
        self.calls.append(path)
        return self._by_path.get(path)


class FakeArchiveSearcherWithStore:
    """带 .store 的归档 searcher mock（真 MemorySearcher 有 .store）。"""

    def __init__(self, store):
        self.store = store


@pytest.fixture
async def patch_stores(monkeypatch):
    monkeypatch.setattr(memory_tools, "get_memory_store", lambda chat_type="p2p": _async_return(FakeKVStore()))
    set_memory_session("p2p", "")
    # 显式锁死降级:测试环境无主模型 container -> mode=past 走降级返候选 JSON 列表。
    # 不靠「环境恰好没初始化 container」的副作用, 避免环境一变隐式 past 测试误挂。
    # 专门测 LLM 提取/降级分支的用例会自行 monkeypatch 覆盖此项。
    import src._container as _c

    def _raise_runtime():
        raise RuntimeError("container not initialized")

    monkeypatch.setattr(_c, "get_container", _raise_runtime)
    yield
    await memory_tools.reset_memory_archive_searcher()


class TestMemoryMode:
    @pytest.mark.asyncio
    async def test_recent_mode_hits_kv(self, patch_stores):
        result = await memory(action="list", query="邮箱", mode="recent")
        data = json.loads(result)
        assert isinstance(data, list)
        assert data == ["k1"]

    @pytest.mark.asyncio
    async def test_recent_auto_fallback_to_past(self, patch_stores):
        """mode=recent 查 KV 空 → 自动 fallback 到 archive（past），返回 archive 结果。"""
        await set_memory_archive_searcher(FakeVecSearcher(), "p2p")
        # query="部署" → FakeKVStore.list_all 返回 [] → fallback 到 archive
        result = await memory(action="list", query="部署", mode="recent")
        data = json.loads(result)
        # fallback 也返回键名字符串数组（与 KV 命中一致）
        assert data == ["deploy_pg"], f"expected str list, got {data}"

    @pytest.mark.asyncio
    async def test_recent_no_fallback_when_kv_has_match(self, patch_stores):
        """KV 有命中时不 fallback 到 archive。"""
        await set_memory_archive_searcher(FakeVecSearcher(), "p2p")
        # query="邮箱" → KV 有 k1 → 不 fallback
        result = await memory(action="list", query="邮箱", mode="recent")
        data = json.loads(result)
        assert data == ["k1"]  # KV 结果，不是 archive 的 deploy_pg

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
        result = await memory(action="list", query="部署", mode="past")
        data = json.loads(result)
        assert len(data) == 1
        assert data[0]["key"] == "deploy_pg"
        assert data[0]["score"] == 0.8
        assert data[0]["category"] == "project"

    @pytest.mark.asyncio
    async def test_past_with_client_returns_llm_extraction(self, patch_stores, monkeypatch):
        """有主模型 client 时，mode=past 走 LLM 提取，返回自然语言文（而非 JSON 列表）。"""
        await set_memory_archive_searcher(FakeVecSearcher(), "p2p")

        class FakeClient:
            async def chat_simple(self, messages, **extra):
                # prompt 应含 query 和候选内容
                assert "部署" in messages[0]["content"]
                assert "deploy_pg" in messages[0]["content"]
                return "[deploy_pg] 用 postgres 部署"

        class FakeAgentLoop:
            _client = FakeClient()

        class FakeContainer:
            agent_loop = FakeAgentLoop()

        import src._container as _c

        monkeypatch.setattr(_c, "get_container", lambda: FakeContainer())

        result = await memory(action="list", query="部署", mode="past")
        assert result == "[deploy_pg] 用 postgres 部署"

    @pytest.mark.asyncio
    async def test_past_without_client_degrades_to_candidate_list(self, patch_stores, monkeypatch):
        """无主模型 client(get_container 未初始化)时,past 降级返回候选 JSON 列表,而非 LLM 提取文。

        显式 monkeypatch 锁死降级路径,不依赖测试环境副作用--若环境意外初始化了 container,
        隐式走降级的测试会失效,此测试仍能守住「无 client -> 候选列表」契约。
        """
        await set_memory_archive_searcher(FakeVecSearcher(), "p2p")

        def _raise_runtime():
            raise RuntimeError("container not initialized")

        import src._container as _c

        monkeypatch.setattr(_c, "get_container", _raise_runtime)
        result = await memory(action="list", query="部署", mode="past")
        data = json.loads(result)
        # 降级返回候选列表 [{key, source, content, score, category, updated_at}]
        assert isinstance(data, list), f"降级应返回候选 JSON 列表, got {result!r}"
        assert data[0]["key"] == "deploy_pg"
        assert data[0]["score"] == 0.8
        assert data[0]["source"] == "semantic"  # _list_past 对无 source 的结果默认填 semantic

    @pytest.mark.asyncio
    async def test_past_chat_simple_failure_degrades_to_candidate_list(self, patch_stores, monkeypatch):
        """主模型 client 存在但 chat_simple 抛异常 -> 降级返回候选 JSON 列表(except 兜底)。"""
        await set_memory_archive_searcher(FakeVecSearcher(), "p2p")

        class FlakyClient:
            async def chat_simple(self, messages, **extra):
                raise RuntimeError("主模型调用失败")

        class FakeAgentLoop:
            _client = FlakyClient()

        class FakeContainer:
            agent_loop = FakeAgentLoop()

        import src._container as _c

        monkeypatch.setattr(_c, "get_container", lambda: FakeContainer())
        result = await memory(action="list", query="部署", mode="past")
        data = json.loads(result)
        assert isinstance(data, list), f"chat_simple 失败应降级返候选列表, got {result!r}"
        assert data[0]["key"] == "deploy_pg"

    @pytest.mark.asyncio
    async def test_past_chat_simple_empty_result_degrades_to_candidate_list(self, patch_stores, monkeypatch):
        """主模型 chat_simple 返回纯空白 -> strip 后为空 -> or 触发降级返候选 JSON 列表。"""
        await set_memory_archive_searcher(FakeVecSearcher(), "p2p")

        class EmptyClient:
            async def chat_simple(self, messages, **extra):
                return "   "  # strip 后为空

        class FakeAgentLoop:
            _client = EmptyClient()

        class FakeContainer:
            agent_loop = FakeAgentLoop()

        import src._container as _c

        monkeypatch.setattr(_c, "get_container", lambda: FakeContainer())
        result = await memory(action="list", query="部署", mode="past")
        data = json.loads(result)
        assert isinstance(data, list), f"空串应降级返候选列表, got {result!r}"
        assert data[0]["key"] == "deploy_pg"

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


class TestMemoryGetArchiveFallback:
    """get 在 KV 未命中时回退归档取回旧记忆（迁移走的键 KV 已无）。"""

    @pytest.mark.asyncio
    async def test_get_falls_back_to_archive_when_kv_misses(self, patch_stores):
        """KV 未命中 + 归档有 -> 回退取回，shape 同 KV recall。"""
        store = FakeArchiveStore(
            {
                "kv:deploy_pg": {
                    "content": "用 postgres 部署",
                    "metadata": {"category": "project", "updated_ts": 1600000000.0, "group_id": ""},
                    "created_at": "2026-01-01T00:00:00+08:00",
                }
            }
        )
        await set_memory_archive_searcher(FakeArchiveSearcherWithStore(store), "p2p")
        got = json.loads(await memory(action="get", key="deploy_pg"))
        assert "error" not in got, f"get 应回退归档取回，got {got}"
        assert got["content"] == "用 postgres 部署"
        assert got["category"] == "project"
        assert got["key"] == "deploy_pg"
        assert got["updated_at"]  # updated_ts 回填
        assert store.calls == ["kv:deploy_pg"], f"应按 _path_for_kv 拼 path 查归档，got {store.calls}"

    @pytest.mark.asyncio
    async def test_get_returns_kv_error_when_archive_misses(self, patch_stores):
        """KV 未命中 + 归档也无 -> 返 KV 的 not-found 错误。"""
        await set_memory_archive_searcher(FakeArchiveSearcherWithStore(FakeArchiveStore({})), "p2p")
        got = json.loads(await memory(action="get", key="missing"))
        assert "error" in got

    @pytest.mark.asyncio
    async def test_get_returns_kv_error_when_archive_disabled(self, patch_stores):
        """无归档 searcher（未启用）-> 返 KV 错误，不崩。"""
        got = json.loads(await memory(action="get", key="anything"))
        assert "error" in got

    @pytest.mark.asyncio
    async def test_get_kv_hit_does_not_touch_archive(self, patch_stores, monkeypatch):
        """KV 命中 -> 直接返，不查归档。"""

        class KvWithContent:
            async def list_all(self, query="", limit=200, group_id=None):
                return [{"key": "k1"}]

            async def recall(self, key, group_id=""):
                return json.dumps(
                    {
                        "key": key,
                        "content": "邮箱 a@b",
                        "category": "contact",
                        "created_at": "2026-01-01",
                        "updated_at": "2026-01-01",
                    }
                )

            async def remember(self, content, key="", category="fact", group_id=""):
                return json.dumps({"ok": True, "key": key})

            async def forget(self, key, group_id=""):
                return json.dumps({"ok": True})

        store = FakeArchiveStore({})
        await set_memory_archive_searcher(FakeArchiveSearcherWithStore(store), "p2p")
        monkeypatch.setattr(memory_tools, "get_memory_store", lambda chat_type="p2p": _async_return(KvWithContent()))
        got = json.loads(await memory(action="get", key="k1"))
        assert "error" not in got
        assert got["content"] == "邮箱 a@b"
        assert store.calls == [], f"KV 命中不应查归档，got {store.calls}"


class TestMemoryModeGroupScope:
    @pytest.mark.asyncio
    async def test_group_scope_filters_by_group_id(self, patch_stores):
        """group scope + 有效 group_id → 按 kv:g:{group_id}: 前缀过滤，只返回该群归档。"""
        await set_memory_archive_searcher(FakeVecSearcherMultiGroup(), "group")
        set_memory_session("group", "groupA")
        result = await memory(action="list", query="redis", mode="past")
        data = json.loads(result)
        # 只返回 groupA 的 2 条（k1, k3），过滤掉 groupB 和 DM
        assert len(data) == 2
        assert {d["key"] for d in data} == {"k1", "k3"}
        assert {d["category"] for d in data} == {"service", "fact"}

    @pytest.mark.asyncio
    async def test_group_scope_colon_in_group_id_key_recovery(self, patch_stores):
        """group_id 带冒号（QQ 实际格式 f"group:{openid}"，见 channels/qq.py:786）时，
        _list_past 反解 path 必须还原原 key。

        回归 bug #1：path="kv:g:group:ABC:note" 经 split(":",3) 得
        ["kv","g","group","ABC:note"] → rkey="ABC:note"（应为 "note"）。
        每个群/频道归档查询都触发。前缀过滤（startswith）没问题，坏的是 key 反解。
        """

        class ColonGidSearcher:
            async def search(self, query, max_results=6, group_id=None):
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
        result = await memory(action="list", query="redis", mode="past")
        data = json.loads(result)
        assert len(data) == 1, f"前缀过滤应命中，got {data}"
        assert data[0]["key"] == "note", f"group_id 带冒号时 key 反解错误: got {data!r}, expected ['note']"

    @pytest.mark.asyncio
    async def test_group_scope_overfetch_truncates_to_cap(self, patch_stores):
        """新契约:searcher.search 返回 semantic+exact 两组各 max_results,总量上限 2*max_results。

        _list_past 传 max_results=3 -> 总量上限 6。mock 返回 10 条时应被截到 6。
        """

        class ManyResultsSearcher:
            async def search(self, query, max_results=6, group_id=None):
                all_docs = [
                    {
                        "content": f"群A 记忆 {i}",
                        "path": f"kv:g:groupA:k{i}",
                        "score": 0.9 - i * 0.01,
                        "metadata": {"category": "fact", "updated_ts": float(i), "group_id": "groupA"},
                    }
                    for i in range(10)
                ]
                return all_docs[: max_results * 2]

        await set_memory_archive_searcher(ManyResultsSearcher(), "group")
        set_memory_session("group", "groupA")
        result = await memory(action="list", query="记忆", mode="past")
        data = json.loads(result)
        assert len(data) == 6  # 2 * max_results(3) = 6

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

            async def search(self, query, max_results=6, group_id=None):
                docs = self.all
                if group_id is not None:
                    docs = [d for d in docs if d["metadata"].get("group_id") == group_id]
                ranked = sorted(docs, key=lambda x: -x["score"])
                return ranked[:max_results]

        await set_memory_archive_searcher(CappedSearcher(), "group")
        set_memory_session("group", "targetA")
        result = await memory(action="list", query="匹配", mode="past")
        data = json.loads(result)
        want_n = [d["key"] for d in data].count("want")
        assert want_n == 1, f"目标群匹配掉出全局 top-20 → 0 结果，但 archive 里有匹配: {data}"

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

            async def close(self):
                pass

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
            result_dm_mysql = json.loads(await memory(action="list", query="mysql", mode="past"))
            assert result_dm_mysql == [], f"DM scope 不应返回 group 数据，got {result_dm_mysql}"

            # DM scope 搜 "redis"（DM store 有）→ 返回 DM 数据
            result_dm_redis = json.loads(await memory(action="list", query="redis", mode="past"))
            assert [d["key"] for d in result_dm_redis] == ["dm_redis"]

            # Group scope 搜 "mysql" → 返回 group 数据（走 group searcher）
            set_memory_session("group", "groupA")
            result_grp_mysql = json.loads(await memory(action="list", query="mysql", mode="past"))
            assert [d["key"] for d in result_grp_mysql] == ["mysql_secret"]

            # Group scope 搜 "redis"（只存在于 DM store）→ 应返回空
            result_grp_redis = json.loads(await memory(action="list", query="redis", mode="past"))
            assert result_grp_redis == [], f"group scope 不应返回 DM 数据，got {result_grp_redis}"
        finally:
            await memory_tools.reset_memory_archive_searcher()
