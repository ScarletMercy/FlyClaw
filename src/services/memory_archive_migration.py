"""KV → archive migration.

周日整理后调用：把超出"最近保留区"的 KV 记忆迁移到 archive store，KV 删行。
保留区 = age <= 7d OR idx < 20（并集）。幂等：path 已在归档库则跳过。

vector_enabled=True  → archive 是 LanceMemoryStore，embed + 存向量
vector_enabled=False → archive 是 MemoryStore（FTS5-only），只存 content，不 embed
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger("flyclaw.memory_archive_migration")


def _path_for_kv(key: str, group_id: str = "", is_group: bool = False) -> str:
    """KV 记忆在归档库的 path 命名。

    is_group=True 时走群前缀（含空 gid → kv:g::{key}），不靠 group_id 真值区分 DM/群。
    """
    if is_group:
        return f"kv:g:{group_id}:{key}"
    return f"kv:{key}"


def _compute_retention(rows: list[dict], now: float, keep_n: int = 20, keep_days: int = 7) -> list[dict]:
    """算待迁移集合。rows 按 updated_ts DESC 排序后，age<=keep_days 或 idx<keep_n 保留。

    返回待迁移列表（保留区的补集）。
    """
    cutoff_age = keep_days * 86400
    sorted_rows = sorted(rows, key=lambda r: r.get("updated_ts", 0), reverse=True)
    migrate = []
    for idx, row in enumerate(sorted_rows):
        age = now - row.get("updated_ts", 0)
        keep = age <= cutoff_age or idx < keep_n
        if not keep:
            migrate.append(row)
    return migrate


async def migrate_kv_to_archive(container: Any) -> dict[str, Any]:
    """周日整理后调用：把超出保留区的 KV 记忆迁移到 archive store。"""
    result: dict[str, Any] = {
        "dm_migrated": 0,
        "dm_failed": 0,
        "groups": [],
        "skipped_reason": "",
    }

    config = container.config
    ms = getattr(config, "memory_store", None)
    if not ms or not ms.enabled:
        result["skipped_reason"] = "memory_store disabled"
        return result

    searchers = getattr(container, "memory_archive_searchers", None)
    if not searchers:
        result["skipped_reason"] = "archive store not initialized"
        return result

    dm_searcher, group_searcher = searchers
    now = time.time()
    keep_n = ms.vector_keep_recent_n
    keep_days = ms.vector_keep_recent_days

    # DM
    dm_result = await _migrate_one_store(
        kv_store=await _get_kv_store("p2p"),
        archive_searcher=dm_searcher,
        now=now,
        keep_n=keep_n,
        keep_days=keep_days,
        group_id="",
    )
    result["dm_migrated"] = dm_result["migrated"]
    result["dm_failed"] = dm_result["failed"]

    # Group: 按 group_id 分组
    group_kv = await _get_kv_store("group")
    from src.tools.memory_tools import GroupMemoryStore

    if isinstance(group_kv, GroupMemoryStore) and group_searcher is not None:
        all_group = []
        offset = 0
        while True:
            batch = await group_kv.list_all(limit=2000, group_id=None, offset=offset)
            if not batch:
                break
            all_group.extend(batch)
            if len(batch) < 2000:
                break
            offset += 2000
        by_group: dict[str, list[dict]] = {}
        for mem in all_group:
            gid = mem.get("group_id", "")
            by_group.setdefault(gid, []).append(mem)

        for gid, gmemories in by_group.items():
            if not gid:
                logger.warning(
                    "KV→archive migration: 跳过空 group_id 的群记忆 %d 条（不迁移，保留 KV）",
                    len(gmemories),
                )
                continue
            g_result = await _migrate_one_store(
                kv_store=group_kv,
                archive_searcher=group_searcher,
                now=now,
                keep_n=keep_n,
                keep_days=keep_days,
                group_id=gid,
                rows=gmemories,
            )
            result["groups"].append({"group_id": gid, "migrated": g_result["migrated"], "failed": g_result["failed"]})

    # 失效 agent 记忆缓存
    agent_loop = getattr(container, "agent_loop", None)
    if agent_loop:
        try:
            agent_loop.invalidate_memory_cache()
        except Exception:
            pass

    total_m = result["dm_migrated"] + sum(g["migrated"] for g in result["groups"])
    total_f = result["dm_failed"] + sum(g["failed"] for g in result["groups"])
    logger.info("KV→archive migration: DM migrated=%d failed=%d", result["dm_migrated"], result["dm_failed"])
    logger.info(
        "KV→archive migration: groups=%d, total migrated=%d failed=%d",
        len(result["groups"]),
        total_m,
        total_f,
    )
    return result


async def _get_kv_store(chat_type: str) -> Any:
    from src.tools.memory_tools import get_memory_store

    return await get_memory_store(chat_type=chat_type)


async def _migrate_one_store(
    kv_store: Any,
    archive_searcher: Any,
    now: float,
    keep_n: int,
    keep_days: int,
    group_id: str,
    rows: list[dict] | None = None,
) -> dict[str, int]:
    """迁移单个 store（DM 或单个 group_id）。

    vector 可用时 embed + 存向量；不可用时只存 content（FTS5-only archive）。
    """
    from src.tools.memory_tools import GroupMemoryStore

    is_group = isinstance(kv_store, GroupMemoryStore)
    if rows is None:
        rows = []
        offset = 0
        while True:
            if is_group:
                batch = await kv_store.list_all(limit=2000, group_id=group_id, offset=offset)
            else:
                batch = await kv_store.list_all(limit=2000, offset=offset)
            if not batch:
                break
            rows.extend(batch)
            if len(batch) < 2000:
                break
            offset += 2000

    to_migrate = _compute_retention(rows, now=now, keep_n=keep_n, keep_days=keep_days)
    if not to_migrate:
        return {"migrated": 0, "failed": 0}

    # 幂等集合：archive 已有的 path
    store = archive_searcher.store
    docs = await store.list_documents()
    existing = {d["path"] for d in docs if d["path"].startswith("kv:")}

    # 分两类：需归档+forget / 仅 forget（之前 add 成功但 forget 失败的孤儿）
    pending_add = []
    pending_forget = []
    for m in to_migrate:
        path = _path_for_kv(m["key"], group_id=group_id, is_group=is_group)
        if path not in existing:
            pending_add.append((m, path))
        else:
            pending_forget.append((m, path))
    if not pending_add and not pending_forget:
        return {"migrated": 0, "failed": 0}

    # 是否走向量路径
    do_vector = archive_searcher.embeddings is not None and store._has_vector_support()

    # 一次性 embed（仅 pending_add，vector 模式）
    embeddings = None
    if do_vector and pending_add:
        contents = [m["content"] for m, _ in pending_add]
        try:
            embeddings = await archive_searcher.embeddings.embed_texts(contents)
        except Exception as e:
            logger.warning("KV→archive embed batch failed (%s): %s，降级 FTS5-only 归档", group_id or "DM", e)
            do_vector = False
            embeddings = None

    migrated = 0
    failed = 0

    # 归档 + forget
    for i, (m, path) in enumerate(pending_add):
        meta = {
            "category": m.get("category", "fact"),
            "updated_ts": m.get("updated_ts", 0),
            "group_id": group_id,
        }
        try:
            await store.add_document(path=path, content=m["content"], metadata=meta, chunk=False)
            if do_vector:
                chunk_ids = await store.get_chunk_ids_for_path(path)
                if chunk_ids:
                    await store.add_embeddings([chunk_ids[0]], [embeddings[i]])
        except Exception as e:
            # add_document/add_embeddings 失败：清半成品 chunk（无向量），避免下轮被
            # list_documents（只查 SQLite）当完整归档 → pending_forget 删 KV 留孤儿
            logger.warning("KV→archive migrate item failed (path=%s): %s", path, e)
            try:
                await store.delete_document(path)
            except Exception as de:
                logger.warning("cleanup half-baked archive chunk failed (path=%s): %s", path, de)
            failed += 1
            continue
        # add 成功，KV 删行（forget 失败不清 archive：下轮 pending_forget 补删，幂等）
        try:
            if is_group:
                await kv_store.forget(m["key"], group_id=group_id)
            else:
                await kv_store.forget(m["key"])
            migrated += 1
        except Exception as e:
            logger.warning("KV→archive forget failed (key=%s): %s", m["key"], e)
            failed += 1

    # 仅 forget：之前 add 成功但 forget 失败的 KV 行，重跑补删（不 re-add，幂等）
    for m, path in pending_forget:
        try:
            if is_group:
                await kv_store.forget(m["key"], group_id=group_id)
            else:
                await kv_store.forget(m["key"])
            migrated += 1
        except Exception as e:
            logger.warning("KV→archive forget retry failed (key=%s): %s", m["key"], e)
            failed += 1

    return {"migrated": migrated, "failed": failed}
