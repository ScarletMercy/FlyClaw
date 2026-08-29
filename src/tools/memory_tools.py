"""Persistent key-value memory backed by SQLite with FTS5 trigram search."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

import aiosqlite

from src.agent.tooldef import ToolDef

logger = logging.getLogger("flyclaw.memory_tools")


class MemoryDeleteNeedsApproval(Exception):
    """Raised by memory_delete to request user confirmation before batch delete."""

    def __init__(self, keys: list[str], previews: list[str]):
        self.keys = keys
        self.previews = previews
        self.command_preview = "\n".join(previews)[:500]
        self.tool_name = "memory_delete"
        self.timeout = 120
        self.auto_deny = True
        self.request_id = ""
        self.denylisted = False
        self.approval_key = "memory_delete"
        self.thread_id = ""
        super().__init__(f"记忆删除需要审批: {len(keys)} 条")


class MemorySaveNeedsApproval(Exception):
    """Raised by memory(save) when pre-validation flags content as unsupported by source.

    保存前自检判定候选记忆疑似臆测/曲解，转人工审批：用户确认后才落库。
    """

    def __init__(self, content: str, source_context: str = "", reason: str = "", mode: str = "model"):
        self.content = content
        self.source_context = source_context
        self.reason = reason
        self.mode = mode
        self.command_preview = content[:500]
        self.tool_name = "memory_save"
        self.timeout = 120
        self.auto_deny = True
        self.request_id = ""
        self.denylisted = False
        self.approval_key = "memory_save"
        self.thread_id = ""
        self.keys: list[str] = []
        super().__init__(f"记忆保存需审批(疑似臆测): {content[:60]}")


_MEMORY_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "preference",
        re.compile(
            r"请?记住|别忘了|以后(?:要|不要|请|别)|"
            r"不要用|不要做|别用|别做|我的风格|"
            r"我(?:偏好|习惯|喜欢)(?:是|:|：|用|写)?",
            re.IGNORECASE,
        ),
    ),
    (
        "identity",
        re.compile(
            r"我(?:的)?(?:名字|姓|号|昵称|网名|ID|身份)(?:是|叫)|"
            r"叫我|我叫|我是(?![a-z\s])|"
            r"我的(?:生日|年龄|地址|家乡|学校|公司|职位|职业|专业)",
            re.IGNORECASE,
        ),
    ),
    (
        "contact",
        re.compile(
            r"(?:邮箱|email|邮件|电话|手机号?|微信号?|QQ|Telegram|discord|github)(?:是|:|：|=)|"
            r"(?:@|邮箱|email).*?(?:是|:|：)|"
            r"1[3-9]\d{9}",
            re.IGNORECASE,
        ),
    ),
    (
        "project",
        re.compile(
            r"(?:我的|我们的)(?:项目|仓库|代码库|产品|系统|服务|网站|app|应用)|"
            r"(?:用了|使用|技术栈|框架|部署在|跑在|运行在)|"
            r"(?:公司|团队|组织|部门)",
            re.IGNORECASE,
        ),
    ),
    (
        "service",
        re.compile(
            r"(?:API|api|接口|地址|URL|url|域名|服务器|端口|数据库|redis|mysql|postgres|mongo)"
            r"(?:是|:|：|=|地址)",
            re.IGNORECASE,
        ),
    ),
]

_CATEGORY_PREFIX_RE = re.compile(r"^\[(\w+)\]\s*")

store: MemoryStore | None = None
_store_initialized: bool = False
_group_store: GroupMemoryStore | None = None
_group_initialized: bool = False
_store_lock: asyncio.Lock = asyncio.Lock()

# Archive searcher 单例（KV 旧记忆归档；vector on 时为 LanceMemoryStore，off 时为 MemoryStore FTS5-only）
_dm_archive_searcher: Any = None
_group_archive_searcher: Any = None
_dm_archive_initialized: bool = False
_group_archive_initialized: bool = False


async def set_memory_archive_searcher(searcher: Any, chat_type: str) -> None:
    """注册 archive searcher 单例（覆盖时 close 旧的，避免句柄泄漏）。"""
    global _dm_archive_searcher, _group_archive_searcher, _dm_archive_initialized, _group_archive_initialized
    old = _group_archive_searcher if chat_type == "group" else _dm_archive_searcher
    if old is not None and old is not searcher:
        try:
            await old.close()
        except Exception:
            pass
    if chat_type == "group":
        _group_archive_searcher = searcher
        _group_archive_initialized = searcher is not None
    else:
        _dm_archive_searcher = searcher
        _dm_archive_initialized = searcher is not None


async def get_memory_archive_searcher(chat_type: str = "p2p") -> Any:
    """按 scope 返回对应 archive searcher；未启用返回 None。"""
    if chat_type == "group":
        return _group_archive_searcher if _group_archive_initialized else None
    return _dm_archive_searcher if _dm_archive_initialized else None


async def reset_memory_archive_searcher() -> None:
    """关闭并重置 archive searcher 单例（热重载调）。"""
    global _dm_archive_searcher, _group_archive_searcher, _dm_archive_initialized, _group_archive_initialized
    for s in (_dm_archive_searcher, _group_archive_searcher):
        if s is not None:
            try:
                await s.close()
            except Exception:
                pass
    _dm_archive_searcher = None
    _group_archive_searcher = None
    _dm_archive_initialized = False
    _group_archive_initialized = False


_current_chat_type: ContextVar[str] = ContextVar("_current_chat_type", default="p2p")
_current_group_id: ContextVar[str] = ContextVar("_current_group_id", default="")
# 当前对话上下文（最近几轮原文），agent loop 执行工具时注入，供 save 自检对照来源
_current_dialog_context: ContextVar[str] = ContextVar("_current_dialog_context", default="")


class MemoryStore:
    def __init__(self, db_path: str | None = None):
        if db_path is None:
            db_path = _default_memories_db()
        self.db_path = str(Path(db_path).expanduser().resolve())
        self._conn: aiosqlite.Connection | None = None
        self._fts_available: bool = False

    async def initialize(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA synchronous=NORMAL")
        await self._conn.execute("PRAGMA busy_timeout=5000")
        await self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                key TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT 'fact',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                updated_ts REAL NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);
        """)
        await self._ensure_fts()
        await self._migrate_category_prefix()
        await self._migrate_add_updated_ts()
        await self._migrate_add_organized()
        await self._conn.commit()
        logger.info("MemoryStore initialized: %s (fts=%s)", self.db_path, self._fts_available)

    async def _ensure_fts(self) -> None:
        try:
            cursor = await self._conn.execute(
                "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='memories_fts'"
            )
            row = await cursor.fetchone()
            if row[0] > 0:
                try:
                    await self._conn.execute("SELECT count(*) FROM memories_fts LIMIT 1")
                    self._fts_available = True
                    return
                except Exception:
                    await self._conn.execute("DROP TABLE IF EXISTS memories_fts")

            await self._conn.execute("CREATE VIRTUAL TABLE memories_fts USING fts5(key, content, tokenize='trigram')")
            await self._conn.execute("INSERT INTO memories_fts (key, content) SELECT key, content FROM memories")
            self._fts_available = True
        except Exception:
            logger.warning("FTS5 trigram not available, search will use LIKE fallback")
            self._fts_available = False

    async def _migrate_category_prefix(self) -> None:
        cursor = await self._conn.execute("SELECT key, content, category FROM memories WHERE content LIKE '[%'")
        rows = await cursor.fetchall()
        if not rows:
            return
        migrated = 0
        for row in rows:
            content = row["content"]
            m = _CATEGORY_PREFIX_RE.match(content)
            if not m:
                continue
            extracted_cat = m.group(1)
            clean_content = _CATEGORY_PREFIX_RE.sub("", content)
            db_cat = row["category"]
            final_cat = extracted_cat if extracted_cat != "fact" else db_cat
            await self._conn.execute(
                "UPDATE memories SET content = ?, category = ? WHERE key = ?",
                (clean_content, final_cat, row["key"]),
            )
            migrated += 1
        if migrated:
            logger.info("Migrated %d memories: stripped [category] prefix from content", migrated)

    async def _migrate_add_updated_ts(self) -> None:
        """Add updated_ts REAL column (epoch seconds) for correct sorting, backfilling from updated_at."""
        cursor = await self._conn.execute("PRAGMA table_info(memories)")
        cols = {row["name"] for row in await cursor.fetchall()}
        if "updated_ts" in cols:
            return
        await self._conn.execute("ALTER TABLE memories ADD COLUMN updated_ts REAL NOT NULL DEFAULT 0")
        cursor = await self._conn.execute("SELECT rowid, updated_at FROM memories")
        rows = await cursor.fetchall()
        for row in rows:
            try:
                dt = datetime.fromisoformat(row["updated_at"])
                if dt.tzinfo is None:
                    dt = dt.astimezone()
                ts = dt.timestamp()
            except Exception:
                ts = 0.0
            await self._conn.execute("UPDATE memories SET updated_ts = ? WHERE rowid = ?", (ts, row["rowid"]))
        await self._conn.commit()
        if rows:
            logger.info("Backfilled updated_ts for %d memories", len(rows))

    async def _migrate_add_organized(self) -> None:
        """加 organized 列（0/1）：记忆整理成功后置 1，整理时跳过，避免重复审查。"""
        cursor = await self._conn.execute("PRAGMA table_info(memories)")
        cols = {row["name"] for row in await cursor.fetchall()}
        if "organized" in cols:
            return
        await self._conn.execute("ALTER TABLE memories ADD COLUMN organized INTEGER NOT NULL DEFAULT 0")
        await self._conn.commit()
        logger.info("Migrated memories: added organized column")

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    @staticmethod
    def _auto_key(content: str) -> str:
        clean = re.sub(r"[^\w\u4e00-\u9fff]+", "_", content[:40]).strip("_")
        return clean or f"mem_{int(datetime.now(timezone.utc).timestamp())}"

    async def _find_key_by_content(self, content: str) -> str | None:
        cursor = await self._conn.execute(
            "SELECT key FROM memories WHERE content = ? LIMIT 1",
            (content,),
        )
        row = await cursor.fetchone()
        return row["key"] if row else None

    async def _find_content_by_key(self, key: str) -> str | None:
        cursor = await self._conn.execute(
            "SELECT content FROM memories WHERE key = ? LIMIT 1",
            (key,),
        )
        row = await cursor.fetchone()
        return row["content"] if row else None

    async def remember(self, content: str, key: str = "", category: str = "fact") -> str:
        if not isinstance(content, str):
            return json.dumps({"error": "content too short or empty"}, ensure_ascii=False)
        content = content.strip()
        if len(content) < 2:
            return json.dumps({"error": "content too short or empty"}, ensure_ascii=False)

        is_dedup = False
        if not key:
            existing_key = await self._find_key_by_content(content)
            if existing_key:
                key = existing_key
                is_dedup = True
            else:
                key = self._auto_key(content)

        content_changed = not is_dedup
        if key and not is_dedup:
            existing_content = await self._find_content_by_key(key)
            if existing_content == content:
                content_changed = False

        dt = datetime.now().astimezone()
        now = dt.replace(microsecond=0).isoformat()
        now_ts = dt.timestamp()
        # 内容变化才重置 organized=0（需复审）；dedup 或同内容写入保留 organized。
        set_clause = (
            "content=excluded.content, category=excluded.category, "
            "updated_at=excluded.updated_at, updated_ts=excluded.updated_ts"
        )
        if content_changed:
            set_clause += ", organized=0"
        await self._conn.execute(
            "INSERT INTO memories (key, content, category, created_at, updated_at, updated_ts, organized) "
            "VALUES (?, ?, ?, ?, ?, ?, 0) "
            f"ON CONFLICT(key) DO UPDATE SET {set_clause}",
            (key, content, category, now, now, now_ts),
        )
        # FTS sync only needed when content actually changed (new insert, not dedup)
        if not is_dedup and self._fts_available:
            try:
                await self._conn.execute(
                    "INSERT OR REPLACE INTO memories_fts (key, content) VALUES (?, ?)",
                    (key, content),
                )
            except Exception:
                pass
        await self._conn.commit()
        result = {"ok": True, "key": key}
        if is_dedup:
            result["dedup"] = True
        return json.dumps(result, ensure_ascii=False)

    async def recall(self, key: str) -> str:
        cursor = await self._conn.execute(
            "SELECT key, content, category, created_at, updated_at FROM memories WHERE key = ?",
            (key,),
        )
        row = await cursor.fetchone()
        if row:
            return json.dumps(
                {
                    "key": row["key"],
                    "content": row["content"],
                    "category": row["category"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                },
                ensure_ascii=False,
            )
        return json.dumps({"error": f"Memory '{key}' not found"}, ensure_ascii=False)

    async def list_all(self, query: str = "", limit: int = 200, offset: int = 0) -> list[dict]:
        if query:
            if self._fts_available and len(query) >= 3:
                try:
                    cursor = await self._conn.execute(
                        "SELECT m.key, m.content, m.category, m.updated_at, m.updated_ts, m.organized FROM memories m "
                        "JOIN memories_fts f ON f.key = m.key "
                        "WHERE memories_fts MATCH ? ORDER BY rank LIMIT ?",
                        (query, limit),
                    )
                    rows = await cursor.fetchall()
                    if rows:
                        return [
                            {
                                "key": r["key"],
                                "content": r["content"],
                                "category": r["category"],
                                "updated_at": r["updated_at"],
                                "updated_ts": r["updated_ts"],
                                "organized": r["organized"],
                            }
                            for r in rows
                        ]
                except Exception:
                    pass
            cursor = await self._conn.execute(
                "SELECT key, content, category, updated_at, updated_ts, organized FROM memories "
                "WHERE content LIKE ? OR key LIKE ? ORDER BY updated_ts DESC LIMIT ?",
                (f"%{query}%", f"%{query}%", limit),
            )
        else:
            cursor = await self._conn.execute(
                "SELECT key, content, category, updated_at, updated_ts, organized FROM memories ORDER BY updated_ts DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
        rows = await cursor.fetchall()
        return [
            {
                "key": r["key"],
                "content": r["content"],
                "category": r["category"],
                "updated_at": r["updated_at"],
                "updated_ts": r["updated_ts"],
                "organized": r["organized"],
            }
            for r in rows
        ]

    async def forget(self, key: str) -> str:
        await self._conn.execute("DELETE FROM memories WHERE key = ?", (key,))
        if self._fts_available:
            try:
                await self._conn.execute("DELETE FROM memories_fts WHERE key = ?", (key,))
            except Exception:
                pass
        await self._conn.commit()
        return json.dumps({"ok": True, "key": key}, ensure_ascii=False)

    async def mark_organized(self, keys: list[str], group_id: str = "") -> int:
        """把给定 key 的 organized 置 1（已整理）。DM store 忽略 group_id。返回更新行数。"""
        if not keys:
            return 0
        placeholders = ",".join("?" * len(keys))
        cursor = await self._conn.execute(
            f"UPDATE memories SET organized = 1 WHERE key IN ({placeholders})",
            list(keys),
        )
        await self._conn.commit()
        return cursor.rowcount or 0


class GroupMemoryStore(MemoryStore):
    """群聊记忆存储，带 group_id 维度隔离（独立 db 文件）。"""

    async def initialize(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA synchronous=NORMAL")
        await self._conn.execute("PRAGMA busy_timeout=5000")
        await self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                key TEXT NOT NULL,
                content TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT 'fact',
                group_id TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                updated_ts REAL NOT NULL DEFAULT 0,
                PRIMARY KEY (key, group_id)
            );
            CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);
            CREATE INDEX IF NOT EXISTS idx_memories_group ON memories(group_id);
        """)
        await self._ensure_fts()
        await self._migrate_add_updated_ts()
        await self._migrate_add_organized()
        await self._conn.commit()
        logger.info("GroupMemoryStore initialized: %s (fts=%s)", self.db_path, self._fts_available)

    async def _ensure_fts(self) -> None:
        try:
            cursor = await self._conn.execute(
                "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='memories_fts'"
            )
            row = await cursor.fetchone()
            if row[0] > 0:
                try:
                    await self._conn.execute("SELECT count(*) FROM memories_fts LIMIT 1")
                    self._fts_available = True
                    return
                except Exception:
                    await self._conn.execute("DROP TABLE IF EXISTS memories_fts")

            await self._conn.execute("CREATE VIRTUAL TABLE memories_fts USING fts5(key, content, tokenize='trigram')")
            await self._conn.execute(
                "INSERT INTO memories_fts (key, content) SELECT group_id || ':' || key, content FROM memories"
            )
            self._fts_available = True
        except Exception:
            logger.warning("FTS5 trigram not available, search will use LIKE fallback")
            self._fts_available = False

    async def _find_key_by_content(self, content: str, group_id: str = "") -> str | None:
        cursor = await self._conn.execute(
            "SELECT key FROM memories WHERE content = ? AND group_id = ? LIMIT 1",
            (content, group_id),
        )
        row = await cursor.fetchone()
        return row["key"] if row else None

    async def _find_content_by_key(self, key: str, group_id: str = "") -> str | None:
        cursor = await self._conn.execute(
            "SELECT content FROM memories WHERE key = ? AND group_id = ? LIMIT 1",
            (key, group_id),
        )
        row = await cursor.fetchone()
        return row["content"] if row else None

    async def remember(self, content: str, key: str = "", category: str = "fact", group_id: str = "") -> str:
        if not isinstance(content, str):
            return json.dumps({"error": "content too short or empty"}, ensure_ascii=False)
        content = content.strip()
        if len(content) < 2:
            return json.dumps({"error": "content too short or empty"}, ensure_ascii=False)

        is_dedup = False
        if not key:
            existing_key = await self._find_key_by_content(content, group_id)
            if existing_key:
                key = existing_key
                is_dedup = True
            else:
                key = self._auto_key(content)

        content_changed = not is_dedup
        if key and not is_dedup:
            existing_content = await self._find_content_by_key(key, group_id)
            if existing_content == content:
                content_changed = False

        dt = datetime.now().astimezone()
        now = dt.replace(microsecond=0).isoformat()
        now_ts = dt.timestamp()
        # 内容变化才重置 organized=0（需复审）；dedup 或同内容写入保留 organized。
        set_clause = (
            "content=excluded.content, category=excluded.category, "
            "updated_at=excluded.updated_at, updated_ts=excluded.updated_ts"
        )
        if content_changed:
            set_clause += ", organized=0"
        await self._conn.execute(
            "INSERT INTO memories (key, content, category, group_id, created_at, updated_at, updated_ts, organized) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 0) "
            f"ON CONFLICT(key, group_id) DO UPDATE SET {set_clause}",
            (key, content, category, group_id, now, now, now_ts),
        )
        if not is_dedup and self._fts_available:
            fts_key = f"{group_id}:{key}"
            try:
                await self._conn.execute(
                    "INSERT OR REPLACE INTO memories_fts (key, content) VALUES (?, ?)",
                    (fts_key, content),
                )
            except Exception:
                pass
        await self._conn.commit()
        result = {"ok": True, "key": key}
        if is_dedup:
            result["dedup"] = True
        return json.dumps(result, ensure_ascii=False)

    async def recall(self, key: str, group_id: str = "") -> str:
        cursor = await self._conn.execute(
            "SELECT key, content, category, created_at, updated_at FROM memories WHERE key = ? AND group_id = ?",
            (key, group_id),
        )
        row = await cursor.fetchone()
        if row:
            return json.dumps(
                {
                    "key": row["key"],
                    "content": row["content"],
                    "category": row["category"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                },
                ensure_ascii=False,
            )
        return json.dumps({"error": f"Memory '{key}' not found"}, ensure_ascii=False)

    async def list_all(
        self, query: str = "", limit: int = 200, group_id: str | None = None, offset: int = 0
    ) -> list[dict]:
        if query:
            if self._fts_available and len(query) >= 3:
                try:
                    if group_id is not None:
                        cursor = await self._conn.execute(
                            "SELECT m.key, m.content, m.category, m.updated_at, m.updated_ts, m.organized "
                            "FROM memories m "
                            "JOIN memories_fts f ON f.key = (m.group_id || ':' || m.key) "
                            "WHERE memories_fts MATCH ? AND m.group_id = ? "
                            "ORDER BY rank LIMIT ?",
                            (query, group_id, limit),
                        )
                    else:
                        cursor = await self._conn.execute(
                            "SELECT m.key, m.content, m.category, m.updated_at, m.group_id, m.updated_ts, m.organized "
                            "FROM memories m "
                            "JOIN memories_fts f ON f.key = (m.group_id || ':' || m.key) "
                            "WHERE memories_fts MATCH ? ORDER BY rank LIMIT ?",
                            (query, limit),
                        )
                    rows = await cursor.fetchall()
                    if rows:
                        return [
                            {
                                "key": r["key"],
                                "content": r["content"],
                                "category": r["category"],
                                "updated_at": r["updated_at"],
                                "updated_ts": r["updated_ts"],
                                "organized": r["organized"],
                            }
                            for r in rows
                        ]
                except Exception:
                    pass
            if group_id is not None:
                cursor = await self._conn.execute(
                    "SELECT key, content, category, updated_at, updated_ts, organized FROM memories "
                    "WHERE (content LIKE ? OR key LIKE ?) AND group_id = ? "
                    "ORDER BY updated_ts DESC LIMIT ?",
                    (f"%{query}%", f"%{query}%", group_id, limit),
                )
            else:
                cursor = await self._conn.execute(
                    "SELECT key, content, category, updated_at, updated_ts, organized FROM memories "
                    "WHERE content LIKE ? OR key LIKE ? ORDER BY updated_ts DESC LIMIT ?",
                    (f"%{query}%", f"%{query}%", limit),
                )
        else:
            if group_id is not None:
                cursor = await self._conn.execute(
                    "SELECT key, content, category, updated_at, updated_ts, organized FROM memories "
                    "WHERE group_id = ? ORDER BY updated_ts DESC LIMIT ? OFFSET ?",
                    (group_id, limit, offset),
                )
            else:
                cursor = await self._conn.execute(
                    "SELECT key, content, category, updated_at, group_id, updated_ts, organized FROM memories "
                    "ORDER BY updated_ts DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                )
                rows = await cursor.fetchall()
                return [
                    {
                        "key": r["key"],
                        "content": r["content"],
                        "category": r["category"],
                        "updated_at": r["updated_at"],
                        "group_id": r["group_id"],
                        "updated_ts": r["updated_ts"],
                        "organized": r["organized"],
                    }
                    for r in rows
                ]
        rows = await cursor.fetchall()
        return [
            {
                "key": r["key"],
                "content": r["content"],
                "category": r["category"],
                "updated_at": r["updated_at"],
                "updated_ts": r["updated_ts"],
                "organized": r["organized"],
            }
            for r in rows
        ]

    async def forget(self, key: str, group_id: str = "") -> str:
        await self._conn.execute(
            "DELETE FROM memories WHERE key = ? AND group_id = ?",
            (key, group_id),
        )
        if self._fts_available:
            fts_key = f"{group_id}:{key}"
            try:
                await self._conn.execute("DELETE FROM memories_fts WHERE key = ?", (fts_key,))
            except Exception:
                pass
        await self._conn.commit()
        return json.dumps({"ok": True, "key": key}, ensure_ascii=False)

    async def mark_organized(self, keys: list[str], group_id: str = "") -> int:
        """群 store 按 group_id 隔离标记 organized。返回更新行数。"""
        if not keys:
            return 0
        placeholders = ",".join("?" * len(keys))
        params = list(keys) + [group_id]
        cursor = await self._conn.execute(
            f"UPDATE memories SET organized = 1 WHERE key IN ({placeholders}) AND group_id = ?",
            params,
        )
        await self._conn.commit()
        return cursor.rowcount or 0


def _default_memories_db() -> str:
    from src.instance import data_dir

    return str(data_dir() / "memories.db")


def _default_group_memories_db() -> str:
    from src.instance import data_dir

    return str(data_dir() / "memories_group.db")


async def get_memory_store(db_path: str | None = None, chat_type: str = "p2p") -> MemoryStore:
    global store, _store_initialized, _group_store, _group_initialized
    if chat_type == "group":
        if _group_initialized:
            return _group_store  # type: ignore[return-value]
        async with _store_lock:
            if _group_initialized:
                return _group_store  # type: ignore[return-value]
            if _group_store is None:
                _group_store = GroupMemoryStore(_default_group_memories_db())
            await _group_store.initialize()
            _group_initialized = True
        return _group_store  # type: ignore[return-value]
    if _store_initialized:
        return store  # type: ignore[return-value]
    async with _store_lock:
        if _store_initialized:
            return store  # type: ignore[return-value]
        if store is None:
            store = MemoryStore(db_path or _default_memories_db())
        await store.initialize()
        _store_initialized = True
    return store  # type: ignore[return-value]


async def reset_memory_store() -> None:
    """关闭并重置模块级 MemoryStore 单例（供热重载调用）。

    调用后下次 get_memory_store() 将用新的 db_path 创建实例。
    """
    global store, _store_initialized, _group_store, _group_initialized
    async with _store_lock:
        if store is not None:
            try:
                await store.close()
            except Exception:
                pass
        if _group_store is not None:
            try:
                await _group_store.close()
            except Exception:
                pass
        store = None
        _store_initialized = False
        _group_store = None
        _group_initialized = False


def set_memory_session(chat_type: str, group_id: str = "") -> None:
    """设置当前 memory scope（由 message.py 在消息入口处调用）。"""
    _current_chat_type.set(chat_type)
    _current_group_id.set(group_id)


def get_memory_scope() -> tuple[str, str]:
    """读取当前 memory scope（chat_type, group_id）。"""
    return _current_chat_type.get(), _current_group_id.get()


def set_memory_dialog_context(ctx: str) -> None:
    """设置当前对话上下文（agent loop 执行工具时注入，供 save 自检对照来源原文）。"""
    _current_dialog_context.set(ctx)


def get_memory_dialog_context() -> str:
    """读取当前对话上下文（save 自检用）。"""
    return _current_dialog_context.get()


def build_dialog_context(messages: list[dict], max_turns: int = 6, max_chars: int = 2000) -> str:
    """从 messages 拼接最近几轮对话原文，供 save 自检对照来源。

    取最近 max_turns 条非空 content，截断到 max_chars。agent loop 执行工具时
    用它构造 source 注入 _current_dialog_context。
    """
    recent = messages[-max_turns:] if len(messages) > max_turns else messages
    lines: list[str] = []
    for m in recent:
        content = m.get("content") or ""
        if not content.strip():
            continue
        role = m.get("role", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)[:max_chars]


def _memory_save_approval_text(
    content: str, source_context: str, timeout: int, zh: bool, model_mode: bool = True
) -> str:
    """构造 memory_save 审批文案：展示候选记忆 + 来源原文，供用户核对是否保存。

    model_mode=True（模型自检 reject）：措辞提示"疑似臆测"。
    model_mode=False（manual 模式）：措辞用"确认保存"（无模型判断）。
    无来源上下文时省略来源行（降级，不阻断审批流程）。
    """
    src = (source_context or "").strip()
    if src:
        src_line = f"来源原文：\n{src[:300]}\n\n" if zh else f"Source:\n{src[:300]}\n\n"
    else:
        src_line = ""
    if zh:
        header = "📝 这条记忆疑似臆测，确认是否保存：" if model_mode else "📝 确认是否保存这条记忆："
        return f"{header}\n\n{content}\n\n{src_line}发送 /y 确认保存，其它消息自动取消（{timeout}秒超时）"
    header = "📝 This memory looks speculative. Confirm saving:" if model_mode else "📝 Confirm saving this memory:"
    return f"{header}\n\n{content}\n\n{src_line}Send /y to save, any other message cancels ({timeout}s timeout)"


def _deny_counts_toward_abort(tool_name: str) -> bool:
    """该工具的 deny 是否计入 consecutive_denies 终止计数。

    memory_save 逐条审批：连续 deny 是用户对不同 content 逐条把关，非死循环重试，不计入。
    其他工具（exec/memory_delete）的连续 deny 视为潜在死循环重试，计入。
    """
    return tool_name != "memory_save"


# ---------------------------------------------------------------------------
# Auto-extract memory from conversation
# ---------------------------------------------------------------------------


def auto_extract_memory(user_input: str, ai_response: str) -> Optional[tuple[str, str]]:
    """从用户输入中提取值得记忆的事实片段。

    匹配到模式后，只返回包含匹配的句子/分句，而非全文。

    Note: ai_response 参数当前未使用，保留供未来扩展。
    """
    if not user_input or len(user_input.strip()) < 4:
        return None
    for category, pattern in _MEMORY_PATTERNS:
        m = pattern.search(user_input)
        if m:
            # 提取包含匹配的句子/分句，而非整段输入
            extracted = _extract_matched_clause(user_input, m.start(), m.end())
            if extracted:  # 防止提取结果为空字符串
                return extracted, category
    return None


def _extract_matched_clause(text: str, match_start: int, match_end: int) -> str:
    """从文本中提取包含匹配位置的子句/句子。

    优先按标点切分取最短包含匹配的片段；若切分后片段过短（<4字）
    则向上扩展到前一个分隔符，确保语义完整。
    """
    if not text:
        return text

    def _is_separator(pos: int) -> bool:
        """判断 pos 位置是否为分句/句子边界。"""
        ch = text[pos]
        # 中文句级标点：句号、叹号、问号、分号、换行
        if ch in "。！？；\n!?;":
            return True
        # 中英文逗号
        if ch in "，,":
            return True
        # 英文句号：仅当后面是空格或行尾时视为句子结束
        # （避免截断 URL、邮箱、版本号中的点号）
        if ch == ".":
            if pos + 1 >= len(text):
                return True
            if text[pos + 1] == " ":
                return True
        return False

    # 从匹配位置向两侧扩展，找到最近的分隔符
    start = match_start
    while start > 0 and not _is_separator(start - 1):
        start -= 1
    # 跳过紧贴匹配的逗号/分隔符（取分隔符后面的内容）
    while start < match_start and text[start] in "，,":
        start += 1
    end = match_end
    while end < len(text) and not _is_separator(end):
        end += 1
    clause = text[start:end].strip().rstrip("，,")
    # 如果提取的片段太短（可能只是触发词），扩展到前一个分隔符
    if len(clause) < 4 and start > 0:
        prev = start - 1
        while prev > 0 and not _is_separator(prev - 1):
            prev -= 1
        clause = text[prev:end].strip().rstrip("，,")
    return clause


# ---------------------------------------------------------------------------
# Passive save helper
# ---------------------------------------------------------------------------


async def save_memory(content: str, key: str = "", category: str = "fact") -> str:
    chat_type, group_id = get_memory_scope()
    s = await get_memory_store(chat_type=chat_type)
    if isinstance(s, GroupMemoryStore):
        return await s.remember(content, key, category, group_id=group_id)
    return await s.remember(content, key, category)


async def _gate_memory_save(content: str) -> Optional[str]:
    """入口1（memory save 工具）的保存前置关卡。

    三种出口：
    - 返回 None：放行，调用方继续 save_memory。
    - 返回 JSON 字符串：已处理（model 自检判定 reject），调用方直接返回该串。
    - raise MemorySaveNeedsApproval：转人工审批（manual 模式），由 agent loop 接管。

    仅入口1 调用。dashboard / auto-extract / 日记 / 整理都直接调 save_memory，不经此关卡
    ——审批协议依赖 agent 工具循环 + 在线交互通道 + thread 上下文，只有入口1 同时具备。
    """
    # session 短路（对齐 delete）：已授权的 args 直接放行，避免 resume 重复自检。
    # approval manager 不可用时（container 未初始化，如部分集成测试）跳过短路、走自检。
    approved = False
    mode = "model"
    try:
        from src._container import get_container
        from src.tools.approval import get_approval_manager
        from src.tools.exec import _current_thread_id

        container = get_container()
        mode = getattr(container.config.memory_store, "save_approval_mode", "model")
        mgr = get_approval_manager()
        approved = mgr.has_session_approval(_current_thread_id.get(""), "memory_save", content[:200])
    except RuntimeError:
        approved = False

    if approved:
        return None

    source = get_memory_dialog_context()
    if mode == "manual":
        # 人工批准模式：跳过自检，直接转人工审批
        raise MemorySaveNeedsApproval(content, source_context=source, reason="manual approval mode", mode=mode)
    # model 模式：模型自检把关。reject 直接丢弃（错误信息挡在库外），不弹审批。
    decision, reason = await _validate_memory(content, source)
    if decision == "reject":
        logger.info("memory save rejected by self-check: %.80s", content)
        return json.dumps({"ok": False, "rejected": True, "reason": reason}, ensure_ascii=False)
    return None


async def _gate_auto_extract(content: str, source_context: str = "") -> tuple[str, str]:
    """入口2（auto_extract 正则提取）的保存前置关卡。

    与入口1（_gate_memory_save）复用同一套 model/manual 审批语义，但不抛异常、
    不依赖 agent 工具循环——auto_extract 在消息回复后触发，无 ApprovalPending 槽位，
    由调用方（message.py）用 detached 后台 task 实现程序式审批（不阻塞主回合）。

    返回 (decision, reason)，decision ∈ {"allow", "reject", "manual"}：
    - allow:  放行，调用方 save_memory
    - reject: model 自检判臆测/曲解，静默丢弃（错误信息挡在库外）
    - manual: 需人工审批（model 模式可疑 / manual 模式全部），调用方起 detached 后台 task
    fail-open：无主模型/调用失败 → allow（不阻断 auto_extract）。
    """
    # 会话级授权短路：本会话已批准该内容 → 不再问（对齐 save 关卡，防重复轰炸）
    try:
        from src._container import get_container
        from src.tools.approval import get_approval_manager
        from src.tools.exec import _current_thread_id

        container = get_container()
        mode = getattr(container.config.memory_store, "save_approval_mode", "model")
        mgr = get_approval_manager()
        if mgr.has_session_approval(_current_thread_id.get(""), "auto_extract", content[:200]):
            return ("allow", "session approved")
    except RuntimeError:
        # container 未初始化（部分集成测试）→ fail-open
        return ("allow", "no container, fail-open")

    if mode == "manual":
        # 人工批准模式：跳过自检，全部转 detached 后台审批
        return ("manual", "manual approval mode")
    # model 模式：模型自检把关。reject 静默丢弃，allow 放行；可疑（allow 但需复核）不在此层处理。
    # 注：_validate_memory 只返回 allow/reject 两态；model 模式下"可疑转人工"由 manual 模式承担。
    decision, reason = await _validate_memory(content, source_context)
    if decision == "reject":
        logger.info("auto_extract rejected by self-check: %.80s", content)
    return (decision, reason)


# ---------------------------------------------------------------------------
# Unified memory tool (single tool with action parameter)
# ---------------------------------------------------------------------------


async def _list_past(query: str) -> list[dict]:
    """mode=past: 搜归档库，返回 [{key, source, content, category, updated_at, score}, ...]（不去重，semantic 在前）。

    语义路与 FTS5 路各自 top max_results=3，合并后不去重（同文档两路命中各保留一条，
    source 区分 semantic/exact）。score: semantic=vec_score, exact=归一化 BM25。
    group scope 空 group_id 或归档未启用时抛 ValueError。
    """
    chat_type, group_id = get_memory_scope()
    if chat_type == "group" and not group_id:
        raise ValueError("group scope 下 group_id 为空")

    searcher = await get_memory_archive_searcher(chat_type)
    if searcher is None:
        raise ValueError("记忆归档未启用")

    if not query.strip():
        return []

    # 群检索:按 group_id 键值匹配(SQL/LanceDB 层过滤),不靠 path 前缀过滤
    search_group_id = group_id if chat_type == "group" else None
    results = await searcher.search(query, max_results=3, group_id=search_group_id)

    # 反解 key:用已知 scope 构造前缀截取,不靠 split(group_id 可能带冒号)
    prefix = f"kv:g:{group_id}:" if chat_type == "group" else "kv:"
    out: list[dict] = []
    for r in results:
        path = r.get("path", "")
        rkey = path[len(prefix) :] if path.startswith(prefix) else path
        meta = r.get("metadata") or {}
        updated_ts = meta.get("updated_ts", 0)
        out.append(
            {
                "key": rkey,
                "source": r.get("source", "semantic"),
                "content": r.get("content", ""),
                "score": r.get("score", 0),
                "category": meta.get("category", "fact"),
                "updated_at": datetime.fromtimestamp(updated_ts).isoformat() if updated_ts else "",
            }
        )
    return out


async def _extract_relevant_with_llm(query: str, candidates: list[dict]) -> str:
    """用主模型按 query 从候选记忆提取相关内容，返回自然语言文。

    无候选 -> 返回 []; 无主模型(get_container 未初始化 / agent_loop 缺失)或调用失败 -> 降级返回候选 JSON 列表。
    主模型取 container.agent_loop._client（ChatClient / FallbackChain 均有 chat_simple）。
    """
    if not candidates:
        return json.dumps([], ensure_ascii=False)
    try:
        from src._container import get_container

        container = get_container()
        client = container.agent_loop._client if container.agent_loop else None
    except RuntimeError:
        client = None
    if client is None:
        return json.dumps(candidates, ensure_ascii=False)  # 测试/无主模型:降级返回原始列表

    lines = []
    for i, m in enumerate(candidates, 1):
        lines.append(
            f"[{i}] key={m.get('key', '')} category={m.get('category', '')} updated={m.get('updated_at', '')}\n"
            f"{m.get('content', '')}"
        )
    context = "\n\n".join(lines)
    prompt = (
        "你是记忆检索助手。根据用户问题，从候选记忆中提取相关内容。\n\n"
        f"用户问题：{query}\n\n候选记忆：\n{context}\n\n"
        "要求：\n"
        "- 只提取与问题相关的记忆内容，按 [key] 要点 的形式简洁列出\n"
        "- 严禁编造，严禁加入候选之外的任何信息\n"
        "- 都不相关时只回复「无相关记忆」"
    )
    try:
        result = await client.chat_simple([{"role": "user", "content": prompt}])
        return result.strip() or json.dumps(candidates, ensure_ascii=False)
    except Exception as e:
        logger.warning("past 主模型提取失败，降级返回原始列表: %s", e)
        return json.dumps(candidates, ensure_ascii=False)


async def _validate_memory(content: str, source_context: str = "") -> tuple[str, str]:
    """保存前自检：判断 content 是否被 source_context 支持，防止臆测/曲解写入记忆库。

    返回 (decision, reason)，decision ∈ {"allow", "reject"}。
    用主模型对照来源原文判断候选记忆是否有据可循。
    fail-open：无来源 / 无主模型 / 调用失败 → 放行，不阻断记忆系统。
    """
    # 无来源上下文 → 无法核对，放行（同时省一次主模型调用）
    if not (source_context or "").strip():
        return ("allow", "no source context, fail-open")

    try:
        from src._container import get_container

        container = get_container()
        client = container.agent_loop._client if container.agent_loop else None
    except RuntimeError:
        return ("allow", "no llm client, fail-open")
    if client is None:
        return ("allow", "no llm client, fail-open")

    prompt = (
        "你是记忆审核员。判断「候选记忆」是否能被「来源原文」支持。\n\n"
        f"候选记忆：{content}\n"
        f"来源原文：{source_context}\n\n"
        "判断标准：\n"
        "- 候选记忆必须能从来源原文中找到依据（用户明确说过或可合理提取）\n"
        "- 臆测、推断、曲解、或截断导致语义反转（如丢否定词）→ reject\n"
        "以 allow 或 reject 开头回复，后跟简短理由。"
    )
    try:
        result = await client.chat_simple([{"role": "user", "content": prompt}])
    except Exception as e:
        logger.warning("memory validation call failed, fail-open: %s", e)
        return ("allow", "validation call failed, fail-open")
    text = (result or "").strip().lower()
    if text.startswith("reject"):
        return ("reject", (result or "").strip())
    return ("allow", (result or "").strip())


async def memory(
    action: Literal["save", "get", "list", "delete"],
    content: str = "",
    key: str = "",
    category: Literal["preference", "identity", "contact", "project", "episodic", "fact"] = "fact",
    query: str = "",
    keys: list[str] | None = None,
    mode: Literal["recent", "past"] = "recent",
) -> str:
    """管理持久记忆（跨会话保存）。用 action 参数指定操作。

    WHEN TO SAVE（主动保存，不要等用户要求）:
    - 用户纠正你或说"记住这个""以后别这样"
    - 用户分享偏好、习惯、个人细节（名字、角色、时区、编码风格）
    - 你发现了环境信息（OS、工具、项目结构）
    - 你学到了约定、API 怪癖、工作流

    ACTIONS:
    - save: 保存记忆（自动去重）。需要 content，可选 key/category
    - get: 按键取回完整记忆内容。键名即记忆摘要，先用 list 查看键名
    - list: 列出记忆（详情用 get 取）。
        mode="recent"（默认）：搜最近保留区（KV，约最近 7 天/20 条内），关键词匹配；
                      返回键名字符串数组；KV 无命中时自动 fallback 到 archive，fallback 同样返回键名（去重）
        mode="past"：搜归档库候选，用主模型按 query 提取相关记忆，返回自然语言提取文
                      （无主模型/失败时降级返回候选 JSON 列表）
        query 为空时 recent 返回全部键名（不 fallback）；past 返回空
    - delete: 删除指定记忆条目。需要 keys 数组

    不要保存：任务进度、闲聊、一次性指令、通用知识。

    Args:
        action: 操作类型: save, get, list, delete
        content: 记忆内容（save 时必填）
        key: 记忆键名，即记忆内容摘要（save 可选，get 必填）
        category: 记忆分类（默认 fact）
        query: list 用关键词过滤记忆
        keys: 要删除的记忆键名列表（delete 必填）
        mode: list 时的检索池。recent=KV 保留区，past=归档库
    """
    normalized = (action or "").strip().lower()

    if normalized == "save":
        if not content:
            return json.dumps({"error": "content is required for save action"}, ensure_ascii=False)
        rejection = await _gate_memory_save(content)
        if rejection is not None:
            return rejection
        return await save_memory(content, key, category)

    chat_type, group_id = get_memory_scope()
    s = await get_memory_store(chat_type=chat_type)
    is_group = isinstance(s, GroupMemoryStore)

    if normalized == "get":
        if not key:
            return json.dumps({"error": "key is required for get action"}, ensure_ascii=False)
        raw = await (s.recall(key, group_id=group_id) if is_group else s.recall(key))
        # KV 未命中 -> 回退归档（迁移走的旧记忆）；取不到则原样返回 KV 错误
        try:
            kv_miss = "error" in json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            kv_miss = False
        if kv_miss:
            searcher = await get_memory_archive_searcher(chat_type)
            store = getattr(searcher, "store", None) if searcher else None
            if store is not None:
                from src.services.memory_archive_migration import _path_for_kv

                doc = await store.get_document_by_path(_path_for_kv(key, group_id=group_id, is_group=is_group))
                if doc is not None:
                    meta = doc.get("metadata") or {}
                    updated_ts = meta.get("updated_ts", 0)
                    return json.dumps(
                        {
                            "key": key,
                            "content": doc.get("content", ""),
                            "category": meta.get("category", "fact"),
                            "created_at": doc.get("created_at", ""),
                            "updated_at": datetime.fromtimestamp(updated_ts).isoformat() if updated_ts else "",
                        },
                        ensure_ascii=False,
                    )
        return raw

    if normalized == "list":
        if mode == "past":
            try:
                candidates = await _list_past(query)
            except ValueError as e:
                return json.dumps({"error": str(e)}, ensure_ascii=False)
            return await _extract_relevant_with_llm(query, candidates)
        # mode="recent"（默认）：查 KV，空则自动 fallback 到 archive
        if is_group:
            items = await s.list_all(query, group_id=group_id)
        else:
            items = await s.list_all(query)
        if not items and query.strip():
            # KV 无命中 → 自动查 archive；archive 未启用则返回空（不报错，保持 recent 行为）
            try:
                past = await _list_past(query)
            except ValueError:
                return json.dumps([], ensure_ascii=False)
            keys: list[str] = []
            seen: set[str] = set()
            for r in past:
                if r["key"] in seen:
                    continue
                seen.add(r["key"])
                keys.append(r["key"])
            return json.dumps(keys, ensure_ascii=False)
        return json.dumps([i["key"] for i in items], ensure_ascii=False)

    if normalized == "delete":
        if not keys:
            return json.dumps({"error": "keys is required for delete action"}, ensure_ascii=False)
        unique_keys = list(dict.fromkeys(k for k in keys if k))
        if not unique_keys:
            return json.dumps({"error": "No valid keys specified"}, ensure_ascii=False)
        found_keys = []
        previews = []
        for k in unique_keys:
            if is_group:
                raw = await s.recall(k, group_id=group_id)
            else:
                raw = await s.recall(k)
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                data = {"error": f"Failed to parse memory data for key '{k}'"}
            if "error" not in data:
                found_keys.append(k)
                c = data.get("content", "")
                previews.append(f"- [{k}]: {c[:80]}")
        if not previews:
            return json.dumps({"error": "None of the specified keys exist"}, ensure_ascii=False)

        # Check session approval before raising (same pattern as exec_command)
        from src.tools.approval import get_approval_manager
        from src.tools.exec import _current_thread_id

        _mgr = get_approval_manager()
        _args_preview = "\n".join(previews)[:200]
        _tid = _current_thread_id.get("")
        if not _mgr.has_session_approval(_tid, "memory_delete", _args_preview):
            raise MemoryDeleteNeedsApproval(found_keys, previews)

        # Approved via session — execute delete directly
        deleted = []
        for k in found_keys:
            if is_group:
                await s.forget(k, group_id=group_id)
            else:
                await s.forget(k)
            deleted.append(k)
        return json.dumps({"ok": True, "deleted": deleted, "count": len(deleted)}, ensure_ascii=False)

    return json.dumps({"error": f"Unknown action '{action}'. Use: save, get, list, delete"}, ensure_ascii=False)


def get_tools() -> list[ToolDef]:
    return [
        ToolDef.from_function(memory),
    ]
