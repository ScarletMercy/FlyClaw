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
from src.utils.tz import now_iso

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

        dt = datetime.now().astimezone()
        now = dt.replace(microsecond=0).isoformat()
        now_ts = dt.timestamp()
        await self._conn.execute(
            "INSERT INTO memories (key, content, category, created_at, updated_at, updated_ts) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET content=excluded.content, "
            "category=excluded.category, updated_at=excluded.updated_at, updated_ts=excluded.updated_ts",
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
                        "SELECT m.key, m.content, m.category, m.updated_at, m.updated_ts FROM memories m "
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
                            }
                            for r in rows
                        ]
                except Exception:
                    pass
            cursor = await self._conn.execute(
                "SELECT key, content, category, updated_at, updated_ts FROM memories "
                "WHERE content LIKE ? OR key LIKE ? ORDER BY updated_ts DESC LIMIT ?",
                (f"%{query}%", f"%{query}%", limit),
            )
        else:
            cursor = await self._conn.execute(
                "SELECT key, content, category, updated_at, updated_ts FROM memories ORDER BY updated_ts DESC LIMIT ? OFFSET ?",
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

        dt = datetime.now().astimezone()
        now = dt.replace(microsecond=0).isoformat()
        now_ts = dt.timestamp()
        await self._conn.execute(
            "INSERT INTO memories (key, content, category, group_id, created_at, updated_at, updated_ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(key, group_id) DO UPDATE SET content=excluded.content, "
            "category=excluded.category, updated_at=excluded.updated_at, updated_ts=excluded.updated_ts",
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
                            "SELECT m.key, m.content, m.category, m.updated_at, m.updated_ts "
                            "FROM memories m "
                            "JOIN memories_fts f ON f.key = (m.group_id || ':' || m.key) "
                            "WHERE memories_fts MATCH ? AND m.group_id = ? "
                            "ORDER BY rank LIMIT ?",
                            (query, group_id, limit),
                        )
                    else:
                        cursor = await self._conn.execute(
                            "SELECT m.key, m.content, m.category, m.updated_at, m.group_id, m.updated_ts "
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
                            }
                            for r in rows
                        ]
                except Exception:
                    pass
            if group_id is not None:
                cursor = await self._conn.execute(
                    "SELECT key, content, category, updated_at, updated_ts FROM memories "
                    "WHERE (content LIKE ? OR key LIKE ?) AND group_id = ? "
                    "ORDER BY updated_ts DESC LIMIT ?",
                    (f"%{query}%", f"%{query}%", group_id, limit),
                )
            else:
                cursor = await self._conn.execute(
                    "SELECT key, content, category, updated_at, updated_ts FROM memories "
                    "WHERE content LIKE ? OR key LIKE ? ORDER BY updated_ts DESC LIMIT ?",
                    (f"%{query}%", f"%{query}%", limit),
                )
        else:
            if group_id is not None:
                cursor = await self._conn.execute(
                    "SELECT key, content, category, updated_at, updated_ts FROM memories "
                    "WHERE group_id = ? ORDER BY updated_ts DESC LIMIT ? OFFSET ?",
                    (group_id, limit, offset),
                )
            else:
                cursor = await self._conn.execute(
                    "SELECT key, content, category, updated_at, group_id, updated_ts FROM memories "
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


# ---------------------------------------------------------------------------
# Unified memory tool (single tool with action parameter)
# ---------------------------------------------------------------------------


async def _list_past(query: str, verbose: bool = False) -> str:
    """mode=past: 搜向量归档库，回填 key/category/updated_at。

    verbose=False（默认）只返回键名字符串数组（与 KV 命中 verbose=False 一致）；
    verbose=True 返回完整对象数组。
    """
    chat_type, group_id = get_memory_scope()
    if chat_type == "group" and not group_id:
        logger.error("memory mode=past: group scope 但 group_id 为空，阻断")
        return json.dumps({"error": "group scope 下 group_id 为空"}, ensure_ascii=False)

    searcher = await get_memory_archive_searcher(chat_type)
    if searcher is None:
        return json.dumps({"error": "记忆归档未启用"}, ensure_ascii=False)

    if not query.strip():
        return json.dumps([], ensure_ascii=False)

    # 群检索：按 group_id 键值匹配（SQL/LanceDB 层过滤），不靠 path 前缀过滤
    search_group_id = group_id if chat_type == "group" else None
    results = await searcher.search(query, max_results=20, min_score=0.2, group_id=search_group_id)

    # 反解 key：用已知 scope 构造前缀截取，不靠 split（group_id 可能带冒号）
    prefix = f"kv:g:{group_id}:" if chat_type == "group" else "kv:"
    formatted = []
    for r in results[:6]:
        path = r.get("path", "")
        rkey = path[len(prefix) :] if path.startswith(prefix) else path
        meta = r.get("metadata") or {}
        updated_ts = meta.get("updated_ts", 0)
        formatted.append(
            {
                "key": rkey,
                "content": r.get("content", ""),
                "category": meta.get("category", "fact"),
                "updated_at": datetime.fromtimestamp(updated_ts).isoformat() if updated_ts else "",
            }
        )
    if verbose:
        return json.dumps(formatted, ensure_ascii=False)
    return json.dumps([f["key"] for f in formatted], ensure_ascii=False)


async def memory(
    action: Literal["save", "get", "list", "delete"],
    content: str = "",
    key: str = "",
    category: Literal["preference", "identity", "contact", "project", "episodic", "fact"] = "fact",
    query: str = "",
    keys: list[str] | None = None,
    mode: Literal["recent", "past"] = "recent",
    verbose: bool = False,
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
    - list: 列出记忆。
        mode="recent"（默认）：搜最近保留区（KV，约最近 7 天/20 条内），关键词匹配；
                      KV 无命中时自动 fallback 到 archive（past）
        mode="past"：只搜已归档的旧记忆（archive，FTS5 + 向量 hybrid 或 FTS5-only）
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
        verbose: list 时是否同时返回完整内容。默认只返回键名
    """
    normalized = (action or "").strip().lower()

    if normalized == "save":
        if not content:
            return json.dumps({"error": "content is required for save action"}, ensure_ascii=False)
        return await save_memory(content, key, category)

    chat_type, group_id = get_memory_scope()
    s = await get_memory_store(chat_type=chat_type)
    is_group = isinstance(s, GroupMemoryStore)

    if normalized == "get":
        if not key:
            return json.dumps({"error": "key is required for get action"}, ensure_ascii=False)
        if is_group:
            return await s.recall(key, group_id=group_id)
        return await s.recall(key)

    if normalized == "list":
        if mode == "past":
            return await _list_past(query, verbose=verbose)
        # mode="recent"（默认）：查 KV，空则自动 fallback 到 archive
        if is_group:
            items = await s.list_all(query, group_id=group_id)
        else:
            items = await s.list_all(query)
        if not items and query.strip():
            # KV 无命中 → 自动查 archive；archive 未启用则返回空（不报错，保持 recent 行为）
            if await get_memory_archive_searcher(chat_type) is not None:
                return await _list_past(query, verbose=verbose)
            return json.dumps([], ensure_ascii=False)
        if verbose:
            return json.dumps(items, ensure_ascii=False)
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

        # Check session/durable approval before raising (same pattern as exec_command)
        from src.tools.approval import get_approval_manager
        from src.tools.exec import _current_thread_id

        _mgr = get_approval_manager()
        _args_preview = "\n".join(previews)[:200]
        _tid = _current_thread_id.get("")
        if not _mgr.has_durable_approval("memory_delete", _args_preview):
            if not _mgr.has_session_approval(_tid, "memory_delete", _args_preview):
                raise MemoryDeleteNeedsApproval(found_keys, previews)

        # Approved via session/durable — execute delete directly
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
