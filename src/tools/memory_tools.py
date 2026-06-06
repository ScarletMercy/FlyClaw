"""Persistent key-value memory backed by SQLite with FTS5 trigram search."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

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
_store_lock: asyncio.Lock = asyncio.Lock()


class MemoryStore:
    def __init__(self, db_path: str = "~/.flyclaw/data/memories.db"):
        self.db_path = str(Path(db_path).expanduser().resolve())
        self._conn: aiosqlite.Connection | None = None
        self._fts_available: bool = False

    async def initialize(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA synchronous=NORMAL")
        await self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                key TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT 'fact',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);
        """)
        await self._ensure_fts()
        await self._migrate_category_prefix()
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

    async def close(self) -> None:
        global _store_initialized
        if self._conn:
            await self._conn.close()
            self._conn = None
        _store_initialized = False

    @staticmethod
    def _auto_key(content: str) -> str:
        clean = re.sub(r"[^\w\u4e00-\u9fff]+", "_", content[:40]).strip("_")
        return clean or f"mem_{int(datetime.now(timezone.utc).timestamp())}"

    async def remember(self, content: str, key: str = "", category: str = "fact") -> str:
        if not key:
            key = self._auto_key(content)
        now = datetime.now(timezone.utc).isoformat()
        await self._conn.execute(
            "INSERT INTO memories (key, content, category, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET content=excluded.content, "
            "category=excluded.category, updated_at=excluded.updated_at",
            (key, content, category, now, now),
        )
        if self._fts_available:
            try:
                await self._conn.execute(
                    "INSERT OR REPLACE INTO memories_fts (key, content) VALUES (?, ?)",
                    (key, content),
                )
            except Exception:
                pass
        await self._conn.commit()
        return json.dumps({"ok": True, "key": key}, ensure_ascii=False)

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

    async def list_all(self, query: str = "", limit: int = 200) -> list[dict]:
        if query:
            if self._fts_available and len(query) >= 3:
                try:
                    cursor = await self._conn.execute(
                        "SELECT m.key, m.content, m.category, m.updated_at FROM memories m "
                        "JOIN memories_fts f ON f.key = m.key "
                        "WHERE memories_fts MATCH ? ORDER BY rank LIMIT ?",
                        (query, limit),
                    )
                    rows = await cursor.fetchall()
                    if rows:
                        return [{"key": r["key"], "content": r["content"], "category": r["category"]} for r in rows]
                except Exception:
                    pass
            cursor = await self._conn.execute(
                "SELECT key, content, category, updated_at FROM memories "
                "WHERE content LIKE ? OR key LIKE ? ORDER BY updated_at DESC LIMIT ?",
                (f"%{query}%", f"%{query}%", limit),
            )
        else:
            cursor = await self._conn.execute(
                "SELECT key, content, category, updated_at FROM memories ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            )
        rows = await cursor.fetchall()
        return [{"key": r["key"], "content": r["content"], "category": r["category"]} for r in rows]

    async def forget(self, key: str) -> str:
        await self._conn.execute("DELETE FROM memories WHERE key = ?", (key,))
        if self._fts_available:
            try:
                await self._conn.execute("DELETE FROM memories_fts WHERE key = ?", (key,))
            except Exception:
                pass
        await self._conn.commit()
        return json.dumps({"ok": True, "key": key}, ensure_ascii=False)


async def get_memory_store(db_path: str = "~/.flyclaw/data/memories.db") -> MemoryStore:
    global store, _store_initialized
    if _store_initialized:
        return store
    async with _store_lock:
        if _store_initialized:
            return store
        if store is None:
            store = MemoryStore(db_path)
        await store.initialize()
        _store_initialized = True
    return store


async def reset_memory_store() -> None:
    """关闭并重置模块级 MemoryStore 单例（供热重载调用）。

    调用后下次 get_memory_store() 将用新的 db_path 创建实例。
    """
    global store, _store_initialized
    async with _store_lock:
        if store is not None:
            try:
                await store.close()
            except Exception:
                pass
        store = None
        _store_initialized = False


# ---------------------------------------------------------------------------
# Auto-extract & LLM judge
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


_JUDGE_PROMPT = """\
判断下面的对话是否包含用户主动提供的、明确的、值得永久记住的事实信息。

只记住以下情况（必须同时满足"明确"和"永久有用"两个条件）：
- 用户直接说了自己的偏好/习惯（如"我喜欢""我习惯""以后请"）
- 用户直接说了自己的身份/联系方式（如名字、邮箱、电话）
- 用户直接说了项目/工作相关的固定信息（如技术栈、服务器地址）

以下情况一律不记：
- 闲聊、寒暄、问答（"在吗""帮我XX""今天天气"）
- 一次性指令（"打开XX""截图""运行XX"）
- 通用知识、讨论、解释
- 情绪表达（"太好了""烦死了"）
- 模糊/不确定的信息
- 对话中没有任何用户主动透露的个人事实

对话:
用户: {user}
助手: {ai}

严格判断。如果有疑问，就不记。

只输出 JSON:
值得记: {{"remember": true, "content": "具体事实内容(一句话)", "category": "preference|identity|contact|project|fact"}}
不值得记: {{"remember": false}}"""


async def judge_memory_with_llm(
    user_input: str,
    ai_response: str,
    model_name: str,
    base_url: str,
    api_key: str,
) -> Optional[tuple[str, str]]:
    from src.agent.client import ChatClient

    model = ChatClient(
        base_url=base_url,
        api_key=api_key,
        model=model_name,
        temperature=0.0,
    )
    prompt = _JUDGE_PROMPT.format(
        user=user_input[:200],
        ai=ai_response[:200],
    )
    try:
        resp = await model.chat(
            [
                {"role": "user", "content": prompt},
            ]
        )
        text = resp.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        data = json.loads(text)
        if not data.get("remember"):
            return None
        content = data.get("content", "").strip()
        category = data.get("category", "fact")
        if content:
            return content, category
        return None
    except Exception as e:
        logger.debug("Memory judge error: %s", e)
        return None
    finally:
        await model.close()


# ---------------------------------------------------------------------------
# Passive save helper
# ---------------------------------------------------------------------------


async def save_memory(content: str, key: str = "", category: str = "fact") -> str:
    s = await get_memory_store()
    return await s.remember(content, key, category)


# ---------------------------------------------------------------------------
# Unified memory tool (hermes pattern: single tool with action parameter)
# ---------------------------------------------------------------------------


async def memory(
    action: Literal["save", "get", "list", "delete"],
    content: str = "",
    key: str = "",
    category: Literal["preference", "identity", "contact", "project", "fact"] = "fact",
    query: str = "",
    keys: list[str] | None = None,
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
    - list: 列出记忆。每条记忆的键名就是该条记忆的内容摘要，默认只返回键名。verbose=true 时同时返回完整内容
    - delete: 删除指定记忆条目。需要 keys 数组

    不要保存：任务进度、闲聊、一次性指令、通用知识。

    Args:
        action: 操作类型: save, get, list, delete
        content: 记忆内容（save 时必填）
        key: 记忆键名，即记忆内容摘要（save 可选，get 必填）
        category: 记忆分类（默认 fact）
        query: list 用关键词过滤记忆
        keys: 要删除的记忆键名列表（delete 必填）
        verbose: list 时是否同时返回完整内容。默认只返回键名
    """
    normalized = (action or "").strip().lower()

    if normalized == "save":
        if not content:
            return json.dumps({"error": "content is required for save action"}, ensure_ascii=False)
        return await save_memory(content, key, category)

    if normalized == "get":
        if not key:
            return json.dumps({"error": "key is required for get action"}, ensure_ascii=False)
        s = await get_memory_store()
        return await s.recall(key)

    if normalized == "list":
        s = await get_memory_store()
        items = await s.list_all(query)
        if verbose:
            return json.dumps(items, ensure_ascii=False)
        return json.dumps([i["key"] for i in items], ensure_ascii=False)

    if normalized == "delete":
        if not keys:
            return json.dumps({"error": "keys is required for delete action"}, ensure_ascii=False)
        unique_keys = list(dict.fromkeys(k for k in keys if k))
        if not unique_keys:
            return json.dumps({"error": "No valid keys specified"}, ensure_ascii=False)
        s = await get_memory_store()
        found_keys = []
        previews = []
        for k in unique_keys:
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
        from src.tools.exec import _current_agent_context

        _mgr = get_approval_manager()
        _args_preview = "\n".join(previews)[:200]
        _tid = _current_agent_context.get({}).get("parent_thread_id", "")
        if not _mgr.has_durable_approval("memory_delete", _args_preview):
            if not _mgr.has_session_approval(_tid, "memory_delete", _args_preview):
                raise MemoryDeleteNeedsApproval(found_keys, previews)

        # Approved via session/durable — execute delete directly
        deleted = []
        for k in found_keys:
            await s.forget(k)
            deleted.append(k)
        return json.dumps({"ok": True, "deleted": deleted, "count": len(deleted)}, ensure_ascii=False)

    return json.dumps({"error": f"Unknown action '{action}'. Use: save, get, list, delete"}, ensure_ascii=False)


def get_tools() -> list[ToolDef]:
    return [
        ToolDef.from_function(memory),
    ]
