"""Procedural memory — learn reusable workflow patterns from agent sessions.

Stores multi-step tool sequences as searchable procedures. Automatically
extracts procedures from successful sessions via event bus listener.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiosqlite

from src.agent.tooldef import ToolDef

logger = logging.getLogger("flyclaw.procedures")


# ── ProcedureStore ────────────────────────────────────────────────

class ProcedureStore:
    """SQLite+FTS5 backed store for learned procedures."""

    def __init__(self, db_path: str, max_procedures: int = 200):
        self.db_path = db_path
        self.max_procedures = max_procedures
        self._conn: Optional[aiosqlite.Connection] = None

    async def initialize(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA synchronous=NORMAL")

        await self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS procedures (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                trigger_patterns TEXT NOT NULL DEFAULT '',
                steps TEXT NOT NULL,
                tags TEXT NOT NULL DEFAULT '',
                source_thread TEXT NOT NULL DEFAULT '',
                use_count INTEGER NOT NULL DEFAULT 0,
                success_count INTEGER NOT NULL DEFAULT 1,
                fail_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
        """)

        try:
            await self._conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS procedures_fts "
                "USING fts5(name, description, trigger_patterns, tags)"
            )
        except Exception as e:
            logger.warning("Failed to create procedures FTS5 table: %s", e)

        await self._conn.commit()
        logger.info("ProcedureStore initialized: %s", self.db_path)

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def add(
        self,
        name: str,
        description: str,
        steps: list[dict],
        trigger_patterns: str = "",
        tags: str = "",
        source_thread: str = "",
    ) -> str:
        cursor = await self._conn.execute(
            "SELECT id FROM procedures WHERE name = ?", (name,)
        )
        existing = await cursor.fetchone()
        if existing:
            logger.debug("Procedure '%s' already exists (%s), skipping", name, existing["id"])
            return existing["id"]

        proc_id = uuid.uuid4().hex[:8]
        now = datetime.now(timezone.utc).isoformat()
        steps_json = json.dumps(steps, ensure_ascii=False)

        await self._conn.execute(
            """INSERT INTO procedures
               (id, name, description, trigger_patterns, steps, tags,
                source_thread, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (proc_id, name, description, trigger_patterns, steps_json,
             tags, source_thread, now, now),
        )

        # TEXT PRIMARY KEY is not rowid alias — query the implicit rowid explicitly
        cursor = await self._conn.execute(
            "SELECT rowid FROM procedures WHERE id = ?", (proc_id,)
        )
        row = await cursor.fetchone()
        rid = row["rowid"] if row else None

        if rid is not None:
            try:
                await self._conn.execute(
                    "INSERT INTO procedures_fts(rowid, name, description, trigger_patterns, tags) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (rid, name, description, trigger_patterns, tags),
                )
            except Exception as e:
                logger.warning("FTS5 insert failed for procedure %s: %s", proc_id, e)

        await self._conn.commit()
        await self._evict()
        logger.info("Procedure added: %s (%s)", name, proc_id)
        return proc_id

    async def search(self, query: str, max_results: int = 5) -> list[dict]:
        if not query.strip():
            return []

        fts_query = self._format_fts_query(query)
        results = []

        if fts_query:
            try:
                cursor = await self._conn.execute(
                    """SELECT p.id, p.name, p.description, p.trigger_patterns,
                              p.steps, p.tags, p.use_count, p.success_count, p.fail_count,
                              bm25(procedures_fts) AS score
                       FROM procedures_fts f
                       JOIN procedures p ON p.rowid = f.rowid
                       WHERE procedures_fts MATCH ?
                       ORDER BY score
                       LIMIT ?""",
                    (fts_query, max_results),
                )
                rows = await cursor.fetchall()
                for row in rows:
                    results.append(self._row_to_dict(row))
            except Exception as e:
                logger.warning("Procedure search failed: %s", e)

        if len(results) < max_results:
            needed = max_results - len(results)
            existing_ids = {r["id"] for r in results}
            extras = await self._fallback_search(query, needed, existing_ids)
            results.extend(extras)

        return results

    async def get(self, proc_id: str) -> Optional[dict]:
        cursor = await self._conn.execute(
            "SELECT * FROM procedures WHERE id = ?", (proc_id,)
        )
        row = await cursor.fetchone()
        return self._row_to_dict(row) if row else None

    async def record_use(self, proc_id: str, success: bool) -> None:
        now = datetime.now(timezone.utc).isoformat()
        if success:
            await self._conn.execute(
                "UPDATE procedures SET use_count = use_count + 1, "
                "success_count = success_count + 1, updated_at = ? WHERE id = ?",
                (now, proc_id),
            )
        else:
            await self._conn.execute(
                "UPDATE procedures SET use_count = use_count + 1, "
                "fail_count = fail_count + 1, updated_at = ? WHERE id = ?",
                (now, proc_id),
            )
        await self._conn.commit()

    async def delete(self, proc_id: str) -> bool:
        cursor = await self._conn.execute(
            "SELECT rowid FROM procedures WHERE id = ?", (proc_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return False

        rid = row["rowid"]
        await self._conn.execute("DELETE FROM procedures WHERE id = ?", (proc_id,))
        try:
            await self._conn.execute(
                "DELETE FROM procedures_fts WHERE rowid = ?", (rid,)
            )
        except Exception:
            pass
        await self._conn.commit()
        return True

    async def list_all(self, tag: str = "", limit: int = 50) -> list[dict]:
        if tag:
            cursor = await self._conn.execute(
                "SELECT * FROM procedures WHERE tags LIKE ? ORDER BY updated_at DESC LIMIT ?",
                (f"%{tag}%", limit),
            )
        else:
            cursor = await self._conn.execute(
                "SELECT * FROM procedures ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            )
        rows = await cursor.fetchall()
        return [self._row_to_dict(r) for r in rows]

    async def count(self) -> int:
        cursor = await self._conn.execute("SELECT COUNT(*) as cnt FROM procedures")
        row = await cursor.fetchone()
        return row["cnt"] if row else 0

    # ── Internal ──────────────────────────────────────────

    def _row_to_dict(self, row) -> dict:
        steps_raw = row["steps"] if isinstance(row["steps"], str) else "[]"
        try:
            steps = json.loads(steps_raw)
        except (json.JSONDecodeError, TypeError):
            steps = []
        return {
            "id": row["id"],
            "name": row["name"],
            "description": row["description"],
            "trigger_patterns": row["trigger_patterns"],
            "steps": steps,
            "tags": row["tags"],
            "source_thread": row["source_thread"],
            "use_count": row["use_count"],
            "success_count": row["success_count"],
            "fail_count": row["fail_count"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    @staticmethod
    def _format_fts_query(query: str) -> str:
        parts = re.split(r'\s+', query)
        fts_reserved = {"AND", "OR", "NOT", "NEAR"}
        cleaned = []
        for p in parts:
            p = p.strip('"*+-^:.')
            if p and p.upper() not in fts_reserved:
                cleaned.append(p)
        return " OR ".join(cleaned) if cleaned else ""

    async def _fallback_search(
        self, query: str, limit: int, exclude_ids: set[str]
    ) -> list[dict]:
        terms = query.lower().split()
        cursor = await self._conn.execute(
            "SELECT * FROM procedures ORDER BY use_count DESC, updated_at DESC"
        )
        rows = await cursor.fetchall()
        results = []
        for row in rows:
            if row["id"] in exclude_ids:
                continue
            text = f"{row['name']} {row['description']} {row['trigger_patterns']} {row['tags']}".lower()
            if any(t in text for t in terms):
                results.append(self._row_to_dict(row))
                if len(results) >= limit:
                    break
        return results

    async def _evict(self) -> None:
        count = await self.count()
        if count <= self.max_procedures:
            return
        excess = count - self.max_procedures
        cursor = await self._conn.execute(
            "SELECT rowid, id FROM procedures ORDER BY use_count ASC, updated_at ASC LIMIT ?",
            (excess,),
        )
        rows = await cursor.fetchall()
        if not rows:
            return
        for row in rows:
            await self._conn.execute("DELETE FROM procedures WHERE id = ?", (row["id"],))
            try:
                await self._conn.execute("DELETE FROM procedures_fts WHERE rowid = ?", (row["rowid"],))
            except Exception:
                pass
        await self._conn.commit()


# ── Singleton ─────────────────────────────────────────────────────

_store: Optional[ProcedureStore] = None
_store_lock: Optional[asyncio.Lock] = None


def _get_store_lock() -> asyncio.Lock:
    global _store_lock
    if _store_lock is None:
        _store_lock = asyncio.Lock()
    return _store_lock


async def get_procedure_store() -> ProcedureStore:
    global _store
    if _store is None:
        async with _get_store_lock():
            if _store is None:
                from src.config import load_config

                config = load_config()
                _store = ProcedureStore(
                    config.procedural_memory.db_path,
                    max_procedures=config.procedural_memory.max_procedures,
                )
                await _store.initialize()
    return _store


async def reset_procedure_store() -> None:
    global _store
    async with _get_store_lock():
        if _store is not None:
            try:
                await _store.close()
            except Exception:
                pass
        _store = None


# ── ProcedureExtractor ────────────────────────────────────────────

_EXTRACTION_PROMPT = """\
You are analyzing an AI assistant's session to extract reusable workflow patterns.

## Session Summary

User request: {user_message}

Tool calls executed:
{tool_sequence}

Final result: {result_summary}

## Task

Determine if this tool sequence represents a reusable multi-step workflow that the assistant might encounter again.

Criteria for a reusable workflow:
- Involves 3+ distinct tool calls that form a coherent sequence
- Solves a common task type (not one-off debugging)
- The steps are generally applicable, not specific to one file/URL

If this IS a reusable workflow, respond with a JSON object:
{{
  "is_procedure": true,
  "name": "short-kebab-case-name",
  "description": "What this workflow accomplishes",
  "trigger_patterns": "comma-separated keywords that would match similar tasks",
  "steps": [
    {{"tool": "tool_name", "args_summary": "key arguments pattern", "purpose": "why this step"}}
  ],
  "tags": "comma-separated category tags"
}}

If NOT reusable (simple, one-off, failed, or trivial), respond with:
{{"is_procedure": false}}

Respond with ONLY the JSON object, no other text."""


def _extract_tool_sequence(messages: list[dict]) -> list[dict]:
    """Extract ordered tool call sequence from messages."""
    sequence = []
    call_id_to_idx: dict[str, int] = {}

    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                tc_id = tc.get("id", "")
                name = fn.get("name", "")
                args_str = fn.get("arguments", "{}")

                try:
                    args = json.loads(args_str) if args_str else {}
                except (json.JSONDecodeError, TypeError):
                    args = {}

                args_preview = json.dumps(args, ensure_ascii=False)[:150]
                idx = len(sequence)
                call_id_to_idx[tc_id] = idx
                sequence.append({
                    "tool": name,
                    "args_summary": args_preview,
                    "call_id": tc_id,
                    "success": None,  # Will be set when result arrives
                })

        elif msg.get("role") == "tool":
            content = msg.get("content", "")
            is_error = isinstance(content, str) and content.startswith("[error]")
            tc_id = msg.get("tool_call_id", "")

            idx = call_id_to_idx.get(tc_id)
            if idx is not None:
                sequence[idx]["success"] = not is_error

    # For calls without a result, assume success (no error result was seen)
    for entry in sequence:
        if entry["success"] is None:
            entry["success"] = True

    return sequence


def _get_user_message(messages: list[dict]) -> str:
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content[:500]
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return part.get("text", "")[:500]
    return ""


def _get_result_summary(messages: list[dict]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            content = msg["content"]
            if isinstance(content, str):
                return content[:500]
    return ""


async def _try_extract_procedure(thread_id: str, messages: list[dict]) -> None:
    """Try to extract a procedure from a completed session. Runs in background."""
    try:
        store = await get_procedure_store()

        cursor = await store._conn.execute(
            "SELECT id FROM procedures WHERE source_thread = ?", (thread_id,)
        )
        if await cursor.fetchone():
            logger.debug("Thread %s already has extracted procedure, skipping", thread_id)
            return

        tool_seq = _extract_tool_sequence(messages)
        if len(tool_seq) < 3:
            return

        successful = [s for s in tool_seq if s.get("success", True)]
        if len(successful) < len(tool_seq) * 0.7:
            return

        user_msg = _get_user_message(messages)
        result_summary = _get_result_summary(messages)
        if not user_msg:
            return

        tool_text = "\n".join(
            f"{i + 1}. {s['tool']}({s['args_summary'][:80]}) -> "
            f"{'ok' if s.get('success', True) else 'fail'}"
            for i, s in enumerate(tool_seq)
        )

        prompt = _EXTRACTION_PROMPT.format(
            user_message=user_msg,
            tool_sequence=tool_text,
            result_summary=result_summary,
        )

        from src.config import load_config
        config = load_config()

        model_name = config.procedural_memory.learn_model or config.model.name
        base_url = config.procedural_memory.learn_model_base_url or config.model.base_url
        api_key = config.procedural_memory.learn_model_api_key or config.model.api_key

        from src.agent.client import create_client
        client = create_client(
            config.model.provider,
            model_name,
            0.0,
            base_url=base_url,
            api_key=api_key,
        )

        try:
            response = await client.chat([{"role": "user", "content": prompt}])
        finally:
            try:
                await client.close()
            except Exception:
                pass

        content = response.content if hasattr(response, "content") else str(response)
        content = content.strip()

        # Extract JSON from code fences if present
        fence_match = re.search(r"```(?:json)?\s*\n(.*?)```", content, re.DOTALL)
        if fence_match:
            content = fence_match.group(1).strip()

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            logger.debug("Procedure extraction returned non-JSON: %.200s", content)
            return

        if not parsed.get("is_procedure"):
            return

        name = parsed.get("name", "")
        description = parsed.get("description", "")
        steps = parsed.get("steps", [])
        trigger_patterns = parsed.get("trigger_patterns", "")
        tags = parsed.get("tags", "")

        if not name or not description or not steps:
            return

        proc_id = await store.add(
            name=name,
            description=description,
            steps=steps,
            trigger_patterns=trigger_patterns,
            tags=tags,
            source_thread=thread_id,
        )
        logger.info(
            "Auto-learned procedure '%s' (%s) from thread %s",
            name, proc_id, thread_id,
        )

    except Exception as e:
        logger.warning("Procedure extraction failed for thread %s: %s", thread_id, e)


async def _on_agent_loop_completed(
    event: str,
    thread_id: str = "",
    total_rounds: int = 0,
    **kwargs,
) -> None:
    """Event handler: auto-learn procedures from successful sessions."""
    from src.config import load_config

    config = load_config()
    if not config.procedural_memory.enabled or not config.procedural_memory.auto_learn:
        return

    min_calls = config.procedural_memory.min_tool_calls
    if total_rounds < min_calls:
        return

    try:
        from src._container import get_container
        container = get_container()
        state_store = container.state_store
        if not state_store:
            return

        state = await state_store.aload(thread_id)
        if not state or not state.messages:
            return

        messages = state.messages

        # Only extract if session ended with a successful assistant response
        last_assistant = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                last_assistant = msg
                break

        if not last_assistant:
            return

        content = last_assistant.get("content", "")
        if isinstance(content, str) and (
            "[error]" in content
            or "[denied]" in content
            or "timed out" in content.lower()
        ):
            return

        task = asyncio.create_task(_try_extract_procedure(thread_id, messages))
        task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)

    except Exception as e:
        logger.debug("Procedure extraction trigger failed: %s", e)


_listener_registered = False


def register_extraction_listener() -> None:
    """Register the event listener for auto-learning procedures."""
    global _listener_registered
    if _listener_registered:
        return

    from src.events import subscribe_async
    subscribe_async("agent_loop.completed", _on_agent_loop_completed)
    _listener_registered = True
    logger.info("Procedural memory extraction listener registered")


# ── Tool functions ────────────────────────────────────────────────

async def procedure_search(query: str, max_results: int = 3) -> str:
    """搜索已知的工作流模式。遇到需要多步骤操作时先搜索是否已有现成流程。

    Args:
        query: 任务描述或关键词。
        max_results: 最多返回几个结果。
    """
    try:
        store = await get_procedure_store()
        results = await store.search(query, max_results=max_results)
        if not results:
            return "No matching procedures found."

        output = []
        for r in results:
            steps_text = "\n".join(
                f"    {i + 1}. {s.get('tool', '?')}: {s.get('purpose', s.get('args_summary', ''))}"
                if isinstance(s, dict) else f"    {i + 1}. {s}"
                for i, s in enumerate(r["steps"])
            )
            output.append(
                f"[{r['id']}] {r['name']}\n"
                f"  Description: {r['description']}\n"
                f"  Triggers: {r['trigger_patterns']}\n"
                f"  Steps:\n{steps_text}\n"
                f"  Used {r['use_count']}x (success {r['success_count']}/{r['use_count'] or 1})"
            )
        return "\n\n".join(output)

    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"


async def procedure_learn(
    name: str,
    description: str,
    steps: str,
    trigger_patterns: str = "",
    tags: str = "",
) -> str:
    """手动保存一个工作流模式供未来复用。

    Args:
        name: 工作流名称。
        description: 工作流描述。
        steps: JSON 数组，每项 {tool, args_summary, purpose}。
        trigger_patterns: 触发关键词（逗号分隔）。
        tags: 标签（逗号分隔）。
    """
    try:
        parsed_steps = json.loads(steps) if isinstance(steps, str) else steps
    except json.JSONDecodeError as e:
        return f"[error] Invalid steps JSON: {e}"

    if not isinstance(parsed_steps, list) or not parsed_steps:
        return "[error] steps must be a non-empty JSON array."

    try:
        store = await get_procedure_store()
        proc_id = await store.add(
            name=name,
            description=description,
            steps=parsed_steps,
            trigger_patterns=trigger_patterns,
            tags=tags,
        )
        return json.dumps({
            "success": True,
            "id": proc_id,
            "name": name,
        }, ensure_ascii=False)
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"


async def procedure_list(tag: str = "", limit: int = 20) -> str:
    """列出已学习的工作流模式。

    Args:
        tag: 按标签过滤（可选）。
        limit: 最多返回数量。
    """
    try:
        store = await get_procedure_store()
        procedures = await store.list_all(tag=tag, limit=limit)
        if not procedures:
            return "No procedures learned yet."

        lines = []
        for p in procedures:
            lines.append(
                f"  [{p['id']}] {p['name']} — {p['description'][:80]} "
                f"(used {p['use_count']}x, tags: {p['tags']})"
            )
        return "\n".join(lines)

    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"


def get_tools() -> list[ToolDef]:
    return [
        ToolDef.from_function(procedure_search),
        ToolDef.from_function(procedure_list),
    ]
