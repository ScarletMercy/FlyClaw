#!/usr/bin/env python3
"""一次性恢复脚本：从 session_index.db 恢复 checkpoints.db

用途：当 checkpoints.db 损坏或丢失时，从 session_index.db 的索引数据恢复会话状态。
注意：这是临时工具，恢复完成后可删除。

使用方法：
    python scripts/recover_checkpoints.py [--dry-run] [--db-path PATH]
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("recover")

# 角色映射：session_index → OpenAI format
ROLE_MAP = {
    "human": "user",
    "user": "user",
    "ai": "assistant",
    "assistant": "assistant",
    "tool": "tool",
    "system": "system",
}


def parse_args():
    parser = argparse.ArgumentParser(description="从 session_index.db 恢复 checkpoints.db")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只分析不写入数据库",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="自定义 checkpoints.db 路径 (默认: ~/.myclaw/data/checkpoints.db)",
    )
    parser.add_argument(
        "--index-path",
        default=None,
        help="自定义 session_index.db 路径 (默认: ~/.myclaw/data/session_index.db)",
    )
    return parser.parse_args()


def get_default_paths():
    data_dir = Path.home() / ".myclaw" / "data"
    return {
        "checkpoints": str(data_dir / "checkpoints.db"),
        "session_index": str(data_dir / "session_index.db"),
    }


def convert_message_to_openai_format(msg_row) -> dict:
    """将 session_index 的消息格式转换为 OpenAI chat format。

    session_index 格式：
        (thread_id, message_id, role, content, tool_name, tool_calls, timestamp)

    OpenAI format：
        user: {"role": "user", "content": "..."}
        assistant: {"role": "assistant", "content": "...", "tool_calls": [...]}
        tool: {"role": "tool", "content": "...", "tool_call_id": "...", "name": "..."}
    """
    thread_id, message_id, role, content, tool_name, tool_calls, timestamp = msg_row

    openai_role = ROLE_MAP.get(role, "user")

    msg = {"role": openai_role}

    if openai_role == "tool":
        # tool 消息需要 tool_call_id 和 name
        msg["content"] = content or ""
        msg["tool_call_id"] = message_id  # 用 message_id 作为 tool_call_id
        if tool_name:
            msg["name"] = tool_name
    elif openai_role == "assistant":
        msg["content"] = content or ""
        # 恢复 tool_calls
        if tool_calls:
            try:
                # session_index 存储的是简化格式，需要转换回 OpenAI 格式
                simplified = json.loads(tool_calls)
                msg["tool_calls"] = [
                    {
                        "id": f"call_{message_id}_{i}",
                        "type": "function",
                        "function": {
                            "name": tc.get("name", "unknown"),
                            "arguments": tc.get("args", "{}"),
                        },
                    }
                    for i, tc in enumerate(simplified)
                ]
            except (json.JSONDecodeError, TypeError):
                pass
    else:
        # user / system
        msg["content"] = content or ""

    return msg


def recover_checkpoints(index_path: str, checkpoints_path: str, dry_run: bool = False) -> dict:
    """从 session_index.db 恢复 checkpoints.db。

    返回统计信息字典。
    """
    stats = {
        "sessions_found": 0,
        "sessions_recovered": 0,
        "messages_recovered": 0,
        "errors": [],
    }

    # 连接源数据库
    if not Path(index_path).exists():
        logger.error("session_index.db 不存在: %s", index_path)
        return stats

    idx_conn = sqlite3.connect(index_path)
    logger.info("已连接 session_index.db: %s", index_path)

    # 获取所有会话
    sessions = idx_conn.execute(
        "SELECT thread_id, channel, sender_id, chat_id, chat_type, "
        "first_message_at, last_message_at, message_count, is_active "
        "FROM sessions ORDER BY last_message_at DESC"
    ).fetchall()

    stats["sessions_found"] = len(sessions)
    logger.info("找到 %d 个会话", len(sessions))

    if dry_run:
        logger.info("[DRY RUN] 不会写入任何数据")
        for s in sessions:
            messages = idx_conn.execute(
                "SELECT thread_id, message_id, role, content, tool_name, tool_calls, timestamp "
                "FROM messages WHERE thread_id = ? ORDER BY timestamp ASC",
                (s[0],),
            ).fetchall()
            logger.info(
                "  %s: %d 条消息 (channel=%s, sender=%s)",
                s[0], len(messages), s[1], s[2],
            )
        idx_conn.close()
        return stats

    # 连接目标数据库
    checkpoints_dir = Path(checkpoints_path).parent
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    cp_conn = sqlite3.connect(checkpoints_path)
    cp_conn.execute("PRAGMA journal_mode=WAL")
    cp_conn.execute("PRAGMA busy_timeout=5000")

    # 创建 sessions 表
    cp_conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            thread_id TEXT PRIMARY KEY,
            messages TEXT NOT NULL,
            metadata TEXT NOT NULL,
            updated_at REAL NOT NULL
        )
    """)
    cp_conn.execute("CREATE INDEX IF NOT EXISTS idx_updated ON sessions(updated_at)")
    cp_conn.commit()

    # 恢复每个会话
    for session in sessions:
        thread_id, channel, sender_id, chat_id, chat_type, first_at, last_at, msg_count, is_active = session

        try:
            # 获取该会话的所有消息
            messages = idx_conn.execute(
                "SELECT thread_id, message_id, role, content, tool_name, tool_calls, timestamp "
                "FROM messages WHERE thread_id = ? ORDER BY timestamp ASC",
                (thread_id,),
            ).fetchall()

            if not messages:
                logger.debug("跳过空会话: %s", thread_id)
                continue

            # 转换消息格式
            openai_messages = []
            for msg_row in messages:
                try:
                    openai_msg = convert_message_to_openai_format(msg_row)
                    openai_messages.append(openai_msg)
                except Exception as e:
                    stats["errors"].append(f"消息转换失败 {msg_row[1]}: {e}")
                    logger.warning("消息转换失败 %s: %s", msg_row[1], e)

            if not openai_messages:
                continue

            # 构建元数据
            metadata = {
                "system_prompt": "",
                "sender_id": sender_id or "",
                "chat_id": chat_id or "",
                "chat_type": chat_type or "p2p",
                "message_id": "",
                "user_role": "",
                "channel": channel or "",
                "pending_approval": None,
            }

            # 写入 checkpoints.db
            cp_conn.execute(
                """INSERT OR REPLACE INTO sessions (thread_id, messages, metadata, updated_at)
                   VALUES (?, ?, ?, ?)""",
                (
                    thread_id,
                    json.dumps(openai_messages, ensure_ascii=False),
                    json.dumps(metadata, ensure_ascii=False),
                    last_at or time.time(),
                ),
            )

            stats["sessions_recovered"] += 1
            stats["messages_recovered"] += len(openai_messages)
            logger.info(
                "恢复会话: %s (%d 条消息)",
                thread_id, len(openai_messages),
            )

        except Exception as e:
            stats["errors"].append(f"会话恢复失败 {thread_id}: {e}")
            logger.error("会话恢复失败 %s: %s", thread_id, e)

    cp_conn.commit()
    cp_conn.close()
    idx_conn.close()

    return stats


def main():
    args = parse_args()
    paths = get_default_paths()

    index_path = args.index_path or paths["session_index"]
    checkpoints_path = args.db_path or paths["checkpoints"]

    logger.info("=" * 60)
    logger.info("MyClaw 会话数据恢复工具")
    logger.info("=" * 60)
    logger.info("源数据库 (session_index): %s", index_path)
    logger.info("目标数据库 (checkpoints): %s", checkpoints_path)
    logger.info("模式: %s", "DRY RUN (只分析)" if args.dry_run else "恢复 (写入数据)")
    logger.info("")

    stats = recover_checkpoints(index_path, checkpoints_path, dry_run=args.dry_run)

    logger.info("")
    logger.info("=" * 60)
    logger.info("恢复统计")
    logger.info("=" * 60)
    logger.info("找到会话数: %d", stats["sessions_found"])
    logger.info("恢复会话数: %d", stats["sessions_recovered"])
    logger.info("恢复消息数: %d", stats["messages_recovered"])

    if stats["errors"]:
        logger.warning("错误数: %d", len(stats["errors"]))
        for err in stats["errors"][:10]:
            logger.warning("  - %s", err)
        if len(stats["errors"]) > 10:
            logger.warning("  ... 还有 %d 个错误", len(stats["errors"]) - 10)

    logger.info("")
    if args.dry_run:
        logger.info("这是 DRY RUN 模式，未写入任何数据。")
        logger.info("移除 --dry-run 参数执行实际恢复。")
    else:
        logger.info("恢复完成！")
        logger.info("请重启 MyClaw 服务以使用恢复的数据。")

    return 0 if not stats["errors"] else 1


if __name__ == "__main__":
    sys.exit(main())
