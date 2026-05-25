"""One-time migration script: fix truncated tool_call arguments in checkpoints.db.

Usage: python scripts/migrate_tool_calls.py [path_to_checkpoints.db]

If no path is provided, defaults to ~/.flyclaw/data/checkpoints.db
"""
import json
import sqlite3
import sys
from pathlib import Path


def fix_args(args_str: str) -> str:
    """Fix truncated JSON arguments. Returns fixed string or original if valid."""
    try:
        json.loads(args_str)
        return args_str
    except (json.JSONDecodeError, TypeError):
        if args_str and args_str[-1] not in ("}", "]"):
            try:
                json.loads(args_str + "}")
                return args_str + "}"
            except (json.JSONDecodeError, TypeError):
                pass
        return "{}"


def migrate(db_path: str):
    db = sqlite3.connect(db_path)
    cursor = db.execute("SELECT thread_id, messages FROM sessions")
    fixed_threads = 0
    fixed_total = 0

    for thread_id, messages_json in cursor:
        messages = json.loads(messages_json)
        changed = False

        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    old_args = fn.get("arguments", "")
                    new_args = fix_args(old_args)
                    if old_args != new_args:
                        fn["arguments"] = new_args
                        changed = True
                        fixed_total += 1

        if changed:
            db.execute(
                "UPDATE sessions SET messages = ? WHERE thread_id = ?",
                (json.dumps(messages, ensure_ascii=False), thread_id)
            )
            fixed_threads += 1

    db.commit()
    db.close()

    print(f"Migration complete:")
    print(f"  Threads fixed: {fixed_threads}")
    print(f"  Arguments fixed: {fixed_total}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        db_path = Path.home() / ".flyclaw" / "data" / "checkpoints.db"
    else:
        db_path = sys.argv[1]

    if not Path(db_path).exists():
        print(f"Database not found: {db_path}")
        sys.exit(1)

    print(f"Migrating: {db_path}")
    migrate(str(db_path))
