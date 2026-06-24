"""Session pruning and cleanup utilities.

Provides automatic and manual session pruning to keep the checkpoints database
from growing unbounded. Design:
- Opt-in auto-pruning at startup
- Manual prune command with configurable age threshold
- VACUUM after prune to reclaim disk space
- Only prunes inactive sessions (is_active=0)
- Async SQLite via aiosqlite for non-blocking operation
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

import aiosqlite

logger = logging.getLogger("flyclaw.session.pruner")

# Prune interval tracking table
_PRUNE_TRACKING_SQL = """
CREATE TABLE IF NOT EXISTS prune_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pruned_at REAL NOT NULL,
    sessions_removed INTEGER NOT NULL,
    older_than_days INTEGER NOT NULL
)
"""


def _get_prune_tracker_path(checkpoints_path: str) -> str:
    """Get the path to the prune tracking database (same dir as checkpoints)."""
    return str(Path(checkpoints_path).parent / "prune_tracking.db")


async def _connect_with_retry(db_path: str, retries: int = 3) -> aiosqlite.Connection:
    """Connect to SQLite with retry logic for concurrent access."""
    for attempt in range(retries):
        try:
            conn = await aiosqlite.connect(db_path, timeout=30)
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA busy_timeout=5000")
            return conn
        except Exception as e:
            if "locked" in str(e).lower() and attempt < retries - 1:
                await asyncio.sleep(1 * (attempt + 1))
                continue
            raise
    return await aiosqlite.connect(db_path, timeout=30)


async def should_prune_now(checkpoints_path: str, min_interval_hours: int) -> bool:
    """Check if enough time has passed since the last prune."""
    if min_interval_hours <= 0:
        return True

    tracker_path = _get_prune_tracker_path(checkpoints_path)
    if not Path(tracker_path).exists():
        return True

    try:
        async with aiosqlite.connect(tracker_path) as conn:
            row = await conn.execute_fetchall("SELECT pruned_at FROM prune_history ORDER BY pruned_at DESC LIMIT 1")

        if not row:
            return True

        last_prune = row[0][0]
        hours_since = (time.time() - last_prune) / 3600
        return hours_since >= min_interval_hours
    except Exception as e:
        logger.warning("Failed to check prune history: %s", e)
        return True


async def record_prune(checkpoints_path: str, sessions_removed: int, older_than_days: int) -> None:
    """Record a prune event for interval tracking."""
    tracker_path = _get_prune_tracker_path(checkpoints_path)
    try:
        async with aiosqlite.connect(tracker_path) as conn:
            await conn.execute(_PRUNE_TRACKING_SQL)
            await conn.execute(
                "INSERT INTO prune_history (pruned_at, sessions_removed, older_than_days) VALUES (?, ?, ?)",
                (time.time(), sessions_removed, older_than_days),
            )
            await conn.commit()
    except Exception as e:
        logger.warning("Failed to record prune event: %s", e)


async def prune_session_index(session_index_path: str, thread_ids: list[str]) -> int:
    """Remove sessions and messages from session_index.db.

    Args:
        session_index_path: Path to session_index.db
        thread_ids: List of thread_ids to remove

    Returns:
        Number of sessions removed
    """
    if not thread_ids or not Path(session_index_path).exists():
        return 0

    try:
        async with aiosqlite.connect(session_index_path, timeout=30) as conn:
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA busy_timeout=5000")

            removed = 0
            for tid in thread_ids:
                cursor = await conn.execute("DELETE FROM sessions WHERE thread_id = ?", (tid,))
                if cursor.rowcount > 0:
                    removed += cursor.rowcount
                await conn.execute("DELETE FROM messages WHERE thread_id = ?", (tid,))

            await conn.commit()

        logger.info("Pruned %d sessions from session_index", removed)
        return removed
    except Exception as e:
        logger.warning("Failed to prune session_index: %s", e)
        return 0


async def prune_sessions(
    checkpoints_path: str,
    older_than_days: int = 90,
    session_index_path: Optional[str] = None,
    dry_run: bool = False,
) -> dict:
    """Prune sessions older than the specified threshold.

    Only prunes inactive sessions (is_active=0 in session_index) to protect
    active conversations.

    Args:
        checkpoints_path: Path to the checkpoints database
        older_than_days: Delete sessions with updated_at older than this
        session_index_path: Optional path to session_index.db for sync cleanup
        dry_run: If True, only report what would be deleted

    Returns:
        Dict with prune statistics
    """
    stats = {
        "total_sessions": 0,
        "sessions_to_prune": 0,
        "sessions_removed": 0,
        "index_sessions_removed": 0,
        "freed_bytes": 0,
        "dry_run": dry_run,
    }

    if not Path(checkpoints_path).exists():
        logger.warning("Checkpoints database not found: %s", checkpoints_path)
        return stats

    cutoff_time = time.time() - (older_than_days * 86400)

    try:
        conn = await _connect_with_retry(checkpoints_path)

        try:
            # Get total session count
            cursor = await conn.execute("SELECT count(*) FROM sessions")
            row = await cursor.fetchone()
            stats["total_sessions"] = row[0]

            # Get inactive sessions from session_index (if available)
            inactive_threads: set[str] = set()
            if session_index_path and Path(session_index_path).exists():
                try:
                    async with aiosqlite.connect(session_index_path, timeout=30) as idx_conn:
                        rows = await idx_conn.execute_fetchall("SELECT thread_id FROM sessions WHERE is_active = 0")
                        inactive_threads = {r[0] for r in rows}
                    logger.debug("Found %d inactive sessions in session_index", len(inactive_threads))
                except Exception as e:
                    logger.warning("Failed to get inactive sessions: %s", e)

            # Find sessions to prune (old AND inactive)
            if inactive_threads:
                # Only prune sessions that are both old AND inactive
                placeholders = ",".join("?" for _ in inactive_threads)
                to_prune = await conn.execute_fetchall(
                    f"SELECT thread_id, length(messages) FROM sessions "
                    f"WHERE updated_at < ? AND thread_id IN ({placeholders})",
                    [cutoff_time] + list(inactive_threads),
                )
            else:
                # Fallback: prune all old sessions (no session_index available)
                to_prune = await conn.execute_fetchall(
                    "SELECT thread_id, length(messages) FROM sessions WHERE updated_at < ?",
                    (cutoff_time,),
                )

            stats["sessions_to_prune"] = len(to_prune)

            if not to_prune:
                logger.info("No sessions to prune (all within %d days or active)", older_than_days)
                return stats

            # Log what will be pruned
            for thread_id, msg_size in to_prune:
                logger.info(
                    "  Would prune: %s (msg_size=%d bytes)",
                    thread_id,
                    msg_size,
                )
                stats["freed_bytes"] += msg_size

            if dry_run:
                logger.info(
                    "[DRY RUN] Would prune %d sessions, freeing ~%d bytes",
                    len(to_prune),
                    stats["freed_bytes"],
                )
                return stats

            # Collect thread IDs to remove
            thread_ids_to_remove = [r[0] for r in to_prune]
            placeholders = ",".join("?" for _ in thread_ids_to_remove)

            # Clean up tool cache files for pruned threads
            try:
                from src.agent.tool_cache import clear_thread_cache

                for tid in thread_ids_to_remove:
                    clear_thread_cache(tid)
            except Exception as e:
                logger.warning("Tool cache cleanup failed (non-fatal): %s", e)

            # Async cleanup session_index.db first (dependent DB; failure here is tolerable)
            if session_index_path:
                try:
                    stats["index_sessions_removed"] = await prune_session_index(
                        session_index_path, thread_ids_to_remove
                    )
                except Exception as e:
                    logger.warning("session_index cleanup failed (non-fatal): %s", e)

            # Then delete from checkpoints.db (source of truth)
            await conn.execute(
                f"DELETE FROM sessions WHERE thread_id IN ({placeholders})",
                thread_ids_to_remove,
            )
            cursor = await conn.execute("SELECT changes()")
            row = await cursor.fetchone()
            stats["sessions_removed"] = row[0]
            await conn.commit()

            logger.info(
                "Pruned %d sessions older than %d days",
                stats["sessions_removed"],
                older_than_days,
            )

            # Record prune event
            await record_prune(checkpoints_path, stats["sessions_removed"], older_than_days)

        finally:
            await conn.close()

    except Exception as e:
        logger.error("Prune failed: %s", e)

    return stats


async def vacuum_database(checkpoints_path: str) -> int:
    """VACUUM the database to reclaim disk space after pruning.

    Uses wal_checkpoint instead of full VACUUM to minimize blocking.

    Returns:
        Size of database file after checkpoint (bytes)
    """
    if not Path(checkpoints_path).exists():
        return 0

    try:
        async with aiosqlite.connect(checkpoints_path, timeout=30) as conn:
            # Use WAL checkpoint instead of full VACUUM (less blocking)
            await conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

        size = Path(checkpoints_path).stat().st_size
        logger.info("Database checkpoint complete, size: %d bytes", size)
        return size
    except Exception as e:
        logger.warning("Database checkpoint failed: %s", e)
        return 0
