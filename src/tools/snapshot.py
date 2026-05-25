"""File snapshot/rollback using a shadow git store.

Before every file-mutating operation (write_file, edit_file), the working
directory is snapshotted into a bare git repository.  Each working directory
gets its own ref and index file so projects stay isolated.

Commands:
    /snapshots           — list snapshots for current workspace
    /snapshots <id>      — show diff for a snapshot
    /rollback <id>       — restore entire workspace to snapshot
    /rollback <id> <f>   — restore a single file from snapshot
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("flyclaw.snapshot")

_EXCLUDE_PATTERNS = [
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".idea", ".vscode", "*.pyc", ".DS_Store", "Thumbs.db",
    "*.egg-info", ".mypy_cache", ".pytest_cache", ".ruff_cache",
]


def _dir_hash(path: str) -> str:
    return hashlib.sha256(path.encode()).hexdigest()[:16]


def _git(*args: str, cwd: str, env: dict | None = None) -> subprocess.CompletedProcess:
    cmd = ["git"] + list(args)
    run_env = dict(os.environ)
    if env:
        run_env.update(env)
    return subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, timeout=30, env=run_env,
    )


class CheckpointManager:
    """Manages file snapshots via a shadow git store."""

    def __init__(self, store_path: str, max_per_dir: int = 20, max_file_size: int = 10_000_000):
        self._store = Path(store_path)
        self._max_per_dir = max_per_dir
        self._max_file_size = max_file_size
        self._lock = threading.Lock()
        self._initialized_dirs: set[str] = set()

    # ── store bootstrap ──

    def _ensure_store(self):
        if self._store.exists() and (self._store / "HEAD").exists():
            return
        self._store.mkdir(parents=True, exist_ok=True)
        r = _git("init", "--bare", cwd=str(self._store))
        if r.returncode != 0:
            logger.error("Failed to init bare store: %s", r.stderr)
            raise RuntimeError(f"git init --bare failed: {r.stderr}")

    def _ensure_initialized(self, work_dir: str):
        if work_dir in self._initialized_dirs:
            return
        self._ensure_store()
        dh = _dir_hash(work_dir)
        ref = f"refs/flyclaw/{dh}"
        # Check if any commits exist for this dir
        r = _git("rev-parse", "--verify", ref, cwd=str(self._store))
        if r.returncode != 0:
            # Need an initial commit
            self._create_initial_commit(work_dir, dh, ref)
        self._initialized_dirs.add(work_dir)

    def _create_initial_commit(self, work_dir: str, dh: str, ref: str):
        idx_file = str(self._store / f"index-{dh}")
        env = {
            "GIT_DIR": str(self._store),
            "GIT_WORK_TREE": work_dir,
            "GIT_INDEX_FILE": idx_file,
        }
        # Add all files
        _git("add", "-A", cwd=work_dir, env=env)
        # Write tree from index
        r_tree = _git("write-tree", cwd=work_dir, env=env)
        if r_tree.returncode != 0:
            logger.debug("write-tree failed (empty dir?): %s", r_tree.stderr.strip())
            return
        tree_sha = r_tree.stdout.strip()
        if not tree_sha:
            return
        # Create an orphan commit (no parent) so directories are independent
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        r_commit = _git(
            "commit-tree", tree_sha, "-m", f"snapshot: initial ({ts})",
            cwd=str(self._store), env=env,
        )
        if r_commit.returncode != 0:
            logger.debug("commit-tree failed: %s", r_commit.stderr.strip())
            return
        sha = r_commit.stdout.strip()
        _git("update-ref", ref, sha, cwd=str(self._store))

    def _env_for(self, work_dir: str) -> dict:
        dh = _dir_hash(work_dir)
        idx_file = str(self._store / f"index-{dh}")
        return {
            "GIT_DIR": str(self._store),
            "GIT_WORK_TREE": work_dir,
            "GIT_INDEX_FILE": idx_file,
        }

    def _ref_for(self, work_dir: str) -> str:
        return f"refs/flyclaw/{_dir_hash(work_dir)}"

    # ── .gitignore for snapshot ──

    def _write_exclude_file(self) -> str:
        exclude_path = self._store / "info" / "exclude"
        exclude_path.parent.mkdir(parents=True, exist_ok=True)
        existing = ""
        if exclude_path.exists():
            existing = exclude_path.read_text(encoding="utf-8")
        if "# flyclaw-snapshot" in existing:
            return str(exclude_path)
        with open(exclude_path, "a", encoding="utf-8") as f:
            f.write("\n# flyclaw-snapshot\n")
            for p in _EXCLUDE_PATTERNS:
                f.write(f"{p}\n")
        return str(exclude_path)

    # ── public API ──

    def ensure_snapshot(self, work_dir: str) -> Optional[str]:
        """Create a snapshot of work_dir. Returns snapshot ID (short SHA) or None.

        Deduplicates: if the latest snapshot matches current state, returns existing ID.
        """
        with self._lock:
            return self._ensure_snapshot_inner(work_dir)

    def _ensure_snapshot_inner(self, work_dir: str) -> Optional[str]:
        work_dir = os.path.abspath(work_dir)
        if not os.path.isdir(work_dir):
            return None

        try:
            self._ensure_initialized(work_dir)
        except Exception as e:
            logger.warning("Snapshot init failed for %s: %s", work_dir, e)
            return None

        env = self._env_for(work_dir)
        ref = self._ref_for(work_dir)

        # Write exclude file so git ignores node_modules etc.
        self._write_exclude_file()

        # Add all files
        _git("add", "-A", cwd=work_dir, env=env)

        # Compare tree with current ref tip for dedup
        r_tree = _git("write-tree", cwd=work_dir, env=env)
        if r_tree.returncode != 0:
            logger.debug("write-tree failed: %s", r_tree.stderr.strip())
            return None
        current_tree = r_tree.stdout.strip()

        r_ref = _git("rev-parse", ref, cwd=str(self._store))
        if r_ref.returncode == 0 and r_ref.stdout.strip():
            ref_sha = r_ref.stdout.strip()
            r_ref_tree = _git("rev-parse", f"{ref_sha}^{{tree}}", cwd=str(self._store))
            if r_ref_tree.returncode == 0 and r_ref_tree.stdout.strip() == current_tree:
                # No changes — return existing ref tip
                return ref_sha[:12]
        else:
            ref_sha = None

        # Write tree and create commit with correct parent
        if not current_tree:
            return None

        parent_args = []
        if ref_sha:
            parent_args = ["-p", ref_sha]

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        r_commit = _git(
            "commit-tree", current_tree, *parent_args, "-m", f"snapshot: {ts}",
            cwd=str(self._store), env=env,
        )
        if r_commit.returncode != 0:
            logger.debug("commit-tree failed: %s", r_commit.stderr.strip())
            return None
        sha = r_commit.stdout.strip()
        if not sha:
            return None

        # Update ref
        _git("update-ref", ref, sha, cwd=str(self._store))

        # Prune old snapshots
        self._prune(work_dir, ref, env)

        return sha[:12]

    def list_snapshots(self, work_dir: str, limit: int = 20) -> list[dict]:
        """Return list of snapshots for work_dir, newest first."""
        work_dir = os.path.abspath(work_dir)
        if not self._store.exists():
            return []

        ref = self._ref_for(work_dir)
        env = self._env_for(work_dir)

        r = _git("rev-list", ref, f"--max-count={limit}", "--format=%h %ci %s", cwd=str(self._store), env=env)
        if r.returncode != 0:
            return []

        results = []
        for line in r.stdout.strip().split("\n"):
            if line.startswith("commit "):
                continue
            parts = line.split(None, 2)
            if len(parts) >= 3:
                results.append({
                    "id": parts[0],
                    "date": f"{parts[1]} {parts[2][:8]}",
                    "message": parts[2][9:] if len(parts[2]) > 9 else parts[2],
                })
        return results

    def diff(self, work_dir: str, snapshot_id: str) -> str:
        """Show diff between current state and a snapshot."""
        work_dir = os.path.abspath(work_dir)
        if not self._store.exists():
            return "No snapshot store found."
        env = self._env_for(work_dir)
        r = _git("diff", snapshot_id, "--", cwd=work_dir, env=env)
        if r.returncode != 0:
            return f"Error generating diff: {r.stderr.strip()}"
        if not r.stdout.strip():
            return "No differences."
        # Truncate very large diffs
        if len(r.stdout) > 20000:
            return r.stdout[:20000] + "\n... (truncated)"
        return r.stdout

    def restore(self, work_dir: str, snapshot_id: str) -> str:
        """Restore entire working directory to a snapshot state."""
        work_dir = os.path.abspath(work_dir)
        if not self._store.exists():
            return "No snapshot store found."

        env = self._env_for(work_dir)
        r = _git("checkout", snapshot_id, "--", ".", cwd=work_dir, env=env)
        if r.returncode != 0:
            return f"Restore failed: {r.stderr.strip()}"
        return f"Restored workspace to snapshot {snapshot_id}"

    def restore_file(self, work_dir: str, snapshot_id: str, file_path: str) -> str:
        """Restore a single file from a snapshot."""
        work_dir = os.path.abspath(work_dir)
        if not self._store.exists():
            return "No snapshot store found."

        env = self._env_for(work_dir)
        r = _git("checkout", snapshot_id, "--", file_path, cwd=work_dir, env=env)
        if r.returncode != 0:
            return f"Restore failed: {r.stderr.strip()}"
        return f"Restored {file_path} from snapshot {snapshot_id}"

    def _prune(self, work_dir: str, ref: str, env: dict):
        """Keep only the last N snapshots for this directory."""
        if self._max_per_dir <= 0:
            return

        # Count commits on this ref
        r = _git("rev-list", "--count", ref, cwd=str(self._store), env=env)
        if r.returncode != 0:
            return

        try:
            count = int(r.stdout.strip())
        except ValueError:
            return

        if count <= self._max_per_dir:
            return

        # Find the commit to keep as new base
        keep = count - self._max_per_dir
        r = _git("rev-list", "--reverse", ref, f"--skip={keep}", "--max-count=1", cwd=str(self._store), env=env)
        if r.returncode != 0:
            return
        new_base = r.stdout.strip()
        if not new_base:
            return

        # Update ref to new base
        _git("update-ref", ref, new_base, cwd=str(self._store))

        # GC unreachable objects
        try:
            _git("reflog", "expire", "--expire=now", "--all", cwd=str(self._store))
            _git("gc", "--prune=now", cwd=str(self._store))
        except Exception:
            pass


# ── singleton ──

_manager: CheckpointManager | None = None


def get_snapshot_manager() -> CheckpointManager | None:
    """Return the global CheckpointManager, or None if snapshots are disabled."""
    global _manager
    if _manager is not None:
        return _manager
    try:
        from src.config import load_config
        cfg = load_config()
        if not cfg.snapshot.enabled:
            return None
        _manager = CheckpointManager(
            store_path=cfg.snapshot.store_path,
            max_per_dir=cfg.snapshot.max_per_dir,
            max_file_size=cfg.snapshot.max_file_size,
        )
        return _manager
    except Exception as e:
        logger.warning("Failed to init snapshot manager: %s", e)
        return None


def reset_snapshot_manager():
    """Reset singleton — for tests or config reload."""
    global _manager
    _manager = None


def snapshot_before_write(work_dir: str) -> Optional[str]:
    """Convenience: take a snapshot before a write operation. Silent on failure."""
    try:
        mgr = get_snapshot_manager()
        if mgr is None:
            return None
        return mgr.ensure_snapshot(work_dir)
    except Exception as e:
        logger.debug("Snapshot before write failed: %s", e)
        return None
