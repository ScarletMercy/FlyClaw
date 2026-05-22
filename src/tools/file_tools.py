from __future__ import annotations

import fnmatch
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("myclaw.file_tools")

_BASE_DIR = os.path.abspath(os.environ.get("MYCLAW_WORKSPACE", "."))
_edit_condition = threading.Condition()

# --- Read dedup and loop detection ---
_read_tracker_lock = threading.Lock()
_read_tracker: dict[str, dict] = {}
_DEDUP_CAP = 500


def set_workspace(path: str):
    """Update the workspace root (called during init from config)."""
    global _BASE_DIR
    _BASE_DIR = os.path.abspath(path)
    logger.info("File tools workspace set to: %s", _BASE_DIR)


def _resolve_path(path: str) -> str:
    """Resolve path relative to workspace. Enforces sandbox when enabled."""
    p = Path(path)
    if not p.is_absolute():
        p = Path(_BASE_DIR) / p
    try:
        real = p.resolve(strict=False)
    except Exception:
        raise ValueError(f"Path '{path}' could not be resolved")
    from src.tools.exec import is_sandbox_enabled
    if not is_sandbox_enabled():
        return str(real)
    base = Path(_BASE_DIR).resolve()
    try:
        real.relative_to(base)
    except ValueError:
        raise ValueError(f"Path '{path}' is outside the workspace")
    return str(real)


def _get_tracker(thread_id: str) -> dict:
    """Get or create read tracker for a thread, with capacity enforcement."""
    task_data = _read_tracker.get(thread_id)
    if task_data is None:
        task_data = {
            "last_key": None,
            "consecutive": 0,
            "dedup": {},
            "dedup_hits": {},
        }
        _read_tracker[thread_id] = task_data
    dedup = task_data.get("dedup")
    if dedup is not None and len(dedup) > _DEDUP_CAP:
        for _ in range(len(dedup) - _DEDUP_CAP):
            dedup.pop(next(iter(dedup)), None)
    dedup_hits = task_data.get("dedup_hits")
    if dedup_hits is not None and len(dedup_hits) > _DEDUP_CAP:
        for _ in range(len(dedup_hits) - _DEDUP_CAP):
            dedup_hits.pop(next(iter(dedup_hits)), None)
    return task_data


def reset_read_dedup(thread_id: str | None = None) -> None:
    """Clear read dedup cache. Called after context compression.

    With thread_id: clear only that thread. Without: clear all.
    """
    with _read_tracker_lock:
        if thread_id:
            task_data = _read_tracker.get(thread_id)
            if task_data:
                if "dedup" in task_data:
                    task_data["dedup"].clear()
                if "dedup_hits" in task_data:
                    task_data["dedup_hits"].clear()
                task_data["consecutive"] = 0
                task_data["last_key"] = None
        else:
            _read_tracker.clear()


def _invalidate_dedup_for_path(resolved: str, thread_id: str) -> None:
    """Remove all dedup cache entries whose resolved path matches."""
    if not thread_id:
        return
    with _read_tracker_lock:
        task_data = _read_tracker.get(thread_id)
        if not task_data:
            return
        dedup = task_data.get("dedup")
        if not dedup:
            return
        stale_keys = [k for k in dedup if k[0] == resolved]
        for k in stale_keys:
            dedup.pop(k, None)
        task_data.setdefault("dedup_hits", {})
        dedup_hits = task_data["dedup_hits"]
        for k in list(dedup_hits):
            if k[0] == resolved:
                dedup_hits.pop(k, None)
        last = task_data.get("last_key")
        if last is not None and len(last) >= 2 and last[1] == resolved:
            task_data["consecutive"] = 0
            task_data["last_key"] = None


def read_file(path: str, offset: int = 0, limit: int = 500) -> str:
    """Read file contents. Returns specified line range.

    Args:
        path: File path (relative to workspace)
        offset: Starting line number (0-based)
        limit: Max number of lines to read
    """
    try:
        resolved = _resolve_path(path)
    except ValueError as e:
        return f"Error: {e}"

    from src.tools.exec import _current_thread_id
    thread_id = _current_thread_id.get("")
    dedup_key = (resolved, offset, limit)
    read_key = ("read", resolved, offset, limit)

    if thread_id:
        with _read_tracker_lock:
            task_data = _get_tracker(thread_id)
            cached_mtime = task_data["dedup"].get(dedup_key)

        if cached_mtime is not None:
            try:
                current_mtime = os.path.getmtime(resolved)
                if current_mtime == cached_mtime:
                    with _read_tracker_lock:
                        task_data = _get_tracker(thread_id)
                        hits = task_data["dedup_hits"].get(dedup_key, 0) + 1
                        task_data["dedup_hits"][dedup_key] = hits

                    if hits >= 2:
                        return (
                            f"BLOCKED: You have called read_file on "
                            f"{path} (offset={offset}, limit={limit}) "
                            f"{hits + 1} times and the file has NOT changed. "
                            "The content from your earlier read_file result is "
                            "still current. Proceed with the information you already have."
                        )

                    return (
                        f"File unchanged since last read: {path} "
                        f"(lines {offset + 1}-{offset + limit}). "
                        "The content from the earlier read_file result in this "
                        "conversation is still current — refer to that instead."
                    )
            except OSError:
                pass

    try:
        with open(resolved, "r", encoding="utf-8") as f:
            lines = f.readlines()
        total = len(lines)
        end = offset + limit
        selected = lines[offset:end]
        result = []
        for i, line in enumerate(selected):
            result.append(f"{offset + i + 1}\t{line.rstrip()}")
        header = f"File: {path} (lines {offset + 1}-{min(end, total)} of {total})\n"
        if end < total:
            header += f"(showing {limit} of {total - offset} remaining lines)\n"
        content = header + "\n".join(result)
    except FileNotFoundError:
        return f"Error: file not found: {path} (resolved to: {resolved}, workspace: {_BASE_DIR})"
    except PermissionError:
        return f"Error: permission denied: {path}"
    except Exception as e:
        return f"Error reading {path}: {e}"

    if thread_id:
        with _read_tracker_lock:
            task_data = _get_tracker(thread_id)
            task_data["dedup_hits"].pop(dedup_key, None)
            if task_data["last_key"] == read_key:
                task_data["consecutive"] += 1
            else:
                task_data["last_key"] = read_key
                task_data["consecutive"] = 1
            count = task_data["consecutive"]

            try:
                task_data["dedup"][dedup_key] = os.path.getmtime(resolved)
            except OSError:
                pass

        if count >= 4:
            return (
                f"BLOCKED: You have read this exact file region "
                f"({path}, offset={offset}, limit={limit}) {count} times in a row. "
                "The content has NOT changed. STOP re-reading and proceed with your task."
            )
        if count >= 3:
            content += (
                f"\n\nWARNING: You have read this exact file region "
                f"({path}, offset={offset}, limit={limit}) {count} times consecutively. "
                "The content has not changed since your last read. "
                "Use the information you already have."
            )

    return content


def write_file(path: str, content: str) -> str:
    """Write content to a file. Creates parent directories if needed.

    Args:
        path: File path (relative to workspace)
        content: Full file content to write
    """
    try:
        resolved = _resolve_path(path)
    except ValueError as e:
        return f"Error: {e}"
    from src.tools.snapshot import snapshot_before_write
    snapshot_before_write(_BASE_DIR)
    with _edit_condition:
        try:
            parent = os.path.dirname(resolved)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(resolved, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            _edit_condition.notify_all()
            lines = content.count("\n") + (0 if content.endswith("\n") else 1)
            from src.tools.exec import _current_thread_id
            _invalidate_dedup_for_path(resolved, _current_thread_id.get(""))
            return f"Written {lines} lines to {path}"
        except PermissionError:
            return f"Error: permission denied: {path}"
        except Exception as e:
            return f"Error writing {path}: {e}"


def edit_file(path: str, old_string: str, new_string: str) -> str:
    """Replace a specific text segment in a file with new text.

    Args:
        path: File path (relative to workspace)
        old_string: Exact text to find and replace
        new_string: Text to replace it with
    """
    try:
        resolved = _resolve_path(path)
    except ValueError as e:
        return f"Error: {e}"
    from src.tools.snapshot import snapshot_before_write
    snapshot_before_write(_BASE_DIR)
    with _edit_condition:
        if not _edit_condition.wait_for(lambda: os.path.exists(resolved), timeout=10.0):
            return f"Error: file not found: {path} (resolved to: {resolved}, workspace: {_BASE_DIR})"

        try:
            with open(resolved, "r", encoding="utf-8") as f:
                content = f.read()
        except FileNotFoundError:
            return f"Error: file not found: {path} (resolved to: {resolved}, workspace: {_BASE_DIR})"
        except Exception as e:
            return f"Error reading {path}: {e}"

        count = content.count(old_string)
        if count == 0:
            return f"Error: text not found in {path}. Make sure old_string matches exactly (including whitespace)."
        if count > 1:
            return f"Error: found {count} matches in {path}. Please provide more context to make old_string unique."

        new_content = content.replace(old_string, new_string, 1)
        try:
            with open(resolved, "w", encoding="utf-8") as f:
                f.write(new_content)
            old_lines = old_string.count("\n") + 1
            new_lines = new_string.count("\n") + 1
            from src.tools.exec import _current_thread_id
            _invalidate_dedup_for_path(resolved, _current_thread_id.get(""))
            return f"Replaced {old_lines} lines with {new_lines} lines in {path}"
        except Exception as e:
            return f"Error writing {path}: {e}"


def list_dir(path: str = ".") -> str:
    """List directory contents.

    Args:
        path: Directory path (relative to workspace), defaults to workspace root
    """
    try:
        resolved = _resolve_path(path)
    except ValueError as e:
        return f"Error: {e}"
    try:
        entries = sorted(os.listdir(resolved))
        if not entries:
            return f"Directory {path} is empty"
        lines = []
        for name in entries:
            full = os.path.join(resolved, name)
            if os.path.isdir(full):
                lines.append(f"  {name}/")
            elif os.path.isfile(full):
                size = os.path.getsize(full)
                if size > 1024 * 1024:
                    lines.append(f"  {name}  ({size / 1024 / 1024:.1f}MB)")
                elif size > 1024:
                    lines.append(f"  {name}  ({size / 1024:.1f}KB)")
                else:
                    lines.append(f"  {name}  ({size}B)")
            else:
                lines.append(f"  {name}")
        return f"Directory: {path}\n" + "\n".join(lines)
    except FileNotFoundError:
        return f"Error: directory not found: {path}"
    except PermissionError:
        return f"Error: permission denied: {path}"
    except Exception as e:
        return f"Error listing {path}: {e}"


def search_files(mode: str = "content", pattern: str = "", path: str = ".",
                  file_pattern: str = "*") -> str:
    """Search files by content (regex) or by name (glob pattern).

    Args:
        mode: "content" to search text with regex, "name" to find files by glob pattern
        pattern: Search pattern (regex for content mode, glob for name mode)
        path: Directory to search in (relative to workspace)
        file_pattern: Glob filter for content mode (e.g. "*.py", "*.md")
    """
    if mode in ("name", "glob"):
        return _glob_impl(pattern or "*", path)
    return _grep_impl(pattern, path, file_pattern)


def _grep_impl(pattern: str, path: str = ".", file_pattern: str = "*") -> str:
    try:
        resolved = _resolve_path(path)
    except ValueError as e:
        return f"Error: {e}"
    if not os.path.isdir(resolved):
        return f"Error: not a directory: {path}"

    import re

    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Error: invalid regex pattern: {e}"

    results = []
    try:
        for root, dirs, files in os.walk(resolved):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("__pycache__", "node_modules", ".git")]
            for fname in files:
                if not fnmatch.fnmatch(fname, file_pattern):
                    continue
                filepath = os.path.join(root, fname)
                rel = os.path.relpath(filepath, resolved)
                try:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        for i, line in enumerate(f, 1):
                            if regex.search(line):
                                results.append(f"{rel}:{i}: {line.rstrip()[:200]}")
                                if len(results) >= 50:
                                    results.append("... (truncated, max 50 matches)")
                                    return "\n".join(results)
                except (PermissionError, OSError):
                    continue
    except Exception as e:
        return f"Error searching: {e}"

    if not results:
        return f"No matches found for '{pattern}' in {path}"
    return f"Found {len(results)} match(es) for '{pattern}':\n" + "\n".join(results)


def _glob_impl(pattern: str, path: str = ".") -> str:
    try:
        resolved = _resolve_path(path)
    except ValueError as e:
        return f"Error: {e}"
    try:
        matches = sorted(Path(resolved).glob(pattern))
        if not matches:
            return f"No files matched '{pattern}' in {path}"
        lines = []
        for m in matches:
            rel = str(m.relative_to(resolved))
            if m.is_dir():
                lines.append(f"  {rel}/")
            else:
                size = m.stat().st_size
                lines.append(f"  {rel}  ({size}B)")
            if len(lines) >= 100:
                lines.append("... (truncated, max 100 matches)")
                break
        return f"Found {len(matches)} match(es) for '{pattern}':\n" + "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def get_tools() -> list:
    from src.agent.tooldef import ToolDef
    return [
        ToolDef.from_function(read_file),
        ToolDef.from_function(write_file),
        ToolDef.from_function(edit_file),
        ToolDef.from_function(list_dir),
        ToolDef.from_function(search_files),
    ]
