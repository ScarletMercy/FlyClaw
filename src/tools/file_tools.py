from __future__ import annotations

import fnmatch
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("flyclaw.file_tools")

_BASE_DIR = os.path.abspath(os.environ.get("FLYCLAW_WORKSPACE", "."))
_edit_condition = threading.Condition()


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
        return header + "\n".join(result)
    except FileNotFoundError:
        return f"Error: file not found: {path} (resolved to: {resolved}, workspace: {_BASE_DIR})"
    except PermissionError:
        return f"Error: permission denied: {path}"
    except Exception as e:
        return f"Error reading {path}: {e}"


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
