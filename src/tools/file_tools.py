from __future__ import annotations

import asyncio
import fnmatch
import logging
import os
from pathlib import Path
from typing import Literal

import aiofiles
import aiofiles.os

logger = logging.getLogger("flyclaw.file_tools")

_BASE_DIR = os.path.abspath(os.environ.get("FLYCLAW_WORKSPACE", "."))
_edit_condition = asyncio.Condition()


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
    from src.agent.tool_cache import cache_root as get_cache_root

    if not is_sandbox_enabled():
        return str(real)
    base = Path(_BASE_DIR).resolve()
    cache_root = get_cache_root()
    allowed_roots = [base, cache_root]
    for root in allowed_roots:
        try:
            real.relative_to(root)
            return str(real)
        except ValueError:
            continue
    raise ValueError(f"当前为沙盒模式，无法访问工作目录之外的路径：{path}（允许范围：{_BASE_DIR}、{cache_root}）")


async def read_file(path: str, offset: int = 0, head_limit: int = 500) -> str:
    """Read file contents. Returns specified line range.

    Args:
        path: File path (relative to workspace)
        offset: Starting line number (0-based)
        head_limit: Max number of lines to read
    """
    try:
        resolved = _resolve_path(path)
    except ValueError as e:
        return f"Error: {e}"

    try:
        # Extension check (no file open needed — no TOCTOU possible)
        ext = os.path.splitext(resolved)[1].lower()
        if ext in _BINARY_EXTS:
            return f"Error: binary file: {path}"
        # Null-byte probe: catch binary files with unknown extensions
        async with aiofiles.open(resolved, "rb") as raw:
            chunk = await raw.read(8192)
            if b"\x00" in chunk:
                return f"Error: binary file: {path}"
        selected: list[str] = []
        total = 0
        async with aiofiles.open(resolved, "r", encoding="utf-8") as f:
            idx = 0
            async for line in f:
                total += 1
                if idx < offset:
                    idx += 1
                    continue
                if len(selected) < head_limit:
                    selected.append(line)
                idx += 1
        if total == 0:
            return f"File: {path} (empty, 0 lines)"
        if offset >= total:
            return f"Error: offset {offset} exceeds file length ({total} lines)"
        end = offset + head_limit
        result = []
        for i, line in enumerate(selected):
            result.append(f"{offset + i + 1}\t{line.rstrip()}")
        header = f"File: {path} (lines {offset + 1}-{min(end, total)} of {total})\n"
        if end < total:
            header += f"(showing {head_limit} of {total - offset} remaining lines)\n"
        return header + "\n".join(result)
    except FileNotFoundError:
        return f"Error: file not found: {path} (resolved to: {resolved}, workspace: {_BASE_DIR})"
    except PermissionError:
        return f"Error: permission denied: {path}"
    except Exception as e:
        return f"Error reading {path}: {e}"


async def write_file(path: str, content: str) -> str:
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

    await snapshot_before_write(_BASE_DIR)
    async with _edit_condition:
        try:
            parent = os.path.dirname(resolved)
            if parent:
                await aiofiles.os.makedirs(parent, exist_ok=True)
            async with aiofiles.open(resolved, "w", encoding="utf-8") as f:
                await f.write(content)
                await f.flush()
                await asyncio.to_thread(os.fsync, f.fileno())
            _edit_condition.notify_all()
            lines = content.count("\n") + (0 if content.endswith("\n") else 1)
            return f"Written {lines} lines to {path}"
        except PermissionError:
            return f"Error: permission denied: {path}"
        except Exception as e:
            return f"Error writing {path}: {e}"


async def edit_file(path: str, old_string: str, new_string: str) -> str:
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

    await snapshot_before_write(_BASE_DIR)
    async with _edit_condition:
        try:
            await asyncio.wait_for(_edit_condition.wait_for(lambda: os.path.exists(resolved)), timeout=10.0)
        except asyncio.TimeoutError:
            return f"Error: file not found: {path} (resolved to: {resolved}, workspace: {_BASE_DIR})"

        try:
            async with aiofiles.open(resolved, "r", encoding="utf-8") as f:
                content = await f.read()
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
            async with aiofiles.open(resolved, "w", encoding="utf-8") as f:
                await f.write(new_content)
            old_lines = old_string.count("\n") + 1
            new_lines = new_string.count("\n") + 1
            return f"Replaced {old_lines} lines with {new_lines} lines in {path}"
        except Exception as e:
            return f"Error writing {path}: {e}"


async def list_dir(path: str = ".") -> str:
    """List directory contents.

    Args:
        path: Directory path (relative to workspace), defaults to workspace root
    """
    try:
        resolved = _resolve_path(path)
    except ValueError as e:
        return f"Error: {e}"
    try:
        entries = sorted(await aiofiles.os.listdir(resolved))
        if not entries:
            return f"Directory {path} is empty"
        lines = []
        for name in entries:
            full = os.path.join(resolved, name)
            if await aiofiles.os.path.isdir(full):
                lines.append(f"  {name}/")
            elif await aiofiles.os.path.isfile(full):
                size = (await aiofiles.os.stat(full)).st_size
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


_EXCLUDED_DIRS = frozenset((".git", "__pycache__", "node_modules"))
_GREP_HARD_LIMIT = 10000
_GLOB_HARD_LIMIT = 10000
_BINARY_EXTS = frozenset(
    (
        ".pyc",
        ".pyo",
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".bin",
        ".dat",
        ".db",
        ".sqlite",
        ".sqlite3",
        ".class",
        ".o",
        ".a",
        ".lib",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".ico",
        ".svg",
        ".webp",
        ".mp3",
        ".mp4",
        ".avi",
        ".mkv",
        ".mov",
        ".wav",
        ".flac",
        ".ogg",
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".xz",
        ".7z",
        ".rar",
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        ".woff",
        ".woff2",
        ".ttf",
        ".eot",
        ".otf",
    )
)


def _is_binary(filepath: str) -> bool:
    """Quick binary check: extension match or null bytes in first 8KB."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext in _BINARY_EXTS:
        return True
    try:
        with open(filepath, "rb") as f:
            chunk = f.read(8192)
        return b"\x00" in chunk
    except (PermissionError, OSError):
        return True


async def _path_not_found_hint(path: str, resolved: str) -> str:
    parent = os.path.dirname(resolved)
    basename = os.path.basename(path)
    hint = ""
    if await aiofiles.os.path.isdir(parent):
        try:
            entries = sorted(await aiofiles.os.listdir(parent))
        except OSError:
            entries = []
        lower_b = basename.lower()
        similar = [e for e in entries if lower_b in e.lower() or e.lower().startswith(lower_b[:3])]
        if similar[:5]:
            hint = f"\n相似路径: {', '.join(similar[:5])}"
    return f"Error: 路径不存在: {path}{hint}"


async def grep(
    pattern: str,
    path: str = ".",
    glob_pattern: str = "*",
    head_limit: int = 50,
    offset: int = 0,
    output_mode: Literal["content", "files_with_matches"] = "content",
) -> str:
    """在文件内容中搜索匹配的文本行（正则表达式）。

    Args:
        pattern: 正则表达式搜索模式
        path: 搜索目标，可以是文件或目录路径（相对于工作区）
        glob_pattern: 目录搜索时的文件名过滤 (glob 模式)，如 "*.py"、"*.md"；path 为文件时忽略
        head_limit: 最大返回条数（默认 50）
        offset: 跳过前 N 条结果（默认 0）
        output_mode: "content" 显示匹配行，"files_with_matches" 仅列出匹配的文件路径
    """
    return await _grep_impl(pattern, path, glob_pattern, head_limit, offset, output_mode)


async def glob(pattern: str = "*", path: str = ".", head_limit: int = 50, offset: int = 0) -> str:
    """按文件名模式递归搜索文件（glob 模式，按修改时间倒序）。

    Args:
        pattern: glob 模式，如 "*.py"、"*config*"
        path: 搜索目录（相对于工作区）
        head_limit: 最大返回条数（默认 50）
        offset: 跳过前 N 条结果（默认 0）
    """
    return await _glob_impl(pattern, path, head_limit, offset)


async def _grep_impl(
    pattern: str,
    path: str = ".",
    glob_pattern: str = "*",
    head_limit: int = 50,
    offset: int = 0,
    output_mode: str = "content",
) -> str:
    try:
        resolved = _resolve_path(path)
    except ValueError as e:
        return f"Error: {e}"
    if not await aiofiles.os.path.exists(resolved):
        return await _path_not_found_hint(path, resolved)

    import re

    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Error: invalid regex pattern: {e}"

    # Single file path — search directly
    if await aiofiles.os.path.isfile(resolved):
        if await asyncio.to_thread(_is_binary, resolved):
            return f"Error: binary file: {path}"
        basename = os.path.basename(resolved)
        if output_mode == "files_with_matches":
            try:
                async with aiofiles.open(resolved, "r", encoding="utf-8", errors="ignore") as f:
                    async for line in f:
                        if regex.search(line):
                            return f"Found 1 of 1 file(s) matching pattern in {path}:\n  {basename}"
            except (PermissionError, OSError):
                return f"Error: permission denied: {path}"
            return f"No files matching '{pattern}'"

        results = []
        try:
            async with aiofiles.open(resolved, "r", encoding="utf-8", errors="ignore") as f:
                i = 0
                async for line in f:
                    i += 1
                    if regex.search(line):
                        results.append(f"{basename}:{i}: {line.rstrip()[:200]}")
                        if len(results) >= _GREP_HARD_LIMIT:
                            break
        except (PermissionError, OSError):
            return f"Error: permission denied: {path}"
        if not results:
            return f"No matches found for '{pattern}' in {path}"
        total = len(results)
        if offset >= total:
            return f"Error: offset {offset} exceeds total matches ({total})"
        page = results[offset : offset + head_limit]
        header = f"Found {len(page)} of {total} match(es) for '{pattern}':\n"
        suffix = ""
        if offset + head_limit < total:
            suffix = f"\n... (截断，共 {total} 条，可用 offset={offset + head_limit} 查看更多)"
        return header + "\n".join(page) + suffix

    if not await aiofiles.os.path.isdir(resolved):
        return f"Error: not a file or directory: {path}"

    if output_mode == "files_with_matches":
        return await _grep_files_with_matches(regex, resolved, path, glob_pattern, head_limit, offset, pattern=pattern)

    # Directory walk — run in thread to avoid blocking event loop
    def _walk_grep():
        results = []
        truncated = False
        try:
            for root, dirs, files in os.walk(resolved):
                if truncated:
                    break
                dirs[:] = [d for d in dirs if not d.startswith(".") and d not in _EXCLUDED_DIRS]
                for fname in files:
                    if truncated:
                        break
                    if not fnmatch.fnmatch(fname, glob_pattern):
                        continue
                    filepath = os.path.join(root, fname)
                    if _is_binary(filepath):
                        continue
                    rel = os.path.relpath(filepath, resolved)
                    try:
                        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                            for i, line in enumerate(f, 1):
                                if regex.search(line):
                                    results.append(f"{rel}:{i}: {line.rstrip()[:200]}")
                                    if len(results) >= _GREP_HARD_LIMIT:
                                        truncated = True
                                        break
                    except (PermissionError, OSError):
                        continue
        except Exception as e:
            return results, truncated, str(e)
        return results, truncated, None

    results, truncated, error = await asyncio.to_thread(_walk_grep)
    if error:
        return f"Error searching: {error}"

    total = len(results)
    if not total:
        return f"No matches found for '{pattern}' in {path}"
    if offset >= total:
        return f"Error: offset {offset} exceeds total matches ({total})"
    page = results[offset : offset + head_limit]
    header = f"Found {len(page)} of {total} match(es) for '{pattern}':\n"
    suffix = ""
    if truncated:
        suffix = f"\n... (结果过多，仅收集前 {total} 条，可用更精确的 pattern 或 glob_pattern 缩小范围)"
    elif offset + head_limit < total:
        suffix = f"\n... (截断，共 {total} 条，可用 offset={offset + head_limit} 查看更多)"
    return header + "\n".join(page) + suffix


async def _grep_files_with_matches(
    regex, resolved: str, path: str, glob_pattern: str, head_limit: int, offset: int, pattern: str = ""
) -> str:
    def _walk():
        files = []
        try:
            for root, dirs, fnames in os.walk(resolved):
                dirs[:] = [d for d in dirs if not d.startswith(".") and d not in _EXCLUDED_DIRS]
                for fname in fnames:
                    if not fnmatch.fnmatch(fname, glob_pattern):
                        continue
                    filepath = os.path.join(root, fname)
                    if _is_binary(filepath):
                        continue
                    try:
                        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                            for line in f:
                                if regex.search(line):
                                    rel = os.path.relpath(filepath, resolved)
                                    files.append(rel)
                                    break
                    except (PermissionError, OSError):
                        continue
                    if len(files) >= _GREP_HARD_LIMIT:
                        break
                if len(files) >= _GREP_HARD_LIMIT:
                    break
        except Exception as e:
            return files, str(e)
        return files, None

    files, error = await asyncio.to_thread(_walk)
    if error:
        return f"Error searching: {error}"

    total = len(files)
    if not total:
        return f"No files matching '{pattern}'"
    if offset >= total:
        return f"Error: offset {offset} exceeds total files ({total})"
    page = files[offset : offset + head_limit]
    header = f"Found {len(page)} of {total} file(s) matching pattern in {path}:\n"
    suffix = ""
    if total >= _GREP_HARD_LIMIT:
        suffix = f"\n... (结果过多，仅收集前 {total} 个文件，可用更精确的 pattern 或 glob_pattern 缩小范围)"
    elif offset + head_limit < total:
        suffix = f"\n... (截断，共 {total} 个文件，可用 offset={offset + head_limit} 查看更多)"
    return header + "\n".join(f"  {f}" for f in page) + suffix


async def _glob_impl(pattern: str, path: str = ".", head_limit: int = 50, offset: int = 0) -> str:
    try:
        resolved = _resolve_path(path)
    except ValueError as e:
        return f"Error: {e}"
    if not await aiofiles.os.path.exists(resolved):
        return await _path_not_found_hint(path, resolved)
    if not await aiofiles.os.path.isdir(resolved):
        return f"Error: not a directory: {path}"

    def _do_glob():
        try:
            all_matches = []
            truncated = False
            for m in Path(resolved).rglob(pattern):
                parts = m.relative_to(resolved).parts
                if any(p.startswith(".") and p not in (".", "..") for p in parts):
                    continue
                if any(p in _EXCLUDED_DIRS for p in parts):
                    continue
                all_matches.append(m)
                if len(all_matches) >= _GLOB_HARD_LIMIT:
                    truncated = True
                    break
            if not all_matches:
                return f"No files matched '{pattern}' in {path}"
            all_matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            total = len(all_matches)
            if offset >= total:
                return f"Error: offset {offset} exceeds total matches ({total})"
            page = all_matches[offset : offset + head_limit]
            lines = []
            for m in page:
                rel = str(m.relative_to(resolved))
                if m.is_dir():
                    lines.append(f"  {rel}/")
                else:
                    size = m.stat().st_size
                    lines.append(f"  {rel}  ({size}B)")
            header = f"Found {len(page)} of {total} match(es) for '{pattern}':\n"
            suffix = ""
            if truncated:
                suffix = f"\n... (结果过多，仅收集前 {total} 条，可用更精确的 pattern 缩小范围)"
            elif offset + head_limit < total:
                suffix = f"\n... (截断，共 {total} 条，可用 offset={offset + head_limit} 查看更多)"
            return header + "\n".join(lines) + suffix
        except Exception as e:
            return f"Error: {e}"

    return await asyncio.to_thread(_do_glob)


def get_tools() -> list:
    from src.agent.tooldef import ToolDef

    return [
        ToolDef.from_function(read_file),
        ToolDef.from_function(write_file),
        ToolDef.from_function(edit_file),
        ToolDef.from_function(list_dir),
        ToolDef.from_function(grep),
        ToolDef.from_function(glob),
    ]
