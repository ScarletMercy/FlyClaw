from __future__ import annotations

import asyncio
import fnmatch
import logging
import os
import re as _re
from pathlib import Path
from typing import Literal

import aiofiles
import aiofiles.os

logger = logging.getLogger("flyclaw.file_tools")

_BASE_DIR = os.path.abspath(os.environ.get("FLYCLAW_WORKSPACE", "."))

# ReDoS 防护: 限制 pattern 长度 + 运行时超时
_MAX_REGEX_LENGTH = 500
_GREP_TIMEOUT = 30.0


_skill_ref_dirs_cache: list | None = None


def _collect_skill_reference_dirs() -> list:
    """Return resolved paths of references/ dirs under each skill directory."""
    global _skill_ref_dirs_cache
    if _skill_ref_dirs_cache is not None:
        return _skill_ref_dirs_cache
    try:
        from src._container import get_container

        container = get_container()
        dirs = []
        for _, skill_dir in container._build_skill_directories():
            ref = (Path(skill_dir) / "references").resolve()
            if ref.is_dir():
                dirs.append(ref)
        _skill_ref_dirs_cache = dirs
        return dirs
    except Exception:
        return []


def _try_skill_ref_readonly(path: str) -> str | None:
    """If *path* falls under a skill references/ dir, return resolved abs path; else None."""
    p = Path(path)
    try:
        if p.is_absolute():
            candidates = [p.resolve(strict=False)]
        else:
            real = (Path(_BASE_DIR).resolve() / p).resolve(strict=False)
            candidates = [real]
            # POSIX correction: AI may drop leading '/' on Linux (mirrors _resolve_path
            # heuristic but without workspace-only constraint so user-skill refs work).
            if os.name == "posix" and not real.exists():
                try:
                    candidates.append((Path("/") / p).resolve(strict=False))
                except (ValueError, OSError):
                    pass
    except OSError:
        return None
    ref_dirs = _collect_skill_reference_dirs()
    for c in candidates:
        for ref_dir in ref_dirs:
            try:
                c.relative_to(ref_dir)
                return str(c)
            except ValueError:
                continue
    return None


def _validate_regex(pattern: str, *, ignore_case: bool = False):
    """编译正则并检查安全性。返回 re.Pattern 或错误字符串。"""
    if len(pattern) > _MAX_REGEX_LENGTH:
        return f"Error: regex pattern too long (max {_MAX_REGEX_LENGTH} chars)"
    flags = _re.IGNORECASE if ignore_case else 0
    try:
        regex = _re.compile(pattern, flags)
    except _re.error as e:
        return f"Error: invalid regex pattern: {e}"
    return regex


def set_workspace(path: str):
    """Update the workspace root (called during init from config)."""
    global _BASE_DIR
    _BASE_DIR = os.path.abspath(path)
    logger.info("File tools workspace set to: %s", _BASE_DIR)


def _resolve_path(path: str) -> str:
    """Resolve path relative to workspace. Enforces sandbox when enabled.

    On Linux the AI may drop the leading ``/`` from an absolute path it saw
    in the system prompt (e.g. ``home/ubuntu/.flyclaw/workspace/file.txt``).
    We detect this by tentatively prepending ``/`` and checking whether the
    result lands inside the workspace (``relative_to``).  If it does *and*
    the normal workspace-relative resolution does not exist on disk, we use
    the corrected interpretation.  Gated to POSIX only — on Windows
    ``Path("/")`` resolves to the current-drive root which can match the
    workspace, so the heuristic is skipped entirely.
    """
    p = Path(path)

    try:
        if p.is_absolute():
            real = p.resolve(strict=False)
        else:
            base = Path(_BASE_DIR).resolve()
            real = (base / p).resolve(strict=False)
            # Fallback (POSIX only): AI dropped leading '/' on a Linux absolute path
            if os.name == "posix" and not real.exists():
                try:
                    abs_candidate = (Path("/") / p).resolve(strict=False)
                    abs_candidate.relative_to(base)  # raises ValueError if outside
                    real = abs_candidate
                except (ValueError, OSError):
                    pass  # not a mis-prefixed path, keep original resolution
    except Exception:
        raise ValueError(f"Path '{path}' could not be resolved")

    from src.tools.exec import is_sandbox_enabled
    from src.agent.tool_cache import cache_root as get_cache_root

    if not is_sandbox_enabled():
        return str(real)

    base = Path(_BASE_DIR).resolve()

    cache_root = get_cache_root()
    for root in (base, cache_root):
        try:
            real.relative_to(root)
            return str(real)
        except ValueError:
            continue

    raise ValueError(f"当前为沙盒模式，无法访问工作目录之外的路径：{path}（允许范围：{_BASE_DIR}、{cache_root}）")


# ── File write ────────────────────────────────────────────────────────


async def _direct_write(path: str, content: str, encoding: str = "utf-8") -> None:
    async with aiofiles.open(path, "w", encoding=encoding) as f:
        await f.write(content)
        await f.flush()


def _count_lines(text: str) -> int:
    """Count lines in text, matching Claude Code semantics."""
    if not text:
        return 0
    return text.count("\n") + (0 if text.endswith("\n") else 1)


async def read_file(path: str, offset: int = 0, head_limit: int = 500) -> str:
    """Read file contents. Returns specified line range.

    Args:
        path: File path (relative to workspace)
        offset: Starting line number (0-based)
        head_limit: Max number of lines to read
    """
    try:
        resolved = _resolve_path(path)
    except ValueError as sandbox_err:
        # _resolve_path blocked the path — try skill references/ as read-only exception
        from src.tools.exec import is_sandbox_enabled

        if is_sandbox_enabled():
            resolved = _try_skill_ref_readonly(path)
            if resolved is None:
                return f"Error: {sandbox_err}"
        else:
            return f"Error: {sandbox_err}"

    try:
        ext = os.path.splitext(resolved)[1].lower()
        if ext in _BINARY_EXTS:
            return f"Error: binary file: {path}"
        # 单 fd：二进制探测 + 文本流式读取，消除 TOCTOU
        fd = os.open(resolved, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        try:
            if b"\x00" in os.read(fd, 8192):
                return f"Error: binary file: {path}"
            os.lseek(fd, 0, os.SEEK_SET)
            selected: list[str] = []
            total = 0
            async with aiofiles.open(fd, "r", encoding="utf-8", errors="replace", closefd=False) as f:
                async for line in f:
                    total += 1
                    if total > offset and len(selected) < head_limit:
                        selected.append(line)
        finally:
            os.close(fd)
        if total == 0:
            return f"File: {path} (empty, 0 lines)"
        if offset > 0 and not selected:
            return f"Error: offset {offset} exceeds file length ({total} lines)"
        end = offset + head_limit
        result = [f"{offset + i + 1}\t{line.rstrip()}" for i, line in enumerate(selected)]
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
    try:
        parent = os.path.dirname(resolved)
        if parent and not await aiofiles.os.path.isdir(parent):
            await aiofiles.os.makedirs(parent, exist_ok=True)
        await _direct_write(resolved, content)
        lines = _count_lines(content)
        return f"Written {lines} line{'s' if lines != 1 else ''} to {path}"
    except PermissionError:
        return f"Error: permission denied: {path}"
    except Exception as e:
        return f"Error writing {path}: {e}"


async def edit_file(path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    """Replace a specific text segment in a file with new text.

    Args:
        path: File path (relative to workspace)
        old_string: Exact text to find and replace
        new_string: Text to replace it with
        replace_all: If True, replace all occurrences; if False (default), require uniqueness
    """
    try:
        resolved = _resolve_path(path)
    except ValueError as e:
        return f"Error: {e}"

    # Read file content
    try:
        async with aiofiles.open(resolved, "r", encoding="utf-8") as f:
            content = await f.read()
    except FileNotFoundError:
        return f"Error: file not found: {path} (resolved to: {resolved}, workspace: {_BASE_DIR})"
    except Exception as e:
        return f"Error reading {path}: {e}"

    # Handle empty old_string: only valid for empty files (file creation)
    if old_string == "":
        if content.strip() != "":
            return f"Error: file already exists: {path}. Cannot create new file with edit_file."
        new_content = new_string
    else:
        count = content.count(old_string)
        if count == 0:
            return f"Error: text not found in {path}. Make sure old_string matches exactly (including whitespace)."
        if count > 1 and not replace_all:
            return f"Error: found {count} matches in {path}. Please provide more context to make old_string unique, or set replace_all=True."

        new_content = content.replace(old_string, new_string, -1 if replace_all else 1)

    # Write
    try:
        await _direct_write(resolved, new_content)
    except PermissionError:
        return f"Error: permission denied: {path}"
    except Exception as e:
        return f"Error writing {path}: {e}"

    old_lines = _count_lines(old_string) if old_string else 0
    new_lines = _count_lines(new_string) if new_string else 0
    suffix = ". All occurrences were replaced." if replace_all else ""
    return f"Replaced {old_lines} line{'s' if old_lines != 1 else ''} with {new_lines} line{'s' if new_lines != 1 else ''} in {path}{suffix}"


async def list_dir(path: str = ".") -> str:
    """List directory contents.

    Args:
        path: Directory path (relative to workspace), defaults to workspace root
    """
    try:
        resolved = _resolve_path(path)
    except ValueError as e:
        from src.tools.exec import is_sandbox_enabled

        if is_sandbox_enabled():
            resolved = _try_skill_ref_readonly(path)
            if resolved is None:
                return f"Error: {e}"
        else:
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
    case_insensitive: bool = True,
) -> str:
    """在文件内容中搜索匹配的文本行（正则表达式）。

    Args:
        pattern: 正则表达式搜索模式
        path: 搜索目标，可以是文件或目录路径（相对于工作区）
        glob_pattern: 目录搜索时的文件名过滤 (glob 模式)，如 "*.py"、"*.md"；path 为文件时忽略
        head_limit: 最大返回条数（默认 50）
        offset: 跳过前 N 条结果（默认 0）
        output_mode: "content" 显示匹配行，"files_with_matches" 仅列出匹配的文件路径
        case_insensitive: 是否忽略大小写（默认 True）
    """
    return await _grep_impl(pattern, path, glob_pattern, head_limit, offset, output_mode, case_insensitive)


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
    case_insensitive: bool = True,
) -> str:
    try:
        resolved = _resolve_path(path)
    except ValueError as e:
        from src.tools.exec import is_sandbox_enabled

        if is_sandbox_enabled():
            resolved = _try_skill_ref_readonly(path)
            if resolved is None:
                return f"Error: {e}"
        else:
            return f"Error: {e}"
    if not await aiofiles.os.path.exists(resolved):
        return await _path_not_found_hint(path, resolved)

    regex_or_err = _validate_regex(pattern, ignore_case=case_insensitive)
    if isinstance(regex_or_err, str):
        return regex_or_err
    regex = regex_or_err

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

    try:
        results, truncated, error = await asyncio.wait_for(asyncio.to_thread(_walk_grep), timeout=_GREP_TIMEOUT)
    except asyncio.TimeoutError:
        return "Error: grep timed out (pattern may cause excessive backtracking on large files)"
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
        from src.tools.exec import is_sandbox_enabled

        if is_sandbox_enabled():
            resolved = _try_skill_ref_readonly(path)
            if resolved is None:
                return f"Error: {e}"
        else:
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

            def _safe_mtime(p: Path) -> float:
                try:
                    return p.stat().st_mtime
                except OSError:
                    return 0.0

            all_matches.sort(key=_safe_mtime, reverse=True)
            total = len(all_matches)
            if offset >= total:
                return f"Error: offset {offset} exceeds total matches ({total})"
            page = all_matches[offset : offset + head_limit]
            lines = []
            for m in page:
                rel = str(m.relative_to(resolved))
                try:
                    st = m.stat()
                except OSError:
                    continue
                if m.is_dir():
                    lines.append(f"  {rel}/")
                else:
                    lines.append(f"  {rel}  ({st.st_size}B)")
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
