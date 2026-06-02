"""Tests for file_tools: write_file, edit_file, read_file."""

import os
import tempfile

import pytest


@pytest.fixture
def workspace(tmp_path):
    """Set up a temporary workspace and configure file_tools to use it."""
    from src.tools import file_tools

    original = file_tools._BASE_DIR
    file_tools.set_workspace(str(tmp_path))
    yield tmp_path
    file_tools._BASE_DIR = original


class TestWriteFile:
    @pytest.mark.asyncio
    async def test_write_and_read(self, workspace):
        from src.tools.file_tools import read_file, write_file

        result = await write_file("hello.txt", "hello world\n")
        assert "Written" in result

        content = await read_file("hello.txt")
        assert "hello world" in content

    @pytest.mark.asyncio
    async def test_write_creates_parent_dirs(self, workspace):
        from src.tools.file_tools import read_file, write_file

        await write_file("sub/dir/file.txt", "nested\n")
        content = await read_file("sub/dir/file.txt")
        assert "nested" in content

    @pytest.mark.asyncio
    async def test_write_overwrites_existing(self, workspace):
        from src.tools.file_tools import read_file, write_file

        await write_file("f.txt", "old content\n")
        await write_file("f.txt", "new content\n")
        content = await read_file("f.txt")
        assert "new content" in content
        assert "old content" not in content


class TestEditFile:
    @pytest.mark.asyncio
    async def test_simple_replace(self, workspace):
        from src.tools.file_tools import edit_file, read_file, write_file

        await write_file("f.txt", "hello world\n")
        result = await edit_file("f.txt", "hello", "goodbye")
        assert "Replaced" in result

        content = await read_file("f.txt")
        assert "goodbye world" in content

    @pytest.mark.asyncio
    async def test_old_string_not_found(self, workspace):
        from src.tools.file_tools import edit_file, write_file

        await write_file("f.txt", "hello world\n")
        result = await edit_file("f.txt", "nonexistent", "replacement")
        assert "Error" in result
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_multiple_matches_without_replace_all(self, workspace):
        from src.tools.file_tools import edit_file, write_file

        await write_file("f.txt", "aaa bbb aaa\n")
        result = await edit_file("f.txt", "aaa", "ccc")
        assert "Error" in result
        assert "2 matches" in result

    @pytest.mark.asyncio
    async def test_multiple_matches_with_replace_all(self, workspace):
        from src.tools.file_tools import edit_file, read_file, write_file

        await write_file("f.txt", "aaa bbb aaa\n")
        result = await edit_file("f.txt", "aaa", "ccc", replace_all=True)
        assert "All occurrences" in result

        content = await read_file("f.txt")
        assert "ccc bbb ccc" in content

    @pytest.mark.asyncio
    async def test_edit_empty_file(self, workspace):
        from src.tools.file_tools import edit_file, read_file, write_file

        await write_file("f.txt", "")
        result = await edit_file("f.txt", "", "new content")
        assert "Replaced" in result

        content = await read_file("f.txt")
        assert "new content" in content
