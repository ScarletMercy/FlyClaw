"""Tests for file_tools: write_file, edit_file, read_file."""

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


class TestGrep:
    @pytest.mark.asyncio
    async def test_default_case_insensitive(self, workspace):
        from src.tools.file_tools import grep, write_file

        await write_file("test.txt", "Hello World\nhello again\nGOODBYE\n")
        result = await grep("hello", path="test.txt")
        assert "Hello World" in result
        assert "hello again" in result

    @pytest.mark.asyncio
    async def test_explicit_case_sensitive(self, workspace):
        from src.tools.file_tools import grep, write_file

        await write_file("test.txt", "Hello World\nhello again\n")
        result = await grep("hello", path="test.txt", case_insensitive=False)
        assert "hello again" in result
        assert "Hello" not in result

    @pytest.mark.asyncio
    async def test_files_with_matches_case_insensitive(self, workspace):
        from src.tools.file_tools import grep, write_file

        await write_file("a.txt", "Foo bar\n")
        await write_file("b.txt", "baz qux\n")
        result = await grep("foo", path=".", output_mode="files_with_matches")
        assert "a.txt" in result
        assert "b.txt" not in result

    @pytest.mark.asyncio
    async def test_inline_flag_override(self, workspace):
        from src.tools.file_tools import grep, write_file

        await write_file("test.txt", "ABC\nabc\nAbc\n")
        result = await grep("(?-i:ABC)", path="test.txt", case_insensitive=True)
        assert "1 of 1" in result
        assert "ABC" in result
        assert "abc" not in result
        assert "Abc" not in result
