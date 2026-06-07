"""Tests for file_tools skill references/ sandbox read-only bypass."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.tools import file_tools


@pytest.fixture(autouse=True)
def _reset_cache():
    """Clear skill ref dirs cache before and after each test."""
    file_tools._skill_ref_dirs_cache = None
    yield
    file_tools._skill_ref_dirs_cache = None


# ---------------------------------------------------------------------------
# _collect_skill_reference_dirs
# ---------------------------------------------------------------------------


class TestCollectSkillReferenceDirs:
    def test_caches_result(self):
        mock_container = MagicMock()
        mock_container._build_skill_directories.return_value = [("user", Path("/fake/skills"))]

        with (
            patch("src._container.get_container", return_value=mock_container),
            patch.object(Path, "is_dir", return_value=True),
        ):
            result1 = file_tools._collect_skill_reference_dirs()
            result2 = file_tools._collect_skill_reference_dirs()

        assert result1 is result2  # same list object from cache
        assert file_tools._skill_ref_dirs_cache is not None

    def test_returns_empty_on_no_container(self):
        with patch("src._container.get_container", side_effect=RuntimeError("no container")):
            result = file_tools._collect_skill_reference_dirs()
        assert result == []

    def test_discovers_references_subdir(self, tmp_path):
        skill_dir = tmp_path / "skills" / "my_skill"
        ref_dir = skill_dir / "references"
        ref_dir.mkdir(parents=True)

        mock_container = MagicMock()
        mock_container._build_skill_directories.return_value = [("user", skill_dir)]

        with patch("src._container.get_container", return_value=mock_container):
            result = file_tools._collect_skill_reference_dirs()

        assert len(result) == 1
        assert result[0] == ref_dir.resolve()

    def test_skips_nonexistent_references(self, tmp_path):
        skill_dir = tmp_path / "skills" / "my_skill"
        skill_dir.mkdir(parents=True)
        # No references/ subdir created

        mock_container = MagicMock()
        mock_container._build_skill_directories.return_value = [("user", skill_dir)]

        with patch("src._container.get_container", return_value=mock_container):
            result = file_tools._collect_skill_reference_dirs()

        assert result == []


# ---------------------------------------------------------------------------
# _try_skill_ref_readonly
# ---------------------------------------------------------------------------


class TestTrySkillRefReadonly:
    def test_absolute_path_under_ref_dir_returns_resolved(self, tmp_path):
        ref_dir = (tmp_path / "skills" / "s" / "references").resolve()
        ref_dir.mkdir(parents=True)
        target = ref_dir / "doc.md"
        target.write_text("data")

        file_tools._skill_ref_dirs_cache = [ref_dir]

        result = file_tools._try_skill_ref_readonly(str(target))
        assert result is not None
        assert Path(result) == target.resolve()

    def test_relative_path_under_workspace_ref_dir(self, tmp_path):
        ref_dir = (tmp_path / "skills" / "s" / "references").resolve()
        ref_dir.mkdir(parents=True)
        (ref_dir / "notes.txt").write_text("hello")

        file_tools._skill_ref_dirs_cache = [ref_dir]
        file_tools._BASE_DIR = str(tmp_path)

        try:
            result = file_tools._try_skill_ref_readonly("skills/s/references/notes.txt")
            assert result is not None
        finally:
            file_tools._BASE_DIR = "."

    def test_path_outside_ref_dir_returns_none(self, tmp_path):
        ref_dir = (tmp_path / "skills" / "s" / "references").resolve()
        ref_dir.mkdir(parents=True)

        file_tools._skill_ref_dirs_cache = [ref_dir]
        outside = tmp_path / "secrets" / "key.pem"
        outside.parent.mkdir(parents=True)
        outside.write_text("secret")

        result = file_tools._try_skill_ref_readonly(str(outside))
        assert result is None

    def test_traversal_attack_returns_none(self, tmp_path):
        ref_dir = (tmp_path / "skills" / "s" / "references").resolve()
        ref_dir.mkdir(parents=True)

        file_tools._skill_ref_dirs_cache = [ref_dir]

        result = file_tools._try_skill_ref_readonly(str(ref_dir / ".." / ".." / ".." / "etc" / "passwd"))
        assert result is None

    def test_empty_ref_dirs_returns_none(self):
        file_tools._skill_ref_dirs_cache = []

        result = file_tools._try_skill_ref_readonly("/any/path/file.txt")
        assert result is None
