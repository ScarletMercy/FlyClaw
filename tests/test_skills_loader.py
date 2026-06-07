"""Tests for src/skills/loader.py — frontmatter parsing, skill loading, discovery."""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.skills.loader import (
    parse_frontmatter,
    is_skill_disabled,
    load_skill,
)


# ── parse_frontmatter ──────────────────────────────────────


class TestParseFrontmatter:
    def test_with_valid_frontmatter(self):
        content = "---\nname: my-skill\ndescription: A test\n---\nBody content"
        meta, body = parse_frontmatter(content)
        assert meta.get("name") == "my-skill"
        assert meta.get("description") == "A test"
        assert "Body content" in body

    def test_no_frontmatter(self):
        content = "Just plain text\nNo frontmatter"
        meta, body = parse_frontmatter(content)
        assert meta == {}
        assert body == content

    def test_empty_frontmatter(self):
        content = "---\n---\nBody"
        meta, body = parse_frontmatter(content)
        assert meta == {}
        assert "Body" in body

    def test_invalid_yaml(self):
        content = "---\n: invalid: yaml: [\n---\nBody"
        meta, body = parse_frontmatter(content)
        # Should gracefully fall back
        assert isinstance(meta, dict)

    def test_multiline_body(self):
        content = "---\nname: test\n---\nLine 1\nLine 2\nLine 3"
        meta, body = parse_frontmatter(content)
        assert "Line 1" in body
        assert "Line 3" in body

    def test_frontmatter_with_extra_fields(self):
        content = "---\nname: test\nuser-invocable: false\ncommand-dispatch: /test\n---\nBody"
        meta, body = parse_frontmatter(content)
        assert meta.get("user-invocable") is False
        assert meta.get("command-dispatch") == "/test"


# ── is_skill_disabled ──────────────────────────────────────


class TestIsSkillDisabled:
    def _make_config(self, disabled=None, channel_disabled=None):
        cfg = MagicMock()
        cfg.skills.disabled = disabled or set()
        cfg.skills.channel_disabled = channel_disabled or {}
        return cfg

    def test_not_disabled(self):
        cfg = self._make_config()
        assert is_skill_disabled("my-skill", cfg) is False

    def test_globally_disabled(self):
        cfg = self._make_config(disabled={"my-skill"})
        assert is_skill_disabled("my-skill", cfg) is True

    def test_channel_disabled(self):
        cfg = self._make_config(channel_disabled={"qq": {"my-skill"}})
        assert is_skill_disabled("my-skill", cfg, channel="qq") is True

    def test_channel_disabled_different_channel(self):
        cfg = self._make_config(channel_disabled={"qq": {"my-skill"}})
        assert is_skill_disabled("my-skill", cfg, channel="weixin") is False

    def test_channel_disabled_no_channel(self):
        cfg = self._make_config(channel_disabled={"qq": {"my-skill"}})
        assert is_skill_disabled("my-skill", cfg) is False


# ── load_skill ─────────────────────────────────────────────


class TestLoadSkill:
    @pytest.mark.asyncio
    async def test_load_valid_skill(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: my-skill\ndescription: Test skill\n---\n# My Skill\n\nDo things.",
            encoding="utf-8",
        )
        skill = await load_skill(skill_dir, "builtin")
        assert skill is not None
        assert skill.name == "my-skill"
        assert skill.description == "Test skill"
        assert "Do things" in skill.body

    @pytest.mark.asyncio
    async def test_load_no_skill_md(self, tmp_path):
        skill_dir = tmp_path / "empty"
        skill_dir.mkdir()
        skill = await load_skill(skill_dir, "builtin")
        assert skill is None

    @pytest.mark.asyncio
    async def test_load_default_name_from_dir(self, tmp_path):
        skill_dir = tmp_path / "auto-named"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\n---\nJust body",
            encoding="utf-8",
        )
        skill = await load_skill(skill_dir, "builtin")
        assert skill is not None
        assert skill.name == "auto-named"

    @pytest.mark.asyncio
    async def test_load_too_large_file(self, tmp_path):
        from src.skills.loader import _MAX_SKILL_FILE_BYTES

        skill_dir = tmp_path / "large"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("x" * (_MAX_SKILL_FILE_BYTES + 1), encoding="utf-8")
        skill = await load_skill(skill_dir, "builtin")
        assert skill is None
