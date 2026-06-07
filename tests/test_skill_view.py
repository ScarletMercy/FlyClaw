"""Tests for skill_view lookup tolerance and format changes.

Covers two bugs:
- Lookup layer: exact string match fails when model copies **bold** names
- Format layer: JSON-wrapped body reduces model compliance
"""

import json

import pytest

from src.skills.types import Skill, SkillMetadata


def _make_skill(name: str, body: str = "Skill body content", description: str = "Test") -> Skill:
    from pathlib import Path

    return Skill(
        name=name,
        description=description,
        file_path=Path(f"/fake/{name}/SKILL.md"),
        base_dir=Path(f"/fake/{name}"),
        source="builtin",
        metadata=SkillMetadata(name=name, description=description),
        body=body,
    )


# ── Lookup layer: _normalize_skill_name ─────────────────────


class TestNormalizeSkillName:
    """Tests for the name normalization helper."""

    def test_plain_name_unchanged(self):
        from src.skills.manager import _normalize_skill_name

        assert _normalize_skill_name("my-skill") == "my-skill"

    def test_strips_whitespace(self):
        from src.skills.manager import _normalize_skill_name

        assert _normalize_skill_name("  my-skill  ") == "my-skill"

    def test_strips_leading_trailing_asterisks(self):
        from src.skills.manager import _normalize_skill_name

        assert _normalize_skill_name("**my-skill**") == "my-skill"

    def test_strips_single_asterisk(self):
        from src.skills.manager import _normalize_skill_name

        assert _normalize_skill_name("*my-skill*") == "my-skill"

    def test_strips_asterisks_and_whitespace(self):
        from src.skills.manager import _normalize_skill_name

        assert _normalize_skill_name("  **my-skill**  ") == "my-skill"

    def test_preserves_internal_hyphens(self):
        from src.skills.manager import _normalize_skill_name

        assert _normalize_skill_name("**my-cool-skill**") == "my-cool-skill"

    def test_empty_after_strip(self):
        from src.skills.manager import _normalize_skill_name

        assert _normalize_skill_name("****") == ""

    def test_only_whitespace(self):
        from src.skills.manager import _normalize_skill_name

        assert _normalize_skill_name("   ") == ""


# ── Lookup layer: _find_skill ───────────────────────────────


class TestFindSkill:
    """Tests for tolerant skill lookup."""

    def test_exact_match(self):
        from src.skills.manager import _find_skill

        skills = [_make_skill("my-skill")]
        assert _find_skill(skills, "my-skill") is not None

    def test_bold_wrapped_name(self):
        from src.skills.manager import _find_skill

        skills = [_make_skill("my-skill")]
        assert _find_skill(skills, "**my-skill**") is not None

    def test_whitespace_padded(self):
        from src.skills.manager import _find_skill

        skills = [_make_skill("my-skill")]
        assert _find_skill(skills, "  my-skill  ") is not None

    def test_case_insensitive_fallback(self):
        from src.skills.manager import _find_skill

        skills = [_make_skill("My-Skill")]
        assert _find_skill(skills, "my-skill") is not None

    def test_not_found_returns_none(self):
        from src.skills.manager import _find_skill

        skills = [_make_skill("my-skill")]
        assert _find_skill(skills, "other-skill") is None

    def test_empty_name_returns_none(self):
        from src.skills.manager import _find_skill

        skills = [_make_skill("my-skill")]
        assert _find_skill(skills, "") is None

    def test_only_asterisks_returns_none(self):
        from src.skills.manager import _find_skill

        skills = [_make_skill("my-skill")]
        assert _find_skill(skills, "****") is None


# ── Format layer: _format_skill_result ──────────────────────


class TestFormatSkillResult:
    """Tests for plain-text result formatting."""

    def test_body_is_top_level_plain_text(self):
        from src.skills.manager import _format_skill_result

        skill = _make_skill("my-skill", body="# My Skill\n\nDo something useful.")
        result = _format_skill_result(skill, view_count=3, use_count=1)

        # Body content should appear at the top, not wrapped in JSON
        assert result.startswith("# My Skill")
        assert "Do something useful." in result

    def test_not_valid_json(self):
        from src.skills.manager import _format_skill_result

        skill = _make_skill("my-skill", body="Body text here")
        result = _format_skill_result(skill)

        # Should NOT be parseable as JSON (it's plain text now)
        with pytest.raises(json.JSONDecodeError):
            json.loads(result)

    def test_metadata_footer_included(self):
        from src.skills.manager import _format_skill_result

        skill = _make_skill("my-skill", body="Body")
        result = _format_skill_result(skill, view_count=5, use_count=2)

        # Metadata should be in a footer section
        assert "my-skill" in result
        assert "Views" in result or "views" in result

    def test_skill_path_info_appended(self):
        from src.skills.manager import _format_skill_result

        skill = _make_skill("my-skill", body="Body")
        result = _format_skill_result(skill)

        assert "Path:" in result

    def test_empty_body_still_has_metadata(self):
        from src.skills.manager import _format_skill_result

        skill = _make_skill("my-skill", body="")
        result = _format_skill_result(skill)

        # Even with empty body, metadata footer should exist
        assert "my-skill" in result

    def test_extra_kwargs_absorbed_without_error(self):
        """get_usage() returns keys beyond view_count/use_count — **_extra must absorb them."""
        from src.skills.manager import _format_skill_result

        skill = _make_skill("my-skill", body="Body")
        # Simulate full usage record with extra keys — must not raise
        result = _format_skill_result(
            skill,
            view_count=1,
            use_count=0,
            last_viewed_at=None,
            last_used_at=None,
            patch_count=3,
            created_by="agent",
            state="active",
            pinned=False,
        )
        assert "Body" in result
        assert "Views: 1" in result
