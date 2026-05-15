"""Tests for skill management tools and disabled filtering."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import AppConfig, SkillsConfig
from src.skills.loader import discover_skills, is_skill_disabled
from src.skills.types import Skill, SkillMetadata
from src.tools.skills_tools import (
    skill_install_tool,
    skill_toggle_tool,
    skill_uninstall_tool,
    skill_view_tool,
    skills_list_tool,
)


@pytest.fixture
def sample_skill(tmp_path):
    skill_dir = tmp_path / "test-skill"
    skill_dir.mkdir()
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        "---\nname: test-skill\ndescription: A test skill\n---\n# Test Skill\nContent here."
    )
    return Skill(
        name="test-skill",
        description="A test skill",
        file_path=skill_md,
        base_dir=skill_dir,
        source="test",
        metadata=SkillMetadata(name="test-skill", description="A test skill"),
        body="# Test Skill\nContent here.",
    )


@pytest.fixture
def mock_config():
    config = AppConfig()
    config.skills = SkillsConfig(
        enabled=True,
        disabled=[],
        channel_disabled={},
    )
    return config


@pytest.fixture
def mock_container(sample_skill, mock_config):
    container = MagicMock()
    container.skills_cache = [sample_skill]
    container.config = mock_config
    container._build_skill_directories.return_value = []
    return container


class TestIsSkillDisabled:
    def test_not_disabled(self, mock_config):
        assert not is_skill_disabled("test-skill", mock_config)

    def test_globally_disabled(self, mock_config):
        mock_config.skills.disabled = ["test-skill"]
        assert is_skill_disabled("test-skill", mock_config)

    def test_channel_disabled(self, mock_config):
        mock_config.skills.channel_disabled = {"feishu": ["test-skill"]}
        assert is_skill_disabled("test-skill", mock_config, channel="feishu")
        assert not is_skill_disabled("test-skill", mock_config, channel="qq")

    def test_channel_not_specified(self, mock_config):
        mock_config.skills.channel_disabled = {"feishu": ["test-skill"]}
        assert not is_skill_disabled("test-skill", mock_config)


class TestSkillsListTool:
    def test_returns_skill_list(self, mock_container):
        with patch("src.tools.skills_tools._get_container", return_value=mock_container):
            tool = skills_list_tool()
            result = json.loads(tool.fn({}))
            assert "skills" in result
            assert len(result["skills"]) == 1
            assert result["skills"][0]["name"] == "test-skill"
            assert result["skills"][0]["disabled"] is False

    def test_shows_disabled_status(self, mock_container, mock_config):
        mock_config.skills.disabled = ["test-skill"]
        with patch("src.tools.skills_tools._get_container", return_value=mock_container):
            tool = skills_list_tool()
            result = json.loads(tool.fn({}))
            assert result["skills"][0]["disabled"] is True


class TestSkillViewTool:
    def test_view_existing_skill(self, mock_container):
        with patch("src.tools.skills_tools._get_container", return_value=mock_container):
            tool = skill_view_tool()
            result = json.loads(tool.fn({"skill_name": "test-skill"}))
            assert result["name"] == "test-skill"
            assert "body" in result

    def test_view_nonexistent_skill(self, mock_container):
        with patch("src.tools.skills_tools._get_container", return_value=mock_container):
            tool = skill_view_tool()
            result = json.loads(tool.fn({"skill_name": "missing"}))
            assert "error" in result


class TestSkillToggleTool:
    def test_disable_skill(self, mock_container, mock_config):
        with patch("src.tools.skills_tools._get_container", return_value=mock_container):
            with patch("src.tools.skills_tools.save_config"):
                with patch("src.tools.skills_tools._reload_all"):
                    tool = skill_toggle_tool()
                    result = json.loads(
                        tool.fn({"skill_name": "test-skill", "enabled": False})
                    )
                    assert result["success"] is True
                    assert result["action"] == "disabled"
                    assert "test-skill" in mock_config.skills.disabled

    def test_enable_skill(self, mock_container, mock_config):
        mock_config.skills.disabled = ["test-skill"]
        with patch("src.tools.skills_tools._get_container", return_value=mock_container):
            with patch("src.tools.skills_tools.save_config"):
                with patch("src.tools.skills_tools._reload_all"):
                    tool = skill_toggle_tool()
                    result = json.loads(
                        tool.fn({"skill_name": "test-skill", "enabled": True})
                    )
                    assert result["success"] is True
                    assert result["action"] == "enabled"
                    assert "test-skill" not in mock_config.skills.disabled

    def test_toggle_channel_disabled(self, mock_container, mock_config):
        with patch("src.tools.skills_tools._get_container", return_value=mock_container):
            with patch("src.tools.skills_tools.save_config"):
                with patch("src.tools.skills_tools._reload_all"):
                    tool = skill_toggle_tool()
                    result = json.loads(
                        tool.fn(
                            {
                                "skill_name": "test-skill",
                                "enabled": False,
                                "channel": "feishu",
                            }
                        )
                    )
                    assert result["success"] is True
                    assert "feishu" in mock_config.skills.channel_disabled
                    assert "test-skill" in mock_config.skills.channel_disabled["feishu"]


class TestSkillInstallTool:
    def test_install_from_local_directory(self, tmp_path, mock_container):
        skill_dir = tmp_path / "new-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: new-skill\n---\nContent")

        user_skills_dir = tmp_path / "user-skills"
        user_skills_dir.mkdir()

        with patch("src.tools.skills_tools._get_container", return_value=mock_container):
            with patch("src.tools.skills_tools._USER_SKILLS_DIR", user_skills_dir):
                with patch("src.tools.skills_tools._reload_all"):
                    tool = skill_install_tool()
                    result = json.loads(tool.fn({"source": str(skill_dir)}))
                    assert result["success"] is True
                    assert (user_skills_dir / "new-skill" / "SKILL.md").exists()

    def test_install_missing_path(self, mock_container):
        with patch("src.tools.skills_tools._get_container", return_value=mock_container):
            tool = skill_install_tool()
            result = json.loads(tool.fn({"source": "/nonexistent/path"}))
            assert "error" in result


class TestSkillUninstallTool:
    def test_uninstall_user_skill(self, tmp_path, mock_container, mock_config):
        user_skills_dir = tmp_path / "user-skills"
        user_skills_dir.mkdir()
        skill_dir = user_skills_dir / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: test-skill\n---\nContent")

        from src.skills.types import Skill, SkillMetadata

        skill = Skill(
            name="test-skill",
            description="Test",
            file_path=skill_dir / "SKILL.md",
            base_dir=skill_dir,
            source="user",
            metadata=SkillMetadata(name="test-skill"),
            body="Content",
        )
        mock_container.skills_cache = [skill]

        with patch("src.tools.skills_tools._get_container", return_value=mock_container):
            with patch("src.tools.skills_tools._USER_SKILLS_DIR", user_skills_dir):
                with patch("src.tools.skills_tools.save_config"):
                    with patch("src.tools.skills_tools._reload_all"):
                        tool = skill_uninstall_tool()
                        result = json.loads(
                            tool.fn({"skill_name": "test-skill"})
                        )
                        assert result["success"] is True
                        assert not skill_dir.exists()
