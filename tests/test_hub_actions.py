import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

from src.skills.types import SkillMeta, SkillBundle, ScanResult


def _make_container(skills_cache=None, hub_enabled=True, guard_enabled=True):
    container = MagicMock()
    container.skills_cache = skills_cache or []
    container.config.skills.hub.enabled = hub_enabled
    container.config.skills.hub.guard_enabled = guard_enabled
    container.config.skills.disabled = []
    return container


def _make_skill(name, base_dir="/tmp/skill"):
    skill = MagicMock()
    skill.name = name
    skill.base_dir = Path(base_dir)
    skill.source = "user"
    return skill


async def _call_action(action, **kwargs):
    from src.skills.manager import get_tools
    tools = get_tools()
    tool_fn = tools[0].fn
    return await tool_fn(action=action, **kwargs)


class TestSearchHub:
    @pytest.mark.asyncio
    async def test_empty_query_error(self):
        with patch("src._container.get_container", return_value=_make_container()):
            result = await _call_action("search_hub")
            data = json.loads(result)
            assert "error" in data

    @pytest.mark.asyncio
    async def test_search_returns_results(self):
        meta = SkillMeta(
            name="test-skill", description="A test skill",
            source="skills-sh", identifier="skills-sh/foo/bar",
            trust_level="community",
        )
        with patch("src._container.get_container", return_value=_make_container()), \
             patch("src.skills.hub.create_sources") as mock_cs, \
             patch("src.skills.hub.parallel_search", return_value=[meta]):
            mock_cs.return_value = [MagicMock()]
            result = await _call_action("search_hub", query="test")
            data = json.loads(result)
            assert len(data["results"]) == 1
            assert data["results"][0]["name"] == "test-skill"

    @pytest.mark.asyncio
    async def test_search_failure(self):
        with patch("src._container.get_container", return_value=_make_container()), \
             patch("src.skills.hub.create_sources", side_effect=Exception("network error")):
            result = await _call_action("search_hub", query="test")
            data = json.loads(result)
            assert "error" in data

    @pytest.mark.asyncio
    async def test_description_none_guard(self):
        meta = SkillMeta(
            name="x", description=None,
            source="s", identifier="s/x", trust_level="community",
        )
        with patch("src._container.get_container", return_value=_make_container()), \
             patch("src.skills.hub.create_sources", return_value=[MagicMock()]), \
             patch("src.skills.hub.parallel_search", return_value=[meta]):
            result = await _call_action("search_hub", query="x")
            data = json.loads(result)
            assert data["results"][0]["description"] == ""


class TestInspectHub:
    @pytest.mark.asyncio
    async def test_missing_identifier(self):
        with patch("src._container.get_container", return_value=_make_container()):
            result = await _call_action("inspect_hub")
            data = json.loads(result)
            assert "error" in data

    @pytest.mark.asyncio
    async def test_inspect_success(self):
        meta = SkillMeta(
            name="my-skill", description="desc",
            source="skills-sh", identifier="skills-sh/foo/bar",
            trust_level="community", repo="foo/bar",
        )
        mock_src = MagicMock()
        mock_src.inspect.return_value = meta
        with patch("src._container.get_container", return_value=_make_container()), \
             patch("src.skills.hub.create_sources", return_value=[mock_src]), \
             patch("src.skills.hub.resolve_source", return_value=mock_src):
            result = await _call_action("inspect_hub", identifier="skills-sh/foo/bar")
            data = json.loads(result)
            assert data["name"] == "my-skill"


class TestInstallHub:
    @pytest.mark.asyncio
    async def test_missing_identifier(self):
        with patch("src._container.get_container", return_value=_make_container()):
            result = await _call_action("install_hub")
            data = json.loads(result)
            assert "error" in data

    @pytest.mark.asyncio
    async def test_install_success(self, tmp_path):
        bundle = SkillBundle(
            name="my-skill", files={"SKILL.md": "# Hello"},
            source="test", identifier="test/x",
            trust_level="community", metadata={},
        )
        scan_result = ScanResult(
            skill_name="my-skill", source="test",
            trust_level="community", verdict="safe",
            findings=[], scanned_at="", summary="",
        )
        mock_src = MagicMock()
        mock_src.fetch.return_value = bundle
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        with patch("src._container.get_container", return_value=_make_container()), \
             patch("src.skills.hub.create_sources", return_value=[mock_src]), \
             patch("src.skills.hub.resolve_source", return_value=mock_src), \
             patch("src.skills.hub.ensure_hub_dirs"), \
             patch("src.skills.hub.quarantine_bundle", return_value=tmp_path / "q"), \
             patch("src.skills.hub.install_from_quarantine", return_value=skills_dir / "my-skill"), \
             patch("src.skills.manager._reload_skills_from_manager"), \
             patch("src.skills.guard.scan_skill", return_value=scan_result), \
             patch("src.skills.guard.should_allow_install", return_value=(True, "ok")):
            (tmp_path / "q").mkdir()
            result = await _call_action("install_hub", identifier="test/x")
            data = json.loads(result)
            assert data.get("success") is True

    @pytest.mark.asyncio
    async def test_install_blocked_by_guard(self, tmp_path):
        bundle = SkillBundle(
            name="evil", files={"SKILL.md": "rm -rf /"},
            source="test", identifier="test/x",
            trust_level="community", metadata={},
        )
        scan_result = ScanResult(
            skill_name="evil", source="test",
            trust_level="community", verdict="dangerous",
            findings=[], scanned_at="", summary="",
        )
        mock_src = MagicMock()
        mock_src.fetch.return_value = bundle
        q_path = tmp_path / "q"
        q_path.mkdir()

        with patch("src._container.get_container", return_value=_make_container()), \
             patch("src.skills.hub.create_sources", return_value=[mock_src]), \
             patch("src.skills.hub.resolve_source", return_value=mock_src), \
             patch("src.skills.hub.ensure_hub_dirs"), \
             patch("src.skills.hub.quarantine_bundle", return_value=q_path), \
             patch("src.skills.guard.scan_skill", return_value=scan_result), \
             patch("src.skills.guard.should_allow_install", return_value=(False, "dangerous")), \
             patch("src.skills.guard.format_scan_report", return_value="report"):
            result = await _call_action("install_hub", identifier="test/x")
            data = json.loads(result)
            assert "blocked" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_fetch_returns_none(self):
        mock_src = MagicMock()
        mock_src.fetch.return_value = None
        with patch("src._container.get_container", return_value=_make_container()), \
             patch("src.skills.hub.create_sources", return_value=[mock_src]), \
             patch("src.skills.hub.resolve_source", return_value=mock_src):
            result = await _call_action("install_hub", identifier="test/x")
            data = json.loads(result)
            assert "error" in data

    @pytest.mark.asyncio
    async def test_guard_disabled(self, tmp_path):
        bundle = SkillBundle(
            name="safe-skill", files={"SKILL.md": "# Ok"},
            source="test", identifier="test/x",
            trust_level="community", metadata={},
        )
        mock_src = MagicMock()
        mock_src.fetch.return_value = bundle
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        q_path = tmp_path / "q"
        q_path.mkdir()

        with patch("src._container.get_container", return_value=_make_container(guard_enabled=False)), \
             patch("src.skills.hub.create_sources", return_value=[mock_src]), \
             patch("src.skills.hub.resolve_source", return_value=mock_src), \
             patch("src.skills.hub.ensure_hub_dirs"), \
             patch("src.skills.hub.quarantine_bundle", return_value=q_path), \
             patch("src.skills.hub.install_from_quarantine", return_value=skills_dir / "safe-skill"), \
             patch("src.skills.manager._reload_skills_from_manager"):
            result = await _call_action("install_hub", identifier="test/x")
            data = json.loads(result)
            assert data.get("success") is True
            assert data["skill"]["scan_verdict"] == "safe"


class TestScanHub:
    @pytest.mark.asyncio
    async def test_missing_name(self):
        with patch("src._container.get_container", return_value=_make_container()):
            result = await _call_action("scan_hub")
            data = json.loads(result)
            assert "error" in data

    @pytest.mark.asyncio
    async def test_scan_from_cache(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# ok", encoding="utf-8")
        skill = _make_skill("my-skill", str(skill_dir))

        with patch("src._container.get_container", return_value=_make_container(skills_cache=[skill])), \
             patch("src.skills.guard.scan_skill") as mock_scan:
            mock_scan.return_value = ScanResult(
                skill_name="my-skill", source="user",
                trust_level="community", verdict="safe",
                findings=[], scanned_at="", summary="",
            )
            result = await _call_action("scan_hub", name="my-skill")
            data = json.loads(result)
            assert data["verdict"] == "safe"

    @pytest.mark.asyncio
    async def test_skill_not_found(self):
        with patch("src._container.get_container", return_value=_make_container(skills_cache=[])), \
             patch("src.skills.hub.HubLockFile") as MockLock:
            MockLock.return_value.get_installed.return_value = None
            result = await _call_action("scan_hub", name="nope")
            data = json.loads(result)
            assert "error" in data
            assert "not found" in data["error"].lower()
