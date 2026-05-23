import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from concurrent.futures import ThreadPoolExecutor

from src.skills.hub import (
    _normalize_bundle_path,
    _validate_skill_name,
    _validate_bundle_rel_path,
    HubLockFile,
    resolve_source,
    create_sources,
    parallel_search,
    quarantine_bundle,
    install_from_quarantine,
    append_audit_log,
    ensure_hub_dirs,
)
from src.skills.types import SkillMeta, SkillBundle, ScanResult


class TestNormalizeBundlePath:
    def test_simple_name(self):
        assert _normalize_bundle_path("my-skill", field_name="test", allow_nested=False) == "my-skill"

    def test_nested_allowed(self):
        assert _normalize_bundle_path("refs/guide.md", field_name="test", allow_nested=True) == "refs/guide.md"

    def test_nested_blocked(self):
        with pytest.raises(ValueError):
            _normalize_bundle_path("a/b", field_name="test", allow_nested=False)

    def test_traversal(self):
        with pytest.raises(ValueError, match="Unsafe"):
            _normalize_bundle_path("../etc/passwd", field_name="test", allow_nested=True)

    def test_absolute(self):
        with pytest.raises(ValueError, match="Unsafe"):
            _normalize_bundle_path("/etc/passwd", field_name="test", allow_nested=True)

    def test_empty(self):
        with pytest.raises(ValueError, match="empty"):
            _normalize_bundle_path("  ", field_name="test", allow_nested=False)

    def test_non_string(self):
        with pytest.raises(ValueError, match="expected a string"):
            _normalize_bundle_path(123, field_name="test", allow_nested=False)

    def test_windows_drive(self):
        with pytest.raises(ValueError, match="Unsafe"):
            _normalize_bundle_path("C:/Windows", field_name="test", allow_nested=True)

    def test_backslash(self):
        assert _normalize_bundle_path("a\\b", field_name="test", allow_nested=True) == "a/b"


class TestValidateSkillName:
    def test_valid(self):
        assert _validate_skill_name("my-skill") == "my-skill"

    def test_rejects_nested(self):
        with pytest.raises(ValueError):
            _validate_skill_name("a/b")


class TestValidateBundleRelPath:
    def test_nested(self):
        assert _validate_bundle_rel_path("refs/guide.md") == "refs/guide.md"

    def test_simple(self):
        assert _validate_bundle_rel_path("SKILL.md") == "SKILL.md"


class TestHubLockFile:
    def _make_lock(self, tmp_path):
        return HubLockFile(path=tmp_path / "lock.json")

    def test_load_missing(self, tmp_path):
        lock = self._make_lock(tmp_path)
        data = lock.load()
        assert data == {"version": 1, "installed": {}}

    def test_save_and_load(self, tmp_path):
        lock = self._make_lock(tmp_path)
        lock.save({"version": 1, "installed": {"x": {"source": "test"}}})
        data = lock.load()
        assert data["installed"]["x"]["source"] == "test"

    def test_record_install(self, tmp_path):
        lock = self._make_lock(tmp_path)
        lock.record_install(
            name="my-skill", source="skills-sh", identifier="skills-sh/foo/bar",
            trust_level="community", scan_verdict="safe", skill_hash="sha256:abc",
            install_path="my-skill", files=["SKILL.md"],
        )
        entry = lock.get_installed("my-skill")
        assert entry is not None
        assert entry["source"] == "skills-sh"
        assert entry["trust_level"] == "community"
        assert "installed_at" in entry

    def test_record_uninstall(self, tmp_path):
        lock = self._make_lock(tmp_path)
        lock.record_install(
            name="my-skill", source="test", identifier="x",
            trust_level="community", scan_verdict="safe", skill_hash="h",
            install_path="my-skill", files=["SKILL.md"],
        )
        assert lock.get_installed("my-skill") is not None
        lock.record_uninstall("my-skill")
        assert lock.get_installed("my-skill") is None

    def test_uninstall_nonexistent(self, tmp_path):
        lock = self._make_lock(tmp_path)
        lock.record_uninstall("nope")

    def test_list_installed(self, tmp_path):
        lock = self._make_lock(tmp_path)
        lock.record_install(
            name="a", source="s1", identifier="x",
            trust_level="community", scan_verdict="safe", skill_hash="h",
            install_path="a", files=["SKILL.md"],
        )
        lock.record_install(
            name="b", source="s2", identifier="y",
            trust_level="trusted", scan_verdict="safe", skill_hash="h2",
            install_path="b", files=["SKILL.md"],
        )
        installed = lock.list_installed()
        names = {e["name"] for e in installed}
        assert names == {"a", "b"}

    def test_corrupted_json(self, tmp_path):
        lock = self._make_lock(tmp_path)
        lock.path.write_text("NOT JSON{{{", encoding="utf-8")
        data = lock.load()
        assert data == {"version": 1, "installed": {}}

    def test_atomic_write(self, tmp_path):
        lock = self._make_lock(tmp_path)
        lock.save({"version": 1, "installed": {}})
        assert not (tmp_path / "lock.json.tmp").exists()
        assert (tmp_path / "lock.json").exists()


class TestResolveSource:
    def _make_sources(self):
        return create_sources()

    def test_skills_sh_prefix(self):
        sources = self._make_sources()
        src = resolve_source("skills-sh/foo/bar", sources)
        assert src is not None
        assert src.source_id() == "skills-sh"

    def test_http_md_url(self):
        sources = self._make_sources()
        src = resolve_source("https://example.com/skill.md", sources)
        assert src is not None
        assert src.source_id() == "url"

    def test_http_non_md_returns_none(self):
        sources = self._make_sources()
        src = resolve_source("https://example.com/skill.html", sources)
        assert src is None

    def test_bare_name_clawhub(self):
        sources = self._make_sources()
        src = resolve_source("my-skill", sources)
        assert src is not None
        assert src.source_id() == "clawhub"

    def test_no_matching_source(self):
        sources = []
        src = resolve_source("anything", sources)
        assert src is None


class TestParallelSearch:
    def _mock_source(self, source_id, results=None):
        src = MagicMock()
        src.source_id.return_value = source_id
        src.search.return_value = results or []
        return src

    def test_merges_results(self):
        s1 = self._mock_source("a", [SkillMeta(name="x", description="d", source="a", identifier="a/x", trust_level="community")])
        s2 = self._mock_source("b", [SkillMeta(name="y", description="d", source="b", identifier="b/y", trust_level="community")])
        results = parallel_search([s1, s2], "test")
        names = {r.name for r in results}
        assert names == {"x", "y"}

    def test_source_filter(self):
        s1 = self._mock_source("a", [SkillMeta(name="x", description="d", source="a", identifier="a/x", trust_level="community")])
        s2 = self._mock_source("b", [SkillMeta(name="y", description="d", source="b", identifier="b/y", trust_level="community")])
        results = parallel_search([s1, s2], "test", source_filter="a")
        assert all(r.source == "a" for r in results)

    def test_empty_sources(self):
        results = parallel_search([], "test")
        assert results == []

    def test_exception_in_source(self):
        s1 = self._mock_source("a")
        s1.search.side_effect = Exception("boom")
        results = parallel_search([s1], "test")
        assert results == []


class TestQuarantineBundle:
    def test_creates_files(self, tmp_path):
        import shutil
        bundle = SkillBundle(
            name="test-skill",
            files={"SKILL.md": "# Hello", "refs/a.md": "ref content"},
            source="test", identifier="test/x",
            trust_level="community", metadata={},
        )
        with patch("src.skills.hub.QUARANTINE_DIR", tmp_path / "q"):
            dest = quarantine_bundle(bundle)
            assert (dest / "SKILL.md").exists()
            assert (dest / "refs" / "a.md").exists()

    def test_rejects_bad_name(self):
        bundle = SkillBundle(
            name="../evil", files={"SKILL.md": "x"},
            source="test", identifier="test/x",
            trust_level="community", metadata={},
        )
        with pytest.raises(ValueError):
            quarantine_bundle(bundle)


class TestInstallFromQuarantine:
    def test_moves_and_records(self, tmp_path):
        q_dir = tmp_path / "quarantine" / "my-skill"
        q_dir.mkdir(parents=True)
        (q_dir / "SKILL.md").write_text("# Hi", encoding="utf-8")

        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        lock_path = skills_dir / "lock.json"

        bundle = SkillBundle(
            name="my-skill", files={"SKILL.md": "# Hi"},
            source="test", identifier="test/x",
            trust_level="community", metadata={},
        )
        scan_result = ScanResult(
            skill_name="my-skill", source="test",
            trust_level="community", verdict="safe",
            findings=[], scanned_at="", summary="",
        )

        with patch("src.skills.hub.SKILLS_DIR", skills_dir), \
             patch("src.skills.hub.QUARANTINE_DIR", tmp_path / "quarantine"), \
             patch("src.skills.hub.append_audit_log"), \
             patch("src.skills.hub.HubLockFile") as MockLock:
            mock_lock = MagicMock()
            MockLock.return_value = mock_lock
            result = install_from_quarantine(q_dir, "my-skill", bundle, scan_result)

        assert (skills_dir / "my-skill" / "SKILL.md").exists()
        assert not q_dir.exists()
        mock_lock.record_install.assert_called_once()


class TestAuditLog:
    def test_appends_line(self, tmp_path):
        log_file = tmp_path / "audit.log"
        with patch("src.skills.hub.AUDIT_LOG", log_file):
            append_audit_log("INSTALL", "my-skill", "test", "community", "safe")
        content = log_file.read_text(encoding="utf-8")
        assert "INSTALL" in content
        assert "my-skill" in content


class TestEnsureHubDirs:
    def test_creates_structure(self, tmp_path):
        hub_dir = tmp_path / "hub"
        with patch("src.skills.hub.HUB_DIR", hub_dir), \
             patch("src.skills.hub.QUARANTINE_DIR", hub_dir / "quarantine"), \
             patch("src.skills.hub.INDEX_CACHE_DIR", hub_dir / "cache"), \
             patch("src.skills.hub.LOCK_FILE", hub_dir / "lock.json"), \
             patch("src.skills.hub.AUDIT_LOG", hub_dir / "audit.log"):
            from src.skills.hub import ensure_hub_dirs as _ensure
            _ensure()
            assert (hub_dir / "quarantine").is_dir()
            assert (hub_dir / "cache").is_dir()
            assert (hub_dir / "lock.json").exists()
            assert (hub_dir / "audit.log").exists()
