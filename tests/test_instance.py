"""Tests for multi-instance support (src/instance.py + config adjustment)."""

import sys
import pytest
from pathlib import Path


class TestParseInstanceFromArgv:
    """Test parse_instance_from_argv strips trailing number from sys.argv."""

    def test_strips_trailing_number(self, monkeypatch):
        from src import instance

        monkeypatch.setattr(sys, "argv", ["flyclaw", "2"])
        result = instance.parse_instance_from_argv()
        assert result == 2
        assert sys.argv == ["flyclaw"]

    def test_no_trailing_number(self, monkeypatch):
        from src import instance

        monkeypatch.setattr(sys, "argv", ["flyclaw"])
        result = instance.parse_instance_from_argv()
        assert result is None
        assert sys.argv == ["flyclaw"]

    def test_trailing_non_number(self, monkeypatch):
        from src import instance

        monkeypatch.setattr(sys, "argv", ["flyclaw", "doctor"])
        result = instance.parse_instance_from_argv()
        assert result is None
        assert sys.argv == ["flyclaw", "doctor"]

    def test_trailing_zero_returns_none(self, monkeypatch):
        from src import instance

        monkeypatch.setattr(sys, "argv", ["flyclaw", "0"])
        result = instance.parse_instance_from_argv()
        assert result is None
        assert sys.argv == ["flyclaw", "0"]

    def test_trailing_negative_not_stripped(self, monkeypatch):
        from src import instance

        monkeypatch.setattr(sys, "argv", ["flyclaw", "-1"])
        result = instance.parse_instance_from_argv()
        assert result is None
        assert sys.argv == ["flyclaw", "-1"]

    def test_large_instance_number(self, monkeypatch):
        from src import instance

        monkeypatch.setattr(sys, "argv", ["flyclaw", "99"])
        result = instance.parse_instance_from_argv()
        assert result == 99
        assert sys.argv == ["flyclaw"]

    def test_flag_not_stripped(self, monkeypatch):
        from src import instance

        monkeypatch.setattr(sys, "argv", ["flyclaw", "--version"])
        result = instance.parse_instance_from_argv()
        assert result is None

    def test_subcommand_with_number(self, monkeypatch):
        from src import instance

        monkeypatch.setattr(sys, "argv", ["flyclaw", "doctor", "2"])
        result = instance.parse_instance_from_argv()
        assert result == 2
        assert sys.argv == ["flyclaw", "doctor"]

    def test_model_switch_not_stripped(self, monkeypatch):
        """flyclaw model switch 2 — "2" is model ID, NOT instance number."""
        from src import instance

        monkeypatch.setattr(sys, "argv", ["flyclaw", "model", "switch", "2"])
        result = instance.parse_instance_from_argv()
        assert result is None
        assert sys.argv == ["flyclaw", "model", "switch", "2"]

    def test_model_list_with_instance(self, monkeypatch):
        """flyclaw model list 2 — "2" IS instance number (list takes no args)."""
        from src import instance

        monkeypatch.setattr(sys, "argv", ["flyclaw", "model", "list", "2"])
        result = instance.parse_instance_from_argv()
        assert result == 2
        assert sys.argv == ["flyclaw", "model", "list"]


class TestInstancePaths:
    """Test path resolution for default and numbered instances."""

    def setup_method(self):
        from src import instance

        instance._instance_number = None

    def teardown_method(self):
        from src import instance

        instance._instance_number = None

    def test_default_config_path(self):
        from src.instance import config_path

        assert config_path() == Path.home() / ".flyclaw" / "config.yaml"

    def test_instance_2_config_path(self):
        from src.instance import config_path

        assert config_path(2) == Path.home() / ".flyclaw2" / "config.yaml"

    def test_default_data_dir(self):
        from src.instance import data_dir

        assert data_dir() == Path.home() / ".flyclaw" / "data"

    def test_instance_3_data_dir(self):
        from src.instance import data_dir

        assert data_dir(3) == Path.home() / ".flyclaw3" / "data"

    def test_default_skills_dir(self):
        from src.instance import skills_dir

        assert skills_dir() == Path.home() / ".flyclaw" / "skills"

    def test_instance_2_skills_dir(self):
        from src.instance import skills_dir

        assert skills_dir(2) == Path.home() / ".flyclaw2" / "skills"

    def test_default_temp_dir(self):
        from src.instance import temp_dir

        assert temp_dir() == Path.home() / ".flyclaw" / "temp"

    def test_instance_2_temp_dir(self):
        from src.instance import temp_dir

        assert temp_dir(2) == Path.home() / ".flyclaw2" / "temp"

    def test_default_service_name(self):
        from src.instance import service_name

        assert service_name() == "flyclaw"

    def test_instance_2_service_name(self):
        from src.instance import service_name

        assert service_name(2) == "flyclaw-2"

    def test_default_label(self):
        from src.instance import instance_label

        assert instance_label() == ""

    def test_instance_2_label(self):
        from src.instance import instance_label

        assert instance_label(2) == "-2"

    def test_set_instance_affects_paths(self):
        from src.instance import set_instance, config_path, data_dir

        set_instance(5)
        assert config_path() == Path.home() / ".flyclaw5" / "config.yaml"
        assert data_dir() == Path.home() / ".flyclaw5" / "data"

    def test_explicit_n_overrides_global(self):
        from src.instance import set_instance, data_dir

        set_instance(5)
        # Explicit parameter takes precedence
        assert data_dir(3) == Path.home() / ".flyclaw3" / "data"


class TestAdjustPathsForInstance:
    """Test _adjust_paths_for_instance redirects default data paths."""

    def teardown_method(self):
        from src import instance

        instance._instance_number = None

    def _p(self, s: str) -> str:
        """Normalize path separators for cross-platform assertions."""
        return Path(s).as_posix()

    def test_default_instance_no_adjustment(self, tmp_path):
        from src.config import load_config
        from src.instance import set_instance

        set_instance(None)
        cfg = load_config(tmp_path / "nonexistent.yaml")
        # Default instance — paths should contain /data/ not /data-N/
        p = self._p(cfg.checkpointer.path)
        assert "/data/" in p or "\\data\\" in p
        assert "/data-" not in p

    def test_instance_2_adjusts_default_paths(self, tmp_path):
        from src.config import load_config
        from src.instance import set_instance

        set_instance(2)
        cfg = load_config(tmp_path / "nonexistent.yaml")
        p_cp = self._p(cfg.checkpointer.path)
        p_mem = self._p(cfg.memory.db_path)
        p_cron = self._p(cfg.cron.store_path)
        p_auth = self._p(cfg.auth.db_path)
        p_ms = self._p(cfg.memory_store.db_path)
        p_task = self._p(cfg.task.db_path)
        p_ss = self._p(cfg.session_search.index_path)
        p_kan = self._p(cfg.kanban.db_dir)
        for label, val in [
            ("checkpointer", p_cp),
            ("memory", p_mem),
            ("cron", p_cron),
            ("auth", p_auth),
            ("memory_store", p_ms),
            ("task", p_task),
            ("session_search", p_ss),
            ("kanban", p_kan),
        ]:
            assert ".flyclaw2" in val, f"{label} path not adjusted: {val}"

    def test_explicit_path_not_adjusted(self, tmp_path):
        from src.config import load_config
        from src.instance import set_instance

        set_instance(2)
        config_file = tmp_path / "config-2.yaml"
        config_file.write_text(
            """
checkpointer:
  path: "/custom/path/checkpoints.db"
"""
        )
        cfg = load_config(config_file)
        # Explicitly set path should NOT be adjusted
        assert ".flyclaw2" not in self._p(cfg.checkpointer.path)
        assert "checkpoints.db" in cfg.checkpointer.path
        # But other defaults should still be adjusted
        assert ".flyclaw2" in self._p(cfg.memory.db_path)

    def test_yaml_config_with_instance_2(self, tmp_path):
        from src.config import load_config
        from src.instance import set_instance

        set_instance(2)
        config_file = tmp_path / "config-2.yaml"
        config_file.write_text(
            """
gateway:
  host: "0.0.0.0"
  port: 18081
model:
  provider: "openai"
  name: "gpt-4o"
"""
        )
        cfg = load_config(config_file)
        assert cfg.gateway.port == 18081
        assert cfg.model.name == "gpt-4o"
        # Data paths should still be auto-adjusted
        assert ".flyclaw2" in self._p(cfg.checkpointer.path)

    def test_instance_2_adjusts_default_port(self, tmp_path):
        """非默认实例自动偏移端口"""
        from src.config import load_config
        from src.instance import set_instance

        set_instance(2)
        cfg = load_config(tmp_path / "nonexistent.yaml")
        assert cfg.gateway.port == 18082  # 18080 + 2

    def test_explicit_port_not_adjusted(self, tmp_path):
        """用户显式设置的端口不偏移"""
        from src.config import load_config
        from src.instance import set_instance

        set_instance(2)
        config_file = tmp_path / "config-2.yaml"
        config_file.write_text("gateway:\n  port: 9090\n")
        cfg = load_config(config_file)
        assert cfg.gateway.port == 9090

    def test_default_instance_port_unchanged(self, tmp_path):
        """默认实例端口不变"""
        from src.config import load_config
        from src.instance import set_instance

        set_instance(None)
        cfg = load_config(tmp_path / "nonexistent.yaml")
        assert cfg.gateway.port == 18080
