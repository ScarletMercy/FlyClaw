"""Tests for config loading and validation."""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestLoadConfig:
    def test_default_config(self, tmp_path):
        """Loading from nonexistent path returns defaults."""
        from src.config import load_config

        cfg = load_config(tmp_path / "nonexistent.yaml")
        assert cfg.gateway.host == "127.0.0.1"
        assert cfg.gateway.port == 18080
        assert cfg.model.provider == "anthropic"
        assert cfg.agents.max_tool_rounds == 15

    def test_yaml_config(self, tmp_path):
        """Loading from a YAML file applies values."""
        from src.config import load_config

        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
gateway:
  host: "0.0.0.0"
  port: 9090
model:
  provider: "openai"
  name: "gpt-4o"
"""
        )
        cfg = load_config(config_file)
        assert cfg.gateway.host == "0.0.0.0"
        assert cfg.gateway.port == 9090
        assert cfg.model.provider == "openai"
        assert cfg.model.name == "gpt-4o"

    def test_env_substitution(self, tmp_path, monkeypatch):
        """Environment variables are substituted in config values."""
        from src.config import load_config

        monkeypatch.setenv("TEST_API_KEY", "sk-12345")
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
model:
  api_key: "${TEST_API_KEY}"
"""
        )
        cfg = load_config(config_file)
        assert cfg.model.api_key == "sk-12345"

    def test_empty_yaml_file(self, tmp_path):
        """Empty YAML file returns defaults."""
        from src.config import load_config

        config_file = tmp_path / "config.yaml"
        config_file.write_text("")
        cfg = load_config(config_file)
        assert cfg.gateway.port == 18080

    def test_invalid_yaml_ignored(self, tmp_path):
        """Invalid YAML values fall back to defaults."""
        from src.config import load_config

        config_file = tmp_path / "config.yaml"
        config_file.write_text("gateway:\n  port: not_a_number\n")
        cfg = load_config(config_file)
        assert cfg.gateway.port == 18080  # falls back to default


class TestConfigModels:
    def test_auth_config_defaults(self):
        from src.config import AuthConfig

        auth = AuthConfig()
        assert auth.enabled is True
        assert auth.default_role == "guest"
        assert auth.pairing_enabled is True
        assert auth.pairing_ttl_seconds == 300

    def test_memory_config_defaults(self):
        from src.config import MemoryConfig

        mem = MemoryConfig()
        assert mem.enabled is False
        assert mem.backend == "sqlite"
        assert mem.vector_weight == 0.7
        assert mem.chunk_tokens == 400

    def test_memory_config_lancedb(self):
        from src.config import MemoryConfig

        mem = MemoryConfig(backend="lancedb", lancedb_uri="/tmp/lance")
        assert mem.backend == "lancedb"
        assert mem.lancedb_uri == "/tmp/lance"

    def test_exec_tool_config(self):
        from src.config import ExecToolConfig

        cfg = ExecToolConfig()
        assert cfg.enabled is True
        assert cfg.timeout_seconds == 30
        assert cfg.sandbox_enabled is True
        assert cfg.max_concurrent == 3
