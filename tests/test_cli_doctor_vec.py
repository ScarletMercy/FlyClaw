"""Tests for cli cmd_status memory archive line."""

from __future__ import annotations

from types import SimpleNamespace

from src.cli import cmd_status


def _make_config(vector_enabled=False, memory_store_enabled=True, **vec_kwargs):
    """构造最小 config，够 cmd_status 用。"""
    ms = SimpleNamespace(
        enabled=memory_store_enabled,
        vector_enabled=vector_enabled,
        vector_model=vec_kwargs.get("vector_model", "text-embedding-3-small"),
        vector_dimensions=vec_kwargs.get("vector_dimensions", 1536),
    )
    return SimpleNamespace(
        model=SimpleNamespace(name="test-model", provider="openai"),
        gateway=SimpleNamespace(port=18080, auth_token=""),
        channels=SimpleNamespace(qq=SimpleNamespace(enabled=False)),
        cron=SimpleNamespace(enabled=False),
        memory=SimpleNamespace(enabled=False),
        memory_store=ms,
        skills=SimpleNamespace(enabled=False),
        plugins=SimpleNamespace(enabled=False),
        tools=SimpleNamespace(exec=SimpleNamespace(approval_mode="disabled")),
        session=SimpleNamespace(idle_reset_minutes=30),
    )


class TestCmdStatusArchive:
    def test_vector_on_prints_hybrid(self, monkeypatch, capsys):
        """enabled+vector_enabled → hybrid: FTS5+向量 + model/dim。"""
        cfg = _make_config(
            vector_enabled=True,
            memory_store_enabled=True,
            vector_model="bge-m3",
            vector_dimensions=1024,
        )
        monkeypatch.setattr("src.cli._load_config", lambda: cfg)
        cmd_status(None)
        out = capsys.readouterr().out
        assert "记忆归档:   已启用 (hybrid: FTS5+向量, model=bge-m3, dim=1024)" in out

    def test_vector_off_prints_fts5_only(self, monkeypatch, capsys):
        """enabled+!vector_enabled → FTS5-only（archive 仍启用，不依赖 vector）。"""
        cfg = _make_config(vector_enabled=False, memory_store_enabled=True)
        monkeypatch.setattr("src.cli._load_config", lambda: cfg)
        cmd_status(None)
        out = capsys.readouterr().out
        assert "记忆归档:   已启用 (FTS5-only)" in out

    def test_disabled_prints_unenabled(self, monkeypatch, capsys):
        """memory_store.enabled=False → 未启用。"""
        cfg = _make_config(memory_store_enabled=False)
        monkeypatch.setattr("src.cli._load_config", lambda: cfg)
        cmd_status(None)
        out = capsys.readouterr().out
        assert "记忆归档:   未启用" in out

    def test_contradictory_prints_warning(self, monkeypatch, capsys):
        """vector_enabled=True 但 memory_store.enabled=False → 矛盾警告。"""
        cfg = _make_config(vector_enabled=True, memory_store_enabled=False)
        monkeypatch.setattr("src.cli._load_config", lambda: cfg)
        cmd_status(None)
        out = capsys.readouterr().out
        assert "⚠" in out
        assert "矛盾" in out
