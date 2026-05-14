import json
import time
from pathlib import Path

import pytest

from src.canvas.file_resolver import FileResolver
from src.canvas.a2ui import A2uiBuilder
from src.canvas.capability import CanvasCapabilityManager


class TestFileResolver:
    def test_resolve_index(self, tmp_path):
        (tmp_path / "index.html").write_text("<h1>Hello</h1>")
        resolver = FileResolver(tmp_path)
        path, mime = resolver.resolve("index.html")
        assert path.name == "index.html"
        assert "text/html" in mime

    def test_resolve_subdir(self, tmp_path):
        sub = tmp_path / "assets"
        sub.mkdir()
        (sub / "style.css").write_text("body{}")
        resolver = FileResolver(tmp_path)
        path, mime = resolver.resolve("assets/style.css")
        assert path.name == "style.css"

    def test_reject_traversal(self, tmp_path):
        resolver = FileResolver(tmp_path)
        with pytest.raises(ValueError, match="traversal"):
            resolver.resolve("../../../etc/passwd")

    def test_default_index_not_found(self, tmp_path):
        resolver = FileResolver(tmp_path)
        path, mime = resolver.resolve("")
        assert path is None


class TestA2uiBuilder:
    def test_text_component(self):
        builder = A2uiBuilder()
        builder.add_text("Hello world", surface_id="main")
        lines = builder.to_jsonl()
        assert len(lines) == 2
        assert "surfaceUpdate" in lines[0]
        assert "beginRendering" in lines[1]

    def test_column_layout(self):
        builder = A2uiBuilder()
        builder.add_column(["comp1", "comp2"], surface_id="s1")
        lines = builder.to_jsonl()
        assert any("Column" in l for l in lines)

    def test_multiple_surfaces(self):
        builder = A2uiBuilder()
        builder.add_text("A", surface_id="s1")
        builder.add_text("B", surface_id="s2")
        lines = builder.to_jsonl()
        surface_updates = [l for l in lines if "surfaceUpdate" in l]
        assert len(surface_updates) == 2

    def test_jsonl_is_valid_json(self):
        builder = A2uiBuilder()
        builder.add_text("Hello")
        builder.add_markdown("# Heading\nParagraph")
        for line in builder.to_jsonl():
            parsed = json.loads(line)
            assert isinstance(parsed, dict)


class TestCanvasCapability:
    def test_mint_and_validate(self):
        mgr = CanvasCapabilityManager()
        token = mgr.mint("node-1")
        assert token
        assert mgr.validate(token, "node-1")

    def test_invalid_token(self):
        mgr = CanvasCapabilityManager()
        assert not mgr.validate("bad-token", "node-1")

    def test_wrong_node(self):
        mgr = CanvasCapabilityManager()
        token = mgr.mint("node-1")
        assert not mgr.validate(token, "node-2")

    def test_expired_token(self):
        mgr = CanvasCapabilityManager(ttl_seconds=0)
        token = mgr.mint("node-1")
        time.sleep(0.05)
        assert not mgr.validate(token, "node-1")

    def test_revoke(self):
        mgr = CanvasCapabilityManager()
        token = mgr.mint("node-1")
        mgr.revoke(token)
        assert not mgr.validate(token, "node-1")

    def test_sliding_renewal(self):
        mgr = CanvasCapabilityManager(ttl_seconds=10)
        token = mgr.mint("node-1")
        assert mgr.validate(token, "node-1")
        assert mgr.validate(token, "node-1")
