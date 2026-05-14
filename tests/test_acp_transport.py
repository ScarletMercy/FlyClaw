from __future__ import annotations

import json
from io import BytesIO

from src.acp.transport import NdjsonTransport


def test_write_message():
    buf = BytesIO()
    transport = NdjsonTransport(writer=buf)
    transport.write({"jsonrpc": "2.0", "method": "initialize"})
    buf.seek(0)
    data = json.loads(buf.read().decode("utf-8"))
    assert data["jsonrpc"] == "2.0"
    assert data["method"] == "initialize"


def test_write_is_ndjson():
    buf = BytesIO()
    transport = NdjsonTransport(writer=buf)
    transport.write({"id": 1})
    transport.write({"id": 2})
    buf.seek(0)
    lines = buf.read().decode("utf-8").strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"id": 1}
    assert json.loads(lines[1]) == {"id": 2}
