from __future__ import annotations

from src.acp.session import AcpSessionManager


def test_create_session():
    mgr = AcpSessionManager()
    sid = mgr.create("agent-1", cwd="/tmp")
    assert sid
    session = mgr.get(sid)
    assert session is not None
    assert session.agent_id == "agent-1"
    assert session.cwd == "/tmp"
    assert session.session_id == sid


def test_get_nonexistent():
    mgr = AcpSessionManager()
    assert mgr.get("no-such-id") is None


def test_close_session():
    mgr = AcpSessionManager()
    sid = mgr.create("agent-1")
    assert mgr.get(sid) is not None
    mgr.close(sid)
    assert mgr.get(sid) is None


def test_list_sessions():
    mgr = AcpSessionManager()
    s1 = mgr.create("a1")
    s2 = mgr.create("a2")
    sessions = mgr.list_sessions()
    assert len(sessions) == 2
    ids = {s.session_id for s in sessions}
    assert s1 in ids
    assert s2 in ids
