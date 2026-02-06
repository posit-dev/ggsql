"""Tests for session management."""

from datetime import timedelta
from ggsql_rest._sessions import Session, SessionManager


def test_session_creation():
    session = Session("test123", timeout_mins=30)
    assert session.id == "test123"
    assert session.tables == []
    assert not session.is_expired()


def test_session_touch():
    session = Session("test123", timeout_mins=30)
    first_access = session.last_accessed
    session.touch()
    assert session.last_accessed >= first_access


def test_session_expiry():
    session = Session("test123", timeout_mins=0)
    # With 0 timeout, session expires immediately
    assert session.is_expired()


def test_session_manager_create():
    mgr = SessionManager(timeout_mins=30)
    session = mgr.create()
    assert session.id is not None
    assert len(session.id) == 32  # uuid hex


def test_session_manager_get():
    mgr = SessionManager(timeout_mins=30)
    session = mgr.create()
    retrieved = mgr.get(session.id)
    assert retrieved is not None
    assert retrieved.id == session.id


def test_session_manager_get_nonexistent():
    mgr = SessionManager(timeout_mins=30)
    assert mgr.get("nonexistent") is None


def test_session_manager_delete():
    mgr = SessionManager(timeout_mins=30)
    session = mgr.create()
    assert mgr.delete(session.id) is True
    assert mgr.get(session.id) is None


def test_session_manager_delete_nonexistent():
    mgr = SessionManager(timeout_mins=30)
    assert mgr.delete("nonexistent") is False


def test_session_manager_cleanup_expired():
    mgr = SessionManager(timeout_mins=0)  # Immediate expiry
    session = mgr.create()
    session_id = session.id
    mgr.cleanup_expired()
    assert mgr.get(session_id) is None
