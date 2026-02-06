"""Session management for isolated DuckDB instances."""

import uuid
from datetime import datetime, timedelta

from ggsql import DuckDBReader


class Session:
    """A user session with an isolated DuckDB instance."""

    def __init__(self, session_id: str, timeout_mins: int = 30):
        self.id = session_id
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.timeout = timedelta(minutes=timeout_mins)
        self.duckdb = DuckDBReader("duckdb://memory")
        self.tables: list[str] = []

    def touch(self) -> None:
        """Update last accessed time."""
        self.last_accessed = datetime.now()

    def is_expired(self) -> bool:
        """Check if session has expired."""
        return datetime.now() - self.last_accessed > self.timeout


class SessionManager:
    """Manages user sessions."""

    def __init__(self, timeout_mins: int = 30):
        self._sessions: dict[str, Session] = {}
        self._timeout_mins = timeout_mins

    def create(self) -> Session:
        """Create a new session."""
        session_id = uuid.uuid4().hex
        session = Session(session_id, self._timeout_mins)
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        """Get a session by ID, or None if not found or expired."""
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if session.is_expired():
            del self._sessions[session_id]
            return None
        session.touch()
        return session

    def delete(self, session_id: str) -> bool:
        """Delete a session. Returns True if deleted, False if not found."""
        return self._sessions.pop(session_id, None) is not None

    def cleanup_expired(self) -> None:
        """Remove all expired sessions."""
        expired = [sid for sid, s in self._sessions.items() if s.is_expired()]
        for sid in expired:
            del self._sessions[sid]
