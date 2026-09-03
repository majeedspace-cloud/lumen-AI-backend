"""Session storage.

`SessionStore` is the interface. `InMemorySessionStore` is today's
implementation. If you outgrow it later, write a `RedisSessionStore`
that implements the same 4 methods and swap it in `get_session_store()`
below — nothing else in the app needs to change.
"""
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from threading import Lock

from app.core.config import get_settings
from app.core.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class SessionData:
    session_id: str
    name: str = "New Chat"  # Add session name for Named Sessions feature
    chat_history: list[dict] = field(default_factory=list)
    vector_store: VectorStore | None = None
    processed_files: set[tuple[str, int]] = field(default_factory=set)
    last_active: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_active = time.time()


class SessionStore(ABC):
    @abstractmethod
    def get_or_create(self, session_id: str) -> SessionData: ...

    @abstractmethod
    def save(self, session_data: SessionData) -> None: ...

    @abstractmethod
    def delete(self, session_id: str) -> None: ...

    @abstractmethod
    def cleanup_expired(self, ttl_seconds: int) -> int:
        """Remove sessions inactive longer than ttl_seconds. Returns count removed."""
        ...

    @abstractmethod
    def list_sessions(self) -> list[dict]:
        """Return list of all sessions with their metadata (id, name, last_active)."""
        ...

    @abstractmethod
    def rename_session(self, session_id: str, new_name: str) -> None:
        """Rename a session."""
        ...


class InMemorySessionStore(SessionStore):
    """Thread-safe in-memory session store.

    Good for a single-container deployment. Data is lost on restart and
    doesn't work across multiple backend processes — see the module
    docstring if you outgrow this.
    """

    def __init__(self):
        self._sessions: dict[str, SessionData] = {}
        self._lock = Lock()

    def get_or_create(self, session_id: str) -> SessionData:
        with self._lock:
            if session_id not in self._sessions:
                logger.info("Creating new session: %s", session_id)
                self._sessions[session_id] = SessionData(session_id=session_id)
            self._sessions[session_id].touch()
            return self._sessions[session_id]

    def save(self, session_data: SessionData) -> None:
        with self._lock:
            session_data.touch()
            self._sessions[session_data.session_id] = session_data

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def cleanup_expired(self, ttl_seconds: int) -> int:
        cutoff = time.time() - ttl_seconds
        with self._lock:
            expired = [sid for sid, s in self._sessions.items() if s.last_active < cutoff]
            for sid in expired:
                del self._sessions[sid]
        if expired:
            logger.info("Cleaned up %d expired sessions", len(expired))
        return len(expired)

    def list_sessions(self) -> list[dict]:
        """Return list of all sessions with their metadata."""
        with self._lock:
            return [
                {
                    "session_id": session_id,
                    "name": session.name,
                    "last_active": session.last_active,
                    "message_count": len(session.chat_history),
                }
                for session_id, session in self._sessions.items()
            ]

    def rename_session(self, session_id: str, new_name: str) -> None:
        """Rename a session."""
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].name = new_name
                logger.info("Renamed session %s to '%s'", session_id, new_name)
            else:
                logger.warning("Attempted to rename non-existent session: %s", session_id)


@lru_cache
def get_session_store() -> SessionStore:
    """Singleton — same store instance shared across all requests in this process."""
    settings = get_settings()
    if settings.session_backend == "redis":
        raise NotImplementedError(
            "Redis backend not implemented yet. Set SESSION_BACKEND=memory, "
            "or implement RedisSessionStore(SessionStore) and wire it in here."
        )
    return InMemorySessionStore()
