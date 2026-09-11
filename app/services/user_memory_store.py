"""Cross-session user memory storage.

Distinct from SessionStore: sessions are disposable conversations (create,
rename, delete freely). This is memory about the *person*, keyed by a
long-lived device_id that survives across every session they ever create —
the whole point is facts learned in one conversation show up in a totally
different one later.
"""
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class UserMemoryData:
    device_id: str
    facts: dict = field(default_factory=dict)
    # The off-switch: when False, the app neither injects existing facts
    # into answers NOR extracts new ones — a full opt-out, not just "stop
    # showing me what you know," since a partial opt-out (still learning
    # silently in the background) isn't what "memory off" should mean.
    enabled: bool = True
    last_updated: float = field(default_factory=time.time)


class UserMemoryStore(ABC):
    @abstractmethod
    def get_or_create(self, device_id: str) -> UserMemoryData: ...

    @abstractmethod
    def save(self, memory: UserMemoryData) -> None: ...

    @abstractmethod
    def clear_facts(self, device_id: str) -> None:
        """Wipe stored facts. Does NOT touch the enabled/disabled setting —
        'forget what you know about me' and 'stop remembering going
        forward' are two different requests."""
        ...

    @abstractmethod
    def set_enabled(self, device_id: str, enabled: bool) -> None: ...


class InMemoryUserMemoryStore(UserMemoryStore):
    def __init__(self):
        self._memories: dict[str, UserMemoryData] = {}
        self._lock = Lock()

    def get_or_create(self, device_id: str) -> UserMemoryData:
        with self._lock:
            if device_id not in self._memories:
                self._memories[device_id] = UserMemoryData(device_id=device_id)
            return self._memories[device_id]

    def save(self, memory: UserMemoryData) -> None:
        with self._lock:
            memory.last_updated = time.time()
            self._memories[memory.device_id] = memory

    def clear_facts(self, device_id: str) -> None:
        with self._lock:
            if device_id in self._memories:
                self._memories[device_id].facts = {}
                self._memories[device_id].last_updated = time.time()
                logger.info("Cleared memory facts for device %s", device_id)

    def set_enabled(self, device_id: str, enabled: bool) -> None:
        with self._lock:
            memory = self._memories.setdefault(device_id, UserMemoryData(device_id=device_id))
            memory.enabled = enabled
            memory.last_updated = time.time()
            logger.info("Memory %s for device %s", "enabled" if enabled else "disabled", device_id)


@lru_cache
def get_user_memory_store() -> UserMemoryStore:
    return InMemoryUserMemoryStore()
