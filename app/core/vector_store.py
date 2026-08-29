"""FAISS-backed vector store for a single document collection (e.g. one session)."""
import logging
from pathlib import Path

import faiss
import numpy as np

from app.core.exceptions import VectorStoreError

logger = logging.getLogger(__name__)


class VectorStore:
    """Wraps a FAISS FlatL2 index plus the chunk text/metadata it indexes.

    One instance = one document collection (e.g. one user session's uploaded PDFs).
    """

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks: list[str] = []
        self.metadata: list[dict] = []

    def add(self, chunks: list[str], embeddings: np.ndarray, metadata: list[dict]) -> None:
        """Add new chunks + their embeddings + metadata to the index.

        Raises:
            VectorStoreError: If the counts don't line up, or embeddings
                have the wrong dimension (this was an unguarded crash risk
                in the original `build_index` — a bad embedding array
                would fail deep inside faiss with a cryptic error).
        """
        if not (len(chunks) == len(embeddings) == len(metadata)):
            raise VectorStoreError(
                f"Mismatched lengths: chunks={len(chunks)}, "
                f"embeddings={len(embeddings)}, metadata={len(metadata)}"
            )
        if len(embeddings) == 0:
            return

        embeddings = np.asarray(embeddings).astype("float32")
        if embeddings.shape[1] != self.dimension:
            raise VectorStoreError(
                f"Embedding dimension {embeddings.shape[1]} doesn't match "
                f"index dimension {self.dimension}. Did the embedding model change?"
            )

        try:
            self.index.add(embeddings)
        except Exception as exc:
            raise VectorStoreError(f"FAISS add() failed: {exc}") from exc

        self.chunks.extend(chunks)
        self.metadata.extend(metadata)

    def search(self, query_embedding: np.ndarray, k: int = 20) -> tuple[np.ndarray, np.ndarray]:
        """Search the index. Returns (distances, indices), both empty if the index has no vectors."""
        if self.index.ntotal == 0:
            return np.array([[]]), np.array([[]])
        try:
            query_embedding = np.asarray([query_embedding]).astype("float32")
            return self.index.search(query_embedding, min(k, self.index.ntotal))
        except Exception as exc:
            raise VectorStoreError(f"FAISS search() failed: {exc}") from exc

    def save(self, dir_path: str | Path) -> None:
        """Persist the index + chunks + metadata to disk."""
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        try:
            faiss.write_index(self.index, str(dir_path / "index.faiss"))
        except Exception as exc:
            raise VectorStoreError(f"Failed to write FAISS index: {exc}") from exc

    @classmethod
    def load(cls, dir_path: str | Path, dimension: int) -> "VectorStore":
        """Load a previously saved index. Raises VectorStoreError if the file is missing/corrupt."""
        index_path = Path(dir_path) / "index.faiss"
        if not index_path.exists():
            raise VectorStoreError(f"No saved index found at {index_path}")
        store = cls(dimension)
        try:
            store.index = faiss.read_index(str(index_path))
        except Exception as exc:
            raise VectorStoreError(f"Failed to read FAISS index: {exc}") from exc
        return store

    @property
    def is_empty(self) -> bool:
        return self.index.ntotal == 0
