"""Sentence-embedding model wrapper.

Loads the SentenceTransformer model once per process and reuses it.
Loading a transformer model is expensive (disk + memory) — you never
want this happening per-request in a FastAPI handler.

Now implements lazy loading to reduce startup memory footprint.
"""
import logging
from threading import Lock

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings
from app.core.exceptions import EmbeddingError

logger = logging.getLogger(__name__)

# Global singleton instance with thread-safe lazy loading
_embedding_model_instance = None
_embedding_model_lock = Lock()


class EmbeddingModel:
    """Thin wrapper around a SentenceTransformer for encoding text."""

    def __init__(self, model_name: str):
        logger.info("Loading embedding model: %s", model_name)
        try:
            self._model = SentenceTransformer(model_name)
        except Exception as exc:
            raise EmbeddingError(f"Failed to load embedding model '{model_name}': {exc}") from exc

    def encode_one(self, text: str) -> np.ndarray:
        """Embed a single string. Raises EmbeddingError on empty input or model failure."""
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text.")
        try:
            return self._model.encode(text, normalize_embeddings=True)
        except Exception as exc:
            raise EmbeddingError(f"Embedding failed: {exc}") from exc

    def encode_many(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed a batch of strings at once — much faster than a Python loop of encode_one."""
        if not texts:
            return np.array([])
        try:
            return self._model.encode(
                texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
            )
        except Exception as exc:
            raise EmbeddingError(f"Batch embedding failed: {exc}") from exc

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()


def get_embedding_model() -> EmbeddingModel:
    """Lazy-loading singleton accessor — model only loads on first call.

    This significantly reduces startup memory footprint for FastAPI Cloud,
    as models aren't loaded until the first actual request needs them.
    Thread-safe to prevent multiple simultaneous loads.
    """
    global _embedding_model_instance

    if _embedding_model_instance is not None:
        return _embedding_model_instance

    with _embedding_model_lock:
        # Double-check pattern to avoid loading if another thread just finished
        if _embedding_model_instance is not None:
            return _embedding_model_instance

        settings = get_settings()
        if settings.skip_model_loading:
            logger.warning("Skipping embedding model load due to skip_model_loading=True")
            raise EmbeddingError("Model loading is disabled")

        _embedding_model_instance = EmbeddingModel(settings.embedding_model_name)
        return _embedding_model_instance


def is_embedding_model_loaded() -> bool:
    """Check if the embedding model has been loaded (useful for health checks)."""
    return _embedding_model_instance is not None
