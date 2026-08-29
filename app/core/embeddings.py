"""Sentence-embedding model wrapper.

Loads the SentenceTransformer model once per process and reuses it.
Loading a transformer model is expensive (disk + memory) — you never
want this happening per-request in a FastAPI handler.
"""
import logging
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings
from app.core.exceptions import EmbeddingError

logger = logging.getLogger(__name__)


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


@lru_cache
def get_embedding_model() -> EmbeddingModel:
    """Singleton accessor — call this everywhere instead of instantiating EmbeddingModel directly."""
    settings = get_settings()
    return EmbeddingModel(settings.embedding_model_name)
