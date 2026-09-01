"""Text embeddings via the Gemini API — no local model, no torch, no GPU deps.

This replaces a local SentenceTransformer model. That approach needed
torch + sentence-transformers loaded into RAM (several hundred MB minimum,
before even loading model weights) — which is exactly what didn't fit on
any free-tier host (512MB-1GB typical). This version just makes an API
call, same as we already do for the LLM. No heavy dependency, no local
model, no memory problem.

Uses Gemini's task_type parameter, which measurably improves retrieval
quality: document chunks are embedded as RETRIEVAL_DOCUMENT, search
queries as RETRIEVAL_QUERY — the model produces embeddings tuned for
each role instead of one generic embedding for everything.
"""
import logging
from functools import lru_cache

import numpy as np
from google import genai
from google.genai import types

from app.core.config import get_settings
from app.core.exceptions import EmbeddingError

logger = logging.getLogger(__name__)

# Gemini's embed_content endpoint accepts a batch, but we cap our own batch
# size defensively rather than assume an undocumented upper limit.
_MAX_BATCH_SIZE = 100


class EmbeddingModel:
    """Thin wrapper around Gemini's embed_content API."""

    def __init__(self, api_key: str, model_name: str, output_dimensionality: int):
        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name
        self._dimension = output_dimensionality

    def _embed(self, texts: list[str], task_type: str) -> np.ndarray:
        if not texts:
            return np.array([])
        try:
            response = self._client.models.embed_content(
                model=self._model_name,
                contents=texts,
                config=types.EmbedContentConfig(
                    output_dimensionality=self._dimension,
                    task_type=task_type,
                ),
            )
            return np.array([e.values for e in response.embeddings], dtype="float32")
        except Exception as exc:
            raise EmbeddingError(f"Embedding failed: {exc}") from exc

    def encode_one(self, text: str, task_type: str = "RETRIEVAL_QUERY") -> np.ndarray:
        """Embed a single string. Raises EmbeddingError on empty input or API failure."""
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text.")
        return self._embed([text], task_type)[0]

    def encode_many(self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
        """Embed a batch of strings, chunked into API-sized batches internally."""
        if not texts:
            return np.array([])
        all_vectors = []
        for i in range(0, len(texts), _MAX_BATCH_SIZE):
            batch = texts[i : i + _MAX_BATCH_SIZE]
            all_vectors.append(self._embed(batch, task_type))
        return np.vstack(all_vectors)

    @property
    def dimension(self) -> int:
        return self._dimension


@lru_cache
def get_embedding_model() -> EmbeddingModel:
    settings = get_settings()
    return EmbeddingModel(
        settings.gemini_api_key, settings.embedding_model_name, settings.embedding_output_dimensionality
    )
