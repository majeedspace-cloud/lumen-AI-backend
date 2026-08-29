"""Cross-encoder reranking — scores (query, chunk) pairs for relevance."""
import logging
from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.core.config import get_settings
from app.core.exceptions import RerankError

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self, model_name: str, hf_token: str):
        logger.info("Loading reranker model: %s", model_name)
        try:
            self._model = CrossEncoder(model_name, token=hf_token)
        except Exception as exc:
            raise RerankError(f"Failed to load reranker '{model_name}': {exc}") from exc

    def rerank(self, query: str, candidates: list[dict], top_n: int = 5) -> list[dict]:
        """Score and sort candidate chunks by relevance to the query.

        Args:
            query: The search query.
            candidates: List of dicts, each must have a "text" key.
            top_n: How many top-scoring candidates to return.

        Returns:
            The top_n candidates, each with a "score" key added, sorted
            descending by score. Empty list if `candidates` is empty.
        """
        if not candidates:
            return []

        texts = [c["text"] for c in candidates if c.get("text", "").strip()]
        valid_candidates = [c for c in candidates if c.get("text", "").strip()]
        if not texts:
            return []

        try:
            pairs = [[query, text] for text in texts]
            scores = self._model.predict(pairs)
        except Exception as exc:
            raise RerankError(f"Cross-encoder prediction failed: {exc}") from exc

        for candidate, score in zip(valid_candidates, scores):
            candidate["score"] = float(score)

        valid_candidates.sort(key=lambda c: c["score"], reverse=True)
        return valid_candidates[:top_n]


@lru_cache
def get_reranker() -> Reranker:
    settings = get_settings()
    return Reranker(settings.reranker_model_name, settings.hf_token)
