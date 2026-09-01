"""Reranking.

v1: a lightweight passthrough — candidates arrive already ordered by RRF
fusion (keyword + semantic), so "reranking" here just trims to top_n. No
model, no dependency, no memory cost. This is a deliberate simplification,
not a missing feature: real cross-encoder or LLM-based reranking is a
good v2 addition (ties in well with the agent-workflow upgrade), added
back once the app is live and actual answer quality data justifies it.
"""
import logging

logger = logging.getLogger(__name__)


class Reranker:
    def rerank(self, query: str, candidates: list[dict], top_n: int = 5) -> list[dict]:
        """Trim to top_n. Kept as a method (not a bare function) so the
        interface matches a future real reranker without callers changing.
        """
        return candidates[:top_n]


def get_reranker() -> Reranker:
    return Reranker()
