"""Hybrid retrieval pipeline: keyword search + semantic search + RRF fusion + reranking."""
import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.core.embeddings import EmbeddingModel
from app.core.exceptions import RetrievalError
from app.core.reranker import Reranker
from app.core.vector_store import VectorStore

logger = logging.getLogger(__name__)


def keyword_search(query: str, chunks: list[str], metadata: list[dict], k: int = 15) -> list[dict]:
    """TF-IDF cosine-similarity search over the chunk collection.

    Returns an empty list (not an error) if there are no chunks — an
    empty collection is a valid state, not a failure.
    """
    if not chunks:
        return []
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(chunks)
        query_vec = vectorizer.transform([query.lower()])
        scores = cosine_similarity(query_vec, matrix)[0]
    except Exception as exc:
        raise RetrievalError(f"Keyword search failed: {exc}") from exc

    top_k = min(k, len(chunks))
    top_indices = scores.argsort()[-top_k:][::-1]
    return [
        {"text": chunks[i], "source": metadata[i]["source"], "chunk_id": metadata[i]["chunk_id"]}
        for i in top_indices
    ]


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], k_constant: int = 60
) -> list[str]:
    """Merge multiple ranked ID lists into one ranking using RRF.

    Args:
        ranked_lists: Each inner list is chunk_ids ordered best-first from
            one retrieval method (e.g. one from keyword search, one from
            semantic search). Any number of lists is supported — the
            original code only fused exactly two.
        k_constant: Smoothing constant, 60 is the standard default.

    Returns:
        chunk_ids ordered by combined RRF score, descending.
    """
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, chunk_id in enumerate(ranked_list):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k_constant + rank + 1)
    return [chunk_id for chunk_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


class HybridRetriever:
    """Combines keyword search, semantic search, RRF fusion, and reranking into one call."""

    def __init__(self, embedding_model: EmbeddingModel, reranker: Reranker):
        self._embedder = embedding_model
        self._reranker = reranker

    def retrieve(
        self,
        query: str,
        store: VectorStore,
        keyword_k: int = 15,
        semantic_k: int = 20,
        rerank_top_n: int = 5,
        rrf_k: int = 60,
    ) -> list[dict]:
        """Run the full hybrid pipeline and return the top reranked chunks.

        Returns an empty list if the store has no data — this is a valid
        state (e.g. no PDF uploaded yet), not an error to raise.
        """
        if store.is_empty:
            return []

        keyword_hits = keyword_search(query, store.chunks, store.metadata, k=keyword_k)

        query_embedding = self._embedder.encode_one(query)
        _distances, faiss_indices = store.search(query_embedding, k=semantic_k)

        keyword_ids = [c["chunk_id"] for c in keyword_hits]
        semantic_ids = [
            store.metadata[i]["chunk_id"] for i in faiss_indices[0] if 0 <= i < len(store.metadata)
        ]

        fused_ids = reciprocal_rank_fusion([keyword_ids, semantic_ids], k_constant=rrf_k)

        id_to_chunk = {
            m["chunk_id"]: {"text": store.chunks[i], "source": m["source"], "chunk_id": m["chunk_id"]}
            for i, m in enumerate(store.metadata)
        }
        candidates = [id_to_chunk[cid] for cid in fused_ids[:25] if cid in id_to_chunk]

        return self._reranker.rerank(query, candidates, top_n=rerank_top_n)
