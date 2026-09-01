"""FastAPI dependency wiring — builds the RAGService from the various singletons."""
from functools import lru_cache

from app.core.config import get_settings
from app.core.embeddings import get_embedding_model
from app.core.llm import get_llm_client
from app.core.reranker import get_reranker
from app.core.retrieval import HybridRetriever
from app.services.intent_router import IntentRouter
from app.services.rag_service import RAGService
from app.services.session_store import SessionStore, get_session_store
from app.services.web_search import get_web_search_service


@lru_cache
def get_intent_router() -> IntentRouter:
    return IntentRouter(get_llm_client())


@lru_cache
def get_rag_service() -> RAGService:
    settings = get_settings()
    embedder = get_embedding_model()
    reranker = get_reranker()
    retriever = HybridRetriever(embedder, reranker)
    return RAGService(
        embedding_model=embedder,
        reranker=reranker,
        retriever=retriever,
        llm_client=get_llm_client(),
        web_search=get_web_search_service(),
        intent_router=get_intent_router(),
        settings=settings,
    )


def get_store() -> SessionStore:
    return get_session_store()
