"""FastAPI dependency wiring — builds the RAGService from the various singletons.

All components now use lazy loading to reduce startup memory footprint.
"""
from threading import Lock

from app.core.config import get_settings
from app.core.embeddings import get_embedding_model
from app.core.llm import get_llm_client
from app.core.reranker import get_reranker
from app.core.retrieval import HybridRetriever
from app.services.intent_router import IntentRouter
from app.services.rag_service import RAGService
from app.services.session_store import SessionStore, get_session_store
from app.services.web_search import get_web_search_service

# Global singleton instances with thread-safe lazy loading
_intent_router_instance = None
_rag_service_instance = None
_intent_router_lock = Lock()
_rag_service_lock = Lock()


def get_intent_router() -> IntentRouter:
    """Lazy-loading singleton for IntentRouter."""
    global _intent_router_instance

    if _intent_router_instance is not None:
        return _intent_router_instance

    with _intent_router_lock:
        if _intent_router_instance is not None:
            return _intent_router_instance

        _intent_router_instance = IntentRouter(get_llm_client())
        return _intent_router_instance


def get_rag_service() -> RAGService:
    """Lazy-loading singleton for RAGService."""
    global _rag_service_instance

    if _rag_service_instance is not None:
        return _rag_service_instance

    with _rag_service_lock:
        if _rag_service_instance is not None:
            return _rag_service_instance

        settings = get_settings()
        embedder = get_embedding_model()
        reranker = get_reranker()
        retriever = HybridRetriever(embedder, reranker)
        _rag_service_instance = RAGService(
            embedding_model=embedder,
            reranker=reranker,
            retriever=retriever,
            llm_client=get_llm_client(),
            web_search=get_web_search_service(),
            intent_router=get_intent_router(),
            settings=settings,
        )
        return _rag_service_instance


def get_store() -> SessionStore:
    return get_session_store()
