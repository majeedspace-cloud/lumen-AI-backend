"""Custom exception types for the RAG pipeline.

Using specific exceptions (instead of bare `except:` like the original
notebook did) means the FastAPI layer can return the right HTTP status
code and error message for each failure mode, and nothing fails silently.
"""


class RAGBaseError(Exception):
    """Base class for all RAG-pipeline errors."""


class DocumentProcessingError(RAGBaseError):
    """Raised when a PDF can't be loaded, is empty, or fails to chunk."""


class EmbeddingError(RAGBaseError):
    """Raised when the embedding model fails to encode text."""


class VectorStoreError(RAGBaseError):
    """Raised when FAISS index build/search/save/load fails."""


class RetrievalError(RAGBaseError):
    """Raised when hybrid retrieval (keyword + semantic + fusion) fails."""


class RerankError(RAGBaseError):
    """Raised when the cross-encoder reranker fails."""
