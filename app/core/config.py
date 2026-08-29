"""Centralized configuration.

Everything that used to be hardcoded (API keys, model names, chunk sizes)
lives here and is pulled from environment variables. Never commit a .env
file — copy .env.example, fill it in locally, and keep the real one out
of git via .gitignore.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- API keys (required, no defaults on purpose so startup fails loudly
    # if you forget to set one instead of silently using a leaked key) ---
    gemini_api_key: str
    hf_token: str
    tavily_api_key: str

    # --- Models ---
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_model_name: str = "gemini-3.1-flash-lite"

    # --- Chunking ---
    chunk_size: int = 500
    chunk_overlap: int = 200

    # --- Retrieval ---
    rrf_k_constant: int = 60
    keyword_search_top_k: int = 15
    semantic_search_top_k: int = 20
    rerank_top_n: int = 5
    low_confidence_threshold: float = -2.0

    # --- Session storage backend: "memory" or "redis" ---
    session_backend: str = "memory"
    redis_url: str = "redis://localhost:6379/0"
    session_ttl_seconds: int = 60 * 60 * 6  # 6 hours
    session_cleanup_interval_seconds: int = 60 * 30  # how often the TTL sweep runs

    # --- Deployment optimization ---
    # Set to True to skip loading ML models during startup (useful for
    # health checks in memory-constrained environments like FastAPI Cloud)
    skip_model_loading: bool = False

    # --- Security ---
    # Comma-separated in .env, e.g. "http://localhost:5173,https://yourapp.com"
    allowed_origins: str = "http://localhost:5173"
    # Shared secret gate for now (MVP-appropriate). Swap for real per-user auth
    # later without touching callers — they all go through verify_api_key().
    api_key: str = ""
    max_upload_mb: int = 15
    max_query_length: int = 2000
    rate_limit_chat: str = "20/minute"
    rate_limit_upload: str = "10/minute"
    # False in production: hides internal exception text from API responses,
    # logs the real detail server-side instead. Flip to True only for local debugging.
    debug_mode: bool = False

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @property
    def allowed_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance — env is read once, not on every call."""
    return Settings()
