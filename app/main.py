"""FastAPI application entrypoint.

Run locally:   uvicorn app.main:app --reload
Run in Docker: see Dockerfile / CMD
"""
import asyncio
import logging
import os
from contextlib import asynccontextmanager

import psutil
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.dependencies import get_store
from app.api.routes import router
from app.core.auth import verify_api_key
from app.core.config import get_settings
from app.core.exceptions import RAGBaseError
from app.core.rate_limit import limiter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()


async def _ttl_cleanup_loop():
    """Runs forever in the background, periodically sweeping expired sessions.

    This is what actually activates session_store.py's cleanup_expired() —
    that method existed since piece 1 but nothing ever called it, so
    expired sessions just sat in memory forever. This loop is the fix.
    """
    store = get_store()
    while True:
        await asyncio.sleep(settings.session_cleanup_interval_seconds)
        try:
            removed = store.cleanup_expired(settings.session_ttl_seconds)
            if removed:
                logger.info("TTL cleanup: removed %d expired session(s)", removed)
        except Exception as exc:
            logger.error("TTL cleanup loop failed: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Starts the TTL cleanup loop when the app boots, cancels it on shutdown.
    This is FastAPI's modern replacement for @app.on_event("startup")."""
    task = asyncio.create_task(_ttl_cleanup_loop())
    logger.info(
        "TTL cleanup active: sweeping every %ds, sessions expire after %ds",
        settings.session_cleanup_interval_seconds,
        settings.session_ttl_seconds,
    )
    logger.info("Application started with lazy loading - ML models will load on first request")
    yield
    task.cancel()
    logger.info("Application shutting down")


app = FastAPI(title="RAG Chatbot API", version="1.0.0", lifespan=lifespan)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Locked to the actual frontend domain(s) via ALLOWED_ORIGINS in .env —
# comma-separated for multiple (e.g. local dev + your deployed frontend).
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "X-API-Key"],
)


@app.exception_handler(RAGBaseError)
async def rag_error_handler(request: Request, exc: RAGBaseError):
    """Pipeline errors become a clean 400. In production (DEBUG_MODE off),
    the client gets a generic message — the real detail is logged
    server-side only, so internal details (file paths, library internals,
    raw exception text) never leak to whoever's calling your API.
    """
    logger.warning("Pipeline error on %s: %s", request.url.path, exc)
    detail = str(exc) if settings.debug_mode else "There was a problem processing your request."
    return JSONResponse(status_code=400, content={"error": type(exc).__name__, "detail": detail})


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    """Last-resort catch-all so a bug never leaks a stack trace to the client."""
    logger.error("Unhandled error on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500, content={"error": "InternalServerError", "detail": "Something went wrong."}
    )


@app.get("/health")
async def health():
    """Lightweight health check that doesn't load ML models.

    Deliberately NOT behind the API key — hosting platforms and uptime
    monitors need to reach this without a secret. This version is optimized
    for memory-constrained environments like FastAPI Cloud.
    """
    from app.core.embeddings import is_embedding_model_loaded
    from app.core.llm import is_llm_client_loaded
    from app.core.reranker import is_reranker_loaded

    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()

    return {
        "status": "ok",
        "memory_usage_mb": memory_info.rss / 1024 / 1024,
        "models_loaded": {
            "embedding": is_embedding_model_loaded(),
            "reranker": is_reranker_loaded(),
            "llm": is_llm_client_loaded(),
        }
    }


app.include_router(router, prefix="/api/v1", dependencies=[Depends(verify_api_key)])
