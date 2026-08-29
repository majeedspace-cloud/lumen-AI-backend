"""Wrapper around the Gemini client.

The google-genai SDK is synchronous (blocking). FastAPI is async — if you
call a blocking function directly inside an `async def` route, it freezes
the ENTIRE server for every user until that call finishes. We avoid this
by running the blocking call in a background thread via
`starlette.concurrency.run_in_threadpool` (used in rag_service.py, not
here — this file just holds the plain sync calls it wraps).

Now implements lazy loading to reduce startup memory footprint.
"""
import logging
from threading import Lock

from google import genai

from app.core.config import get_settings
from app.core.exceptions import RAGBaseError

logger = logging.getLogger(__name__)

# Global singleton instance with thread-safe lazy loading
_llm_client_instance = None
_llm_client_lock = Lock()


class LLMError(RAGBaseError):
    """Raised when a call to the LLM fails."""


class GeminiClient:
    def __init__(self, api_key: str, model_name: str):
        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name

    def generate(self, system_prompt: str, user_message: str, temperature: float = 0.3) -> str:
        """Single-turn generation. Raises LLMError on failure — never fails silently."""
        try:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=user_message,
                config={"system_instruction": system_prompt, "temperature": temperature},
            )
            return response.text or ""
        except Exception as exc:
            logger.error("Gemini call failed: %s", exc)
            raise LLMError(f"LLM generation failed: {exc}") from exc

    def generate_stream(self, system_prompt: str, user_message: str, temperature: float = 0.3):
        """Streaming generation — yields text chunks as they arrive instead of
        waiting for the full response. This is a regular (sync) Python
        generator; FastAPI/Starlette knows how to consume sync generators
        in a background thread automatically (see routes.py), so this
        doesn't need to be async itself.

        Raises LLMError immediately (before yielding anything) if the call
        can't even start. Once streaming has begun, a mid-stream failure is
        logged and the generator just stops — the caller gets whatever
        text arrived before the failure, rather than losing it all.
        """
        try:
            stream = self._client.models.generate_content_stream(
                model=self._model_name,
                contents=user_message,
                config={"system_instruction": system_prompt, "temperature": temperature},
            )
        except Exception as exc:
            logger.error("Gemini stream failed to start: %s", exc)
            raise LLMError(f"LLM streaming failed to start: {exc}") from exc

        try:
            for chunk in stream:
                if chunk.text:
                    yield chunk.text
        except Exception as exc:
            logger.error("Gemini stream interrupted mid-response: %s", exc)


def get_llm_client() -> GeminiClient:
    """Lazy-loading singleton accessor — client only loads on first call.

    This significantly reduces startup memory footprint for FastAPI Cloud,
    as the client isn't initialized until the first actual request needs it.
    Thread-safe to prevent multiple simultaneous initializations.
    """
    global _llm_client_instance

    if _llm_client_instance is not None:
        return _llm_client_instance

    with _llm_client_lock:
        # Double-check pattern to avoid initialization if another thread just finished
        if _llm_client_instance is not None:
            return _llm_client_instance

        settings = get_settings()
        _llm_client_instance = GeminiClient(settings.gemini_api_key, settings.llm_model_name)
        return _llm_client_instance


def is_llm_client_loaded() -> bool:
    """Check if the LLM client has been loaded (useful for health checks)."""
    return _llm_client_instance is not None
