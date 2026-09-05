"""Wrapper around the Gemini client.

The google-genai SDK is synchronous (blocking). FastAPI is async — if you
call a blocking function directly inside an `async def` route, it freezes
the ENTIRE server for every user until that call finishes. We avoid this
by running the blocking call in a background thread via
`starlette.concurrency.run_in_threadpool` (used in rag_service.py, not
here — this file just holds the plain sync calls it wraps).
"""
import logging
from functools import lru_cache

from google import genai
from google.genai import types

from app.core.config import get_settings
from app.core.exceptions import RAGBaseError

logger = logging.getLogger(__name__)


class LLMError(RAGBaseError):
    """Raised when a call to the LLM fails."""


def _build_contents(history: list[dict] | None, user_message: str) -> list[types.Content]:
    """Turns saved chat history + the current message into what Gemini's
    multi-turn API actually expects: a list of Content objects, each
    tagged with who said it.

    This is the actual fix for "it forgets everything" — previously we
    only ever sent the current message as a bare string, so from Gemini's
    point of view every single call looked like the start of a brand new
    conversation. `history` here is session.chat_history, which was
    already being saved correctly — it just was never being read back.
    """
    contents = []
    for msg in history or []:
        role = "model" if msg["role"] == "assistant" else "user"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_message)]))
    return contents


class GeminiClient:
    def __init__(self, api_key: str, model_name: str):
        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name

    def generate(
        self, system_prompt: str, user_message: str, temperature: float = 0.3, history: list[dict] | None = None
    ) -> str:
        """Single-turn generation. Raises LLMError on failure — never fails silently.

        Pass `history` (session.chat_history) to make this call aware of
        prior turns in the conversation — omit it for a genuinely
        stateless call (e.g. the intent classifier, which should judge
        each message fresh, not be biased by earlier ones).
        """
        try:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=_build_contents(history, user_message),
                config={"system_instruction": system_prompt, "temperature": temperature},
            )
            return response.text or ""
        except Exception as exc:
            logger.error("Gemini call failed: %s", exc)
            raise LLMError(f"LLM generation failed: {exc}") from exc

    def generate_stream(
        self, system_prompt: str, user_message: str, temperature: float = 0.3, history: list[dict] | None = None
    ):
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
                contents=_build_contents(history, user_message),
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


@lru_cache
def get_llm_client() -> GeminiClient:
    settings = get_settings()
    return GeminiClient(settings.gemini_api_key, settings.llm_model_name)
