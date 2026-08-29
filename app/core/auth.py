"""Authentication.

Today: a single shared secret key (fastest thing that actually blocks
randoms from hitting your API and burning your Gemini/Tavily quota).

Later, when you have real users: replace the body of verify_api_key()
with real token/session validation. Every route already depends on this
one function (see routes.py) — so upgrading auth later means editing
ONE function, not every endpoint.
"""

from fastapi import Header, HTTPException

from app.core.config import get_settings


async def verify_api_key(x_api_key: str = Header(default="")) -> None:
    """FastAPI dependency — raises 401 if the request's X-API-Key header
    doesn't match the configured secret.

    In DEBUG_MODE (local dev), a missing api_key setting just disables the
    check — no extra setup needed to run locally. Outside debug mode, a
    missing api_key setting is a MISCONFIGURATION, not "open by choice" —
    it fails loudly (500) instead of silently running your production API
    unprotected, which is the actually dangerous version of this bug.
    """
    settings = get_settings()

    if not settings.api_key:
        if settings.debug_mode:
            return  # local dev convenience only
        raise HTTPException(
            status_code=500,
            detail="Server misconfigured: API_KEY must be set when DEBUG_MODE is off.",
        )

    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
