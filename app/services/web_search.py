"""Web search via Tavily, for questions that need live/current info."""
import logging
from functools import lru_cache

from tavily import TavilyClient

from app.core.config import get_settings
from app.core.exceptions import RAGBaseError

logger = logging.getLogger(__name__)


class WebSearchError(RAGBaseError):
    """Raised when the web search provider fails."""


class WebSearchService:
    def __init__(self, api_key: str):
        self._client = TavilyClient(api_key=api_key)

    def search(self, query: str, max_results: int = 5) -> dict:
        """Returns {"answer": str, "results": [{"title", "url", "content"}, ...]}."""
        try:
            result = self._client.search(
                query=query, max_results=max_results, search_depth="advanced", include_answer=True
            )
        except Exception as exc:
            logger.error("Tavily search failed: %s", exc)
            raise WebSearchError(f"Web search failed: {exc}") from exc

        return {
            "answer": result.get("answer", ""),
            "results": [
                {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
                for r in result.get("results", [])
            ],
        }


@lru_cache
def get_web_search_service() -> WebSearchService:
    settings = get_settings()
    return WebSearchService(settings.tavily_api_key)
