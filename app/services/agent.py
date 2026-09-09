"""Multi-step agent reasoning — v2, redesigned after finding real issues in v1.

What changed and why:

1. v1 re-decided (via 2-3 extra LLM calls) something the intent classifier
   already correctly determined one call earlier, then threw that answer
   away. This version plans directly from the already-computed intent —
   zero extra LLM calls for the initial plan.

2. v1's docstring promised "search again with different strategy if the
   first search doesn't yield enough" but never actually did this — verified
   by testing: an empty document search just went straight to answering
   with nothing. This version actually does it: if the planned search(es)
   come back completely empty and there's an untried source available, it
   automatically tries that source. This is a cheap deterministic check
   (empty results -> try the other source), not another LLM call — the
   fix costs nothing extra, it was just never wired up.

3. v1 had run() and run_stream() as two separate copies of the same
   reasoning loop. This version has ONE implementation (run_stream, a
   generator) — run() just drains it and collects the result.
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

from app.core.llm import GeminiClient
from app.core.prompts import SYSTEM_PROMPT
from app.core.retrieval import HybridRetriever
from app.core.vector_store import VectorStore
from app.services.intent_router import Intent
from app.services.web_search import WebSearchService

logger = logging.getLogger(__name__)


@dataclass
class _AgentResult:
    pdf_chunks: list = field(default_factory=list)
    web_results: list = field(default_factory=list)
    searched_pdf: bool = False
    searched_web: bool = False


class DocumentSearchTool:
    def __init__(self, retriever: HybridRetriever, vector_store: Optional[VectorStore], **retrieval_kwargs):
        self._retriever = retriever
        self._vector_store = vector_store
        self._kwargs = retrieval_kwargs

    def search(self, query: str) -> list[dict]:
        if self._vector_store is None or self._vector_store.is_empty:
            return []
        return self._retriever.retrieve(query, self._vector_store, **self._kwargs)


class WebSearchTool:
    def __init__(self, web_search: WebSearchService):
        self._web_search = web_search

    def search(self, query: str) -> list[dict]:
        try:
            result = self._web_search.search(query)
            return result.get("results", [])
        except Exception as exc:
            logger.warning("Web search failed, treating as empty: %s", exc)
            return []


class MultiStepAgent:
    """Plans its search from the intent classifier's result, executes it,
    and falls back to the other source if the plan comes back empty and a
    fallback exists. That fallback behavior is the genuinely adaptive part —
    everything else is a straightforward, cheap, deterministic pipeline.
    """

    def __init__(
        self,
        llm_client: GeminiClient,
        retriever: HybridRetriever,
        web_search: WebSearchService,
        max_steps: int = 5,
        keyword_k: int = 15,
        semantic_k: int = 20,
        rerank_top_n: int = 5,
        rrf_k: int = 60,
    ):
        self._llm = llm_client
        self._retriever = retriever
        self._web_search = web_search
        self._retrieval_kwargs = dict(
            keyword_k=keyword_k, semantic_k=semantic_k, rerank_top_n=rerank_top_n, rrf_k=rrf_k
        )

    def _plan(self, intent: Intent, has_pdf: bool) -> list[str]:
        """The initial search plan, derived directly from intent — no LLM
        call needed, since the intent classifier already answered this."""
        if intent == Intent.NEEDS_BOTH:
            return ["pdf", "web"] if has_pdf else ["web"]
        if intent == Intent.NEEDS_PDF:
            return ["pdf"] if has_pdf else ["web"]  # no doc uploaded - only real option is web
        return ["web"]  # NEEDS_WEB

    def run_stream(
        self,
        query: str,
        vector_store: Optional[VectorStore],
        chat_history: list[dict],
        intent: Intent,
    ):
        """The one real implementation. Yields status/sources/token/done
        events — routes.py turns each into an SSE event for the frontend.
        run() below just drains this generator instead of duplicating it.
        """
        has_pdf = vector_store is not None and not vector_store.is_empty
        doc_tool = DocumentSearchTool(self._retriever, vector_store, **self._retrieval_kwargs)
        web_tool = WebSearchTool(self._web_search)

        plan = self._plan(intent, has_pdf)
        result = _AgentResult()

        for source in plan:
            if source == "pdf":
                yield {"type": "status", "text": "Searching your document..."}
                result.pdf_chunks = doc_tool.search(query)
                result.searched_pdf = True
            else:
                yield {"type": "status", "text": "Searching the web..."}
                result.web_results = web_tool.search(query)
                result.searched_web = True

        # The actual fix for "never adapts when a search comes up empty":
        # if nothing useful came back and there's a source we haven't
        # tried yet, try it now before giving up.
        if not result.pdf_chunks and not result.web_results:
            if not result.searched_web:
                yield {"type": "status", "text": "Nothing useful found yet — checking the web..."}
                result.web_results = web_tool.search(query)
                result.searched_web = True
            elif not result.searched_pdf and has_pdf:
                yield {"type": "status", "text": "Nothing useful found yet — checking your document..."}
                result.pdf_chunks = doc_tool.search(query)
                result.searched_pdf = True

        pdf_sources = sorted({c["source"] for c in result.pdf_chunks})
        yield {"type": "sources", "pdf": pdf_sources, "web": result.web_results}
        yield {"type": "status", "text": "Writing your answer..."}

        context_parts = []
        if result.pdf_chunks:
            context_parts.append(
                "=== PDF CONTEXT ===\n"
                + "\n\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(result.pdf_chunks))
            )
        if result.web_results:
            context_parts.append(
                "=== WEB CONTEXT ===\n"
                + "\n\n".join(f"- {r['title']}: {r['content'][:300]}" for r in result.web_results)
            )
        context = "\n\n".join(context_parts) if context_parts else "(no context found)"
        user_message = f"Context:\n{context}\n\nQuestion: {query}"

        for chunk in self._llm.generate_stream(SYSTEM_PROMPT, user_message, history=chat_history):
            yield {"type": "token", "text": chunk}

        yield {"type": "done"}

    def run(
        self,
        query: str,
        vector_store: Optional[VectorStore],
        chat_history: list[dict],
        intent: Intent,
    ) -> dict:
        """Non-streaming version. Drains run_stream() instead of
        duplicating its logic — one implementation, not two."""
        full_answer = ""
        pdf_sources: list[str] = []
        web_results: list[dict] = []

        for event in self.run_stream(query, vector_store, chat_history, intent):
            if event["type"] == "token":
                full_answer += event["text"]
            elif event["type"] == "sources":
                pdf_sources = event["pdf"]
                web_results = event["web"]

        return {"answer": full_answer, "sources": {"pdf": pdf_sources, "web": web_results}}
