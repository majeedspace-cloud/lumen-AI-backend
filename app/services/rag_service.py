"""RAG orchestration — v1.1: LLM-based intent routing (replaces keyword regex).

Flow per message:
  1. Classify intent (casual / needs_pdf / needs_web / needs_both) via one
     cheap LLM call — see intent_router.py for why this beats keyword regex.
  2. Casual -> skip retrieval entirely, just chat.
  3. Otherwise -> pull PDF context and/or web context per the classification,
     build one prompt, ask Gemini, return the answer + which sources were used.

Still NOT the full multi-step agent (no self-critique/retry loop) — that's
the next upgrade once this is solid. This version fixes routing correctness
without the extra cost/latency of a multi-step reasoning loop.
"""

import logging

from app.core.config import Settings
from app.core.document_loader import chunk_text, load_pdf_text
from app.core.embeddings import EmbeddingModel
from app.core.exceptions import RAGBaseError
from app.core.llm import GeminiClient
from app.core.prompts import CASUAL_SYSTEM_PROMPT, SYSTEM_PROMPT
from app.core.reranker import Reranker
from app.core.retrieval import HybridRetriever
from app.core.vector_store import VectorStore
from app.services.agent import MultiStepAgent
from app.services.intent_router import Intent, IntentRouter
from app.services.memory_extractor import MemoryExtractor
from app.services.session_store import SessionData
from app.services.user_memory_store import UserMemoryStore
from app.services.web_search import WebSearchService

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        reranker: Reranker,
        retriever: HybridRetriever,
        llm_client: GeminiClient,
        web_search: WebSearchService,
        intent_router: IntentRouter,
        settings: Settings,
        user_memory_store: UserMemoryStore | None = None,
        memory_extractor: MemoryExtractor | None = None,
    ):
        self._embedder = embedding_model
        self._reranker = reranker
        self._retriever = retriever
        self._llm = llm_client
        self._web_search = web_search
        self._intent_router = intent_router
        self._settings = settings
        self._user_memory_store = user_memory_store
        self._memory_extractor = memory_extractor
        
        # Initialize multi-step agent if enabled
        if settings.enable_multi_step_agent:
            self._agent = MultiStepAgent(
                llm_client=llm_client,
                retriever=retriever,
                web_search=web_search,
                max_steps=settings.agent_max_steps,
                keyword_k=settings.keyword_search_top_k,
                semantic_k=settings.semantic_search_top_k,
                rerank_top_n=settings.rerank_top_n,
                rrf_k=settings.rrf_k_constant,
            )
        else:
            self._agent = None

    # ---------------- Cross-session memory ----------------

    def _get_memory_block(self, device_id: str | None) -> str:
        """A short text block describing what's known about this person,
        prepended to whatever message goes to the LLM. Empty string if
        memory is off, unavailable, or nothing's been learned yet — so
        callers can always safely prepend this with no special-casing.
        """
        if not device_id or self._user_memory_store is None:
            return ""
        memory = self._user_memory_store.get_or_create(device_id)
        if not memory.enabled or not memory.facts:
            return ""
        facts_text = ", ".join(f"{k}: {v}" for k, v in memory.facts.items())
        return f"[What you remember about this user: {facts_text}]\n\n"

    def _update_memory(self, device_id: str | None, query: str) -> None:
        """Runs after answering — extracts any new durable fact from the
        user's message and merges it into their stored profile. Silently
        does nothing if memory is off/unavailable; never blocks a chat
        response over a failed extraction (see MemoryExtractor.extract).
        """
        if not device_id or self._user_memory_store is None or self._memory_extractor is None:
            return
        memory = self._user_memory_store.get_or_create(device_id)
        if not memory.enabled:
            logger.debug("Memory disabled for device %s, skipping extraction", device_id)
            return
        new_facts = self._memory_extractor.extract(query)
        if new_facts:
            memory.facts = self._memory_extractor.merge(memory.facts, new_facts)
            self._user_memory_store.save(memory)
            logger.info("Learned new fact(s) for device %s: %s", device_id, list(new_facts.keys()))

    # ---------------- Document ingestion ----------------

    def ingest_pdf(self, session: SessionData, file_path: str, filename: str, file_size: int) -> int:
        """Process an uploaded PDF into the session's vector store.

        Returns the number of chunks added. Returns 0 (no-op, not an error)
        if this exact file was already processed for this session.
        """
        if (filename, file_size) in session.processed_files:
            logger.info("File '%s' already processed for session %s, skipping", filename, session.session_id)
            return 0

        text = load_pdf_text(file_path)
        chunks = chunk_text(text, self._settings.chunk_size, self._settings.chunk_overlap)
        
        # Check if document would generate too many chunks for API limits
        if len(chunks) > self._settings.max_chunks_per_document:
            logger.warning(
                "Document '%s' would generate %d chunks (max: %d). Truncating to fit API limits.",
                filename, len(chunks), self._settings.max_chunks_per_document
            )
            chunks = chunks[:self._settings.max_chunks_per_document]
        
        embeddings = self._embedder.encode_many(chunks, task_type="RETRIEVAL_DOCUMENT")

        if session.vector_store is None:
            session.vector_store = VectorStore(dimension=self._embedder.dimension)

        base_index = len(session.vector_store.chunks)
        metadata = [
            {"chunk_id": f"{filename}_chunk_{base_index + i}", "source": filename}
            for i in range(len(chunks))
        ]
        session.vector_store.add(chunks, embeddings, metadata)
        session.processed_files.add((filename, file_size))

        logger.info("Ingested '%s': %d chunks added", filename, len(chunks))
        return len(chunks)

    # ---------------- Document management ----------------

    def list_documents(self, session: SessionData) -> list[dict]:
        """Return per-file chunk counts for everything indexed in this session."""
        if session.vector_store is None:
            return []
        counts: dict[str, int] = {}
        for meta in session.vector_store.metadata:
            counts[meta["source"]] = counts.get(meta["source"], 0) + 1
        return [{"filename": name, "chunks": count} for name, count in counts.items()]

    def delete_document(self, session: SessionData, filename: str) -> bool:
        """Remove a document's chunks from the session's index.

        FAISS's flat index doesn't support deleting individual vectors, so
        we rebuild the index from the chunks we're keeping (their raw text
        is already stored, just re-embed it — cheap since deletes are rare).

        Returns False if the filename wasn't found in this session (a normal
        "nothing to do" outcome, not an error).
        """
        if session.vector_store is None:
            return False

        keep_chunks, keep_metadata, found = [], [], False
        for text, meta in zip(session.vector_store.chunks, session.vector_store.metadata):
            if meta["source"] == filename:
                found = True
                continue
            keep_chunks.append(text)
            keep_metadata.append(meta)

        if not found:
            return False

        new_store = VectorStore(dimension=self._embedder.dimension)
        if keep_chunks:
            embeddings = self._embedder.encode_many(keep_chunks, task_type="RETRIEVAL_DOCUMENT")
            new_store.add(keep_chunks, embeddings, keep_metadata)

        session.vector_store = new_store
        session.processed_files = {f for f in session.processed_files if f[0] != filename}
        logger.info("Deleted '%s' from session %s", filename, session.session_id)
        return True

    # ---------------- Chat ----------------

    def chat(self, session: SessionData, query: str, device_id: str | None = None) -> dict:
        """Answer a query, routing via intent classification instead of keyword rules.

        Returns {"answer": str, "sources": {"pdf": [...], "web": [...]}}.
        """
        has_pdf = session.vector_store is not None and not session.vector_store.is_empty
        intent = self._intent_router.classify(query, has_pdf)
        logger.info("Query classified as: %s", intent.value)
        memory_block = self._get_memory_block(device_id)

        if intent == Intent.CASUAL:
            answer = self._llm.generate(
                CASUAL_SYSTEM_PROMPT, memory_block + query, history=session.chat_history
            )
            self._append_history(session, query, answer)
            self._update_memory(device_id, query)
            return {"answer": answer, "sources": {"pdf": [], "web": []}}

        # Use multi-step agent if enabled, otherwise fall back to single-pass.
        # Passing `intent` here is the fix for the redundant-LLM-calls issue:
        # the agent used to re-decide this from scratch (2-3 extra calls) —
        # now it plans directly from what we already determined above.
        if self._agent:
            logger.info("Using multi-step agent for query")
            result = self._agent.run(query, session.vector_store, session.chat_history, intent, memory_block)
            self._append_history(session, query, result["answer"])
            self._update_memory(device_id, query)
            return {
                "answer": result["answer"],
                "sources": result["sources"],
            }
        else:
            # Original single-pass logic
            context_parts = []
            pdf_sources: list[str] = []
            web_sources: list[dict] = []

            want_pdf = intent in (Intent.NEEDS_PDF, Intent.NEEDS_BOTH) and has_pdf
            want_web = intent in (Intent.NEEDS_WEB, Intent.NEEDS_BOTH)

            if want_pdf:
                chunks = self._retriever.retrieve(
                    query,
                    session.vector_store,
                    keyword_k=self._settings.keyword_search_top_k,
                    semantic_k=self._settings.semantic_search_top_k,
                    rerank_top_n=self._settings.rerank_top_n,
                    rrf_k=self._settings.rrf_k_constant,
                )
                if chunks:
                    context_parts.append(
                        "=== PDF CONTEXT ===\n"
                        + "\n\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(chunks))
                    )
                    pdf_sources = sorted({c["source"] for c in chunks})

            if want_web:
                try:
                    web_result = self._web_search.search(query)
                    if web_result["results"]:
                        web_context = "\n\n".join(
                            f"- {r['title']}: {r['content'][:300]}" for r in web_result["results"]
                        )
                        context_parts.append(f"=== WEB CONTEXT ===\n{web_context}")
                        web_sources = web_result["results"]
                except RAGBaseError as exc:
                    logger.warning("Web search failed, continuing without it: %s", exc)

            context = "\n\n".join(context_parts) if context_parts else "(no context found)"
            user_message = f"{memory_block}Context:\n{context}\n\nQuestion: {query}"
            answer = self._llm.generate(SYSTEM_PROMPT, user_message, history=session.chat_history)

            self._append_history(session, query, answer)
            self._update_memory(device_id, query)
            return {"answer": answer, "sources": {"pdf": pdf_sources, "web": web_sources}}

    # ---------------- Streaming chat ----------------

    def chat_stream(self, session: SessionData, query: str, device_id: str | None = None):
        """Generator version of chat() for streaming (SSE) responses.

        Yields small dicts describing what's happening, in order:
          {"type": "status", "text": "..."}           -- a step starting (drives a live "thinking" UI)
          {"type": "sources", "pdf": [...], "web": [...]}  -- once retrieval is done, before the answer starts
          {"type": "token", "text": "..."}             -- one chunk of the answer as it's generated
          {"type": "done"}                              -- stream finished, nothing more coming

        routes.py turns each of these into an SSE event for the frontend.
        The routing logic (classify -> retrieve -> generate) is identical
        to chat() above — only the LLM call at the end streams instead of
        blocking, and progress status events are emitted along the way.
        """
        has_pdf = session.vector_store is not None and not session.vector_store.is_empty

        intent = self._intent_router.classify(query, has_pdf)
        logger.info("Query classified as: %s", intent.value)
        memory_block = self._get_memory_block(device_id)

        if intent == Intent.CASUAL:
            full_answer = ""
            for chunk in self._llm.generate_stream(
                CASUAL_SYSTEM_PROMPT, memory_block + query, history=session.chat_history
            ):
                full_answer += chunk
                yield {"type": "token", "text": chunk}
            self._append_history(session, query, full_answer)
            self._update_memory(device_id, query)
            yield {"type": "done"}
            return

        # Use multi-step agent if enabled, otherwise fall back to single-pass.
        # Same fix as in chat() above — pass the already-computed intent
        # instead of letting the agent re-derive it.
        if self._agent:
            logger.info("Using multi-step agent for streaming query")
            full_answer = ""
            for event in self._agent.run_stream(
                query, session.vector_store, session.chat_history, intent, memory_block
            ):
                yield event
                if event["type"] == "token":
                    full_answer += event["text"]
            self._append_history(session, query, full_answer)
            self._update_memory(device_id, query)
            return
        else:
            # Original single-pass logic
            context_parts = []
            pdf_sources: list[str] = []
            web_sources: list[dict] = []

            want_pdf = intent in (Intent.NEEDS_PDF, Intent.NEEDS_BOTH) and has_pdf
            want_web = intent in (Intent.NEEDS_WEB, Intent.NEEDS_BOTH)

            if want_pdf:
                yield {"type": "status", "text": "Searching your document..."}
                chunks = self._retriever.retrieve(
                    query,
                    session.vector_store,
                    keyword_k=self._settings.keyword_search_top_k,
                    semantic_k=self._settings.semantic_search_top_k,
                    rerank_top_n=self._settings.rerank_top_n,
                    rrf_k=self._settings.rrf_k_constant,
                )
                if chunks:
                    context_parts.append(
                        "=== PDF CONTEXT ===\n"
                        + "\n\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(chunks))
                    )
                    pdf_sources = sorted({c["source"] for c in chunks})

            if want_web:
                yield {"type": "status", "text": "Searching the web..."}
                try:
                    web_result = self._web_search.search(query)
                    if web_result["results"]:
                        web_context = "\n\n".join(
                            f"- {r['title']}: {r['content'][:300]}" for r in web_result["results"]
                        )
                        context_parts.append(f"=== WEB CONTEXT ===\n{web_context}")
                        web_sources = web_result["results"]
                except RAGBaseError as exc:
                    logger.warning("Web search failed, continuing without it: %s", exc)

            yield {"type": "sources", "pdf": pdf_sources, "web": web_sources}
            yield {"type": "status", "text": "Writing your answer..."}

            context = "\n\n".join(context_parts) if context_parts else "(no context found)"
            user_message = f"{memory_block}Context:\n{context}\n\nQuestion: {query}"

            full_answer = ""
            for chunk in self._llm.generate_stream(SYSTEM_PROMPT, user_message, history=session.chat_history):
                full_answer += chunk
                yield {"type": "token", "text": chunk}

            self._append_history(session, query, full_answer)
            self._update_memory(device_id, query)
            yield {"type": "done"}

    @staticmethod
    def _append_history(session: SessionData, query: str, answer: str) -> None:
        session.chat_history.append({"role": "user", "content": query})
        session.chat_history.append({"role": "assistant", "content": answer})
        session.chat_history = session.chat_history[-12:]  # keep last 6 turns
