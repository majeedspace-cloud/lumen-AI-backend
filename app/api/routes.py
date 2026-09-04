"""Chat, upload, and document-management endpoints.

Blocking work (embeddings, FAISS, the LLM call — none of it is async-native)
is run via `run_in_threadpool` so it doesn't block FastAPI's event loop
and freeze other users' requests while one request is processing.
"""
import json
import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from app.api.dependencies import get_rag_service, get_store
from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    CreateSessionRequest,
    CreateSessionResponse,
    DeleteDocumentResponse,
    DeleteSessionResponse,
    DocumentInfo,
    DocumentListResponse,
    RenameSessionRequest,
    RenameSessionResponse,
    SessionDetailResponse,
    SessionInfo,
    SessionListResponse,
    UploadResponse,
)
from app.core.config import get_settings
from app.core.rate_limit import limiter
from app.services.rag_service import RAGService
from app.services.session_store import SessionStore

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()


@router.post("/chat", response_model=ChatResponse)
@limiter.limit(settings.rate_limit_chat)
async def chat(
    request: Request,
    body: ChatRequest,
    rag: RAGService = Depends(get_rag_service),
    store: SessionStore = Depends(get_store),
):
    session = store.get_or_create(body.session_id)
    
    # Auto-name session if it's still "New Chat" and this is the first message
    if session.name == "New Chat" and len(session.chat_history) == 0:
        # Generate a name from the first few words of the query
        words = body.query.split()[:4]
        auto_name = " ".join(words).capitalize()
        if len(auto_name) > 30:
            auto_name = auto_name[:27] + "..."
        session.name = auto_name
    
    result = await run_in_threadpool(rag.chat, session, body.query)
    store.save(session)
    return ChatResponse(
        answer=result["answer"],
        pdf_sources=result["sources"]["pdf"],
        web_sources=result["sources"]["web"],
    )


@router.post("/upload", response_model=UploadResponse)
@limiter.limit(settings.rate_limit_upload)
async def upload_pdf(
    request: Request,
    session_id: str = Form(...),
    file: UploadFile = File(...),
    rag: RAGService = Depends(get_rag_service),
    store: SessionStore = Depends(get_store),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported right now.")

    contents = await file.read()

    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large — max {settings.max_upload_mb}MB, got {len(contents) / 1024 / 1024:.1f}MB.",
        )

    session = store.get_or_create(session_id)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        chunks_added = await run_in_threadpool(
            rag.ingest_pdf, session, tmp_path, file.filename, len(contents)
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    store.save(session)

    message = (
        "Already processed this file earlier in this session."
        if chunks_added == 0
        else f"Indexed {chunks_added} chunks."
    )
    return UploadResponse(filename=file.filename, chunks_added=chunks_added, message=message)


def _format_sse(event_type: str, data: dict) -> str:
    """Format one Server-Sent Event. The 'event:' line names it (so the
    frontend can listen for specific types), 'data:' carries the JSON
    payload. The blank line at the end is required by the SSE spec — it's
    how the client knows one event ended.
    """
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


@router.post("/chat/stream")
@limiter.limit(settings.rate_limit_chat)
async def chat_stream(
    request: Request,
    body: ChatRequest,
    rag: RAGService = Depends(get_rag_service),
    store: SessionStore = Depends(get_store),
):
    """Same as /chat, but streams the answer as it's generated instead of
    waiting for the full response. Uses Server-Sent Events (SSE).

    Note: `rag.chat_stream` is a normal (sync) Python generator. Starlette's
    StreamingResponse automatically runs sync generators in a background
    thread (`iterate_in_threadpool`) so this doesn't block the event loop —
    same protection run_in_threadpool gives the non-streaming endpoints,
    just handled for us automatically here.
    """
    session = store.get_or_create(body.session_id)

    def event_stream():
        for event in rag.chat_stream(session, body.query):
            event_type = event["type"]
            yield _format_sse(event_type, event)
        store.save(session)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # tells nginx (if you deploy behind it) not to buffer this
        },
    )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    session_id: str,
    rag: RAGService = Depends(get_rag_service),
    store: SessionStore = Depends(get_store),
):
    """Powers the sidebar's document list — filenames + chunk counts for this session."""
    session = store.get_or_create(session_id)
    docs = await run_in_threadpool(rag.list_documents, session)
    return DocumentListResponse(documents=[DocumentInfo(**d) for d in docs])


@router.delete("/documents/{filename}", response_model=DeleteDocumentResponse)
async def delete_document(
    filename: str,
    session_id: str,
    rag: RAGService = Depends(get_rag_service),
    store: SessionStore = Depends(get_store),
):
    """Powers the sidebar's delete button — removes a doc's chunks from this session's index."""
    session = store.get_or_create(session_id)
    deleted = await run_in_threadpool(rag.delete_document, session, filename)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"'{filename}' not found in this session.")
    store.save(session)
    return DeleteDocumentResponse(filename=filename, deleted=True)


# ---------------- Session Management Endpoints ----------------

@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(store: SessionStore = Depends(get_store)):
    """Return list of all sessions with their metadata for the sidebar."""
    sessions = await run_in_threadpool(store.list_sessions)
    return SessionListResponse(sessions=sessions)


@router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
async def get_session_detail(
    session_id: str,
    rag: RAGService = Depends(get_rag_service),
    store: SessionStore = Depends(get_store),
):
    """Get detailed session information including chat history and documents."""
    session = await run_in_threadpool(store.get, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    docs = await run_in_threadpool(rag.list_documents, session)
    return SessionDetailResponse(
        session_id=session.session_id,
        name=session.name,
        chat_history=session.chat_history,
        documents=[DocumentInfo(**d) for d in docs]
    )


@router.post("/sessions", response_model=CreateSessionResponse)
async def create_session(
    body: CreateSessionRequest,
    store: SessionStore = Depends(get_store),
):
    """Create a new session with an optional name."""
    import uuid
    session_id = str(uuid.uuid4())
    session = store.get_or_create(session_id)
    session.name = body.name
    store.save(session)
    logger.info("Created new session: %s with name '%s'", session_id, body.name)
    return CreateSessionResponse(session_id=session_id, name=body.name)


@router.put("/sessions/{session_id}/rename", response_model=RenameSessionResponse)
async def rename_session(
    session_id: str,
    body: RenameSessionRequest,
    store: SessionStore = Depends(get_store),
):
    """Rename an existing session."""
    await run_in_threadpool(store.rename_session, session_id, body.new_name)
    session = store.get_or_create(session_id)
    return RenameSessionResponse(session_id=session_id, name=body.new_name)


@router.delete("/sessions/{session_id}", response_model=DeleteSessionResponse)
async def delete_session(
    session_id: str,
    store: SessionStore = Depends(get_store),
):
    """Delete a session and all its data."""
    await run_in_threadpool(store.delete, session_id)
    logger.info("Deleted session: %s", session_id)
    return DeleteSessionResponse(session_id=session_id, deleted=True)
