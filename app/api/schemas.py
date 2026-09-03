"""Request/response schemas for the API."""
from pydantic import BaseModel, Field

from app.core.config import get_settings

_settings = get_settings()


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Client-generated session/conversation ID")
    query: str = Field(
        ...,
        min_length=1,
        max_length=_settings.max_query_length,
        description="The user's question",
    )


class ChatResponse(BaseModel):
    answer: str
    pdf_sources: list[str]
    web_sources: list[dict]


class UploadResponse(BaseModel):
    filename: str
    chunks_added: int
    message: str


class DocumentInfo(BaseModel):
    filename: str
    chunks: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]


class DeleteDocumentResponse(BaseModel):
    filename: str
    deleted: bool


class SessionInfo(BaseModel):
    session_id: str
    name: str
    last_active: float
    message_count: int


class SessionListResponse(BaseModel):
    sessions: list[SessionInfo]


class CreateSessionRequest(BaseModel):
    name: str = Field(default="New Chat", description="Name for the new session")


class CreateSessionResponse(BaseModel):
    session_id: str
    name: str


class RenameSessionRequest(BaseModel):
    new_name: str = Field(..., description="New name for the session")


class RenameSessionResponse(BaseModel):
    session_id: str
    name: str


class DeleteSessionResponse(BaseModel):
    session_id: str
    deleted: bool


class ErrorResponse(BaseModel):
    error: str
    detail: str
