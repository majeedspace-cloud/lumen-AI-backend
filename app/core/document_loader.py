"""PDF loading and text chunking utilities."""
import logging
from pathlib import Path

import pymupdf  # PyMuPDF

from app.core.exceptions import DocumentProcessingError

logger = logging.getLogger(__name__)


def load_pdf_text(file_path: str | Path) -> str:
    """Extract all text from a PDF file.

    Args:
        file_path: Path to the PDF on disk.

    Returns:
        The full extracted text, pages joined in order.

    Raises:
        DocumentProcessingError: If the file doesn't exist, isn't a valid
            PDF, or contains no extractable text (e.g. a scanned image PDF
            with no OCR layer).
    """
    path = Path(file_path)
    if not path.exists():
        raise DocumentProcessingError(f"File not found: {path}")

    try:
        doc = pymupdf.open(path)
    except Exception as exc:
        raise DocumentProcessingError(f"Could not open '{path.name}' as a PDF: {exc}") from exc

    try:
        text = "".join(page.get_text() for page in doc)
    finally:
        doc.close()

    if not text.strip():
        raise DocumentProcessingError(
            f"No extractable text found in '{path.name}'. "
            "It may be a scanned/image-only PDF that needs OCR."
        )
    return text


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 200) -> list[str]:
    """Split text into overlapping word-count chunks.

    Args:
        text: Raw text to split.
        chunk_size: Words per chunk.
        overlap: Words shared between consecutive chunks (helps retrieval
            not miss context that straddles a chunk boundary).

    Returns:
        List of text chunks. Empty list if `text` is blank.

    Raises:
        DocumentProcessingError: If overlap >= chunk_size, which would
            cause an infinite loop (the original bug risk in the notebook
            version — never validated this).
    """
    if overlap >= chunk_size:
        raise DocumentProcessingError(
            f"chunk_overlap ({overlap}) must be smaller than chunk_size ({chunk_size})"
        )

    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += step

    logger.info("Chunked text into %d chunks (size=%d, overlap=%d)", len(chunks), chunk_size, overlap)
    return chunks
