"""LangGraph tools for the RAG agent."""

import logging
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

from backend.core.config import Settings
from backend.core.dependency import (
    get_docstore,
    get_model,
    get_settings,
    get_vector_store,
)
from backend.servies.chat_service import ChatService
from backend.servies.file_service import PDFFileService

logger = logging.getLogger(__name__)

# Global service instances (initialized lazily)
_chat_service: Optional[ChatService] = None


def _get_chat_service() -> ChatService:
    """Get or create a ChatService instance."""
    global _chat_service
    if _chat_service is None:
        cfg = get_settings()
        vector_store = get_vector_store(cfg)
        docstore = get_docstore(cfg)
        model_service = get_model(cfg)
        file_service = PDFFileService()
        _chat_service = ChatService(
            cfg=cfg,
            vector_store=vector_store,
            file_service=file_service,
            model_service=model_service,
            docstore=docstore,
        )
    return _chat_service


@tool
def ingest_document(file_path: str) -> str:
    """Ingest a PDF document into the knowledge base.

    This tool processes a PDF file, extracts text, tables, and images,
    creates embeddings, and stores them in the vector database.

    Args:
        file_path: The absolute path to the PDF file to ingest.

    Returns:
        A message indicating success with the number of pages and chunks processed.
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"Error: File not found at {file_path}"
        if not path.suffix.lower() == ".pdf":
            return f"Error: File must be a PDF. Got {path.suffix}"

        svc = _get_chat_service()
        result = svc.ingest(str(path))

        return (
            f"Successfully ingested '{path.name}': "
            f"{result.processed_pages} pages processed, "
            f"{result.chunks_indexed} chunks indexed."
        )
    except Exception as e:
        logger.exception("Error ingesting document: %s", e)
        return f"Error ingesting document: {str(e)}"


@tool
def search_knowledge_base(query: str, k: int = 4) -> str:
    """Search the knowledge base for relevant information.

    This tool searches the vector database for documents relevant to the query
    and returns the matching context.

    Args:
        query: The search query to find relevant documents.
        k: Number of results to return (default: 4).

    Returns:
        The relevant context from the knowledge base, or a message if nothing found.
    """
    try:
        svc = _get_chat_service()
        context_chunks = svc.show_context(query, k=k)

        if not context_chunks:
            return "No relevant documents found in the knowledge base."

        results = []
        for i, chunk in enumerate(context_chunks, 1):
            source_info = ""
            if chunk.source:
                source_info += f" (Source: {chunk.source}"
                if chunk.page_number:
                    source_info += f", Page {chunk.page_number}"
                source_info += ")"
            results.append(f"[{i}]{source_info}\n{chunk.text}")

        return "\n\n---\n\n".join(results)
    except Exception as e:
        logger.exception("Error searching knowledge base: %s", e)
        return f"Error searching knowledge base: {str(e)}"


# Export tools for use in the graph
rag_tools = [ingest_document, search_knowledge_base]
