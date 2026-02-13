"""
=============================================================================
HUMMINGBIRD-LEA - Knowledge Base API
Powered by CoDre-X | B & D Servicing LLC
=============================================================================
API endpoints for managing the knowledge base (RAG system).

Endpoints:
- POST /api/knowledge/upload - Upload and index a document
- GET /api/knowledge/documents - List indexed documents
- DELETE /api/knowledge/documents/{source} - Delete a document
- POST /api/knowledge/search - Search the knowledge base
- GET /api/knowledge/stats - Get knowledge base statistics
=============================================================================
"""

import os
import shutil
from typing import Optional, List
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from pydantic import BaseModel, Field

from core.utils.config import get_settings
from core.utils.auth import User, get_current_user, get_optional_user
from core.services.rag import (
    get_rag_engine,
    IndexingResult,
    SearchResults,
)

router = APIRouter()
settings = get_settings()


# =============================================================================
# Request/Response Models
# =============================================================================

class SearchRequest(BaseModel):
    """Request to search the knowledge base"""
    query: str = Field(..., description="Search query", min_length=1)
    top_k: int = Field(5, description="Number of results", ge=1, le=20)


class SearchResultItem(BaseModel):
    """A single search result"""
    content: str
    score: float
    source: str
    filename: str


class SearchResponse(BaseModel):
    """Response from knowledge search"""
    query: str
    results: List[SearchResultItem]
    total_found: int
    search_time_ms: float


class DocumentInfo(BaseModel):
    """Information about an indexed document"""
    source: str
    filename: str
    document_type: str
    chunk_count: int
    indexed_at: Optional[str] = None


class IndexingResponse(BaseModel):
    """Response from document indexing"""
    success: bool
    filename: str
    document_type: str
    chunks_created: int
    chunks_indexed: int
    duration_seconds: float
    error: Optional[str] = None


class KnowledgeStats(BaseModel):
    """Knowledge base statistics"""
    collection_name: str
    total_chunks: int
    total_documents: int
    persist_directory: str


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/upload", response_model=IndexingResponse)
async def upload_document(
    file: UploadFile = File(..., description="Document to upload and index"),
    user: Optional[User] = Depends(get_optional_user),
):
    """
    Upload and index a document into the knowledge base.

    **Supported formats:** PDF, DOCX, TXT, MD, CSV, JSON

    The document will be:
    1. Saved to the uploads directory
    2. Loaded and parsed
    3. Split into chunks
    4. Embedded using nomic-embed-text
    5. Stored in ChromaDB for retrieval
    """
    # Validate file type
    allowed_extensions = {".pdf", ".docx", ".txt", ".md", ".csv", ".json"}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {file_ext}. Allowed: {', '.join(allowed_extensions)}"
        )

    # Check file size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Seek back to start

    if file_size > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {settings.max_upload_size_mb}MB"
        )

    # Save file to uploads directory
    upload_path = settings.upload_path / file.filename

    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}"
        )

    # Index the document
    try:
        engine = get_rag_engine()
        result = await engine.index_document(upload_path)

        return IndexingResponse(
            success=result.success,
            filename=result.filename,
            document_type=result.document_type,
            chunks_created=result.chunks_created,
            chunks_indexed=result.chunks_indexed,
            duration_seconds=result.duration_seconds,
            error=result.error,
        )

    except Exception as e:
        # Clean up uploaded file on error
        if upload_path.exists():
            upload_path.unlink()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to index document: {str(e)}"
        )


@router.get("/documents", response_model=List[DocumentInfo])
async def list_documents(
    user: Optional[User] = Depends(get_optional_user),
):
    """
    List all documents in the knowledge base.

    Returns information about each indexed document including:
    - Source file path
    - Filename
    - Document type
    - Number of chunks
    - When it was indexed
    """
    engine = get_rag_engine()
    documents = engine.list_documents()

    return [
        DocumentInfo(
            source=doc["source"],
            filename=doc["filename"],
            document_type=doc["document_type"],
            chunk_count=doc["chunk_count"],
            indexed_at=doc.get("indexed_at"),
        )
        for doc in documents
    ]


@router.delete("/documents/{filename}")
async def delete_document(
    filename: str,
    user: User = Depends(get_current_user),
):
    """
    Delete a document from the knowledge base.

    **Requires authentication.**

    This removes all chunks associated with the document from the vector store.
    """
    engine = get_rag_engine()

    # Find the document by filename
    documents = engine.list_documents()
    matching_docs = [d for d in documents if d["filename"] == filename]

    if not matching_docs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {filename}"
        )

    # Delete from vector store
    source = matching_docs[0]["source"]
    deleted_count = engine.delete_document(source)

    # Optionally delete the file from uploads
    upload_path = settings.upload_path / filename
    if upload_path.exists():
        try:
            upload_path.unlink()
        except Exception:
            pass  # Ignore file deletion errors

    return {
        "success": True,
        "filename": filename,
        "chunks_deleted": deleted_count,
    }


@router.post("/search", response_model=SearchResponse)
async def search_knowledge(
    request: SearchRequest,
    user: Optional[User] = Depends(get_optional_user),
):
    """
    Search the knowledge base.

    Returns the most relevant document chunks matching the query,
    ordered by semantic similarity.
    """
    engine = get_rag_engine()

    try:
        results = await engine.search(request.query, top_k=request.top_k)

        return SearchResponse(
            query=request.query,
            results=[
                SearchResultItem(
                    content=r.content,
                    score=r.score,
                    source=r.source,
                    filename=r.filename,
                )
                for r in results.results
            ],
            total_found=results.total_found,
            search_time_ms=results.search_time_ms,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.get("/context")
async def get_context(
    query: str,
    top_k: int = 3,
    user: Optional[User] = Depends(get_optional_user),
):
    """
    Get RAG context for a query.

    Returns formatted context suitable for inclusion in an agent prompt.
    This is what agents use internally to enhance their responses.
    """
    engine = get_rag_engine()

    try:
        context = await engine.get_context(query, top_k=top_k)

        return {
            "query": query,
            "context_text": context.context_text,
            "sources": context.sources,
            "chunks_used": context.chunks_used,
            "relevance_scores": context.relevance_scores,
            "formatted_prompt": context.format_for_prompt() if context.context_text else None,
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get context: {str(e)}"
        )


@router.get("/stats", response_model=KnowledgeStats)
async def get_stats(
    user: Optional[User] = Depends(get_optional_user),
):
    """
    Get knowledge base statistics.

    Returns:
    - Total number of indexed chunks
    - Total number of documents
    - Collection configuration
    """
    engine = get_rag_engine()
    stats = engine.get_stats()

    return KnowledgeStats(
        collection_name=stats["collection_name"],
        total_chunks=stats["total_chunks"],
        total_documents=stats["total_documents"],
        persist_directory=stats["persist_directory"],
    )


@router.post("/clear")
async def clear_knowledge_base(
    user: User = Depends(get_current_user),
):
    """
    Clear the entire knowledge base.

    **Requires authentication.**
    **Warning:** This will delete all indexed documents!
    """
    engine = get_rag_engine()
    engine.clear_index()

    return {
        "success": True,
        "message": "Knowledge base cleared",
    }


@router.post("/index-directory")
async def index_directory(
    directory: str = Form(..., description="Path to directory to index"),
    recursive: bool = Form(True, description="Search subdirectories"),
    user: User = Depends(get_current_user),
):
    """
    Index all documents in a directory.

    **Requires authentication.**

    Indexes all supported documents (PDF, DOCX, TXT, etc.) in the specified directory.
    """
    path = Path(directory)

    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Directory not found: {directory}"
        )

    if not path.is_dir():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Not a directory: {directory}"
        )

    engine = get_rag_engine()

    try:
        results = await engine.index_directory(path, recursive=recursive)

        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        return {
            "success": True,
            "total_files": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "results": [
                {
                    "filename": r.filename,
                    "success": r.success,
                    "chunks_indexed": r.chunks_indexed,
                    "error": r.error,
                }
                for r in results
            ],
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to index directory: {str(e)}"
        )
