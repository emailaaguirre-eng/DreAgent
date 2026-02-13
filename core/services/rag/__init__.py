"""
=============================================================================
HUMMINGBIRD-LEA - RAG System (Phase 3)
Powered by CoDre-X | B & D Servicing LLC
=============================================================================
Retrieval Augmented Generation (RAG) system for knowledge-enhanced responses.

Components:
- Document Loaders: Load PDF, DOCX, TXT, CSV, JSON files
- Chunking: Split documents into semantic chunks
- Embeddings: Generate vectors using nomic-embed-text
- Vector Store: ChromaDB for persistent similarity search
- RAG Engine: Unified interface for indexing and retrieval

Usage:
    from core.services.rag import get_rag_engine

    # Get the engine
    engine = get_rag_engine()

    # Index documents
    await engine.index_document("path/to/document.pdf")
    await engine.index_directory("path/to/documents/")

    # Get context for a query
    context = await engine.get_context("What are economic incentives?")

    # Use in agent prompt
    prompt = f"{base_prompt}\n\n{context.format_for_prompt()}"
=============================================================================
"""

# Document Loaders
from .loaders import (
    # Types
    DocumentType,
    DocumentMetadata,
    Document,
    DocumentChunk,
    # Loaders
    BaseLoader,
    TextLoader,
    PDFLoader,
    DocxLoader,
    CSVLoader,
    JSONLoader,
    UniversalLoader,
    # Factory
    get_loader,
)

# Chunking
from .chunking import (
    # Types
    ChunkingStrategy,
    ChunkingConfig,
    # Chunkers
    BaseChunker,
    FixedSizeChunker,
    SentenceChunker,
    ParagraphChunker,
    RecursiveChunker,
    # Functions
    get_chunker,
    chunk_document,
)

# Embeddings
from .embeddings import (
    # Types
    EmbeddingResult,
    BatchEmbeddingResult,
    # Classes
    EmbeddingCache,
    EmbeddingPipeline,
    # Functions
    get_embedding_pipeline,
    embed_text,
)

# Vector Store
from .vectorstore import (
    # Types
    DistanceMetric,
    SearchResult,
    SearchResults,
    # Classes
    ChromaVectorStore,
    # Functions
    get_vectorstore,
    reset_vectorstore,
)

# RAG Engine
from .engine import (
    # Types
    IndexingResult,
    RAGContext,
    RAGConfig,
    # Classes
    RAGEngine,
    # Functions
    get_rag_engine,
    reset_rag_engine,
)

# Agent Integration
from .agent_mixin import (
    RAGMixin,
    RAGEnhancedPromptBuilder,
    create_rag_system_prompt_addition,
    should_use_rag,
)


# =============================================================================
# Convenience Functions
# =============================================================================

async def index_document(file_path: str) -> "IndexingResult":
    """
    Convenience function to index a document.

    Args:
        file_path: Path to the document

    Returns:
        IndexingResult
    """
    engine = get_rag_engine()
    return await engine.index_document(file_path)


async def search_knowledge(query: str, top_k: int = 5) -> "SearchResults":
    """
    Convenience function to search the knowledge base.

    Args:
        query: Search query
        top_k: Number of results

    Returns:
        SearchResults
    """
    engine = get_rag_engine()
    return await engine.search(query, top_k=top_k)


async def get_rag_context(query: str) -> "RAGContext":
    """
    Convenience function to get RAG context for a query.

    Args:
        query: The question

    Returns:
        RAGContext with formatted context
    """
    engine = get_rag_engine()
    return await engine.get_context(query)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Document Loaders
    "DocumentType",
    "DocumentMetadata",
    "Document",
    "DocumentChunk",
    "BaseLoader",
    "TextLoader",
    "PDFLoader",
    "DocxLoader",
    "CSVLoader",
    "JSONLoader",
    "UniversalLoader",
    "get_loader",

    # Chunking
    "ChunkingStrategy",
    "ChunkingConfig",
    "BaseChunker",
    "FixedSizeChunker",
    "SentenceChunker",
    "ParagraphChunker",
    "RecursiveChunker",
    "get_chunker",
    "chunk_document",

    # Embeddings
    "EmbeddingResult",
    "BatchEmbeddingResult",
    "EmbeddingCache",
    "EmbeddingPipeline",
    "get_embedding_pipeline",
    "embed_text",

    # Vector Store
    "DistanceMetric",
    "SearchResult",
    "SearchResults",
    "ChromaVectorStore",
    "get_vectorstore",
    "reset_vectorstore",

    # RAG Engine
    "IndexingResult",
    "RAGContext",
    "RAGConfig",
    "RAGEngine",
    "get_rag_engine",
    "reset_rag_engine",

    # Convenience Functions
    "index_document",
    "search_knowledge",
    "get_rag_context",

    # Agent Integration
    "RAGMixin",
    "RAGEnhancedPromptBuilder",
    "create_rag_system_prompt_addition",
    "should_use_rag",
]
