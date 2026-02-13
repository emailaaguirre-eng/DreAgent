"""
=============================================================================
HUMMINGBIRD-LEA - RAG Engine
Powered by CoDre-X | B & D Servicing LLC
=============================================================================
Main RAG (Retrieval Augmented Generation) engine that ties all components
together: document loading, chunking, embedding, and retrieval.

Usage:
    engine = RAGEngine()

    # Index a document
    await engine.index_document("path/to/document.pdf")

    # Query with context
    context = await engine.get_context("What are tax incentives?")

    # Use context in agent response
    response = agent.respond(question, context=context)
=============================================================================
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Union, Callable

from core.utils.config import get_settings
from .loaders import (
    UniversalLoader,
    Document,
    DocumentChunk,
    DocumentMetadata,
    DocumentType,
    get_loader,
)
from .chunking import (
    ChunkingConfig,
    ChunkingStrategy,
    get_chunker,
    chunk_document,
)
from .embeddings import (
    EmbeddingPipeline,
    get_embedding_pipeline,
)
from .vectorstore import (
    ChromaVectorStore,
    SearchResults,
    SearchResult,
    get_vectorstore,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class IndexingResult:
    """Result of indexing a document"""
    source: str
    filename: str
    document_type: str
    chunks_created: int
    chunks_indexed: int
    duration_seconds: float
    success: bool
    error: Optional[str] = None


@dataclass
class RAGContext:
    """Context retrieved for a query"""
    query: str
    context_text: str           # Formatted context for prompt
    sources: List[str]          # Source documents used
    chunks_used: int
    relevance_scores: List[float]
    search_time_ms: float

    def format_for_prompt(self) -> str:
        """Format context for inclusion in agent prompt"""
        if not self.context_text:
            return ""

        return f"""
## Relevant Knowledge Base Information

The following information was retrieved from the knowledge base and may be relevant to your response:

{self.context_text}

---
Sources: {', '.join(self.sources)}

Use this information to inform your response, but always prioritize accuracy.
If the retrieved information doesn't fully answer the question, acknowledge the limitation.
"""


@dataclass
class RAGConfig:
    """Configuration for RAG engine"""
    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 50
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE

    # Retrieval
    top_k: int = 5                      # Number of chunks to retrieve
    min_relevance_score: float = 0.3    # Minimum similarity score
    max_context_length: int = 2000      # Max chars for context

    # Search
    rerank: bool = False                # Whether to rerank results
    hybrid_search: bool = False         # Combine semantic + keyword search


# =============================================================================
# RAG Engine
# =============================================================================

class RAGEngine:
    """
    Main RAG engine for Hummingbird-LEA.

    Provides:
    - Document indexing (load, chunk, embed, store)
    - Semantic search and retrieval
    - Context generation for agent prompts
    - Document management (list, delete)
    """

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        collection_name: str = "hummingbird_knowledge",
    ):
        """
        Initialize the RAG engine.

        Args:
            config: RAG configuration
            collection_name: ChromaDB collection name
        """
        self.config = config or RAGConfig()
        self.collection_name = collection_name

        # Initialize components
        self.loader = get_loader()
        self.embedding_pipeline = get_embedding_pipeline()
        self.vectorstore = get_vectorstore(collection_name)

        # Chunking config from RAG config
        self.chunking_config = ChunkingConfig(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

        logger.info(f"RAGEngine initialized with collection: {collection_name}")

    async def index_document(
        self,
        file_path: Union[str, Path],
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> IndexingResult:
        """
        Index a document for retrieval.

        Args:
            file_path: Path to the document
            progress_callback: Optional callback(stage, current, total)

        Returns:
            IndexingResult with indexing statistics
        """
        start_time = datetime.utcnow()
        path = Path(file_path)

        try:
            # Stage 1: Load document
            if progress_callback:
                progress_callback("loading", 0, 4)

            logger.info(f"Loading document: {path}")
            document = self.loader.load(path)

            # Stage 2: Chunk document
            if progress_callback:
                progress_callback("chunking", 1, 4)

            logger.info(f"Chunking document: {document.metadata.filename}")
            chunker = get_chunker(self.config.chunking_strategy, self.chunking_config)
            chunks = chunker.chunk(document)

            logger.info(f"Created {len(chunks)} chunks")

            # Stage 3: Generate embeddings
            if progress_callback:
                progress_callback("embedding", 2, 4)

            logger.info(f"Generating embeddings for {len(chunks)} chunks")
            embedded_chunks = await self.embedding_pipeline.embed_chunks(chunks)

            logger.info(f"Generated {len(embedded_chunks)} embeddings")

            # Stage 4: Store in vector database
            if progress_callback:
                progress_callback("storing", 3, 4)

            logger.info(f"Storing {len(embedded_chunks)} chunks in vector store")
            indexed_count = self.vectorstore.add_chunks(embedded_chunks)

            if progress_callback:
                progress_callback("complete", 4, 4)

            duration = (datetime.utcnow() - start_time).total_seconds()

            logger.info(
                f"Indexing complete: {document.metadata.filename}, "
                f"{indexed_count} chunks in {duration:.2f}s"
            )

            return IndexingResult(
                source=str(path.absolute()),
                filename=document.metadata.filename,
                document_type=document.metadata.document_type.value,
                chunks_created=len(chunks),
                chunks_indexed=indexed_count,
                duration_seconds=duration,
                success=True,
            )

        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.error(f"Indexing failed for {path}: {e}")

            return IndexingResult(
                source=str(path),
                filename=path.name,
                document_type="unknown",
                chunks_created=0,
                chunks_indexed=0,
                duration_seconds=duration,
                success=False,
                error=str(e),
            )

    async def index_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = True,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> List[IndexingResult]:
        """
        Index all documents in a directory.

        Args:
            directory: Path to directory
            recursive: Whether to search subdirectories
            progress_callback: Optional callback(filename, current, total)

        Returns:
            List of IndexingResults
        """
        path = Path(directory)

        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")

        # Find all supported files
        supported_extensions = {".txt", ".md", ".pdf", ".docx", ".csv", ".json"}
        files = []

        pattern = "**/*" if recursive else "*"
        for file_path in path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                files.append(file_path)

        logger.info(f"Found {len(files)} documents to index in {path}")

        results = []
        for i, file_path in enumerate(files):
            if progress_callback:
                progress_callback(file_path.name, i, len(files))

            result = await self.index_document(file_path)
            results.append(result)

        successful = sum(1 for r in results if r.success)
        logger.info(f"Directory indexing complete: {successful}/{len(files)} successful")

        return results

    async def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_source: Optional[str] = None,
    ) -> SearchResults:
        """
        Search the knowledge base.

        Args:
            query: Search query
            top_k: Number of results (default from config)
            filter_source: Optional source file filter

        Returns:
            SearchResults with matched chunks
        """
        top_k = top_k or self.config.top_k

        # Generate query embedding
        query_result = await self.embedding_pipeline.embed(query)

        if not query_result:
            logger.error("Failed to generate query embedding")
            return SearchResults(
                query=query,
                results=[],
                total_found=0,
                search_time_ms=0,
            )

        # Build filter
        filter_metadata = None
        if filter_source:
            filter_metadata = {"source": filter_source}

        # Search vector store
        results = self.vectorstore.search(
            query_embedding=query_result.embedding,
            query_text=query,
            top_k=top_k,
            filter_metadata=filter_metadata,
        )

        # Filter by minimum relevance score
        filtered_results = [
            r for r in results.results
            if r.score >= self.config.min_relevance_score
        ]

        results.results = filtered_results
        results.total_found = len(filtered_results)

        return results

    async def get_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> RAGContext:
        """
        Get relevant context for a query.

        This is the main method for RAG - it retrieves relevant chunks
        and formats them for inclusion in an agent prompt.

        Args:
            query: The user's question
            top_k: Number of chunks to retrieve
            max_length: Maximum context length

        Returns:
            RAGContext with formatted context
        """
        top_k = top_k or self.config.top_k
        max_length = max_length or self.config.max_context_length

        # Search for relevant chunks
        search_results = await self.search(query, top_k=top_k)

        if not search_results.results:
            return RAGContext(
                query=query,
                context_text="",
                sources=[],
                chunks_used=0,
                relevance_scores=[],
                search_time_ms=search_results.search_time_ms,
            )

        # Build context from results
        context_parts = []
        sources = set()
        scores = []
        current_length = 0

        for i, result in enumerate(search_results.results):
            # Check length limit
            if current_length + len(result.content) > max_length:
                break

            # Add source header
            source_label = f"[From: {result.filename}]"
            context_parts.append(f"{source_label}\n{result.content}")

            sources.add(result.filename)
            scores.append(result.score)
            current_length += len(result.content) + len(source_label) + 10

        context_text = "\n\n---\n\n".join(context_parts)

        return RAGContext(
            query=query,
            context_text=context_text,
            sources=list(sources),
            chunks_used=len(context_parts),
            relevance_scores=scores,
            search_time_ms=search_results.search_time_ms,
        )

    def delete_document(self, source: str) -> int:
        """
        Delete a document from the index.

        Args:
            source: Source file path

        Returns:
            Number of chunks deleted
        """
        return self.vectorstore.delete_by_source(source)

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all indexed documents.

        Returns:
            List of document info dicts
        """
        return self.vectorstore.get_sources()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get RAG engine statistics.

        Returns:
            Statistics dict
        """
        vs_stats = self.vectorstore.get_stats()

        return {
            **vs_stats,
            "config": {
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "top_k": self.config.top_k,
                "min_relevance_score": self.config.min_relevance_score,
            },
            "embedding_cache_size": self.embedding_pipeline.cache_size,
        }

    def clear_index(self):
        """Clear all indexed documents"""
        self.vectorstore.delete_collection()
        logger.info("RAG index cleared")


# =============================================================================
# Factory Function
# =============================================================================

_rag_engine_instance: Optional[RAGEngine] = None


def get_rag_engine(
    config: Optional[RAGConfig] = None,
    collection_name: str = "hummingbird_knowledge",
) -> RAGEngine:
    """Get or create the RAG engine singleton"""
    global _rag_engine_instance

    if _rag_engine_instance is None:
        _rag_engine_instance = RAGEngine(config, collection_name)

    return _rag_engine_instance


def reset_rag_engine():
    """Reset the RAG engine singleton"""
    global _rag_engine_instance
    _rag_engine_instance = None
