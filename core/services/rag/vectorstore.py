"""
=============================================================================
HUMMINGBIRD-LEA - Vector Store (ChromaDB)
Powered by CoDre-X | B & D Servicing LLC
=============================================================================
Vector database integration using ChromaDB for semantic search.

Features:
- Persistent storage of document embeddings
- Semantic similarity search
- Metadata filtering
- Collection management
=============================================================================
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
from enum import Enum

from core.utils.config import get_settings
from .loaders import DocumentChunk, DocumentMetadata, DocumentType

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class DistanceMetric(Enum):
    """Distance metrics for similarity search"""
    COSINE = "cosine"       # Best for normalized embeddings
    L2 = "l2"               # Euclidean distance
    IP = "ip"               # Inner product


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SearchResult:
    """A single search result"""
    content: str
    score: float                    # Similarity score (0-1 for cosine)
    chunk_id: str
    source: str                     # Original document source
    metadata: Dict[str, Any]

    @property
    def filename(self) -> str:
        """Get filename from metadata"""
        return self.metadata.get("filename", "unknown")

    @property
    def document_type(self) -> str:
        """Get document type from metadata"""
        return self.metadata.get("document_type", "unknown")


@dataclass
class SearchResults:
    """Collection of search results"""
    query: str
    results: List[SearchResult]
    total_found: int
    search_time_ms: float

    def get_context(self, max_results: int = 3) -> str:
        """Get concatenated context from top results"""
        contexts = []
        for i, result in enumerate(self.results[:max_results], 1):
            contexts.append(
                f"[Source {i}: {result.filename}]\n{result.content}"
            )
        return "\n\n---\n\n".join(contexts)


# =============================================================================
# ChromaDB Vector Store
# =============================================================================

class ChromaVectorStore:
    """
    Vector store using ChromaDB for persistent embedding storage and search.

    Usage:
        store = ChromaVectorStore()

        # Add documents
        await store.add_chunks(chunks)

        # Search
        results = await store.search("What is economic incentive?", top_k=5)

        # Get context for RAG
        context = results.get_context()
    """

    DEFAULT_COLLECTION = "hummingbird_knowledge"

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = DEFAULT_COLLECTION,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
    ):
        """
        Initialize ChromaDB vector store.

        Args:
            persist_directory: Directory for persistent storage
            collection_name: Name of the collection
            distance_metric: Distance metric for similarity
        """
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError:
            raise ImportError(
                "chromadb is required for vector storage. "
                "Install with: pip install chromadb"
            )

        settings = get_settings()

        # Set up persistence directory
        if persist_directory:
            self.persist_dir = Path(persist_directory)
        else:
            self.persist_dir = settings.knowledge_path / "vectordb"

        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            )
        )

        self.collection_name = collection_name
        self.distance_metric = distance_metric

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": distance_metric.value}
        )

        logger.info(
            f"ChromaDB initialized: collection='{collection_name}', "
            f"persist_dir='{self.persist_dir}', "
            f"existing_docs={self.collection.count()}"
        )

    def add_chunks(
        self,
        chunks: List[DocumentChunk],
        batch_size: int = 100,
    ) -> int:
        """
        Add document chunks to the vector store.

        Args:
            chunks: List of DocumentChunks with embeddings
            batch_size: Batch size for insertion

        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0

        added = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            ids = []
            embeddings = []
            documents = []
            metadatas = []

            for chunk in batch:
                if chunk.embedding is None:
                    logger.warning(f"Skipping chunk without embedding: {chunk.chunk_index}")
                    continue

                # Generate unique ID
                chunk_id = self._generate_chunk_id(chunk)

                ids.append(chunk_id)
                embeddings.append(chunk.embedding)
                documents.append(chunk.content)
                metadatas.append(self._chunk_to_metadata(chunk))

            if ids:
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )
                added += len(ids)

        logger.info(f"Added {added} chunks to collection '{self.collection_name}'")
        return added

    def search(
        self,
        query_embedding: List[float],
        query_text: Optional[str] = None,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> SearchResults:
        """
        Search for similar documents.

        Args:
            query_embedding: Query embedding vector
            query_text: Original query text (for logging)
            top_k: Number of results to return
            filter_metadata: Optional metadata filter

        Returns:
            SearchResults with matched documents
        """
        import time
        start_time = time.time()

        # Build query
        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }

        if filter_metadata:
            query_params["where"] = filter_metadata

        # Execute search
        results = self.collection.query(**query_params)

        # Parse results
        search_results = []

        if results and results["ids"] and results["ids"][0]:
            ids = results["ids"][0]
            documents = results["documents"][0] if results["documents"] else []
            metadatas = results["metadatas"][0] if results["metadatas"] else []
            distances = results["distances"][0] if results["distances"] else []

            for i, chunk_id in enumerate(ids):
                # Convert distance to similarity score
                # For cosine: similarity = 1 - distance
                distance = distances[i] if i < len(distances) else 0
                score = 1 - distance if self.distance_metric == DistanceMetric.COSINE else 1 / (1 + distance)

                search_results.append(SearchResult(
                    content=documents[i] if i < len(documents) else "",
                    score=score,
                    chunk_id=chunk_id,
                    source=metadatas[i].get("source", "") if i < len(metadatas) else "",
                    metadata=metadatas[i] if i < len(metadatas) else {},
                ))

        search_time = (time.time() - start_time) * 1000

        logger.debug(
            f"Search completed: query='{query_text[:50] if query_text else 'N/A'}...', "
            f"results={len(search_results)}, time={search_time:.2f}ms"
        )

        return SearchResults(
            query=query_text or "",
            results=search_results,
            total_found=len(search_results),
            search_time_ms=search_time,
        )

    def delete_by_source(self, source: str) -> int:
        """
        Delete all chunks from a specific source document.

        Args:
            source: Source file path

        Returns:
            Number of chunks deleted
        """
        # Get IDs to delete
        results = self.collection.get(
            where={"source": source},
            include=["metadatas"],
        )

        if results and results["ids"]:
            ids_to_delete = results["ids"]
            self.collection.delete(ids=ids_to_delete)
            logger.info(f"Deleted {len(ids_to_delete)} chunks from source: {source}")
            return len(ids_to_delete)

        return 0

    def delete_collection(self):
        """Delete the entire collection"""
        self.client.delete_collection(self.collection_name)
        # Recreate empty collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance_metric.value}
        )
        logger.info(f"Collection '{self.collection_name}' deleted and recreated")

    def get_sources(self) -> List[Dict[str, Any]]:
        """
        Get list of all indexed sources (documents).

        Returns:
            List of source info dicts
        """
        # Get all metadata
        results = self.collection.get(include=["metadatas"])

        if not results or not results["metadatas"]:
            return []

        # Aggregate by source
        sources = {}
        for meta in results["metadatas"]:
            source = meta.get("source", "unknown")
            if source not in sources:
                sources[source] = {
                    "source": source,
                    "filename": meta.get("filename", "unknown"),
                    "document_type": meta.get("document_type", "unknown"),
                    "chunk_count": 0,
                    "indexed_at": meta.get("indexed_at"),
                }
            sources[source]["chunk_count"] += 1

        return list(sources.values())

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        count = self.collection.count()
        sources = self.get_sources()

        return {
            "collection_name": self.collection_name,
            "total_chunks": count,
            "total_documents": len(sources),
            "persist_directory": str(self.persist_dir),
            "distance_metric": self.distance_metric.value,
        }

    def _generate_chunk_id(self, chunk: DocumentChunk) -> str:
        """Generate unique ID for a chunk"""
        import hashlib

        # Create ID from source + chunk index
        source = chunk.metadata.source
        index = chunk.chunk_index
        content_hash = hashlib.md5(chunk.content[:100].encode()).hexdigest()[:8]

        return f"{source}:{index}:{content_hash}"

    def _chunk_to_metadata(self, chunk: DocumentChunk) -> Dict[str, Any]:
        """Convert chunk metadata to ChromaDB format"""
        return {
            "source": chunk.metadata.source,
            "filename": chunk.metadata.filename,
            "document_type": chunk.metadata.document_type.value,
            "chunk_index": chunk.chunk_index,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            "word_count": chunk.word_count,
            "indexed_at": datetime.utcnow().isoformat(),
        }


# =============================================================================
# Factory Functions
# =============================================================================

_vectorstore_instance: Optional[ChromaVectorStore] = None


def get_vectorstore(
    collection_name: str = ChromaVectorStore.DEFAULT_COLLECTION,
) -> ChromaVectorStore:
    """Get or create the vector store singleton"""
    global _vectorstore_instance

    if _vectorstore_instance is None or _vectorstore_instance.collection_name != collection_name:
        _vectorstore_instance = ChromaVectorStore(collection_name=collection_name)

    return _vectorstore_instance


def reset_vectorstore():
    """Reset the vector store singleton"""
    global _vectorstore_instance
    _vectorstore_instance = None
