"""
=============================================================================
HUMMINGBIRD-LEA - Embedding Pipeline
Powered by CoDre-X | B & D Servicing LLC
=============================================================================
Embedding generation using Ollama's nomic-embed-text model.

Features:
- Async batch embedding generation
- Caching for frequently embedded texts
- Progress tracking for large documents
=============================================================================
"""

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any
from datetime import datetime

from core.providers.ollama import get_ollama_client, ModelType
from .loaders import DocumentChunk

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class EmbeddingResult:
    """Result of embedding a text"""
    text: str
    embedding: List[float]
    model: str
    dimension: int
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def text_hash(self) -> str:
        """Get hash of the text for caching"""
        return hashlib.md5(self.text.encode()).hexdigest()


@dataclass
class BatchEmbeddingResult:
    """Result of batch embedding operation"""
    embeddings: List[EmbeddingResult]
    total_texts: int
    successful: int
    failed: int
    duration_seconds: float
    model: str


# =============================================================================
# Embedding Cache
# =============================================================================

class EmbeddingCache:
    """
    Simple in-memory cache for embeddings.
    Helps avoid re-embedding the same text.
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: Dict[str, EmbeddingResult] = {}
        self._access_order: List[str] = []

    def _get_key(self, text: str, model: str) -> str:
        """Generate cache key"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"{model}:{text_hash}"

    def get(self, text: str, model: str) -> Optional[EmbeddingResult]:
        """Get cached embedding if exists"""
        key = self._get_key(text, model)
        result = self._cache.get(key)

        if result:
            # Update access order (LRU)
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)

        return result

    def set(self, result: EmbeddingResult):
        """Cache an embedding result"""
        key = self._get_key(result.text, result.model)

        # Evict oldest if at capacity
        while len(self._cache) >= self.max_size and self._access_order:
            oldest_key = self._access_order.pop(0)
            self._cache.pop(oldest_key, None)

        self._cache[key] = result
        self._access_order.append(key)

    def clear(self):
        """Clear the cache"""
        self._cache.clear()
        self._access_order.clear()

    @property
    def size(self) -> int:
        """Current cache size"""
        return len(self._cache)


# =============================================================================
# Embedding Pipeline
# =============================================================================

class EmbeddingPipeline:
    """
    Pipeline for generating embeddings using Ollama.

    Uses nomic-embed-text model by default, which produces
    768-dimensional embeddings optimized for semantic search.

    Usage:
        pipeline = EmbeddingPipeline()

        # Single text
        result = await pipeline.embed("Hello world")

        # Batch of texts
        results = await pipeline.embed_batch(["Text 1", "Text 2", "Text 3"])

        # Document chunks
        chunks = await pipeline.embed_chunks(document_chunks)
    """

    def __init__(
        self,
        model: Optional[str] = None,
        use_cache: bool = True,
        cache_size: int = 1000,
        batch_size: int = 10,
        retry_attempts: int = 3,
    ):
        """
        Initialize the embedding pipeline.

        Args:
            model: Embedding model name (default: from settings)
            use_cache: Whether to cache embeddings
            cache_size: Maximum cache size
            batch_size: Batch size for concurrent embedding
            retry_attempts: Number of retry attempts on failure
        """
        self.ollama = get_ollama_client()
        self.model = model or self.ollama.get_model(ModelType.EMBED)
        self.use_cache = use_cache
        self.cache = EmbeddingCache(max_size=cache_size) if use_cache else None
        self.batch_size = batch_size
        self.retry_attempts = retry_attempts

        logger.info(f"EmbeddingPipeline initialized with model: {self.model}")

    async def embed(self, text: str) -> Optional[EmbeddingResult]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            EmbeddingResult or None if failed
        """
        # Check cache
        if self.cache:
            cached = self.cache.get(text, self.model)
            if cached:
                logger.debug(f"Cache hit for text: {text[:50]}...")
                return cached

        # Generate embedding
        for attempt in range(self.retry_attempts):
            try:
                response = await self.ollama.generate_embedding(text, self.model)

                if response and response.embedding:
                    result = EmbeddingResult(
                        text=text,
                        embedding=response.embedding,
                        model=response.model,
                        dimension=len(response.embedding),
                    )

                    # Cache result
                    if self.cache:
                        self.cache.set(result)

                    return result

            except Exception as e:
                logger.warning(f"Embedding attempt {attempt + 1} failed: {e}")
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(1)  # Brief delay before retry

        logger.error(f"Failed to embed text after {self.retry_attempts} attempts")
        return None

    async def embed_batch(
        self,
        texts: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BatchEmbeddingResult:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed
            progress_callback: Optional callback(current, total) for progress

        Returns:
            BatchEmbeddingResult with all embeddings
        """
        start_time = datetime.utcnow()
        embeddings = []
        failed = 0

        total = len(texts)

        # Process in batches for concurrency control
        for i in range(0, total, self.batch_size):
            batch = texts[i:i + self.batch_size]

            # Create tasks for concurrent embedding
            tasks = [self.embed(text) for text in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Batch embedding error: {result}")
                    failed += 1
                elif result is None:
                    failed += 1
                else:
                    embeddings.append(result)

            # Report progress
            if progress_callback:
                progress_callback(min(i + self.batch_size, total), total)

        duration = (datetime.utcnow() - start_time).total_seconds()

        logger.info(
            f"Batch embedding complete: {len(embeddings)}/{total} successful "
            f"in {duration:.2f}s"
        )

        return BatchEmbeddingResult(
            embeddings=embeddings,
            total_texts=total,
            successful=len(embeddings),
            failed=failed,
            duration_seconds=duration,
            model=self.model,
        )

    async def embed_chunks(
        self,
        chunks: List[DocumentChunk],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[DocumentChunk]:
        """
        Generate embeddings for document chunks and attach them.

        Args:
            chunks: List of DocumentChunks to embed
            progress_callback: Optional callback for progress

        Returns:
            List of DocumentChunks with embeddings attached
        """
        texts = [chunk.content for chunk in chunks]

        batch_result = await self.embed_batch(texts, progress_callback)

        # Create mapping of text to embedding
        embedding_map = {
            result.text: result.embedding
            for result in batch_result.embeddings
        }

        # Attach embeddings to chunks
        embedded_chunks = []
        for chunk in chunks:
            embedding = embedding_map.get(chunk.content)
            if embedding:
                chunk.embedding = embedding
                embedded_chunks.append(chunk)
            else:
                logger.warning(f"No embedding for chunk {chunk.chunk_index}")

        return embedded_chunks

    async def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings from this model.

        Returns:
            Embedding dimension (e.g., 768 for nomic-embed-text)
        """
        # Embed a test string to get dimension
        result = await self.embed("test")
        if result:
            return result.dimension
        return 768  # Default for nomic-embed-text

    def clear_cache(self):
        """Clear the embedding cache"""
        if self.cache:
            self.cache.clear()
            logger.info("Embedding cache cleared")

    @property
    def cache_size(self) -> int:
        """Get current cache size"""
        return self.cache.size if self.cache else 0


# =============================================================================
# Factory Function
# =============================================================================

_pipeline_instance: Optional[EmbeddingPipeline] = None


def get_embedding_pipeline() -> EmbeddingPipeline:
    """Get or create the embedding pipeline singleton"""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = EmbeddingPipeline()
    return _pipeline_instance


async def embed_text(text: str) -> Optional[List[float]]:
    """
    Convenience function to embed a single text.

    Args:
        text: Text to embed

    Returns:
        Embedding vector or None if failed
    """
    pipeline = get_embedding_pipeline()
    result = await pipeline.embed(text)
    return result.embedding if result else None
