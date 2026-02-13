"""
=============================================================================
HUMMINGBIRD-LEA - Phase 3 RAG System Tests
Powered by CoDre-X | B & D Servicing LLC
=============================================================================
Tests for Phase 3 RAG components:
- Document loaders
- Text chunking
- Embedding pipeline
- Vector store
- RAG engine
=============================================================================
"""

import pytest
import tempfile
from pathlib import Path

# Import Phase 3 components
from core.services.rag import (
    # Document loading
    UniversalLoader,
    TextLoader,
    Document,
    DocumentType,
    get_loader,
    # Chunking
    ChunkingConfig,
    ChunkingStrategy,
    RecursiveChunker,
    SentenceChunker,
    get_chunker,
    chunk_document,
    # Embeddings
    EmbeddingPipeline,
    EmbeddingCache,
    # Vector store
    ChromaVectorStore,
    SearchResult,
    # RAG engine
    RAGEngine,
    RAGConfig,
    RAGContext,
    # Agent integration
    RAGMixin,
    should_use_rag,
)


# =============================================================================
# Document Loader Tests
# =============================================================================

class TestDocumentLoaders:
    """Tests for document loaders"""

    def test_text_loader_txt(self, tmp_path):
        """Test loading a .txt file"""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("This is a test document.\nWith multiple lines.")

        loader = TextLoader()
        assert loader.supports(test_file)

        doc = loader.load(test_file)
        assert doc.content == "This is a test document.\nWith multiple lines."
        assert doc.metadata.document_type == DocumentType.TXT
        assert doc.metadata.filename == "test.txt"

    def test_text_loader_md(self, tmp_path):
        """Test loading a .md file"""
        test_file = tmp_path / "test.md"
        test_file.write_text("# Heading\n\nParagraph text.")

        loader = TextLoader()
        assert loader.supports(test_file)

        doc = loader.load(test_file)
        assert "# Heading" in doc.content
        assert doc.metadata.document_type == DocumentType.MD

    def test_universal_loader(self, tmp_path):
        """Test the universal loader"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Universal loader test.")

        loader = get_loader()
        doc = loader.load(test_file)

        assert doc.content == "Universal loader test."

    def test_unsupported_file_type(self, tmp_path):
        """Test that unsupported file types raise an error"""
        test_file = tmp_path / "test.xyz"
        test_file.write_text("Unknown format")

        loader = get_loader()
        with pytest.raises(ValueError):
            loader.load(test_file)


# =============================================================================
# Chunking Tests
# =============================================================================

class TestChunking:
    """Tests for text chunking"""

    def test_recursive_chunker_basic(self, tmp_path):
        """Test basic recursive chunking"""
        # Create a document with enough content
        content = "First paragraph with some content.\n\n"
        content += "Second paragraph with more content.\n\n"
        content += "Third paragraph continues here."

        test_file = tmp_path / "test.txt"
        test_file.write_text(content)

        loader = get_loader()
        doc = loader.load(test_file)

        config = ChunkingConfig(chunk_size=100, chunk_overlap=10, min_chunk_size=20)
        chunker = RecursiveChunker(config)

        chunks = chunker.chunk(doc)
        assert len(chunks) >= 1
        assert all(chunk.content for chunk in chunks)

    def test_sentence_chunker(self, tmp_path):
        """Test sentence-based chunking"""
        content = "First sentence. Second sentence. Third sentence here. Fourth one too."

        test_file = tmp_path / "test.txt"
        test_file.write_text(content)

        loader = get_loader()
        doc = loader.load(test_file)

        config = ChunkingConfig(chunk_size=50, min_chunk_size=10)
        chunker = SentenceChunker(config)

        chunks = chunker.chunk(doc)
        assert len(chunks) >= 1

    def test_chunking_preserves_metadata(self, tmp_path):
        """Test that chunks preserve document metadata"""
        test_file = tmp_path / "metadata_test.txt"
        test_file.write_text("Test content for metadata.")

        loader = get_loader()
        doc = loader.load(test_file)

        chunks = chunk_document(doc)

        for chunk in chunks:
            assert chunk.metadata.source == doc.metadata.source
            assert chunk.metadata.filename == doc.metadata.filename


# =============================================================================
# Embedding Cache Tests
# =============================================================================

class TestEmbeddingCache:
    """Tests for embedding cache"""

    def test_cache_basic_operations(self):
        """Test basic cache operations"""
        from core.services.rag.embeddings import EmbeddingResult

        cache = EmbeddingCache(max_size=10)

        # Create a test result
        result = EmbeddingResult(
            text="test text",
            embedding=[0.1, 0.2, 0.3],
            model="test-model",
            dimension=3,
        )

        # Set and get
        cache.set(result)
        cached = cache.get("test text", "test-model")

        assert cached is not None
        assert cached.embedding == [0.1, 0.2, 0.3]

    def test_cache_eviction(self):
        """Test cache eviction when full"""
        from core.services.rag.embeddings import EmbeddingResult

        cache = EmbeddingCache(max_size=2)

        # Add 3 items to a cache of size 2
        for i in range(3):
            result = EmbeddingResult(
                text=f"text {i}",
                embedding=[float(i)],
                model="test",
                dimension=1,
            )
            cache.set(result)

        # First item should be evicted
        assert cache.get("text 0", "test") is None
        assert cache.get("text 2", "test") is not None


# =============================================================================
# Vector Store Tests
# =============================================================================

class TestChromaVectorStore:
    """Tests for ChromaDB vector store"""

    def test_vectorstore_initialization(self, tmp_path):
        """Test vector store initialization"""
        store = ChromaVectorStore(
            persist_directory=str(tmp_path / "vectordb"),
            collection_name="test_collection",
        )

        assert store.collection is not None
        assert store.collection_name == "test_collection"

    def test_vectorstore_stats(self, tmp_path):
        """Test getting vector store stats"""
        store = ChromaVectorStore(
            persist_directory=str(tmp_path / "vectordb"),
            collection_name="test_stats",
        )

        stats = store.get_stats()
        assert "collection_name" in stats
        assert "total_chunks" in stats
        assert stats["total_chunks"] == 0  # Empty collection


# =============================================================================
# RAG Engine Tests
# =============================================================================

class TestRAGEngine:
    """Tests for the RAG engine"""

    def test_rag_config_defaults(self):
        """Test RAG config default values"""
        config = RAGConfig()

        assert config.chunk_size == 512
        assert config.chunk_overlap == 50
        assert config.top_k == 5
        assert config.min_relevance_score == 0.3

    def test_rag_context_format(self):
        """Test RAG context formatting"""
        context = RAGContext(
            query="test query",
            context_text="Some relevant context",
            sources=["doc1.pdf", "doc2.txt"],
            chunks_used=2,
            relevance_scores=[0.9, 0.8],
            search_time_ms=10.5,
        )

        formatted = context.format_for_prompt()
        assert "Knowledge Base" in formatted
        assert "relevant context" in formatted.lower() or "Some relevant context" in formatted


# =============================================================================
# RAG Mixin Tests
# =============================================================================

class TestRAGMixin:
    """Tests for RAG agent mixin"""

    @pytest.mark.asyncio
    async def test_should_use_rag_questions(self):
        """Test RAG usage detection for questions"""
        assert await should_use_rag("What is economic incentive?") == True
        assert await should_use_rag("How does tax credit work?") == True
        assert await should_use_rag("Tell me about site selection") == True

    @pytest.mark.asyncio
    async def test_should_use_rag_greetings(self):
        """Test RAG not used for greetings"""
        assert await should_use_rag("Hi") == False
        assert await should_use_rag("Hello there") == False


# =============================================================================
# Integration Tests (require Ollama)
# =============================================================================

class TestRAGIntegration:
    """Integration tests that require Ollama to be running"""

    @pytest.mark.skip(reason="Requires Ollama to be running")
    @pytest.mark.asyncio
    async def test_full_rag_pipeline(self, tmp_path):
        """Test full RAG pipeline: load -> chunk -> embed -> store -> search"""
        from core.services.rag import get_rag_engine

        # Create test document
        test_file = tmp_path / "test_doc.txt"
        test_file.write_text(
            "Economic incentives are financial benefits offered to businesses. "
            "Tax credits reduce the amount of tax a company owes. "
            "Site selection involves choosing the best location for a facility."
        )

        # Initialize RAG engine with test collection
        engine = RAGEngine(
            config=RAGConfig(chunk_size=100),
            collection_name="test_integration",
        )

        # Index the document
        result = await engine.index_document(test_file)
        assert result.success
        assert result.chunks_indexed > 0

        # Search
        search_results = await engine.search("What are tax credits?")
        assert search_results.total_found > 0

        # Get context
        context = await engine.get_context("economic incentives")
        assert context.chunks_used > 0

        # Cleanup
        engine.clear_index()


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
