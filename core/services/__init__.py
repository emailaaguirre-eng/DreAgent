"""
=============================================================================
HUMMINGBIRD-LEA - Services
Powered by CoDre-X | B & D Servicing LLC
=============================================================================
Core services for the application.

Phase 3: RAG System - COMPLETE
Phase 4: Vision & OCR - COMPLETE
Phase 5: Document Generation - COMPLETE
=============================================================================
"""

# Phase 3: RAG System
from .rag import (
    # Engine
    RAGEngine,
    RAGConfig,
    RAGContext,
    IndexingResult,
    get_rag_engine,
    reset_rag_engine,
    # Convenience functions
    index_document,
    search_knowledge,
    get_rag_context,
    # Document loading
    get_loader,
    Document,
    DocumentChunk,
    DocumentType,
    # Chunking
    get_chunker,
    ChunkingConfig,
    ChunkingStrategy,
    # Embeddings
    get_embedding_pipeline,
    embed_text,
    # Vector store
    get_vectorstore,
    SearchResults,
    SearchResult,
)

__all__ = [
    # RAG Engine
    "RAGEngine",
    "RAGConfig",
    "RAGContext",
    "IndexingResult",
    "get_rag_engine",
    "reset_rag_engine",
    # Convenience
    "index_document",
    "search_knowledge",
    "get_rag_context",
    # Document loading
    "get_loader",
    "Document",
    "DocumentChunk",
    "DocumentType",
    # Chunking
    "get_chunker",
    "ChunkingConfig",
    "ChunkingStrategy",
    # Embeddings
    "get_embedding_pipeline",
    "embed_text",
    # Vector store
    "get_vectorstore",
    "SearchResults",
    "SearchResult",
    # Vision & OCR (Phase 4)
    "ImageAnalysisEngine",
    "get_analysis_engine",
    "VisionService",
    "get_vision_service",
    "OCRService",
    "get_ocr_service",
    "VisionMixin",
    "analyze_image",
    "extract_image_text",
    # Document Generation (Phase 5)
    "DocumentEngine",
    "get_document_engine",
    "generate_document",
    "generate_proposal",
    "DocumentFormat",
    "DocumentMetadata",
    "DocumentSection",
    "DocumentResult",
    "ProposalData",
    "SiteSelectionData",
    "IncentiveSummaryData",
    "PowerPointGenerator",
    "WordGenerator",
    "PDFGenerator",
    "DocumentMixin",
    "should_generate_document",
    "detect_document_type",
]

# Phase 4: Vision & OCR
from .vision import (
    # Analysis engine
    ImageAnalysisEngine,
    ImageAnalysisResult,
    get_analysis_engine,
    analyze_image,
    extract_image_text,
    # Vision service
    VisionService,
    VisionResult,
    get_vision_service,
    # OCR service
    OCRService,
    OCRResult,
    get_ocr_service,
    # Agent mixin
    VisionMixin,
    should_use_vision,
    detect_image_intent,
)

# Phase 5: Document Generation
from .documents import (
    # Engine
    DocumentEngine,
    get_document_engine,
    generate_document,
    generate_proposal,
    # Data classes
    DocumentFormat,
    DocumentMetadata,
    DocumentSection,
    DocumentResult,
    ProposalData,
    SiteSelectionData,
    IncentiveSummaryData,
    # Generators
    PowerPointGenerator,
    WordGenerator,
    PDFGenerator,
    # Agent mixin
    DocumentMixin,
    should_generate_document,
    detect_document_type,
)
