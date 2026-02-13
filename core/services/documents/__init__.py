"""
=============================================================================
HUMMINGBIRD-LEA - Document Generation Services
Powered by CoDre-X | B & D Servicing LLC
=============================================================================
Document generation services for creating professional documents.

Supported Formats:
- PowerPoint (PPTX) - python-pptx
- Word (DOCX) - python-docx
- PDF - reportlab

EIAG Templates:
- Incentive Proposals
- Site Selection Reports
- Tax Credit Summaries
- Economic Impact Analysis
- Client Presentations
- Executive Summaries
- Formal Letters
=============================================================================
"""

# Base classes and utilities
from .base import (
    # Enums
    DocumentFormat,
    DocumentCategory,
    EIAGDocumentType,
    # Styles
    ColorScheme,
    FontScheme,
    # Data classes
    DocumentMetadata,
    DocumentSection,
    DocumentResult,
    ProposalData,
    SiteSelectionData,
    IncentiveSummaryData,
    # Utilities
    get_template_manager,
    sanitize_filename,
    format_currency,
    format_percentage,
)

# PowerPoint generator
from .powerpoint import (
    PowerPointGenerator,
    get_powerpoint_generator,
)

# Word generator
from .word import (
    WordGenerator,
    get_word_generator,
)

# PDF generator
from .pdf import (
    PDFGenerator,
    get_pdf_generator,
)

# Unified engine
from .engine import (
    DocumentEngine,
    EIAGTemplates,
    GenerationRequest,
    BatchResult,
    get_document_engine,
    generate_document,
    generate_proposal,
)

# Agent mixin
from .agent_mixin import (
    DocumentMixin,
    should_generate_document,
    detect_document_type,
    detect_output_format,
    process_document_request,
    extract_document_params,
)


__all__ = [
    # Enums
    "DocumentFormat",
    "DocumentCategory",
    "EIAGDocumentType",
    # Styles
    "ColorScheme",
    "FontScheme",
    # Data classes
    "DocumentMetadata",
    "DocumentSection",
    "DocumentResult",
    "ProposalData",
    "SiteSelectionData",
    "IncentiveSummaryData",
    # Utilities
    "get_template_manager",
    "sanitize_filename",
    "format_currency",
    "format_percentage",
    # PowerPoint
    "PowerPointGenerator",
    "get_powerpoint_generator",
    # Word
    "WordGenerator",
    "get_word_generator",
    # PDF
    "PDFGenerator",
    "get_pdf_generator",
    # Engine
    "DocumentEngine",
    "EIAGTemplates",
    "GenerationRequest",
    "BatchResult",
    "get_document_engine",
    "generate_document",
    "generate_proposal",
    # Agent mixin
    "DocumentMixin",
    "should_generate_document",
    "detect_document_type",
    "detect_output_format",
    "process_document_request",
    "extract_document_params",
]
