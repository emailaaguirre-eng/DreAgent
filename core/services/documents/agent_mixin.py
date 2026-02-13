"""
=============================================================================
HUMMINGBIRD-LEA - Document Generation Agent Mixin
Powered by CoDre-X | B & D Servicing LLC
=============================================================================
Mixin class to add document generation capabilities to agents.

Usage:
    class GrantAgent(BaseAgent, DocumentMixin):
        pass
=============================================================================
"""

import logging
from typing import Optional, List, Dict, Any, Union
from pathlib import Path

from .base import (
    DocumentFormat,
    DocumentCategory,
    DocumentMetadata,
    DocumentSection,
    DocumentResult,
    ProposalData,
    SiteSelectionData,
    IncentiveSummaryData,
)
from .engine import DocumentEngine, get_document_engine

logger = logging.getLogger(__name__)


# =============================================================================
# Document Intent Detection
# =============================================================================

def should_generate_document(message: str) -> bool:
    """
    Determine if a message requests document generation.

    Args:
        message: User's message

    Returns:
        True if document generation is requested
    """
    message_lower = message.lower()

    # Document generation keywords
    doc_keywords = [
        "generate", "create", "make", "produce", "draft", "write",
        "document", "report", "proposal", "presentation", "letter",
        "pdf", "word", "powerpoint", "pptx", "docx",
        "summary", "analysis", "comparison",
    ]

    # Check for document-related keywords
    return any(keyword in message_lower for keyword in doc_keywords)


def detect_document_type(message: str) -> Optional[str]:
    """
    Detect what type of document is being requested.

    Args:
        message: User's message

    Returns:
        Document type string or None
    """
    message_lower = message.lower()

    # Check for specific document types
    if "proposal" in message_lower:
        if "incentive" in message_lower or "tax" in message_lower:
            return "incentive_proposal"
        return "proposal"

    if "site selection" in message_lower or "site comparison" in message_lower:
        return "site_selection"

    if "incentive summary" in message_lower or "tax credit summary" in message_lower:
        return "incentive_summary"

    if "presentation" in message_lower or "powerpoint" in message_lower or "pptx" in message_lower:
        return "presentation"

    if "letter" in message_lower:
        return "letter"

    if "executive summary" in message_lower:
        return "executive_summary"

    if "report" in message_lower:
        return "report"

    if "analysis" in message_lower:
        return "analysis"

    return None


def detect_output_format(message: str) -> DocumentFormat:
    """
    Detect the desired output format from a message.

    Args:
        message: User's message

    Returns:
        DocumentFormat (defaults to PDF)
    """
    message_lower = message.lower()

    if "powerpoint" in message_lower or "pptx" in message_lower or "presentation" in message_lower:
        return DocumentFormat.PPTX

    if "word" in message_lower or "docx" in message_lower:
        return DocumentFormat.DOCX

    # Default to PDF
    return DocumentFormat.PDF


# =============================================================================
# Document Mixin
# =============================================================================

class DocumentMixin:
    """
    Mixin class that adds document generation capabilities to agents.

    Provides methods for:
    - Document generation (PDF, DOCX, PPTX)
    - EIAG template generation
    - Document type detection
    - Format conversion

    Usage:
        class GrantAgent(BaseAgent, DocumentMixin):
            async def process(self, message, context):
                if should_generate_document(message):
                    doc_type = detect_document_type(message)
                    # Handle document generation
                return await super().process(message, context)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._document_engine: Optional[DocumentEngine] = None

    @property
    def document_engine(self) -> DocumentEngine:
        """Get the document engine (lazy init)"""
        if self._document_engine is None:
            self._document_engine = get_document_engine()
        return self._document_engine

    def generate_document(
        self,
        title: str,
        content: Union[str, List[DocumentSection]],
        format: DocumentFormat = DocumentFormat.PDF,
        client_name: Optional[str] = None,
        project_name: Optional[str] = None,
    ) -> DocumentResult:
        """
        Generate a document.

        Args:
            title: Document title
            content: Content or sections
            format: Output format
            client_name: Optional client name
            project_name: Optional project name

        Returns:
            DocumentResult
        """
        return self.document_engine.generate(
            title=title,
            content=content,
            format=format,
            client_name=client_name,
            project_name=project_name,
        )

    def generate_proposal(
        self,
        data: ProposalData,
        format: DocumentFormat = DocumentFormat.PDF,
    ) -> DocumentResult:
        """
        Generate an EIAG incentive proposal.

        Args:
            data: Proposal data
            format: Output format

        Returns:
            DocumentResult
        """
        return self.document_engine.generate_eiag_proposal(data, format)

    def generate_site_selection(
        self,
        data: SiteSelectionData,
        format: DocumentFormat = DocumentFormat.PDF,
    ) -> DocumentResult:
        """
        Generate a site selection report.

        Args:
            data: Site selection data
            format: Output format

        Returns:
            DocumentResult
        """
        return self.document_engine.generate_eiag_site_selection(data, format)

    def generate_incentive_summary(
        self,
        data: IncentiveSummaryData,
        format: DocumentFormat = DocumentFormat.PDF,
    ) -> DocumentResult:
        """
        Generate an incentive summary.

        Args:
            data: Incentive summary data
            format: Output format

        Returns:
            DocumentResult
        """
        return self.document_engine.generate_eiag_incentive_summary(data, format)

    def generate_presentation(
        self,
        title: str,
        client_name: str,
        agenda: List[str],
        key_points: List[Dict[str, Any]],
        recommendations: List[str],
        next_steps: List[str],
    ) -> DocumentResult:
        """
        Generate a client presentation.

        Args:
            title: Presentation title
            client_name: Client name
            agenda: Agenda items
            key_points: Key points
            recommendations: Recommendations
            next_steps: Next steps

        Returns:
            DocumentResult
        """
        return self.document_engine.generate_eiag_presentation(
            title=title,
            client_name=client_name,
            agenda=agenda,
            key_points=key_points,
            recommendations=recommendations,
            next_steps=next_steps,
        )

    def generate_letter(
        self,
        recipient_name: str,
        recipient_title: str,
        recipient_company: str,
        subject: str,
        body_paragraphs: List[str],
        sender_name: str,
        sender_title: str,
    ) -> DocumentResult:
        """
        Generate a formal letter.

        Args:
            recipient_name: Recipient name
            recipient_title: Recipient title
            recipient_company: Recipient company
            subject: Letter subject
            body_paragraphs: Body paragraphs
            sender_name: Sender name
            sender_title: Sender title

        Returns:
            DocumentResult
        """
        return self.document_engine.generate_eiag_letter(
            recipient_name=recipient_name,
            recipient_title=recipient_title,
            recipient_company=recipient_company,
            subject=subject,
            body_paragraphs=body_paragraphs,
            sender_name=sender_name,
            sender_title=sender_title,
        )

    def quick_proposal(
        self,
        client: str,
        project: str,
        incentives: List[Dict[str, str]],
        total_value: str,
        format: DocumentFormat = DocumentFormat.PDF,
    ) -> DocumentResult:
        """
        Generate a quick proposal with minimal input.

        Args:
            client: Client name
            project: Project name
            incentives: List of incentives
            total_value: Total value
            format: Output format

        Returns:
            DocumentResult
        """
        return self.document_engine.quick_proposal(
            client=client,
            project=project,
            incentives=incentives,
            total_value=total_value,
            format=format,
        )

    def get_available_templates(self) -> List[Dict[str, Any]]:
        """Get list of available document templates"""
        return self.document_engine.list_templates()

    def format_document_result(self, result: DocumentResult) -> str:
        """
        Format a document result for display to user.

        Args:
            result: DocumentResult

        Returns:
            Formatted string
        """
        if result.success:
            return (
                f"Document generated successfully:\n"
                f"- File: {result.filename}\n"
                f"- Format: {result.format.value.upper()}\n"
                f"- Size: {result.file_size_bytes / 1024:.1f} KB\n"
                f"- Location: {result.file_path}"
            )
        else:
            return f"Document generation failed: {result.error}"


# =============================================================================
# Convenience Functions
# =============================================================================

async def process_document_request(
    message: str,
    context: Optional[Dict[str, Any]] = None,
) -> Optional[DocumentResult]:
    """
    Process a document generation request.

    Args:
        message: User's message
        context: Optional context with additional info

    Returns:
        DocumentResult or None if not a document request
    """
    if not should_generate_document(message):
        return None

    engine = get_document_engine()
    doc_type = detect_document_type(message)
    output_format = detect_output_format(message)

    context = context or {}

    # Handle based on document type
    if doc_type == "incentive_proposal" and context.get("proposal_data"):
        return engine.generate_eiag_proposal(
            context["proposal_data"],
            format=output_format,
        )

    if doc_type == "site_selection" and context.get("site_data"):
        return engine.generate_eiag_site_selection(
            context["site_data"],
            format=output_format,
        )

    # Default: generate simple document
    title = context.get("title", "Generated Document")
    content = context.get("content", message)

    return engine.generate(
        title=title,
        content=content,
        format=output_format,
    )


def extract_document_params(message: str) -> Dict[str, Any]:
    """
    Extract document parameters from a message.

    Args:
        message: User's message

    Returns:
        Dict with extracted parameters
    """
    params = {
        "doc_type": detect_document_type(message),
        "format": detect_output_format(message),
        "should_generate": should_generate_document(message),
    }

    # Try to extract client/project names
    message_lower = message.lower()

    # Look for "for [client]" patterns
    if " for " in message_lower:
        parts = message.split(" for ", 1)
        if len(parts) > 1:
            params["client_hint"] = parts[1].split()[0] if parts[1] else None

    return params
