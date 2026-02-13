"""
=============================================================================
HUMMINGBIRD-LEA - Document Generation Engine
Powered by CoDre-X | B & D Servicing LLC
=============================================================================
Unified document generation engine combining all format generators.

Features:
- Multi-format output (PPTX, DOCX, PDF)
- EIAG template support
- AI-assisted content generation
- Batch document generation
=============================================================================
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from enum import Enum

from .base import (
    DocumentFormat,
    DocumentCategory,
    EIAGDocumentType,
    DocumentMetadata,
    DocumentSection,
    DocumentResult,
    ColorScheme,
    ProposalData,
    SiteSelectionData,
    IncentiveSummaryData,
    get_template_manager,
)
from .powerpoint import PowerPointGenerator, get_powerpoint_generator
from .word import WordGenerator, get_word_generator
from .pdf import PDFGenerator, get_pdf_generator

logger = logging.getLogger(__name__)


# =============================================================================
# EIAG Template Helpers
# =============================================================================

class EIAGTemplates:
    """
    Pre-built EIAG document templates.

    Provides easy-to-use methods for generating common EIAG documents.
    """

    @staticmethod
    def create_proposal_data(
        client_name: str,
        project_name: str,
        executive_summary: str,
        background: str,
        incentives: List[Dict[str, str]],
        total_value: str,
        investment: str,
        net_benefit: str,
        roi: str,
        timeline: List[Dict[str, str]],
        next_steps: List[str],
        contact_name: str,
        contact_email: str,
        contact_phone: str,
    ) -> ProposalData:
        """Create proposal data from parameters"""
        return ProposalData(
            client_name=client_name,
            project_name=project_name,
            executive_summary=executive_summary,
            background=background,
            proposed_incentives=incentives,
            financial_summary={
                "total_value": total_value,
                "investment": investment,
                "net_benefit": net_benefit,
                "roi": roi,
            },
            timeline=timeline,
            next_steps=next_steps,
            contact_info={
                "name": contact_name,
                "email": contact_email,
                "phone": contact_phone,
            },
        )

    @staticmethod
    def create_site_selection_data(
        client_name: str,
        project_name: str,
        executive_summary: str,
        sites: List[Dict[str, str]],
        criteria: List[Dict[str, str]],
        comparison_matrix: List[List[str]],
        recommendation: str,
        detailed_analysis: Optional[List[Dict[str, Any]]] = None,
    ) -> SiteSelectionData:
        """Create site selection data from parameters"""
        return SiteSelectionData(
            client_name=client_name,
            project_name=project_name,
            executive_summary=executive_summary,
            sites_evaluated=sites,
            evaluation_criteria=criteria,
            comparison_matrix=comparison_matrix,
            recommendation=recommendation,
            detailed_analysis=detailed_analysis or [],
        )

    @staticmethod
    def create_incentive_summary_data(
        client_name: str,
        project_name: str,
        total_value: str,
        incentives: List[Dict[str, str]],
        eligibility: List[str],
        timeline: List[Dict[str, str]],
        compliance: List[str],
        contact_name: str,
        contact_email: str,
        contact_phone: str,
        caveats: Optional[List[str]] = None,
        assumptions: Optional[List[str]] = None,
    ) -> IncentiveSummaryData:
        """Create incentive summary data from parameters"""
        return IncentiveSummaryData(
            client_name=client_name,
            project_name=project_name,
            total_incentive_value=total_value,
            incentive_breakdown=incentives,
            eligibility_requirements=eligibility,
            application_timeline=timeline,
            compliance_requirements=compliance,
            contact_info={
                "name": contact_name,
                "email": contact_email,
                "phone": contact_phone,
            },
            caveats=caveats,
            assumptions=assumptions,
        )


# =============================================================================
# Document Generation Engine
# =============================================================================

@dataclass
class GenerationRequest:
    """Request for document generation"""
    title: str
    content: Union[str, List[DocumentSection], Dict[str, Any]]
    format: DocumentFormat = DocumentFormat.PDF
    template: Optional[str] = None
    client_name: Optional[str] = None
    project_name: Optional[str] = None
    category: DocumentCategory = DocumentCategory.REPORT


@dataclass
class BatchResult:
    """Result from batch document generation"""
    total: int
    successful: int
    failed: int
    results: List[DocumentResult]
    processing_time_ms: float


class DocumentEngine:
    """
    Unified document generation engine.

    Provides a single interface for generating documents in any format
    with support for EIAG templates.

    Usage:
        engine = DocumentEngine()

        # Generate a simple document
        result = engine.generate(
            title="My Report",
            content="Report content here...",
            format=DocumentFormat.PDF
        )

        # Generate from EIAG template
        result = engine.generate_eiag_proposal(
            client_name="Acme Corp",
            project_name="Manufacturing Expansion",
            ...
        )

        # Generate in multiple formats
        results = engine.generate_multi_format(
            title="Report",
            content=sections,
            formats=[DocumentFormat.PDF, DocumentFormat.DOCX]
        )
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        colors: Optional[ColorScheme] = None,
    ):
        """
        Initialize the document engine.

        Args:
            output_dir: Directory for output files
            colors: Color scheme for documents
        """
        self.output_dir = output_dir or Path("data/documents")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.colors = colors or ColorScheme.eiag_brand()

        # Initialize generators lazily
        self._pptx_generator = None
        self._word_generator = None
        self._pdf_generator = None

        # Template manager
        self.templates = get_template_manager()

        logger.info("DocumentEngine initialized")

    @property
    def pptx(self) -> PowerPointGenerator:
        """Get PowerPoint generator"""
        if self._pptx_generator is None:
            self._pptx_generator = PowerPointGenerator(
                output_dir=self.output_dir,
                colors=self.colors,
            )
        return self._pptx_generator

    @property
    def word(self) -> WordGenerator:
        """Get Word generator"""
        if self._word_generator is None:
            self._word_generator = WordGenerator(
                output_dir=self.output_dir,
                colors=self.colors,
            )
        return self._word_generator

    @property
    def pdf(self) -> PDFGenerator:
        """Get PDF generator"""
        if self._pdf_generator is None:
            self._pdf_generator = PDFGenerator(
                output_dir=self.output_dir,
                colors=self.colors,
            )
        return self._pdf_generator

    def _get_generator(self, format: DocumentFormat):
        """Get the appropriate generator for a format"""
        if format == DocumentFormat.PPTX:
            return self.pptx
        elif format == DocumentFormat.DOCX:
            return self.word
        elif format == DocumentFormat.PDF:
            return self.pdf
        else:
            raise ValueError(f"Unsupported format: {format}")

    def generate(
        self,
        title: str,
        content: Union[str, List[DocumentSection]],
        format: DocumentFormat = DocumentFormat.PDF,
        client_name: Optional[str] = None,
        project_name: Optional[str] = None,
        category: DocumentCategory = DocumentCategory.REPORT,
    ) -> DocumentResult:
        """
        Generate a document.

        Args:
            title: Document title
            content: String content or list of sections
            format: Output format
            client_name: Optional client name
            project_name: Optional project name
            category: Document category

        Returns:
            DocumentResult
        """
        # Create metadata
        metadata = DocumentMetadata(
            title=title,
            client_name=client_name,
            project_name=project_name,
            category=category,
        )

        # Convert string content to sections if needed
        if isinstance(content, str):
            sections = [
                DocumentSection(
                    title="Content",
                    content=content,
                    level=1,
                )
            ]
        else:
            sections = content

        # Generate document
        generator = self._get_generator(format)
        return generator.generate(metadata, sections)

    def generate_multi_format(
        self,
        title: str,
        content: Union[str, List[DocumentSection]],
        formats: List[DocumentFormat],
        **kwargs,
    ) -> List[DocumentResult]:
        """
        Generate a document in multiple formats.

        Args:
            title: Document title
            content: Content or sections
            formats: List of formats to generate
            **kwargs: Additional arguments for generate()

        Returns:
            List of DocumentResult
        """
        results = []
        for format in formats:
            result = self.generate(title, content, format, **kwargs)
            results.append(result)
        return results

    # =========================================================================
    # EIAG Document Generation
    # =========================================================================

    def generate_eiag_proposal(
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
        generator = self._get_generator(format)
        return generator.generate_proposal(data)

    def generate_eiag_site_selection(
        self,
        data: SiteSelectionData,
        format: DocumentFormat = DocumentFormat.PDF,
    ) -> DocumentResult:
        """
        Generate an EIAG site selection report.

        Args:
            data: Site selection data
            format: Output format

        Returns:
            DocumentResult
        """
        generator = self._get_generator(format)
        return generator.generate_site_selection(data)

    def generate_eiag_incentive_summary(
        self,
        data: IncentiveSummaryData,
        format: DocumentFormat = DocumentFormat.PDF,
    ) -> DocumentResult:
        """
        Generate an EIAG incentive summary.

        Args:
            data: Incentive summary data
            format: Output format

        Returns:
            DocumentResult
        """
        generator = self._get_generator(format)
        return generator.generate_incentive_summary(data)

    def generate_eiag_presentation(
        self,
        title: str,
        client_name: str,
        agenda: List[str],
        key_points: List[Dict[str, Any]],
        recommendations: List[str],
        next_steps: List[str],
    ) -> DocumentResult:
        """
        Generate an EIAG client presentation.

        Args:
            title: Presentation title
            client_name: Client name
            agenda: Agenda items
            key_points: Key points with content
            recommendations: Recommendations
            next_steps: Next steps

        Returns:
            DocumentResult
        """
        return self.pptx.generate_client_presentation(
            title=title,
            client_name=client_name,
            agenda=agenda,
            key_points=key_points,
            recommendations=recommendations,
            next_steps=next_steps,
        )

    def generate_eiag_letter(
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
        Generate a formal EIAG letter.

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
        return self.word.generate_letter(
            recipient_name=recipient_name,
            recipient_title=recipient_title,
            recipient_company=recipient_company,
            subject=subject,
            body_paragraphs=body_paragraphs,
            sender_name=sender_name,
            sender_title=sender_title,
        )

    def generate_eiag_executive_summary(
        self,
        title: str,
        client_name: str,
        project_name: str,
        summary_points: List[Dict[str, str]],
        key_metrics: Dict[str, str],
        recommendations: List[str],
    ) -> DocumentResult:
        """
        Generate a one-page executive summary.

        Args:
            title: Summary title
            client_name: Client name
            project_name: Project name
            summary_points: Summary points
            key_metrics: Key metrics
            recommendations: Recommendations

        Returns:
            DocumentResult
        """
        return self.pdf.generate_executive_summary(
            title=title,
            client_name=client_name,
            project_name=project_name,
            summary_points=summary_points,
            key_metrics=key_metrics,
            recommendations=recommendations,
        )

    # =========================================================================
    # Quick Generation Methods
    # =========================================================================

    def quick_proposal(
        self,
        client: str,
        project: str,
        incentives: List[Dict[str, str]],
        total_value: str,
        format: DocumentFormat = DocumentFormat.PDF,
    ) -> DocumentResult:
        """
        Quick proposal generation with minimal input.

        Args:
            client: Client name
            project: Project name
            incentives: List of incentives
            total_value: Total incentive value
            format: Output format

        Returns:
            DocumentResult
        """
        data = ProposalData(
            client_name=client,
            project_name=project,
            executive_summary=f"This proposal outlines economic incentive opportunities for {project}.",
            background=f"{client} is evaluating options for {project}.",
            proposed_incentives=incentives,
            financial_summary={"total_value": total_value},
            timeline=[],
            next_steps=["Review proposal", "Schedule follow-up meeting"],
            contact_info={"name": "EIAG Team", "email": "info@eiag.com", "phone": ""},
        )
        return self.generate_eiag_proposal(data, format)

    def quick_site_comparison(
        self,
        client: str,
        project: str,
        sites: List[str],
        recommendation: str,
        format: DocumentFormat = DocumentFormat.PDF,
    ) -> DocumentResult:
        """
        Quick site comparison with minimal input.

        Args:
            client: Client name
            project: Project name
            sites: List of site names
            recommendation: Recommended site
            format: Output format

        Returns:
            DocumentResult
        """
        sites_data = [{"name": s, "location": "", "size": "", "features": ""} for s in sites]

        data = SiteSelectionData(
            client_name=client,
            project_name=project,
            executive_summary=f"Analysis of {len(sites)} sites for {project}.",
            sites_evaluated=sites_data,
            evaluation_criteria=[],
            comparison_matrix=[["Site"] + sites],
            recommendation=recommendation,
            detailed_analysis=[],
        )
        return self.generate_eiag_site_selection(data, format)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def list_templates(self) -> List[Dict[str, Any]]:
        """List all available templates"""
        return self.templates.list_templates()

    def get_output_files(self) -> List[Path]:
        """Get list of generated output files"""
        return list(self.output_dir.glob("*.*"))

    def cleanup_old_files(self, days: int = 7) -> int:
        """
        Clean up files older than specified days.

        Args:
            days: Delete files older than this many days

        Returns:
            Number of files deleted
        """
        import time
        cutoff = time.time() - (days * 24 * 60 * 60)
        deleted = 0

        for file in self.output_dir.glob("*.*"):
            if file.stat().st_mtime < cutoff:
                try:
                    file.unlink()
                    deleted += 1
                except Exception as e:
                    logger.warning(f"Failed to delete {file}: {e}")

        logger.info(f"Cleaned up {deleted} old files")
        return deleted


# =============================================================================
# Factory Function
# =============================================================================

_document_engine: Optional[DocumentEngine] = None


def get_document_engine() -> DocumentEngine:
    """Get or create the document engine singleton"""
    global _document_engine
    if _document_engine is None:
        _document_engine = DocumentEngine()
    return _document_engine


# =============================================================================
# Convenience Functions
# =============================================================================

def generate_document(
    title: str,
    content: Union[str, List[DocumentSection]],
    format: DocumentFormat = DocumentFormat.PDF,
    **kwargs,
) -> DocumentResult:
    """
    Convenience function to generate a document.

    Args:
        title: Document title
        content: Content or sections
        format: Output format
        **kwargs: Additional arguments

    Returns:
        DocumentResult
    """
    engine = get_document_engine()
    return engine.generate(title, content, format, **kwargs)


def generate_proposal(
    client_name: str,
    project_name: str,
    incentives: List[Dict[str, str]],
    total_value: str,
    format: DocumentFormat = DocumentFormat.PDF,
) -> DocumentResult:
    """
    Convenience function to generate a quick proposal.

    Args:
        client_name: Client name
        project_name: Project name
        incentives: List of incentives
        total_value: Total value
        format: Output format

    Returns:
        DocumentResult
    """
    engine = get_document_engine()
    return engine.quick_proposal(client_name, project_name, incentives, total_value, format)
