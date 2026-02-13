"""
=============================================================================
HUMMINGBIRD-LEA - PDF Generation Service
Powered by CoDre-X | B & D Servicing LLC
=============================================================================
PDF document generation using reportlab.

Features:
- Professional PDF layouts
- EIAG-branded documents
- Tables, headers, footers
- Template-based generation
=============================================================================
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import time
import io

from .base import (
    BaseDocumentGenerator,
    DocumentFormat,
    DocumentMetadata,
    DocumentSection,
    DocumentResult,
    ColorScheme,
    FontScheme,
    ProposalData,
    SiteSelectionData,
    IncentiveSummaryData,
)

logger = logging.getLogger(__name__)


# =============================================================================
# PDF Generator
# =============================================================================

class PDFGenerator(BaseDocumentGenerator):
    """
    PDF document generator using reportlab.

    Usage:
        generator = PDFGenerator()

        # Generate from sections
        result = generator.generate(metadata, sections)

        # Generate EIAG proposal PDF
        result = generator.generate_proposal(proposal_data)
    """

    @property
    def format(self) -> DocumentFormat:
        return DocumentFormat.PDF

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        colors: Optional[ColorScheme] = None,
        fonts: Optional[FontScheme] = None,
    ):
        super().__init__(output_dir, colors, fonts)

        # Check for reportlab availability
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate
            self._reportlab_available = True
        except ImportError:
            self._reportlab_available = False
            logger.warning("reportlab not installed. Install with: pip install reportlab")

    def _check_reportlab(self):
        """Check if reportlab is available"""
        if not self._reportlab_available:
            raise ImportError(
                "reportlab is required for PDF generation. "
                "Install with: pip install reportlab"
            )

    def _hex_to_color(self, hex_color: str):
        """Convert hex color to reportlab Color"""
        from reportlab.lib.colors import HexColor
        return HexColor(hex_color)

    def generate(
        self,
        metadata: DocumentMetadata,
        sections: List[DocumentSection],
    ) -> DocumentResult:
        """
        Generate a PDF document from sections.

        Args:
            metadata: Document metadata
            sections: List of document sections

        Returns:
            DocumentResult with file path
        """
        self._check_reportlab()
        start_time = time.time()

        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib.colors import HexColor
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            PageBreak, ListFlowable, ListItem
        )
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

        try:
            # Generate filename and path
            filename = self._generate_filename(metadata)
            output_path = self._get_output_path(filename)

            # Create document
            doc = SimpleDocTemplate(
                str(output_path),
                pagesize=letter,
                rightMargin=0.75*inch,
                leftMargin=0.75*inch,
                topMargin=0.75*inch,
                bottomMargin=0.75*inch,
            )

            # Create styles
            styles = self._create_styles()

            # Build story (content)
            story = []

            # Add title page
            story.extend(self._create_title_page(metadata, styles))
            story.append(PageBreak())

            # Add table of contents
            story.extend(self._create_toc(sections, styles))
            story.append(PageBreak())

            # Add content sections
            for section in sections:
                story.extend(self._create_section(section, styles))
                story.append(Spacer(1, 0.25*inch))

            # Build PDF
            doc.build(
                story,
                onFirstPage=lambda c, d: self._add_header_footer(c, d, metadata, True),
                onLaterPages=lambda c, d: self._add_header_footer(c, d, metadata, False),
            )

            processing_time = (time.time() - start_time) * 1000

            return DocumentResult(
                success=True,
                file_path=output_path,
                format=self.format,
                file_size_bytes=output_path.stat().st_size,
                page_count=len(sections) + 2,  # Approximate
                processing_time_ms=processing_time,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"PDF generation error: {e}")
            return DocumentResult(
                success=False,
                error=str(e),
            )

    def _create_styles(self) -> Dict[str, Any]:
        """Create paragraph styles"""
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
        from reportlab.lib.colors import HexColor

        base_styles = getSampleStyleSheet()

        styles = {
            "title": ParagraphStyle(
                "CustomTitle",
                parent=base_styles["Title"],
                fontSize=28,
                textColor=HexColor(self.colors.primary),
                spaceAfter=12,
                alignment=TA_CENTER,
                fontName="Helvetica-Bold",
            ),
            "subtitle": ParagraphStyle(
                "CustomSubtitle",
                parent=base_styles["Normal"],
                fontSize=18,
                textColor=HexColor(self.colors.secondary),
                spaceAfter=24,
                alignment=TA_CENTER,
                fontName="Helvetica",
            ),
            "heading1": ParagraphStyle(
                "CustomHeading1",
                parent=base_styles["Heading1"],
                fontSize=18,
                textColor=HexColor(self.colors.primary),
                spaceBefore=18,
                spaceAfter=12,
                fontName="Helvetica-Bold",
            ),
            "heading2": ParagraphStyle(
                "CustomHeading2",
                parent=base_styles["Heading2"],
                fontSize=14,
                textColor=HexColor(self.colors.secondary),
                spaceBefore=12,
                spaceAfter=8,
                fontName="Helvetica-Bold",
            ),
            "heading3": ParagraphStyle(
                "CustomHeading3",
                parent=base_styles["Heading3"],
                fontSize=12,
                textColor=HexColor(self.colors.text),
                spaceBefore=8,
                spaceAfter=6,
                fontName="Helvetica-Bold",
            ),
            "body": ParagraphStyle(
                "CustomBody",
                parent=base_styles["Normal"],
                fontSize=11,
                textColor=HexColor(self.colors.text),
                spaceAfter=8,
                alignment=TA_JUSTIFY,
                fontName="Helvetica",
                leading=14,
            ),
            "bullet": ParagraphStyle(
                "CustomBullet",
                parent=base_styles["Normal"],
                fontSize=11,
                textColor=HexColor(self.colors.text),
                leftIndent=20,
                spaceAfter=4,
                fontName="Helvetica",
            ),
            "footer": ParagraphStyle(
                "CustomFooter",
                parent=base_styles["Normal"],
                fontSize=9,
                textColor=HexColor(self.colors.text_light),
                alignment=TA_CENTER,
                fontName="Helvetica",
            ),
            "toc": ParagraphStyle(
                "TOCEntry",
                parent=base_styles["Normal"],
                fontSize=11,
                textColor=HexColor(self.colors.text),
                leftIndent=0,
                spaceAfter=6,
                fontName="Helvetica",
            ),
        }

        return styles

    def _create_title_page(
        self,
        metadata: DocumentMetadata,
        styles: Dict[str, Any],
    ) -> List:
        """Create title page elements"""
        from reportlab.platypus import Paragraph, Spacer
        from reportlab.lib.units import inch

        elements = []

        # Add spacing at top
        elements.append(Spacer(1, 2*inch))

        # Title
        elements.append(Paragraph(metadata.title, styles["title"]))

        # Subtitle (project/client name)
        if metadata.project_name or metadata.client_name:
            subtitle = metadata.project_name or metadata.client_name
            elements.append(Paragraph(subtitle, styles["subtitle"]))

        # Spacing
        elements.append(Spacer(1, 3*inch))

        # Date and author
        date_text = metadata.created_at.strftime("%B %d, %Y")
        elements.append(Paragraph(f"Prepared by: {metadata.author}", styles["footer"]))
        elements.append(Paragraph(date_text, styles["footer"]))
        elements.append(Spacer(1, 0.5*inch))
        elements.append(Paragraph(metadata.company, styles["footer"]))

        return elements

    def _create_toc(
        self,
        sections: List[DocumentSection],
        styles: Dict[str, Any],
    ) -> List:
        """Create table of contents"""
        from reportlab.platypus import Paragraph, Spacer
        from reportlab.lib.units import inch

        elements = []

        elements.append(Paragraph("Table of Contents", styles["heading1"]))
        elements.append(Spacer(1, 0.25*inch))

        for i, section in enumerate(sections, 1):
            toc_entry = f"{i}. {section.title}"
            elements.append(Paragraph(toc_entry, styles["toc"]))

            if section.subsections:
                for j, subsection in enumerate(section.subsections, 1):
                    sub_entry = f"    {i}.{j} {subsection.title}"
                    elements.append(Paragraph(sub_entry, styles["toc"]))

        return elements

    def _create_section(
        self,
        section: DocumentSection,
        styles: Dict[str, Any],
    ) -> List:
        """Create section elements"""
        from reportlab.platypus import Paragraph, Spacer, ListFlowable, ListItem
        from reportlab.lib.units import inch

        elements = []

        # Add heading based on level
        heading_style = styles.get(f"heading{section.level}", styles["heading1"])
        elements.append(Paragraph(section.title, heading_style))

        # Add content
        if section.content:
            elements.append(Paragraph(section.content, styles["body"]))

        # Add bullet items
        if section.items:
            bullet_items = []
            for item in section.items:
                bullet_items.append(
                    ListItem(Paragraph(item, styles["bullet"]))
                )
            elements.append(
                ListFlowable(
                    bullet_items,
                    bulletType='bullet',
                    start='bulletchar',
                )
            )

        # Add table
        if section.table_data:
            table = self._create_table(section.table_data)
            if table:
                elements.append(Spacer(1, 0.1*inch))
                elements.append(table)

        # Process subsections
        if section.subsections:
            for subsection in section.subsections:
                subsection.level = section.level + 1
                elements.extend(self._create_section(subsection, styles))

        return elements

    def _create_table(self, table_data: List[List[str]]):
        """Create a table"""
        from reportlab.platypus import Table, TableStyle
        from reportlab.lib.colors import HexColor, white
        from reportlab.lib.units import inch

        if not table_data:
            return None

        # Create table
        table = Table(table_data, repeatRows=1)

        # Style the table
        style = TableStyle([
            # Header row
            ('BACKGROUND', (0, 0), (-1, 0), HexColor(self.colors.primary)),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('TOPPADDING', (0, 0), (-1, 0), 8),

            # Data rows
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ('TOPPADDING', (0, 1), (-1, -1), 6),

            # Grid
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor(self.colors.text_light)),

            # Alignment
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ])

        # Alternate row colors
        for i in range(1, len(table_data)):
            if i % 2 == 0:
                style.add('BACKGROUND', (0, i), (-1, i), HexColor(self.colors.background_alt))

        table.setStyle(style)
        return table

    def _add_header_footer(self, canvas, doc, metadata: DocumentMetadata, is_first: bool):
        """Add header and footer to page"""
        from reportlab.lib.colors import HexColor
        from reportlab.lib.units import inch

        canvas.saveState()

        # Skip header/footer on title page
        if is_first:
            canvas.restoreState()
            return

        page_width, page_height = doc.pagesize

        # Header - company name
        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(HexColor(self.colors.text_light))
        canvas.drawRightString(
            page_width - 0.75*inch,
            page_height - 0.5*inch,
            metadata.company
        )

        # Header line
        canvas.setStrokeColor(HexColor(self.colors.primary))
        canvas.setLineWidth(1)
        canvas.line(
            0.75*inch,
            page_height - 0.6*inch,
            page_width - 0.75*inch,
            page_height - 0.6*inch
        )

        # Footer - page number and confidential
        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(HexColor(self.colors.text_light))

        # Confidential notice
        canvas.drawString(
            0.75*inch,
            0.5*inch,
            "CONFIDENTIAL"
        )

        # Page number
        page_num = canvas.getPageNumber()
        canvas.drawCentredString(
            page_width / 2,
            0.5*inch,
            f"Page {page_num}"
        )

        # Date
        date_text = metadata.created_at.strftime("%B %d, %Y")
        canvas.drawRightString(
            page_width - 0.75*inch,
            0.5*inch,
            date_text
        )

        canvas.restoreState()

    # =========================================================================
    # EIAG Template Methods
    # =========================================================================

    def generate_proposal(self, data: ProposalData) -> DocumentResult:
        """
        Generate an EIAG incentive proposal PDF.

        Args:
            data: Proposal data

        Returns:
            DocumentResult
        """
        metadata = DocumentMetadata(
            title=f"Economic Incentive Proposal",
            subject=f"Incentive proposal for {data.project_name}",
            client_name=data.client_name,
            project_name=data.project_name,
        )

        sections = [
            DocumentSection(
                title="Executive Summary",
                content=data.executive_summary,
                level=1,
            ),
            DocumentSection(
                title="Project Background",
                content=data.background,
                level=1,
            ),
            DocumentSection(
                title="Proposed Incentives",
                content="The following incentives have been identified for this project:",
                level=1,
                table_data=[
                    ["Incentive Program", "Estimated Value", "Type", "Timeline"],
                    *[
                        [
                            inc.get("name", ""),
                            inc.get("value", ""),
                            inc.get("type", ""),
                            inc.get("timeline", "")
                        ]
                        for inc in data.proposed_incentives
                    ]
                ],
            ),
            DocumentSection(
                title="Financial Summary",
                content="",
                level=1,
                items=[
                    f"Total Estimated Incentive Value: {data.financial_summary.get('total_value', 'TBD')}",
                    f"Required Capital Investment: {data.financial_summary.get('investment', 'TBD')}",
                    f"Net Benefit to Company: {data.financial_summary.get('net_benefit', 'TBD')}",
                    f"Projected ROI: {data.financial_summary.get('roi', 'TBD')}",
                ],
            ),
            DocumentSection(
                title="Implementation Timeline",
                content="",
                level=1,
                table_data=[
                    ["Phase", "Timeline", "Key Milestone"],
                    *[
                        [
                            item.get("phase", ""),
                            item.get("timeline", ""),
                            item.get("milestone", "")
                        ]
                        for item in data.timeline
                    ]
                ],
            ),
            DocumentSection(
                title="Next Steps",
                content="To proceed with this proposal:",
                level=1,
                items=data.next_steps,
            ),
            DocumentSection(
                title="Contact Information",
                content="",
                level=1,
                items=[
                    f"Contact: {data.contact_info.get('name', '')}",
                    f"Email: {data.contact_info.get('email', '')}",
                    f"Phone: {data.contact_info.get('phone', '')}",
                ],
            ),
        ]

        return self.generate(metadata, sections)

    def generate_incentive_summary(self, data: IncentiveSummaryData) -> DocumentResult:
        """
        Generate an incentive summary PDF.

        Args:
            data: Incentive summary data

        Returns:
            DocumentResult
        """
        metadata = DocumentMetadata(
            title=f"Economic Incentive Summary",
            subject=f"Incentive summary for {data.project_name}",
            client_name=data.client_name,
            project_name=data.project_name,
        )

        sections = [
            DocumentSection(
                title="Overview",
                content=f"Total Estimated Incentive Value: {data.total_incentive_value}",
                level=1,
            ),
            DocumentSection(
                title="Incentive Breakdown",
                content="",
                level=1,
                table_data=[
                    ["Program", "Value", "Type", "Duration", "Status"],
                    *[
                        [
                            inc.get("name", ""),
                            inc.get("value", ""),
                            inc.get("type", ""),
                            inc.get("duration", ""),
                            inc.get("status", "")
                        ]
                        for inc in data.incentive_breakdown
                    ]
                ],
            ),
            DocumentSection(
                title="Eligibility Requirements",
                content="",
                level=1,
                items=data.eligibility_requirements,
            ),
            DocumentSection(
                title="Compliance Requirements",
                content="",
                level=1,
                items=data.compliance_requirements,
            ),
        ]

        return self.generate(metadata, sections)

    def generate_executive_summary(
        self,
        title: str,
        client_name: str,
        project_name: str,
        summary_points: List[Dict[str, str]],
        key_metrics: Dict[str, str],
        recommendations: List[str],
    ) -> DocumentResult:
        """
        Generate a one-page executive summary PDF.

        Args:
            title: Summary title
            client_name: Client name
            project_name: Project name
            summary_points: List of dicts with 'heading' and 'content'
            key_metrics: Dict of metric name to value
            recommendations: List of recommendation strings

        Returns:
            DocumentResult
        """
        metadata = DocumentMetadata(
            title=title,
            client_name=client_name,
            project_name=project_name,
        )

        sections = []

        # Add summary points as sections
        for point in summary_points:
            sections.append(DocumentSection(
                title=point.get("heading", ""),
                content=point.get("content", ""),
                level=2,
            ))

        # Add key metrics
        metrics_items = [f"{k}: {v}" for k, v in key_metrics.items()]
        sections.append(DocumentSection(
            title="Key Metrics",
            content="",
            level=1,
            items=metrics_items,
        ))

        # Add recommendations
        sections.append(DocumentSection(
            title="Recommendations",
            content="",
            level=1,
            items=recommendations,
        ))

        return self.generate(metadata, sections)


# =============================================================================
# Factory Function
# =============================================================================

_pdf_generator: Optional[PDFGenerator] = None


def get_pdf_generator() -> PDFGenerator:
    """Get or create the PDF generator singleton"""
    global _pdf_generator
    if _pdf_generator is None:
        _pdf_generator = PDFGenerator()
    return _pdf_generator
