"""
=============================================================================
HUMMINGBIRD-LEA - Document Generation Base
Powered by CoDre-X | B & D Servicing LLC
=============================================================================
Base classes and utilities for document generation.

Features:
- Common document metadata
- Template management
- Output file handling
- Style definitions
=============================================================================
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class DocumentFormat(Enum):
    """Supported document formats"""
    PPTX = "pptx"       # PowerPoint
    DOCX = "docx"       # Word
    PDF = "pdf"         # PDF
    XLSX = "xlsx"       # Excel (future)


class DocumentCategory(Enum):
    """Categories of documents"""
    PROPOSAL = "proposal"
    REPORT = "report"
    PRESENTATION = "presentation"
    LETTER = "letter"
    MEMO = "memo"
    SUMMARY = "summary"
    ANALYSIS = "analysis"
    TEMPLATE = "template"


class EIAGDocumentType(Enum):
    """EIAG-specific document types"""
    INCENTIVE_PROPOSAL = "incentive_proposal"
    SITE_SELECTION_REPORT = "site_selection_report"
    TAX_CREDIT_SUMMARY = "tax_credit_summary"
    ECONOMIC_IMPACT_ANALYSIS = "economic_impact_analysis"
    INCENTIVE_COMPARISON = "incentive_comparison"
    CLIENT_PRESENTATION = "client_presentation"
    EXECUTIVE_SUMMARY = "executive_summary"


# =============================================================================
# Style Definitions
# =============================================================================

@dataclass
class ColorScheme:
    """Color scheme for documents"""
    primary: str = "#1a365d"        # Dark blue
    secondary: str = "#2b6cb0"      # Medium blue
    accent: str = "#48bb78"         # Green
    text: str = "#2d3748"           # Dark gray
    text_light: str = "#718096"     # Light gray
    background: str = "#ffffff"     # White
    background_alt: str = "#f7fafc" # Light gray bg

    # EIAG Brand colors
    @classmethod
    def eiag_brand(cls) -> "ColorScheme":
        """Get EIAG brand color scheme"""
        return cls(
            primary="#1a365d",      # Navy blue
            secondary="#2b6cb0",    # Royal blue
            accent="#38a169",       # Forest green
            text="#1a202c",         # Near black
            text_light="#4a5568",   # Gray
            background="#ffffff",
            background_alt="#edf2f7",
        )


@dataclass
class FontScheme:
    """Font scheme for documents"""
    heading: str = "Calibri"
    body: str = "Calibri"
    heading_size: int = 24
    subheading_size: int = 18
    body_size: int = 11
    small_size: int = 9


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DocumentMetadata:
    """Metadata for generated documents"""
    title: str
    author: str = "Hummingbird-LEA"
    company: str = "B & D Servicing LLC"
    subject: str = ""
    keywords: List[str] = field(default_factory=list)
    category: DocumentCategory = DocumentCategory.REPORT
    created_at: datetime = field(default_factory=datetime.utcnow)
    version: str = "1.0"

    # EIAG-specific fields
    client_name: Optional[str] = None
    project_name: Optional[str] = None
    eiag_type: Optional[EIAGDocumentType] = None


@dataclass
class DocumentSection:
    """A section within a document"""
    title: str
    content: str
    level: int = 1                  # Heading level (1-3)
    items: Optional[List[str]] = None  # Bullet points
    table_data: Optional[List[List[str]]] = None  # Table rows
    subsections: Optional[List["DocumentSection"]] = None


@dataclass
class DocumentResult:
    """Result from document generation"""
    success: bool
    file_path: Optional[Path] = None
    format: Optional[DocumentFormat] = None
    file_size_bytes: int = 0
    page_count: int = 0
    processing_time_ms: float = 0
    error: Optional[str] = None
    metadata: Optional[DocumentMetadata] = None

    @property
    def filename(self) -> str:
        """Get the filename"""
        return self.file_path.name if self.file_path else ""


# =============================================================================
# Template Data Classes
# =============================================================================

@dataclass
class ProposalData:
    """Data for generating a proposal document"""
    client_name: str
    project_name: str
    executive_summary: str
    background: str
    proposed_incentives: List[Dict[str, Any]]
    financial_summary: Dict[str, Any]
    timeline: List[Dict[str, str]]
    next_steps: List[str]
    contact_info: Dict[str, str]

    # Optional fields
    introduction: Optional[str] = None
    methodology: Optional[str] = None
    appendices: Optional[List[Dict[str, str]]] = None


@dataclass
class SiteSelectionData:
    """Data for site selection report"""
    client_name: str
    project_name: str
    executive_summary: str
    sites_evaluated: List[Dict[str, Any]]
    evaluation_criteria: List[Dict[str, Any]]
    comparison_matrix: List[List[str]]
    recommendation: str
    detailed_analysis: List[Dict[str, Any]]

    # Optional fields
    market_overview: Optional[str] = None
    risk_assessment: Optional[str] = None


@dataclass
class IncentiveSummaryData:
    """Data for tax credit/incentive summary"""
    client_name: str
    project_name: str
    total_incentive_value: str
    incentive_breakdown: List[Dict[str, Any]]
    eligibility_requirements: List[str]
    application_timeline: List[Dict[str, str]]
    compliance_requirements: List[str]
    contact_info: Dict[str, str]

    # Optional fields
    caveats: Optional[List[str]] = None
    assumptions: Optional[List[str]] = None


# =============================================================================
# Base Document Generator
# =============================================================================

class BaseDocumentGenerator(ABC):
    """
    Abstract base class for document generators.

    Subclasses implement specific format generation (PPTX, DOCX, PDF).
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        colors: Optional[ColorScheme] = None,
        fonts: Optional[FontScheme] = None,
    ):
        """
        Initialize the document generator.

        Args:
            output_dir: Directory for output files
            colors: Color scheme to use
            fonts: Font scheme to use
        """
        self.output_dir = output_dir or Path("data/documents")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.colors = colors or ColorScheme.eiag_brand()
        self.fonts = fonts or FontScheme()

    @property
    @abstractmethod
    def format(self) -> DocumentFormat:
        """Get the document format this generator produces"""
        pass

    @abstractmethod
    def generate(
        self,
        metadata: DocumentMetadata,
        sections: List[DocumentSection],
    ) -> DocumentResult:
        """
        Generate a document from sections.

        Args:
            metadata: Document metadata
            sections: List of document sections

        Returns:
            DocumentResult with file path and status
        """
        pass

    def _generate_filename(
        self,
        metadata: DocumentMetadata,
        suffix: Optional[str] = None,
    ) -> str:
        """Generate a filename from metadata"""
        # Clean the title for filename
        title = metadata.title.lower()
        title = "".join(c if c.isalnum() or c == " " else "" for c in title)
        title = title.replace(" ", "_")[:50]

        # Add timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Build filename
        parts = [title, timestamp]
        if suffix:
            parts.append(suffix)

        return f"{'_'.join(parts)}.{self.format.value}"

    def _get_output_path(self, filename: str) -> Path:
        """Get full output path for a file"""
        return self.output_dir / filename


# =============================================================================
# Template Manager
# =============================================================================

class TemplateManager:
    """
    Manages document templates.

    Provides access to predefined EIAG templates and custom templates.
    """

    def __init__(self, template_dir: Optional[Path] = None):
        """
        Initialize the template manager.

        Args:
            template_dir: Directory containing templates
        """
        self.template_dir = template_dir or Path("config/templates")
        self._templates: Dict[str, Dict[str, Any]] = {}

        # Register built-in templates
        self._register_builtin_templates()

    def _register_builtin_templates(self):
        """Register built-in EIAG templates"""
        self._templates["incentive_proposal"] = {
            "name": "Incentive Proposal",
            "description": "Standard EIAG incentive proposal template",
            "category": DocumentCategory.PROPOSAL,
            "eiag_type": EIAGDocumentType.INCENTIVE_PROPOSAL,
            "sections": [
                "Executive Summary",
                "Project Background",
                "Proposed Incentives",
                "Financial Summary",
                "Implementation Timeline",
                "Next Steps",
            ],
        }

        self._templates["site_selection"] = {
            "name": "Site Selection Report",
            "description": "Site evaluation and recommendation report",
            "category": DocumentCategory.REPORT,
            "eiag_type": EIAGDocumentType.SITE_SELECTION_REPORT,
            "sections": [
                "Executive Summary",
                "Evaluation Criteria",
                "Sites Evaluated",
                "Comparison Matrix",
                "Detailed Analysis",
                "Recommendation",
            ],
        }

        self._templates["tax_credit_summary"] = {
            "name": "Tax Credit Summary",
            "description": "Summary of available tax credits and incentives",
            "category": DocumentCategory.SUMMARY,
            "eiag_type": EIAGDocumentType.TAX_CREDIT_SUMMARY,
            "sections": [
                "Overview",
                "Incentive Breakdown",
                "Eligibility Requirements",
                "Application Process",
                "Compliance Requirements",
            ],
        }

        self._templates["economic_impact"] = {
            "name": "Economic Impact Analysis",
            "description": "Analysis of economic impact for a project",
            "category": DocumentCategory.ANALYSIS,
            "eiag_type": EIAGDocumentType.ECONOMIC_IMPACT_ANALYSIS,
            "sections": [
                "Executive Summary",
                "Methodology",
                "Direct Impacts",
                "Indirect Impacts",
                "Fiscal Impact",
                "Conclusions",
            ],
        }

        self._templates["client_presentation"] = {
            "name": "Client Presentation",
            "description": "PowerPoint presentation for clients",
            "category": DocumentCategory.PRESENTATION,
            "eiag_type": EIAGDocumentType.CLIENT_PRESENTATION,
            "sections": [
                "Title Slide",
                "Agenda",
                "Executive Summary",
                "Key Findings",
                "Recommendations",
                "Next Steps",
                "Q&A",
            ],
        }

    def get_template(self, template_name: str) -> Optional[Dict[str, Any]]:
        """Get a template by name"""
        return self._templates.get(template_name)

    def list_templates(self) -> List[Dict[str, Any]]:
        """List all available templates"""
        return [
            {"name": name, **template}
            for name, template in self._templates.items()
        ]

    def get_template_sections(self, template_name: str) -> List[str]:
        """Get the sections for a template"""
        template = self._templates.get(template_name)
        return template.get("sections", []) if template else []


# =============================================================================
# Factory Functions
# =============================================================================

_template_manager: Optional[TemplateManager] = None


def get_template_manager() -> TemplateManager:
    """Get or create the template manager singleton"""
    global _template_manager
    if _template_manager is None:
        _template_manager = TemplateManager()
    return _template_manager


# =============================================================================
# Utility Functions
# =============================================================================

def sanitize_filename(name: str) -> str:
    """Sanitize a string for use as a filename"""
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, "")

    # Replace spaces with underscores
    name = name.replace(" ", "_")

    # Limit length
    return name[:100]


def format_currency(amount: Union[int, float], currency: str = "$") -> str:
    """Format a number as currency"""
    if isinstance(amount, float):
        return f"{currency}{amount:,.2f}"
    return f"{currency}{amount:,}"


def format_percentage(value: Union[int, float]) -> str:
    """Format a number as percentage"""
    if isinstance(value, float):
        return f"{value:.1f}%"
    return f"{value}%"
