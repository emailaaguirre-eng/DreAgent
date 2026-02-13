"""
=============================================================================
HUMMINGBIRD-LEA - Phase 5 Document Generation Tests
Powered by CoDre-X | B & D Servicing LLC
=============================================================================
Tests for Phase 5 Document Generation components:
- Base document utilities
- PowerPoint generation
- Word document generation
- PDF generation
- EIAG templates
- Document engine
- Agent mixin
=============================================================================
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime

# Import Phase 5 components
from core.services.documents import (
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
    # Generators
    PowerPointGenerator,
    WordGenerator,
    PDFGenerator,
    get_powerpoint_generator,
    get_word_generator,
    get_pdf_generator,
    # Engine
    DocumentEngine,
    EIAGTemplates,
    get_document_engine,
    generate_document,
    generate_proposal,
    # Agent mixin
    DocumentMixin,
    should_generate_document,
    detect_document_type,
    detect_output_format,
)


# =============================================================================
# Enum Tests
# =============================================================================

class TestEnums:
    """Tests for document enums"""

    def test_document_format_values(self):
        """Test DocumentFormat enum values"""
        assert DocumentFormat.PPTX.value == "pptx"
        assert DocumentFormat.DOCX.value == "docx"
        assert DocumentFormat.PDF.value == "pdf"

    def test_document_category_values(self):
        """Test DocumentCategory enum values"""
        assert DocumentCategory.PROPOSAL.value == "proposal"
        assert DocumentCategory.REPORT.value == "report"
        assert DocumentCategory.PRESENTATION.value == "presentation"

    def test_eiag_document_type_values(self):
        """Test EIAGDocumentType enum values"""
        assert EIAGDocumentType.INCENTIVE_PROPOSAL.value == "incentive_proposal"
        assert EIAGDocumentType.SITE_SELECTION_REPORT.value == "site_selection_report"
        assert EIAGDocumentType.TAX_CREDIT_SUMMARY.value == "tax_credit_summary"


# =============================================================================
# Style Tests
# =============================================================================

class TestStyles:
    """Tests for document styles"""

    def test_color_scheme_defaults(self):
        """Test ColorScheme default values"""
        colors = ColorScheme()
        assert colors.primary.startswith("#")
        assert colors.secondary.startswith("#")
        assert colors.text.startswith("#")

    def test_color_scheme_eiag_brand(self):
        """Test EIAG brand color scheme"""
        colors = ColorScheme.eiag_brand()
        assert colors.primary == "#1a365d"  # Navy blue
        assert colors.accent == "#38a169"   # Forest green

    def test_font_scheme_defaults(self):
        """Test FontScheme default values"""
        fonts = FontScheme()
        assert fonts.heading == "Calibri"
        assert fonts.body == "Calibri"
        assert fonts.body_size == 11


# =============================================================================
# Data Class Tests
# =============================================================================

class TestDataClasses:
    """Tests for document data classes"""

    def test_document_metadata_defaults(self):
        """Test DocumentMetadata defaults"""
        metadata = DocumentMetadata(title="Test Document")
        assert metadata.title == "Test Document"
        assert metadata.author == "Hummingbird-LEA"
        assert metadata.company == "B & D Servicing LLC"
        assert metadata.category == DocumentCategory.REPORT

    def test_document_metadata_with_eiag(self):
        """Test DocumentMetadata with EIAG fields"""
        metadata = DocumentMetadata(
            title="Proposal",
            client_name="Acme Corp",
            project_name="Expansion",
            eiag_type=EIAGDocumentType.INCENTIVE_PROPOSAL,
        )
        assert metadata.client_name == "Acme Corp"
        assert metadata.project_name == "Expansion"
        assert metadata.eiag_type == EIAGDocumentType.INCENTIVE_PROPOSAL

    def test_document_section_basic(self):
        """Test DocumentSection creation"""
        section = DocumentSection(
            title="Introduction",
            content="This is the introduction.",
            level=1,
        )
        assert section.title == "Introduction"
        assert section.content == "This is the introduction."
        assert section.level == 1

    def test_document_section_with_items(self):
        """Test DocumentSection with bullet items"""
        section = DocumentSection(
            title="Key Points",
            content="",
            items=["Point 1", "Point 2", "Point 3"],
        )
        assert len(section.items) == 3
        assert "Point 1" in section.items

    def test_document_section_with_table(self):
        """Test DocumentSection with table data"""
        section = DocumentSection(
            title="Data",
            content="",
            table_data=[
                ["Header 1", "Header 2"],
                ["Data 1", "Data 2"],
            ],
        )
        assert len(section.table_data) == 2
        assert section.table_data[0][0] == "Header 1"

    def test_document_result_success(self):
        """Test DocumentResult success case"""
        result = DocumentResult(
            success=True,
            file_path=Path("/tmp/test.pdf"),
            format=DocumentFormat.PDF,
            file_size_bytes=1024,
            page_count=5,
        )
        assert result.success is True
        assert result.filename == "test.pdf"

    def test_document_result_failure(self):
        """Test DocumentResult failure case"""
        result = DocumentResult(
            success=False,
            error="Generation failed",
        )
        assert result.success is False
        assert result.error == "Generation failed"


# =============================================================================
# EIAG Data Class Tests
# =============================================================================

class TestEIAGDataClasses:
    """Tests for EIAG-specific data classes"""

    def test_proposal_data(self):
        """Test ProposalData creation"""
        data = ProposalData(
            client_name="Acme Corp",
            project_name="Manufacturing Expansion",
            executive_summary="Summary here",
            background="Background info",
            proposed_incentives=[
                {"name": "Tax Credit", "value": "$1M", "type": "State", "timeline": "2024"}
            ],
            financial_summary={"total_value": "$5M"},
            timeline=[{"phase": "Phase 1", "timeline": "Q1 2024", "milestone": "Start"}],
            next_steps=["Step 1", "Step 2"],
            contact_info={"name": "John", "email": "john@test.com", "phone": "123"},
        )
        assert data.client_name == "Acme Corp"
        assert len(data.proposed_incentives) == 1
        assert len(data.next_steps) == 2

    def test_site_selection_data(self):
        """Test SiteSelectionData creation"""
        data = SiteSelectionData(
            client_name="Acme Corp",
            project_name="New Facility",
            executive_summary="Summary",
            sites_evaluated=[
                {"name": "Site A", "location": "City A", "size": "100 acres", "features": "Features"}
            ],
            evaluation_criteria=[
                {"name": "Cost", "weight": "30%", "description": "Total cost"}
            ],
            comparison_matrix=[["Site", "Score"], ["Site A", "85"]],
            recommendation="Site A is recommended",
        )
        assert data.client_name == "Acme Corp"
        assert len(data.sites_evaluated) == 1

    def test_incentive_summary_data(self):
        """Test IncentiveSummaryData creation"""
        data = IncentiveSummaryData(
            client_name="Acme Corp",
            project_name="Project X",
            total_incentive_value="$10M",
            incentive_breakdown=[
                {"name": "Credit A", "value": "$5M", "type": "Tax", "duration": "5 years", "status": "Available"}
            ],
            eligibility_requirements=["Req 1", "Req 2"],
            application_timeline=[{"step": "Apply", "timeline": "Q1", "action": "Submit"}],
            compliance_requirements=["Compliance 1"],
            contact_info={"name": "Jane", "email": "jane@test.com", "phone": "456"},
        )
        assert data.total_incentive_value == "$10M"
        assert len(data.eligibility_requirements) == 2


# =============================================================================
# Utility Function Tests
# =============================================================================

class TestUtilities:
    """Tests for utility functions"""

    def test_sanitize_filename(self):
        """Test filename sanitization"""
        assert sanitize_filename("test file") == "test_file"
        assert sanitize_filename("test/file:name") == "testfilename"
        assert len(sanitize_filename("a" * 200)) <= 100

    def test_format_currency(self):
        """Test currency formatting"""
        assert format_currency(1000) == "$1,000"
        assert format_currency(1234.56) == "$1,234.56"
        assert format_currency(1000000, "€") == "€1,000,000"

    def test_format_percentage(self):
        """Test percentage formatting"""
        assert format_percentage(50) == "50%"
        assert format_percentage(33.33) == "33.3%"


# =============================================================================
# Template Manager Tests
# =============================================================================

class TestTemplateManager:
    """Tests for template manager"""

    def test_template_manager_singleton(self):
        """Test template manager singleton"""
        manager1 = get_template_manager()
        manager2 = get_template_manager()
        assert manager1 is manager2

    def test_list_templates(self):
        """Test listing templates"""
        manager = get_template_manager()
        templates = manager.list_templates()
        assert len(templates) >= 4
        template_names = [t["name"] for t in templates]
        assert "Incentive Proposal" in template_names or "incentive_proposal" in [t.get("name", "") for t in templates]

    def test_get_template(self):
        """Test getting a specific template"""
        manager = get_template_manager()
        template = manager.get_template("incentive_proposal")
        assert template is not None
        assert "sections" in template

    def test_get_template_sections(self):
        """Test getting template sections"""
        manager = get_template_manager()
        sections = manager.get_template_sections("incentive_proposal")
        assert len(sections) > 0
        assert "Executive Summary" in sections


# =============================================================================
# Intent Detection Tests
# =============================================================================

class TestIntentDetection:
    """Tests for document generation intent detection"""

    def test_should_generate_document_positive(self):
        """Test positive document generation detection"""
        assert should_generate_document("Create a proposal for Acme Corp") is True
        assert should_generate_document("Generate a PDF report") is True
        assert should_generate_document("Make a PowerPoint presentation") is True
        assert should_generate_document("Draft a letter to the client") is True

    def test_should_generate_document_negative(self):
        """Test negative document generation detection"""
        assert should_generate_document("Hello") is False
        assert should_generate_document("What is the weather?") is False
        assert should_generate_document("Tell me about tax credits") is False

    def test_detect_document_type(self):
        """Test document type detection"""
        assert detect_document_type("Create an incentive proposal") == "incentive_proposal"
        assert detect_document_type("Generate a site selection report") == "site_selection"
        assert detect_document_type("Make a presentation") == "presentation"
        assert detect_document_type("Write a letter") == "letter"
        assert detect_document_type("Create a report") == "report"

    def test_detect_output_format(self):
        """Test output format detection"""
        assert detect_output_format("Generate a PowerPoint") == DocumentFormat.PPTX
        assert detect_output_format("Create a Word document") == DocumentFormat.DOCX
        assert detect_output_format("Make a PDF") == DocumentFormat.PDF
        assert detect_output_format("Create a document") == DocumentFormat.PDF  # Default


# =============================================================================
# Generator Singleton Tests
# =============================================================================

class TestGeneratorSingletons:
    """Tests for generator singleton patterns"""

    def test_powerpoint_generator_singleton(self):
        """Test PowerPoint generator singleton"""
        gen1 = get_powerpoint_generator()
        gen2 = get_powerpoint_generator()
        assert gen1 is gen2
        assert gen1.format == DocumentFormat.PPTX

    def test_word_generator_singleton(self):
        """Test Word generator singleton"""
        gen1 = get_word_generator()
        gen2 = get_word_generator()
        assert gen1 is gen2
        assert gen1.format == DocumentFormat.DOCX

    def test_pdf_generator_singleton(self):
        """Test PDF generator singleton"""
        gen1 = get_pdf_generator()
        gen2 = get_pdf_generator()
        assert gen1 is gen2
        assert gen1.format == DocumentFormat.PDF

    def test_document_engine_singleton(self):
        """Test document engine singleton"""
        engine1 = get_document_engine()
        engine2 = get_document_engine()
        assert engine1 is engine2


# =============================================================================
# EIAG Templates Helper Tests
# =============================================================================

class TestEIAGTemplates:
    """Tests for EIAG template helpers"""

    def test_create_proposal_data(self):
        """Test creating proposal data from helper"""
        data = EIAGTemplates.create_proposal_data(
            client_name="Acme",
            project_name="Project",
            executive_summary="Summary",
            background="Background",
            incentives=[{"name": "Credit", "value": "$1M", "type": "Tax", "timeline": "2024"}],
            total_value="$5M",
            investment="$10M",
            net_benefit="$15M",
            roi="150%",
            timeline=[{"phase": "1", "timeline": "Q1", "milestone": "Start"}],
            next_steps=["Step 1"],
            contact_name="John",
            contact_email="john@test.com",
            contact_phone="123",
        )
        assert isinstance(data, ProposalData)
        assert data.client_name == "Acme"
        assert data.financial_summary["total_value"] == "$5M"


# =============================================================================
# Integration Tests (require libraries)
# =============================================================================

class TestDocumentGeneration:
    """Integration tests for document generation"""

    @pytest.fixture
    def temp_output_dir(self, tmp_path):
        """Create a temporary output directory"""
        output_dir = tmp_path / "documents"
        output_dir.mkdir()
        return output_dir

    def test_engine_initialization(self, temp_output_dir):
        """Test engine initialization"""
        engine = DocumentEngine(output_dir=temp_output_dir)
        assert engine.output_dir == temp_output_dir
        assert engine.colors is not None

    def test_engine_list_templates(self, temp_output_dir):
        """Test listing templates from engine"""
        engine = DocumentEngine(output_dir=temp_output_dir)
        templates = engine.list_templates()
        assert len(templates) >= 4

    @pytest.mark.skipif(True, reason="Requires python-docx")
    def test_word_generation(self, temp_output_dir):
        """Test Word document generation"""
        engine = DocumentEngine(output_dir=temp_output_dir)

        sections = [
            DocumentSection(title="Introduction", content="Test content"),
            DocumentSection(title="Details", content="More content", items=["Item 1", "Item 2"]),
        ]

        result = engine.generate(
            title="Test Document",
            content=sections,
            format=DocumentFormat.DOCX,
        )

        assert result.success
        assert result.file_path.exists()
        assert result.file_path.suffix == ".docx"

    @pytest.mark.skipif(True, reason="Requires reportlab")
    def test_pdf_generation(self, temp_output_dir):
        """Test PDF generation"""
        engine = DocumentEngine(output_dir=temp_output_dir)

        result = engine.generate(
            title="Test PDF",
            content="This is test content for the PDF.",
            format=DocumentFormat.PDF,
        )

        assert result.success
        assert result.file_path.exists()
        assert result.file_path.suffix == ".pdf"

    @pytest.mark.skipif(True, reason="Requires python-pptx")
    def test_powerpoint_generation(self, temp_output_dir):
        """Test PowerPoint generation"""
        engine = DocumentEngine(output_dir=temp_output_dir)

        sections = [
            DocumentSection(title="Slide 1", content="Content for slide 1"),
            DocumentSection(title="Slide 2", content="", items=["Point 1", "Point 2"]),
        ]

        result = engine.generate(
            title="Test Presentation",
            content=sections,
            format=DocumentFormat.PPTX,
        )

        assert result.success
        assert result.file_path.exists()
        assert result.file_path.suffix == ".pptx"


# =============================================================================
# DocumentMixin Tests
# =============================================================================

class TestDocumentMixin:
    """Tests for DocumentMixin"""

    def test_mixin_properties(self):
        """Test DocumentMixin lazy property initialization"""
        class MockAgent(DocumentMixin):
            def __init__(self):
                super().__init__()

        agent = MockAgent()
        assert agent._document_engine is None

    def test_mixin_get_templates(self):
        """Test getting templates through mixin"""
        class MockAgent(DocumentMixin):
            def __init__(self):
                super().__init__()

        agent = MockAgent()
        templates = agent.get_available_templates()
        assert len(templates) >= 4

    def test_mixin_format_result_success(self):
        """Test formatting successful result"""
        class MockAgent(DocumentMixin):
            def __init__(self):
                super().__init__()

        agent = MockAgent()
        result = DocumentResult(
            success=True,
            file_path=Path("/tmp/test.pdf"),
            format=DocumentFormat.PDF,
            file_size_bytes=2048,
        )
        formatted = agent.format_document_result(result)
        assert "successfully" in formatted
        assert "test.pdf" in formatted
        assert "PDF" in formatted

    def test_mixin_format_result_failure(self):
        """Test formatting failed result"""
        class MockAgent(DocumentMixin):
            def __init__(self):
                super().__init__()

        agent = MockAgent()
        result = DocumentResult(
            success=False,
            error="Test error",
        )
        formatted = agent.format_document_result(result)
        assert "failed" in formatted
        assert "Test error" in formatted


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
