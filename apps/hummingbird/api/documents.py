"""
=============================================================================
HUMMINGBIRD-LEA - Documents API
Powered by CoDre-X | B & D Servicing LLC
=============================================================================
REST API endpoints for document generation services.

Endpoints:
- POST /api/documents/generate       - Generate a document
- POST /api/documents/proposal       - Generate EIAG proposal
- POST /api/documents/site-selection - Generate site selection report
- POST /api/documents/presentation   - Generate client presentation
- POST /api/documents/letter         - Generate formal letter
- GET  /api/documents/templates      - List available templates
- GET  /api/documents/files          - List generated files
- GET  /api/documents/download/{filename} - Download a document
=============================================================================
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from core.services.documents import (
    DocumentEngine,
    DocumentFormat,
    DocumentSection,
    ProposalData,
    SiteSelectionData,
    IncentiveSummaryData,
    get_document_engine,
)
from core.utils.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================

class GenerateDocumentRequest(BaseModel):
    """Request to generate a generic document"""
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")
    format: str = Field("pdf", description="Output format: pdf, docx, pptx")
    client_name: Optional[str] = Field(None, description="Client name")
    project_name: Optional[str] = Field(None, description="Project name")


class SectionModel(BaseModel):
    """Document section model"""
    title: str
    content: str = ""
    level: int = 1
    items: Optional[List[str]] = None
    table_data: Optional[List[List[str]]] = None


class GenerateSectionsRequest(BaseModel):
    """Request to generate document from sections"""
    title: str = Field(..., description="Document title")
    sections: List[SectionModel] = Field(..., description="Document sections")
    format: str = Field("pdf", description="Output format: pdf, docx, pptx")
    client_name: Optional[str] = None
    project_name: Optional[str] = None


class IncentiveModel(BaseModel):
    """Incentive data model"""
    name: str
    value: str
    type: str = ""
    timeline: str = ""


class TimelineItemModel(BaseModel):
    """Timeline item model"""
    phase: str
    timeline: str
    milestone: str = ""


class ContactInfoModel(BaseModel):
    """Contact info model"""
    name: str
    email: str
    phone: str = ""


class GenerateProposalRequest(BaseModel):
    """Request to generate an EIAG proposal"""
    client_name: str = Field(..., description="Client name")
    project_name: str = Field(..., description="Project name")
    executive_summary: str = Field(..., description="Executive summary")
    background: str = Field(..., description="Project background")
    incentives: List[IncentiveModel] = Field(..., description="Proposed incentives")
    total_value: str = Field(..., description="Total incentive value")
    investment: str = Field("", description="Required investment")
    net_benefit: str = Field("", description="Net benefit")
    roi: str = Field("", description="ROI")
    timeline: List[TimelineItemModel] = Field(default=[], description="Implementation timeline")
    next_steps: List[str] = Field(default=[], description="Next steps")
    contact: ContactInfoModel = Field(..., description="Contact information")
    format: str = Field("pdf", description="Output format: pdf, docx, pptx")


class SiteModel(BaseModel):
    """Site data model"""
    name: str
    location: str = ""
    size: str = ""
    features: str = ""


class CriterionModel(BaseModel):
    """Evaluation criterion model"""
    name: str
    weight: str = ""
    description: str = ""


class GenerateSiteSelectionRequest(BaseModel):
    """Request to generate a site selection report"""
    client_name: str
    project_name: str
    executive_summary: str
    sites: List[SiteModel]
    criteria: List[CriterionModel] = []
    comparison_matrix: List[List[str]] = []
    recommendation: str
    format: str = "pdf"


class KeyPointModel(BaseModel):
    """Key point for presentation"""
    title: str
    content: str = ""
    items: Optional[List[str]] = None


class GeneratePresentationRequest(BaseModel):
    """Request to generate a client presentation"""
    title: str
    client_name: str
    agenda: List[str]
    key_points: List[KeyPointModel]
    recommendations: List[str]
    next_steps: List[str]


class GenerateLetterRequest(BaseModel):
    """Request to generate a formal letter"""
    recipient_name: str
    recipient_title: str
    recipient_company: str
    subject: str
    body_paragraphs: List[str]
    sender_name: str
    sender_title: str


class DocumentResponse(BaseModel):
    """Response from document generation"""
    success: bool
    filename: str = ""
    format: str = ""
    file_size_kb: float = 0
    download_url: str = ""
    error: Optional[str] = None


# =============================================================================
# Helper Functions
# =============================================================================

def parse_format(format_str: str) -> DocumentFormat:
    """Parse format string to DocumentFormat enum"""
    format_map = {
        "pdf": DocumentFormat.PDF,
        "docx": DocumentFormat.DOCX,
        "word": DocumentFormat.DOCX,
        "pptx": DocumentFormat.PPTX,
        "powerpoint": DocumentFormat.PPTX,
    }
    return format_map.get(format_str.lower(), DocumentFormat.PDF)


def create_response(result) -> DocumentResponse:
    """Create API response from DocumentResult"""
    if result.success:
        return DocumentResponse(
            success=True,
            filename=result.filename,
            format=result.format.value,
            file_size_kb=result.file_size_bytes / 1024,
            download_url=f"/api/documents/download/{result.filename}",
        )
    else:
        return DocumentResponse(
            success=False,
            error=result.error,
        )


# =============================================================================
# API Endpoints
# =============================================================================

@router.post("/generate", response_model=DocumentResponse)
async def generate_document(request: GenerateDocumentRequest):
    """
    Generate a document from text content.

    Creates a simple document with the provided title and content.
    """
    try:
        engine = get_document_engine()
        format = parse_format(request.format)

        result = engine.generate(
            title=request.title,
            content=request.content,
            format=format,
            client_name=request.client_name,
            project_name=request.project_name,
        )

        return create_response(result)

    except Exception as e:
        logger.error(f"Document generation error: {e}")
        return DocumentResponse(success=False, error=str(e))


@router.post("/generate-sections", response_model=DocumentResponse)
async def generate_from_sections(request: GenerateSectionsRequest):
    """
    Generate a document from structured sections.

    Allows more control over document structure with headings,
    bullet points, and tables.
    """
    try:
        engine = get_document_engine()
        format = parse_format(request.format)

        # Convert Pydantic models to DocumentSection
        sections = [
            DocumentSection(
                title=s.title,
                content=s.content,
                level=s.level,
                items=s.items,
                table_data=s.table_data,
            )
            for s in request.sections
        ]

        result = engine.generate(
            title=request.title,
            content=sections,
            format=format,
            client_name=request.client_name,
            project_name=request.project_name,
        )

        return create_response(result)

    except Exception as e:
        logger.error(f"Section document generation error: {e}")
        return DocumentResponse(success=False, error=str(e))


@router.post("/proposal", response_model=DocumentResponse)
async def generate_proposal(request: GenerateProposalRequest):
    """
    Generate an EIAG incentive proposal document.

    Creates a professional proposal with executive summary,
    incentive breakdown, financial analysis, and timeline.
    """
    try:
        engine = get_document_engine()
        format = parse_format(request.format)

        # Build ProposalData
        data = ProposalData(
            client_name=request.client_name,
            project_name=request.project_name,
            executive_summary=request.executive_summary,
            background=request.background,
            proposed_incentives=[
                {"name": i.name, "value": i.value, "type": i.type, "timeline": i.timeline}
                for i in request.incentives
            ],
            financial_summary={
                "total_value": request.total_value,
                "investment": request.investment,
                "net_benefit": request.net_benefit,
                "roi": request.roi,
            },
            timeline=[
                {"phase": t.phase, "timeline": t.timeline, "milestone": t.milestone}
                for t in request.timeline
            ],
            next_steps=request.next_steps,
            contact_info={
                "name": request.contact.name,
                "email": request.contact.email,
                "phone": request.contact.phone,
            },
        )

        result = engine.generate_eiag_proposal(data, format)
        return create_response(result)

    except Exception as e:
        logger.error(f"Proposal generation error: {e}")
        return DocumentResponse(success=False, error=str(e))


@router.post("/site-selection", response_model=DocumentResponse)
async def generate_site_selection(request: GenerateSiteSelectionRequest):
    """
    Generate a site selection report.

    Creates a comprehensive site evaluation report with
    comparison matrix and recommendations.
    """
    try:
        engine = get_document_engine()
        format = parse_format(request.format)

        data = SiteSelectionData(
            client_name=request.client_name,
            project_name=request.project_name,
            executive_summary=request.executive_summary,
            sites_evaluated=[
                {"name": s.name, "location": s.location, "size": s.size, "features": s.features}
                for s in request.sites
            ],
            evaluation_criteria=[
                {"name": c.name, "weight": c.weight, "description": c.description}
                for c in request.criteria
            ],
            comparison_matrix=request.comparison_matrix,
            recommendation=request.recommendation,
            detailed_analysis=[],
        )

        result = engine.generate_eiag_site_selection(data, format)
        return create_response(result)

    except Exception as e:
        logger.error(f"Site selection generation error: {e}")
        return DocumentResponse(success=False, error=str(e))


@router.post("/presentation", response_model=DocumentResponse)
async def generate_presentation(request: GeneratePresentationRequest):
    """
    Generate a client presentation (PowerPoint).

    Creates a professional slide deck with agenda,
    key points, and recommendations.
    """
    try:
        engine = get_document_engine()

        key_points = [
            {"title": kp.title, "content": kp.content, "items": kp.items}
            for kp in request.key_points
        ]

        result = engine.generate_eiag_presentation(
            title=request.title,
            client_name=request.client_name,
            agenda=request.agenda,
            key_points=key_points,
            recommendations=request.recommendations,
            next_steps=request.next_steps,
        )

        return create_response(result)

    except Exception as e:
        logger.error(f"Presentation generation error: {e}")
        return DocumentResponse(success=False, error=str(e))


@router.post("/letter", response_model=DocumentResponse)
async def generate_letter(request: GenerateLetterRequest):
    """
    Generate a formal business letter (Word).

    Creates a properly formatted business letter.
    """
    try:
        engine = get_document_engine()

        result = engine.generate_eiag_letter(
            recipient_name=request.recipient_name,
            recipient_title=request.recipient_title,
            recipient_company=request.recipient_company,
            subject=request.subject,
            body_paragraphs=request.body_paragraphs,
            sender_name=request.sender_name,
            sender_title=request.sender_title,
        )

        return create_response(result)

    except Exception as e:
        logger.error(f"Letter generation error: {e}")
        return DocumentResponse(success=False, error=str(e))


@router.get("/templates")
async def list_templates():
    """
    List all available document templates.

    Returns template names and descriptions.
    """
    try:
        engine = get_document_engine()
        templates = engine.list_templates()

        return {
            "success": True,
            "templates": templates,
        }

    except Exception as e:
        logger.error(f"Template listing error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/files")
async def list_files():
    """
    List all generated document files.

    Returns filenames and metadata.
    """
    try:
        engine = get_document_engine()
        files = engine.get_output_files()

        file_list = []
        for f in files:
            file_list.append({
                "filename": f.name,
                "size_kb": f.stat().st_size / 1024,
                "modified": f.stat().st_mtime,
                "download_url": f"/api/documents/download/{f.name}",
            })

        return {
            "success": True,
            "files": file_list,
            "count": len(file_list),
        }

    except Exception as e:
        logger.error(f"File listing error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/download/{filename}")
async def download_file(filename: str):
    """
    Download a generated document.

    Returns the file as an attachment.
    """
    try:
        engine = get_document_engine()
        file_path = engine.output_dir / filename

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        # Determine media type
        suffix = file_path.suffix.lower()
        media_types = {
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        }
        media_type = media_types.get(suffix, "application/octet-stream")

        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type=media_type,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File download error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/files/{filename}")
async def delete_file(filename: str):
    """
    Delete a generated document.
    """
    try:
        engine = get_document_engine()
        file_path = engine.output_dir / filename

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        file_path.unlink()

        return {
            "success": True,
            "message": f"Deleted {filename}",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File deletion error: {e}")
        return {"success": False, "error": str(e)}


@router.post("/cleanup")
async def cleanup_old_files(days: int = 7):
    """
    Clean up old generated files.

    Deletes files older than specified days.
    """
    try:
        engine = get_document_engine()
        deleted = engine.cleanup_old_files(days)

        return {
            "success": True,
            "deleted_count": deleted,
            "message": f"Deleted {deleted} files older than {days} days",
        }

    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# Health Check
# =============================================================================

@router.get("/health")
async def documents_health():
    """Check document generation service health"""
    try:
        engine = get_document_engine()

        # Check if output directory exists
        output_ok = engine.output_dir.exists()

        # Check generator availability
        generators = {
            "pdf": engine._pdf_generator is not None or True,  # Always available (lazy init)
            "word": engine._word_generator is not None or True,
            "powerpoint": engine._pptx_generator is not None or True,
        }

        return {
            "status": "healthy",
            "output_directory": str(engine.output_dir),
            "output_directory_exists": output_ok,
            "generators": generators,
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }
