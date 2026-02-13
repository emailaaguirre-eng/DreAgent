"""
=============================================================================
HUMMINGBIRD-LEA - Vision API
Powered by CoDre-X | B & D Servicing LLC
=============================================================================
REST API endpoints for vision and OCR services.

Endpoints:
- POST /api/vision/analyze      - Analyze an image
- POST /api/vision/describe     - Get image description
- POST /api/vision/ask          - Ask a question about an image
- POST /api/vision/extract-text - Extract text from image (OCR)
- POST /api/vision/upload       - Upload and analyze an image file
=============================================================================
"""

import logging
import base64
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

from core.services.vision import (
    ImageAnalysisEngine,
    get_analysis_engine,
    AnalysisMode,
    ImageCategory,
    OCRLanguage,
)
from core.utils.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================

class AnalyzeImageRequest(BaseModel):
    """Request to analyze an image"""
    image_base64: str = Field(..., description="Base64 encoded image")
    mode: str = Field("auto", description="Analysis mode: auto, vision, ocr, combined")
    question: Optional[str] = Field(None, description="Specific question about the image")
    extract_text: bool = Field(True, description="Whether to extract text with OCR")
    detailed: bool = Field(False, description="Whether to provide detailed analysis")


class DescribeImageRequest(BaseModel):
    """Request to describe an image"""
    image_base64: str = Field(..., description="Base64 encoded image")
    detailed: bool = Field(False, description="Whether to provide detailed description")


class AskImageRequest(BaseModel):
    """Request to ask a question about an image"""
    image_base64: str = Field(..., description="Base64 encoded image")
    question: str = Field(..., description="Question to answer about the image")
    context: Optional[str] = Field(None, description="Additional context")


class ExtractTextRequest(BaseModel):
    """Request to extract text from an image"""
    image_base64: str = Field(..., description="Base64 encoded image")
    structured: bool = Field(False, description="Whether to return structured data")


class ImageAnalysisResponse(BaseModel):
    """Response from image analysis"""
    success: bool
    description: str = ""
    extracted_text: str = ""
    category: str = ""
    has_text: bool = False
    text_confidence: float = 0.0
    processing_time_ms: float = 0.0
    error: Optional[str] = None


class TextExtractionResponse(BaseModel):
    """Response from text extraction"""
    success: bool
    text: str = ""
    confidence: float = 0.0
    word_count: int = 0
    lines: Optional[List[dict]] = None
    error: Optional[str] = None


# =============================================================================
# Helper Functions
# =============================================================================

def parse_analysis_mode(mode_str: str) -> AnalysisMode:
    """Parse analysis mode from string"""
    mode_map = {
        "auto": AnalysisMode.AUTO,
        "vision": AnalysisMode.VISION_ONLY,
        "ocr": AnalysisMode.OCR_ONLY,
        "combined": AnalysisMode.COMBINED,
    }
    return mode_map.get(mode_str.lower(), AnalysisMode.AUTO)


async def save_uploaded_file(file: UploadFile) -> Path:
    """Save an uploaded file and return its path"""
    # Validate file type
    allowed_types = {"image/jpeg", "image/png", "image/gif", "image/webp", "image/bmp"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}"
        )

    # Create upload directory
    upload_dir = settings.upload_path / "vision"
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique filename
    import uuid
    ext = Path(file.filename).suffix if file.filename else ".jpg"
    filename = f"{uuid.uuid4()}{ext}"
    file_path = upload_dir / filename

    # Save file
    content = await file.read()
    file_path.write_bytes(content)

    return file_path


# =============================================================================
# API Endpoints
# =============================================================================

@router.post("/analyze", response_model=ImageAnalysisResponse)
async def analyze_image(request: AnalyzeImageRequest):
    """
    Analyze an image with automatic or specified mode.

    Combines vision understanding with OCR text extraction.
    """
    try:
        engine = get_analysis_engine()

        result = await engine.analyze(
            image=request.image_base64,
            mode=parse_analysis_mode(request.mode),
            question=request.question,
            extract_text=request.extract_text,
            detailed=request.detailed,
        )

        return ImageAnalysisResponse(
            success=True,
            description=result.description,
            extracted_text=result.extracted_text,
            category=result.category.value,
            has_text=result.has_text,
            text_confidence=result.text_confidence,
            processing_time_ms=result.processing_time_ms,
        )

    except Exception as e:
        logger.error(f"Image analysis error: {e}")
        return ImageAnalysisResponse(
            success=False,
            error=str(e),
        )


@router.post("/describe", response_model=ImageAnalysisResponse)
async def describe_image(request: DescribeImageRequest):
    """
    Get a description of an image using vision model.

    Does not extract text - use /analyze or /extract-text for OCR.
    """
    try:
        engine = get_analysis_engine()

        description = await engine.describe_only(
            image=request.image_base64,
            detailed=request.detailed,
        )

        return ImageAnalysisResponse(
            success=True,
            description=description,
        )

    except Exception as e:
        logger.error(f"Image description error: {e}")
        return ImageAnalysisResponse(
            success=False,
            error=str(e),
        )


@router.post("/ask", response_model=ImageAnalysisResponse)
async def ask_about_image(request: AskImageRequest):
    """
    Ask a specific question about an image.

    Uses the vision model to answer questions about image content.
    """
    try:
        engine = get_analysis_engine()

        result = await engine.ask(
            image=request.image_base64,
            question=request.question,
            context=request.context,
        )

        return ImageAnalysisResponse(
            success=True,
            description=result.description,
            processing_time_ms=result.processing_time_ms,
        )

    except Exception as e:
        logger.error(f"Image question error: {e}")
        return ImageAnalysisResponse(
            success=False,
            error=str(e),
        )


@router.post("/extract-text", response_model=TextExtractionResponse)
async def extract_text(request: ExtractTextRequest):
    """
    Extract text from an image using OCR.

    For documents with lots of text, use structured=true for organized output.
    """
    try:
        engine = get_analysis_engine()

        result = await engine.extract_text_only(
            image=request.image_base64,
            structured=request.structured,
        )

        if request.structured:
            return TextExtractionResponse(
                success=True,
                text=result.get("raw_text", ""),
                confidence=result.get("total_confidence", 0.0),
                lines=result.get("lines", []),
            )
        else:
            return TextExtractionResponse(
                success=True,
                text=result.get("text", ""),
                confidence=result.get("confidence", 0.0),
                word_count=result.get("word_count", 0),
            )

    except Exception as e:
        logger.error(f"Text extraction error: {e}")
        return TextExtractionResponse(
            success=False,
            error=str(e),
        )


@router.post("/upload")
async def upload_and_analyze(
    file: UploadFile = File(...),
    mode: str = Form("auto"),
    question: Optional[str] = Form(None),
    extract_text: bool = Form(True),
    detailed: bool = Form(False),
):
    """
    Upload an image file and analyze it.

    Accepts multipart form data with an image file.
    """
    try:
        # Save uploaded file
        file_path = await save_uploaded_file(file)

        try:
            engine = get_analysis_engine()

            result = await engine.analyze(
                image=file_path,
                mode=parse_analysis_mode(mode),
                question=question,
                extract_text=extract_text,
                detailed=detailed,
            )

            return {
                "success": True,
                "description": result.description,
                "extracted_text": result.extracted_text,
                "category": result.category.value,
                "has_text": result.has_text,
                "text_confidence": result.text_confidence,
                "processing_time_ms": result.processing_time_ms,
                "filename": file.filename,
            }

        finally:
            # Clean up uploaded file
            try:
                file_path.unlink()
            except Exception:
                pass

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload analysis error: {e}")
        return {
            "success": False,
            "error": str(e),
        }


@router.post("/detect-objects")
async def detect_objects(request: DescribeImageRequest):
    """
    Detect and list objects in an image.

    Returns a list of detected objects with their approximate locations.
    """
    try:
        engine = get_analysis_engine()

        objects = await engine.detect_objects(request.image_base64)

        return {
            "success": True,
            "objects": objects,
            "count": len(objects),
        }

    except Exception as e:
        logger.error(f"Object detection error: {e}")
        return {
            "success": False,
            "error": str(e),
            "objects": [],
            "count": 0,
        }


@router.post("/analyze-document")
async def analyze_document(
    file: UploadFile = File(...),
    structured: bool = Form(False),
):
    """
    Analyze a document image (optimized for text-heavy images).

    Combines vision understanding with comprehensive text extraction.
    """
    try:
        # Save uploaded file
        file_path = await save_uploaded_file(file)

        try:
            engine = get_analysis_engine()

            result = await engine.analyze_document(
                image=file_path,
                extract_structured=structured,
            )

            return {
                "success": True,
                "description": result.description,
                "extracted_text": result.extracted_text,
                "text_confidence": result.text_confidence,
                "processing_time_ms": result.processing_time_ms,
                "filename": file.filename,
            }

        finally:
            # Clean up uploaded file
            try:
                file_path.unlink()
            except Exception:
                pass

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document analysis error: {e}")
        return {
            "success": False,
            "error": str(e),
        }


# =============================================================================
# Health Check
# =============================================================================

@router.get("/health")
async def vision_health():
    """Check vision service health"""
    try:
        from core.providers.ollama import get_ollama_client

        ollama = get_ollama_client()
        is_available = await ollama.is_available()

        return {
            "status": "healthy" if is_available else "degraded",
            "ollama_available": is_available,
            "vision_model": ollama.get_model_config().get("vision", "unknown"),
            "message": "Vision service is ready" if is_available else "Ollama not available",
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }
