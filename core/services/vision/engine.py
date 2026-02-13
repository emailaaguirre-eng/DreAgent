"""
=============================================================================
HUMMINGBIRD-LEA - Image Analysis Engine
Powered by CoDre-X | B & D Servicing LLC
=============================================================================
Unified image analysis engine combining Vision and OCR services.

Features:
- Combined vision + OCR analysis
- Smart analysis type detection
- Document vs photo detection
- Confidence-weighted results
=============================================================================
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from enum import Enum

from .vision import VisionService, VisionResult, AnalysisType, get_vision_service
from .ocr import OCRService, OCRResult, OCRLanguage, get_ocr_service
from .image_utils import (
    ImageProcessor,
    ProcessedImage,
    ImageInfo,
    get_image_processor,
    validate_image_file,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class ImageCategory(Enum):
    """Categories of images for analysis routing"""
    DOCUMENT = "document"       # Text-heavy documents
    PHOTO = "photo"             # General photographs
    SCREENSHOT = "screenshot"   # Screen captures
    DIAGRAM = "diagram"         # Charts, flowcharts, etc.
    MIXED = "mixed"             # Contains both text and imagery
    UNKNOWN = "unknown"


class AnalysisMode(Enum):
    """Analysis mode selection"""
    AUTO = "auto"               # Automatically detect best approach
    VISION_ONLY = "vision"      # Use only vision model
    OCR_ONLY = "ocr"            # Use only OCR
    COMBINED = "combined"       # Use both and merge results


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ImageAnalysisResult:
    """Combined result from image analysis"""
    # Core content
    description: str                    # Visual description
    extracted_text: str                 # OCR text (if any)

    # Analysis metadata
    category: ImageCategory
    analysis_mode: AnalysisMode

    # Vision results
    vision_result: Optional[VisionResult] = None

    # OCR results
    ocr_result: Optional[OCRResult] = None

    # Combined metrics
    has_text: bool = False
    text_confidence: float = 0.0
    vision_confidence: float = 1.0

    # Processing info
    image_info: Optional[Dict] = None
    processing_time_ms: float = 0
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def summary(self) -> str:
        """Get a summary combining description and text"""
        parts = []

        if self.description:
            parts.append(f"**Description:** {self.description}")

        if self.has_text and self.extracted_text:
            parts.append(f"**Extracted Text:**\n{self.extracted_text}")

        return "\n\n".join(parts) if parts else "No analysis available."

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "description": self.description,
            "extracted_text": self.extracted_text,
            "category": self.category.value,
            "analysis_mode": self.analysis_mode.value,
            "has_text": self.has_text,
            "text_confidence": self.text_confidence,
            "vision_confidence": self.vision_confidence,
            "processing_time_ms": self.processing_time_ms,
            "image_info": self.image_info,
        }


@dataclass
class AnalysisRequest:
    """Request for image analysis"""
    image: Union[str, Path, bytes]
    mode: AnalysisMode = AnalysisMode.AUTO
    question: Optional[str] = None      # Specific question about the image
    extract_text: bool = True           # Whether to run OCR
    detailed: bool = False              # Detailed vs concise analysis
    languages: Optional[List[OCRLanguage]] = None  # OCR languages


# =============================================================================
# Image Analysis Engine
# =============================================================================

class ImageAnalysisEngine:
    """
    Unified engine for image analysis combining Vision and OCR.

    Usage:
        engine = ImageAnalysisEngine()

        # Automatic analysis
        result = await engine.analyze("path/to/image.jpg")
        print(result.description)
        print(result.extracted_text)

        # Ask a question
        result = await engine.ask("path/to/image.jpg", "What color is the car?")

        # Document analysis
        result = await engine.analyze_document("path/to/doc.png")
    """

    def __init__(
        self,
        vision_service: Optional[VisionService] = None,
        ocr_service: Optional[OCRService] = None,
    ):
        """
        Initialize the analysis engine.

        Args:
            vision_service: Vision service instance (or use default)
            ocr_service: OCR service instance (or use default)
        """
        self.vision = vision_service or get_vision_service()
        self.ocr = ocr_service or get_ocr_service()
        self.processor = get_image_processor()

        logger.info("ImageAnalysisEngine initialized")

    async def analyze(
        self,
        image: Union[str, Path, bytes],
        mode: AnalysisMode = AnalysisMode.AUTO,
        question: Optional[str] = None,
        extract_text: bool = True,
        detailed: bool = False,
    ) -> ImageAnalysisResult:
        """
        Analyze an image with automatic mode detection.

        Args:
            image: Image file path, bytes, or base64 string
            mode: Analysis mode (auto, vision, ocr, combined)
            question: Optional question to answer about the image
            extract_text: Whether to extract text with OCR
            detailed: Whether to provide detailed analysis

        Returns:
            ImageAnalysisResult with combined analysis
        """
        import time
        start_time = time.time()

        # Validate image
        if isinstance(image, (str, Path)):
            path = Path(image)
            if path.exists():
                is_valid, error = validate_image_file(path)
                if not is_valid:
                    raise ValueError(error)

        # Determine analysis mode
        if mode == AnalysisMode.AUTO:
            mode = await self._detect_best_mode(image, question)

        # Initialize result components
        vision_result = None
        ocr_result = None
        description = ""
        extracted_text = ""
        category = ImageCategory.UNKNOWN

        # Run analysis based on mode
        if mode in (AnalysisMode.VISION_ONLY, AnalysisMode.COMBINED):
            vision_result = await self._run_vision_analysis(
                image, question, detailed
            )
            description = vision_result.content

        if mode in (AnalysisMode.OCR_ONLY, AnalysisMode.COMBINED) and extract_text:
            try:
                ocr_result = await self.ocr.extract_text(image)
                extracted_text = ocr_result.text
            except Exception as e:
                logger.warning(f"OCR failed: {e}")
                # Continue with vision-only results

        # Determine category based on results
        category = self._categorize_image(vision_result, ocr_result)

        # Get image info
        image_info = None
        if vision_result and vision_result.image_info:
            image_info = vision_result.image_info

        processing_time = (time.time() - start_time) * 1000

        return ImageAnalysisResult(
            description=description,
            extracted_text=extracted_text,
            category=category,
            analysis_mode=mode,
            vision_result=vision_result,
            ocr_result=ocr_result,
            has_text=bool(extracted_text and len(extracted_text) > 5),
            text_confidence=ocr_result.total_confidence if ocr_result else 0.0,
            vision_confidence=vision_result.confidence if vision_result else 0.0,
            image_info=image_info,
            processing_time_ms=processing_time,
        )

    async def ask(
        self,
        image: Union[str, Path, bytes],
        question: str,
        context: Optional[str] = None,
    ) -> ImageAnalysisResult:
        """
        Ask a specific question about an image.

        Args:
            image: Image to analyze
            question: Question to answer
            context: Additional context

        Returns:
            ImageAnalysisResult with the answer
        """
        vision_result = await self.vision.ask(image, question, context)

        return ImageAnalysisResult(
            description=vision_result.content,
            extracted_text="",
            category=ImageCategory.UNKNOWN,
            analysis_mode=AnalysisMode.VISION_ONLY,
            vision_result=vision_result,
            vision_confidence=vision_result.confidence,
            image_info=vision_result.image_info,
            processing_time_ms=vision_result.processing_time_ms,
        )

    async def analyze_document(
        self,
        image: Union[str, Path, bytes],
        extract_structured: bool = False,
    ) -> ImageAnalysisResult:
        """
        Analyze a document image (optimized for text-heavy images).

        Args:
            image: Document image
            extract_structured: Whether to extract structured data

        Returns:
            ImageAnalysisResult with document analysis
        """
        import time
        start_time = time.time()

        # Run both vision (for understanding) and OCR (for text)
        vision_result = await self.vision.analyze_document(image)

        # Use structured extraction for documents
        if extract_structured:
            ocr_data = await self.ocr.extract_structured(image)
            extracted_text = "\n".join(
                line["text"] for line in ocr_data.get("lines", [])
            )
            text_confidence = ocr_data.get("total_confidence", 0.0)
            ocr_result = None  # Structured data instead
        else:
            ocr_result = await self.ocr.extract_text(image, paragraph=True)
            extracted_text = ocr_result.text
            text_confidence = ocr_result.total_confidence

        processing_time = (time.time() - start_time) * 1000

        return ImageAnalysisResult(
            description=vision_result.content,
            extracted_text=extracted_text,
            category=ImageCategory.DOCUMENT,
            analysis_mode=AnalysisMode.COMBINED,
            vision_result=vision_result,
            ocr_result=ocr_result,
            has_text=True,
            text_confidence=text_confidence,
            vision_confidence=vision_result.confidence,
            image_info=vision_result.image_info,
            processing_time_ms=processing_time,
        )

    async def extract_text_only(
        self,
        image: Union[str, Path, bytes],
        structured: bool = False,
    ) -> Dict[str, Any]:
        """
        Extract only text from an image (OCR only).

        Args:
            image: Image to extract text from
            structured: Whether to return structured data

        Returns:
            Dict with extracted text and metadata
        """
        if structured:
            return await self.ocr.extract_structured(image)
        else:
            result = await self.ocr.extract_text(image)
            return {
                "text": result.text,
                "confidence": result.total_confidence,
                "word_count": result.word_count,
                "regions": len(result.regions),
            }

    async def describe_only(
        self,
        image: Union[str, Path, bytes],
        detailed: bool = False,
    ) -> str:
        """
        Get only a description of the image (vision only).

        Args:
            image: Image to describe
            detailed: Whether to provide detailed description

        Returns:
            Description text
        """
        result = await self.vision.describe(image, detailed=detailed)
        return result.content

    async def detect_objects(
        self,
        image: Union[str, Path, bytes],
    ) -> List[str]:
        """
        Detect objects in an image.

        Args:
            image: Image to analyze

        Returns:
            List of detected objects
        """
        result = await self.vision.detect_objects(image)

        # Parse the response into a list
        # Vision model returns text, we parse it
        content = result.content
        objects = []

        # Simple parsing - split by newlines and clean up
        lines = content.split("\n")
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                # Remove bullet points and numbers
                clean = line.lstrip("-•*0123456789. ")
                if clean:
                    objects.append(clean)

        return objects

    async def _detect_best_mode(
        self,
        image: Union[str, Path, bytes],
        question: Optional[str],
    ) -> AnalysisMode:
        """Detect the best analysis mode for an image"""
        # If there's a specific question, use vision
        if question:
            return AnalysisMode.VISION_ONLY

        # For now, default to combined mode for comprehensive results
        # In the future, we could do quick image analysis to detect type
        return AnalysisMode.COMBINED

    async def _run_vision_analysis(
        self,
        image: Union[str, Path, bytes],
        question: Optional[str],
        detailed: bool,
    ) -> VisionResult:
        """Run vision analysis with appropriate prompt"""
        if question:
            return await self.vision.ask(image, question)
        else:
            return await self.vision.describe(image, detailed=detailed)

    def _categorize_image(
        self,
        vision_result: Optional[VisionResult],
        ocr_result: Optional[OCRResult],
    ) -> ImageCategory:
        """Categorize image based on analysis results"""
        has_significant_text = (
            ocr_result and
            ocr_result.word_count > 20 and
            ocr_result.total_confidence > 0.5
        )

        if has_significant_text:
            # Check if it's a document or mixed content
            if ocr_result.word_count > 100:
                return ImageCategory.DOCUMENT
            else:
                return ImageCategory.MIXED

        # Check vision description for clues
        if vision_result:
            content_lower = vision_result.content.lower()

            if any(word in content_lower for word in ["screenshot", "screen", "interface", "window"]):
                return ImageCategory.SCREENSHOT

            if any(word in content_lower for word in ["chart", "graph", "diagram", "flowchart"]):
                return ImageCategory.DIAGRAM

            if any(word in content_lower for word in ["document", "form", "paper", "letter"]):
                return ImageCategory.DOCUMENT

        return ImageCategory.PHOTO


# =============================================================================
# Factory Function
# =============================================================================

_analysis_engine: Optional[ImageAnalysisEngine] = None


def get_analysis_engine() -> ImageAnalysisEngine:
    """Get or create the image analysis engine singleton"""
    global _analysis_engine
    if _analysis_engine is None:
        _analysis_engine = ImageAnalysisEngine()
    return _analysis_engine


# =============================================================================
# Convenience Functions
# =============================================================================

async def analyze_image(
    image: Union[str, Path, bytes],
    question: Optional[str] = None,
) -> ImageAnalysisResult:
    """
    Convenience function to analyze an image.

    Args:
        image: Image to analyze
        question: Optional question to answer

    Returns:
        ImageAnalysisResult
    """
    engine = get_analysis_engine()
    return await engine.analyze(image, question=question)


async def describe_image(
    image: Union[str, Path, bytes],
) -> str:
    """
    Convenience function to describe an image.

    Args:
        image: Image to describe

    Returns:
        Description text
    """
    engine = get_analysis_engine()
    return await engine.describe_only(image)


async def extract_image_text(
    image: Union[str, Path, bytes],
) -> str:
    """
    Convenience function to extract text from an image.

    Args:
        image: Image to extract text from

    Returns:
        Extracted text
    """
    engine = get_analysis_engine()
    result = await engine.extract_text_only(image)
    return result.get("text", "")
