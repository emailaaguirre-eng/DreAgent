"""
=============================================================================
HUMMINGBIRD-LEA - Vision Services
Powered by CoDre-X | B & D Servicing LLC
=============================================================================
Vision and OCR services for image understanding and text extraction.

Components:
- VisionService: Image understanding with llava
- OCRService: Text extraction with EasyOCR
- ImageAnalysisEngine: Unified analysis combining vision + OCR
- ImageProcessor: Image processing utilities
=============================================================================
"""

# Image utilities
from .image_utils import (
    ImageFormat,
    ImageInfo,
    ProcessedImage,
    ImageProcessor,
    get_image_processor,
    validate_image_file,
    validate_base64_image,
    MAX_IMAGE_SIZE,
    MAX_FILE_SIZE_MB,
)

# Vision service
from .vision import (
    AnalysisType,
    VisionResult,
    VisionQuestion,
    VisionService,
    get_vision_service,
    describe_image,
    ask_about_image,
)

# OCR service
from .ocr import (
    OCRLanguage,
    TextRegion,
    OCRResult,
    OCRService,
    get_ocr_service,
    extract_text_from_image,
    get_ocr_regions,
)

# Unified analysis engine
from .engine import (
    ImageCategory,
    AnalysisMode,
    ImageAnalysisResult,
    AnalysisRequest,
    ImageAnalysisEngine,
    get_analysis_engine,
    analyze_image,
    extract_image_text,
)

# Agent mixin
from .agent_mixin import (
    VisionMixin,
    should_use_vision,
    detect_image_intent,
    process_image_for_agent,
)


__all__ = [
    # Image utilities
    "ImageFormat",
    "ImageInfo",
    "ProcessedImage",
    "ImageProcessor",
    "get_image_processor",
    "validate_image_file",
    "validate_base64_image",
    "MAX_IMAGE_SIZE",
    "MAX_FILE_SIZE_MB",
    # Vision service
    "AnalysisType",
    "VisionResult",
    "VisionQuestion",
    "VisionService",
    "get_vision_service",
    "describe_image",
    "ask_about_image",
    # OCR service
    "OCRLanguage",
    "TextRegion",
    "OCRResult",
    "OCRService",
    "get_ocr_service",
    "extract_text_from_image",
    "get_ocr_regions",
    # Unified analysis engine
    "ImageCategory",
    "AnalysisMode",
    "ImageAnalysisResult",
    "AnalysisRequest",
    "ImageAnalysisEngine",
    "get_analysis_engine",
    "analyze_image",
    "extract_image_text",
    # Agent mixin
    "VisionMixin",
    "should_use_vision",
    "detect_image_intent",
    "process_image_for_agent",
]
