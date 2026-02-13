"""
=============================================================================
HUMMINGBIRD-LEA - OCR Service
Powered by CoDre-X | B & D Servicing LLC
=============================================================================
OCR (Optical Character Recognition) service using EasyOCR.

Features:
- Multi-language text extraction
- Bounding box detection
- Confidence scoring
- Structured text output
=============================================================================
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class OCRLanguage(Enum):
    """Supported OCR languages"""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    CHINESE_SIMPLIFIED = "ch_sim"
    CHINESE_TRADITIONAL = "ch_tra"
    JAPANESE = "ja"
    KOREAN = "ko"
    ARABIC = "ar"
    RUSSIAN = "ru"


# Default languages to use
DEFAULT_LANGUAGES = [OCRLanguage.ENGLISH]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TextRegion:
    """A detected text region in an image"""
    text: str
    confidence: float               # 0.0 to 1.0
    bounding_box: List[List[int]]   # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    position: Tuple[int, int]       # (x, y) of top-left corner
    size: Tuple[int, int]           # (width, height) of region

    @property
    def top_left(self) -> Tuple[int, int]:
        """Get top-left corner"""
        return (self.bounding_box[0][0], self.bounding_box[0][1])

    @property
    def bottom_right(self) -> Tuple[int, int]:
        """Get bottom-right corner"""
        return (self.bounding_box[2][0], self.bounding_box[2][1])


@dataclass
class OCRResult:
    """Result from OCR processing"""
    text: str                       # Full extracted text
    regions: List[TextRegion]       # Individual text regions
    languages: List[str]            # Languages used
    total_confidence: float         # Average confidence
    word_count: int
    processing_time_ms: float
    image_size: Tuple[int, int]     # (width, height)
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def line_count(self) -> int:
        """Get number of text lines detected"""
        return len(self.regions)

    def get_high_confidence_text(self, threshold: float = 0.7) -> str:
        """Get only high-confidence text"""
        high_conf = [r.text for r in self.regions if r.confidence >= threshold]
        return " ".join(high_conf)

    def get_text_by_position(self, sort_by: str = "top_to_bottom") -> str:
        """Get text sorted by position"""
        if sort_by == "top_to_bottom":
            sorted_regions = sorted(self.regions, key=lambda r: r.position[1])
        elif sort_by == "left_to_right":
            sorted_regions = sorted(self.regions, key=lambda r: r.position[0])
        else:
            sorted_regions = self.regions

        return "\n".join(r.text for r in sorted_regions)


# =============================================================================
# OCR Service
# =============================================================================

class OCRService:
    """
    OCR service using EasyOCR for text extraction.

    Usage:
        service = OCRService()

        # Extract text from image
        result = await service.extract_text("path/to/image.png")
        print(result.text)

        # Get individual regions
        for region in result.regions:
            print(f"{region.text} (confidence: {region.confidence:.2f})")
    """

    def __init__(
        self,
        languages: Optional[List[OCRLanguage]] = None,
        gpu: bool = False,
    ):
        """
        Initialize the OCR service.

        Args:
            languages: List of languages to detect
            gpu: Whether to use GPU acceleration
        """
        self.languages = languages or DEFAULT_LANGUAGES
        self.gpu = gpu
        self._reader = None  # Lazy initialization

        # Language codes for EasyOCR
        self.lang_codes = [lang.value for lang in self.languages]

        logger.info(f"OCRService initialized with languages: {self.lang_codes}")

    def _get_reader(self):
        """Get or create the EasyOCR reader (lazy init)"""
        if self._reader is None:
            try:
                import easyocr
            except ImportError:
                raise ImportError(
                    "EasyOCR is required for OCR. Install with: pip install easyocr"
                )

            logger.info("Initializing EasyOCR reader (this may take a moment)...")
            self._reader = easyocr.Reader(
                self.lang_codes,
                gpu=self.gpu,
                verbose=False,
            )
            logger.info("EasyOCR reader initialized")

        return self._reader

    async def extract_text(
        self,
        image: Union[str, Path, bytes],
        detail_level: int = 1,
        paragraph: bool = False,
    ) -> OCRResult:
        """
        Extract text from an image.

        Args:
            image: Image file path or bytes
            detail_level: 0=simple, 1=standard, 2=high detail
            paragraph: Whether to merge into paragraphs

        Returns:
            OCRResult with extracted text
        """
        import time
        import asyncio
        start_time = time.time()

        # Run OCR in thread pool (EasyOCR is sync)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._run_ocr,
            image,
            detail_level,
            paragraph,
        )

        processing_time = (time.time() - start_time) * 1000

        # Build result
        regions = []
        texts = []

        for detection in result["detections"]:
            bbox, text, confidence = detection

            # Calculate position and size
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            position = (min(x_coords), min(y_coords))
            size = (max(x_coords) - min(x_coords), max(y_coords) - min(y_coords))

            region = TextRegion(
                text=text,
                confidence=confidence,
                bounding_box=bbox,
                position=position,
                size=size,
            )
            regions.append(region)
            texts.append(text)

        # Calculate average confidence
        total_confidence = (
            sum(r.confidence for r in regions) / len(regions)
            if regions else 0.0
        )

        # Build full text
        full_text = " ".join(texts) if not paragraph else "\n".join(texts)

        return OCRResult(
            text=full_text,
            regions=regions,
            languages=self.lang_codes,
            total_confidence=total_confidence,
            word_count=len(full_text.split()),
            processing_time_ms=processing_time,
            image_size=result["image_size"],
        )

    def _run_ocr(
        self,
        image: Union[str, Path, bytes],
        detail_level: int,
        paragraph: bool,
    ) -> Dict[str, Any]:
        """Run OCR synchronously"""
        from PIL import Image
        import io
        import numpy as np

        reader = self._get_reader()

        # Load image
        if isinstance(image, bytes):
            pil_image = Image.open(io.BytesIO(image))
        elif isinstance(image, (str, Path)):
            pil_image = Image.open(image)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        image_size = pil_image.size

        # Convert to numpy array for EasyOCR
        image_array = np.array(pil_image)

        # Run OCR
        results = reader.readtext(
            image_array,
            detail=detail_level,
            paragraph=paragraph,
        )

        return {
            "detections": results,
            "image_size": image_size,
        }

    async def extract_structured(
        self,
        image: Union[str, Path, bytes],
    ) -> Dict[str, Any]:
        """
        Extract text with structured layout information.

        Returns a dict with text organized by position.

        Args:
            image: Image to process

        Returns:
            Dict with structured text data
        """
        result = await self.extract_text(image, detail_level=1)

        # Sort regions by vertical position
        sorted_regions = sorted(result.regions, key=lambda r: r.position[1])

        # Group into lines (regions with similar y positions)
        lines = []
        current_line = []
        current_y = None
        line_threshold = 20  # pixels

        for region in sorted_regions:
            if current_y is None:
                current_y = region.position[1]
                current_line = [region]
            elif abs(region.position[1] - current_y) < line_threshold:
                current_line.append(region)
            else:
                # Sort current line by x position
                current_line.sort(key=lambda r: r.position[0])
                lines.append(current_line)
                current_line = [region]
                current_y = region.position[1]

        if current_line:
            current_line.sort(key=lambda r: r.position[0])
            lines.append(current_line)

        # Build structured output
        structured = {
            "lines": [],
            "raw_text": result.text,
            "total_confidence": result.total_confidence,
        }

        for line_regions in lines:
            line_text = " ".join(r.text for r in line_regions)
            line_conf = sum(r.confidence for r in line_regions) / len(line_regions)

            structured["lines"].append({
                "text": line_text,
                "confidence": line_conf,
                "word_count": len(line_regions),
                "y_position": line_regions[0].position[1],
            })

        return structured

    async def detect_language(
        self,
        image: Union[str, Path, bytes],
    ) -> str:
        """
        Detect the primary language in an image.

        Note: This is a simple heuristic based on character patterns.

        Args:
            image: Image to analyze

        Returns:
            Detected language code
        """
        result = await self.extract_text(image)

        if not result.text:
            return "unknown"

        # Simple language detection based on character ranges
        text = result.text

        # Check for CJK characters
        cjk_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        if cjk_count > len(text) * 0.3:
            return "chinese"

        # Check for Japanese
        hiragana = sum(1 for c in text if '\u3040' <= c <= '\u309f')
        katakana = sum(1 for c in text if '\u30a0' <= c <= '\u30ff')
        if (hiragana + katakana) > len(text) * 0.1:
            return "japanese"

        # Check for Korean
        korean = sum(1 for c in text if '\uac00' <= c <= '\ud7af')
        if korean > len(text) * 0.3:
            return "korean"

        # Check for Arabic
        arabic = sum(1 for c in text if '\u0600' <= c <= '\u06ff')
        if arabic > len(text) * 0.3:
            return "arabic"

        # Check for Cyrillic (Russian, etc.)
        cyrillic = sum(1 for c in text if '\u0400' <= c <= '\u04ff')
        if cyrillic > len(text) * 0.3:
            return "russian"

        # Default to English/Latin
        return "english"


# =============================================================================
# Factory Function
# =============================================================================

_ocr_service: Optional[OCRService] = None


def get_ocr_service(
    languages: Optional[List[OCRLanguage]] = None,
) -> OCRService:
    """Get or create the OCR service singleton"""
    global _ocr_service

    if _ocr_service is None:
        _ocr_service = OCRService(languages=languages)

    return _ocr_service


# =============================================================================
# Convenience Functions
# =============================================================================

async def extract_text_from_image(
    image: Union[str, Path, bytes],
) -> str:
    """
    Convenience function to extract text from an image.

    Args:
        image: Image to process

    Returns:
        Extracted text
    """
    service = get_ocr_service()
    result = await service.extract_text(image)
    return result.text


async def get_ocr_regions(
    image: Union[str, Path, bytes],
) -> List[TextRegion]:
    """
    Convenience function to get text regions from an image.

    Args:
        image: Image to process

    Returns:
        List of TextRegion objects
    """
    service = get_ocr_service()
    result = await service.extract_text(image)
    return result.regions
