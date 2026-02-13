"""
=============================================================================
HUMMINGBIRD-LEA - Vision Agent Mixin
Powered by CoDre-X | B & D Servicing LLC
=============================================================================
Mixin class to add vision capabilities to agents.

Usage:
    class MyAgent(BaseAgent, VisionMixin):
        pass
=============================================================================
"""

import logging
from typing import Optional, Union, Dict, Any, List
from pathlib import Path

from .engine import (
    ImageAnalysisEngine,
    ImageAnalysisResult,
    AnalysisMode,
    get_analysis_engine,
)
from .vision import VisionService, VisionResult, get_vision_service
from .ocr import OCRService, OCRResult, get_ocr_service

logger = logging.getLogger(__name__)


# =============================================================================
# Vision Detection
# =============================================================================

def should_use_vision(message: str, has_image: bool = False) -> bool:
    """
    Determine if vision capabilities should be used for a request.

    Args:
        message: User's message
        has_image: Whether an image was provided

    Returns:
        True if vision should be used
    """
    # Always use vision if an image is provided
    if has_image:
        return True

    # Check for vision-related keywords
    vision_keywords = [
        "image", "picture", "photo", "screenshot",
        "see", "look at", "show me", "what's in",
        "describe", "analyze", "read this",
        "ocr", "text in", "extract text",
        "document", "scan", "recognize",
    ]

    message_lower = message.lower()
    return any(keyword in message_lower for keyword in vision_keywords)


def detect_image_intent(message: str) -> str:
    """
    Detect what the user wants to do with an image.

    Args:
        message: User's message

    Returns:
        Intent type: describe, question, extract_text, analyze_document, general
    """
    message_lower = message.lower()

    # Text extraction intent
    if any(word in message_lower for word in ["extract", "ocr", "read text", "what does it say"]):
        return "extract_text"

    # Document analysis intent
    if any(word in message_lower for word in ["document", "form", "scan", "receipt", "invoice"]):
        return "analyze_document"

    # Question about image
    if "?" in message or any(word in message_lower for word in ["what", "who", "where", "how many", "is there"]):
        return "question"

    # Description request
    if any(word in message_lower for word in ["describe", "tell me about", "what's in"]):
        return "describe"

    return "general"


# =============================================================================
# Vision Mixin
# =============================================================================

class VisionMixin:
    """
    Mixin class that adds vision capabilities to agents.

    Provides methods for:
    - Image analysis and description
    - Text extraction (OCR)
    - Visual question answering
    - Document analysis

    Usage:
        class LeaAgent(BaseAgent, VisionMixin):
            async def process(self, message, context):
                if context.has_image:
                    vision_context = await self.analyze_image_for_context(
                        context.image,
                        message
                    )
                    message = f"{message}\\n\\n{vision_context}"
                return await super().process(message, context)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vision_engine: Optional[ImageAnalysisEngine] = None
        self._vision_service: Optional[VisionService] = None
        self._ocr_service: Optional[OCRService] = None

    @property
    def vision_engine(self) -> ImageAnalysisEngine:
        """Get the vision analysis engine (lazy init)"""
        if self._vision_engine is None:
            self._vision_engine = get_analysis_engine()
        return self._vision_engine

    @property
    def vision_service(self) -> VisionService:
        """Get the vision service (lazy init)"""
        if self._vision_service is None:
            self._vision_service = get_vision_service()
        return self._vision_service

    @property
    def ocr_service(self) -> OCRService:
        """Get the OCR service (lazy init)"""
        if self._ocr_service is None:
            self._ocr_service = get_ocr_service()
        return self._ocr_service

    async def analyze_image(
        self,
        image: Union[str, Path, bytes],
        question: Optional[str] = None,
        mode: AnalysisMode = AnalysisMode.AUTO,
    ) -> ImageAnalysisResult:
        """
        Analyze an image.

        Args:
            image: Image to analyze
            question: Optional question about the image
            mode: Analysis mode

        Returns:
            ImageAnalysisResult
        """
        return await self.vision_engine.analyze(
            image=image,
            question=question,
            mode=mode,
        )

    async def describe_image(
        self,
        image: Union[str, Path, bytes],
        detailed: bool = False,
    ) -> str:
        """
        Get a description of an image.

        Args:
            image: Image to describe
            detailed: Whether to provide detailed description

        Returns:
            Description text
        """
        return await self.vision_engine.describe_only(image, detailed)

    async def ask_about_image(
        self,
        image: Union[str, Path, bytes],
        question: str,
        context: Optional[str] = None,
    ) -> str:
        """
        Ask a question about an image.

        Args:
            image: Image to analyze
            question: Question to answer
            context: Additional context

        Returns:
            Answer text
        """
        result = await self.vision_engine.ask(image, question, context)
        return result.description

    async def extract_text_from_image(
        self,
        image: Union[str, Path, bytes],
    ) -> str:
        """
        Extract text from an image using OCR.

        Args:
            image: Image to extract text from

        Returns:
            Extracted text
        """
        result = await self.ocr_service.extract_text(image)
        return result.text

    async def analyze_document_image(
        self,
        image: Union[str, Path, bytes],
    ) -> Dict[str, Any]:
        """
        Analyze a document image.

        Args:
            image: Document image

        Returns:
            Dict with document analysis
        """
        result = await self.vision_engine.analyze_document(image)
        return {
            "description": result.description,
            "text": result.extracted_text,
            "confidence": result.text_confidence,
        }

    async def analyze_image_for_context(
        self,
        image: Union[str, Path, bytes],
        message: str,
    ) -> str:
        """
        Analyze an image and format the result for use in agent context.

        This is useful for enhancing agent responses with image understanding.

        Args:
            image: Image to analyze
            message: User's message (used to detect intent)

        Returns:
            Formatted context string
        """
        intent = detect_image_intent(message)

        try:
            if intent == "extract_text":
                text = await self.extract_text_from_image(image)
                return f"[Extracted Text from Image]\n{text}"

            elif intent == "analyze_document":
                doc = await self.analyze_document_image(image)
                parts = [
                    "[Document Analysis]",
                    f"Type: {doc['description'][:200]}",
                ]
                if doc.get("text"):
                    parts.append(f"Content:\n{doc['text'][:1000]}")
                return "\n".join(parts)

            elif intent == "question":
                answer = await self.ask_about_image(image, message)
                return f"[Image Analysis]\n{answer}"

            else:
                # General description
                description = await self.describe_image(image)
                return f"[Image Description]\n{description}"

        except Exception as e:
            logger.warning(f"Image analysis failed: {e}")
            return "[Image analysis unavailable]"

    async def check_image_for_text(
        self,
        image: Union[str, Path, bytes],
        min_confidence: float = 0.5,
    ) -> tuple[bool, str]:
        """
        Check if an image contains significant text.

        Args:
            image: Image to check
            min_confidence: Minimum OCR confidence

        Returns:
            Tuple of (has_text, extracted_text)
        """
        try:
            result = await self.ocr_service.extract_text(image)

            has_text = (
                result.word_count > 5 and
                result.total_confidence >= min_confidence
            )

            return has_text, result.text if has_text else ""

        except Exception as e:
            logger.warning(f"Text detection failed: {e}")
            return False, ""


# =============================================================================
# Convenience Functions
# =============================================================================

async def process_image_for_agent(
    image: Union[str, Path, bytes],
    message: str,
) -> Dict[str, Any]:
    """
    Process an image for use with an agent.

    Args:
        image: Image to process
        message: User's message

    Returns:
        Dict with analysis results and formatted context
    """
    engine = get_analysis_engine()
    intent = detect_image_intent(message)

    result = await engine.analyze(
        image=image,
        question=message if intent == "question" else None,
        extract_text=intent in ("extract_text", "analyze_document", "general"),
    )

    return {
        "description": result.description,
        "extracted_text": result.extracted_text,
        "has_text": result.has_text,
        "category": result.category.value,
        "intent": intent,
        "context": _format_for_context(result, intent),
    }


def _format_for_context(result: ImageAnalysisResult, intent: str) -> str:
    """Format analysis result for agent context"""
    parts = []

    if intent == "extract_text" and result.has_text:
        parts.append(f"[Extracted Text]\n{result.extracted_text}")
    elif intent == "question":
        parts.append(f"[Visual Answer]\n{result.description}")
    else:
        parts.append(f"[Image Description]\n{result.description}")
        if result.has_text:
            parts.append(f"[Text in Image]\n{result.extracted_text[:500]}")

    return "\n\n".join(parts)
