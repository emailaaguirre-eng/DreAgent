"""
=============================================================================
HUMMINGBIRD-LEA - Vision Service
Powered by CoDre-X | B & D Servicing LLC
=============================================================================
Vision service using Ollama's llava model for image understanding.

Features:
- Image description and analysis
- Visual question answering
- Object and text detection
- Scene understanding
=============================================================================
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from enum import Enum

from core.providers.ollama import get_ollama_client, ModelType
from .image_utils import (
    ImageProcessor,
    ProcessedImage,
    get_image_processor,
    validate_image_file,
    validate_base64_image,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class AnalysisType(Enum):
    """Types of image analysis"""
    DESCRIBE = "describe"           # General description
    DETAILED = "detailed"           # Detailed analysis
    OBJECTS = "objects"             # Object detection
    TEXT = "text"                   # Text/OCR detection
    SCENE = "scene"                 # Scene understanding
    QUESTION = "question"           # Answer a question
    DOCUMENT = "document"           # Document analysis
    CUSTOM = "custom"               # Custom prompt


# Prompt templates for different analysis types
ANALYSIS_PROMPTS = {
    AnalysisType.DESCRIBE: (
        "Describe this image in a clear, concise paragraph. "
        "Focus on the main subjects and important details."
    ),
    AnalysisType.DETAILED: (
        "Provide a detailed analysis of this image. Include:\n"
        "1. Main subjects and their characteristics\n"
        "2. Background and setting\n"
        "3. Colors, lighting, and visual style\n"
        "4. Any text visible in the image\n"
        "5. Overall mood or impression"
    ),
    AnalysisType.OBJECTS: (
        "List all objects you can identify in this image. "
        "For each object, note its approximate location "
        "(left, right, center, top, bottom) and any notable characteristics."
    ),
    AnalysisType.TEXT: (
        "Extract and transcribe all visible text in this image. "
        "Preserve the layout and formatting as much as possible. "
        "If no text is visible, say 'No text detected'."
    ),
    AnalysisType.SCENE: (
        "Analyze the scene in this image. Describe:\n"
        "1. What type of place or setting this is\n"
        "2. What activity or event might be happening\n"
        "3. The time of day or weather if visible\n"
        "4. Any notable features of the environment"
    ),
    AnalysisType.DOCUMENT: (
        "This appears to be a document or form. Please:\n"
        "1. Identify the type of document\n"
        "2. Extract all visible text, preserving structure\n"
        "3. Note any important fields, headers, or sections\n"
        "4. Summarize the key information if possible"
    ),
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class VisionResult:
    """Result from vision analysis"""
    content: str                    # The analysis text
    analysis_type: AnalysisType
    confidence: float = 1.0         # Confidence score (0-1)
    model: str = ""                 # Model used
    image_info: Optional[Dict] = None  # Image metadata
    processing_time_ms: float = 0
    tokens_used: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def has_text(self) -> bool:
        """Check if text was detected"""
        lower = self.content.lower()
        return "no text" not in lower and len(self.content) > 20


@dataclass
class VisionQuestion:
    """A question about an image"""
    question: str
    image_data: str                 # Base64 encoded
    context: Optional[str] = None   # Additional context


# =============================================================================
# Vision Service
# =============================================================================

class VisionService:
    """
    Service for image understanding using llava.

    Usage:
        service = VisionService()

        # Describe an image
        result = await service.describe("path/to/image.jpg")

        # Ask a question
        result = await service.ask(
            image="path/to/image.jpg",
            question="What color is the car?"
        )

        # Extract text
        result = await service.extract_text("path/to/document.png")
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.3,
    ):
        """
        Initialize the vision service.

        Args:
            model: Vision model name (default: from settings)
            temperature: Generation temperature (lower = more focused)
        """
        self.ollama = get_ollama_client()
        self.model = model or self.ollama.get_model(ModelType.VISION)
        self.temperature = temperature
        self.processor = get_image_processor()

        logger.info(f"VisionService initialized with model: {self.model}")

    async def analyze(
        self,
        image: Union[str, Path, bytes],
        analysis_type: AnalysisType = AnalysisType.DESCRIBE,
        custom_prompt: Optional[str] = None,
    ) -> VisionResult:
        """
        Analyze an image with the specified analysis type.

        Args:
            image: Image file path, bytes, or base64 string
            analysis_type: Type of analysis to perform
            custom_prompt: Custom prompt (for CUSTOM type)

        Returns:
            VisionResult with analysis
        """
        import time
        start_time = time.time()

        # Process the image
        processed = await self._process_image(image)

        # Get the prompt
        if analysis_type == AnalysisType.CUSTOM and custom_prompt:
            prompt = custom_prompt
        else:
            prompt = ANALYSIS_PROMPTS.get(analysis_type, ANALYSIS_PROMPTS[AnalysisType.DESCRIBE])

        # Call vision model
        response = await self.ollama.analyze_image(
            image_base64=processed.base64_data,
            prompt=prompt,
            model=self.model,
        )

        processing_time = (time.time() - start_time) * 1000

        return VisionResult(
            content=response.content,
            analysis_type=analysis_type,
            model=response.model,
            image_info={
                "original_size": processed.original_size,
                "processed_size": processed.processed_size,
                "was_resized": processed.was_resized,
                "format": processed.info.format.value,
            },
            processing_time_ms=processing_time,
            tokens_used=response.eval_count or 0,
        )

    async def describe(
        self,
        image: Union[str, Path, bytes],
        detailed: bool = False,
    ) -> VisionResult:
        """
        Describe an image.

        Args:
            image: Image to describe
            detailed: Whether to provide detailed description

        Returns:
            VisionResult with description
        """
        analysis_type = AnalysisType.DETAILED if detailed else AnalysisType.DESCRIBE
        return await self.analyze(image, analysis_type)

    async def ask(
        self,
        image: Union[str, Path, bytes],
        question: str,
        context: Optional[str] = None,
    ) -> VisionResult:
        """
        Ask a question about an image.

        Args:
            image: Image to analyze
            question: Question to answer
            context: Optional additional context

        Returns:
            VisionResult with the answer
        """
        # Build the prompt
        prompt = question
        if context:
            prompt = f"Context: {context}\n\nQuestion: {question}"

        return await self.analyze(
            image,
            analysis_type=AnalysisType.QUESTION,
            custom_prompt=prompt,
        )

    async def extract_text(
        self,
        image: Union[str, Path, bytes],
    ) -> VisionResult:
        """
        Extract text from an image using vision.

        Note: For better OCR accuracy, use the OCR service instead.

        Args:
            image: Image to extract text from

        Returns:
            VisionResult with extracted text
        """
        return await self.analyze(image, AnalysisType.TEXT)

    async def analyze_document(
        self,
        image: Union[str, Path, bytes],
    ) -> VisionResult:
        """
        Analyze a document image.

        Args:
            image: Document image

        Returns:
            VisionResult with document analysis
        """
        return await self.analyze(image, AnalysisType.DOCUMENT)

    async def detect_objects(
        self,
        image: Union[str, Path, bytes],
    ) -> VisionResult:
        """
        Detect and list objects in an image.

        Args:
            image: Image to analyze

        Returns:
            VisionResult with object list
        """
        return await self.analyze(image, AnalysisType.OBJECTS)

    async def understand_scene(
        self,
        image: Union[str, Path, bytes],
    ) -> VisionResult:
        """
        Understand the scene in an image.

        Args:
            image: Image to analyze

        Returns:
            VisionResult with scene understanding
        """
        return await self.analyze(image, AnalysisType.SCENE)

    async def compare_images(
        self,
        image1: Union[str, Path, bytes],
        image2: Union[str, Path, bytes],
        aspect: Optional[str] = None,
    ) -> VisionResult:
        """
        Compare two images.

        Note: This processes images sequentially since llava
        handles one image at a time.

        Args:
            image1: First image
            image2: Second image
            aspect: Specific aspect to compare (optional)

        Returns:
            VisionResult with comparison
        """
        # Analyze both images
        result1 = await self.describe(image1, detailed=True)
        result2 = await self.describe(image2, detailed=True)

        # Build comparison prompt
        comparison_prompt = f"""Based on these two image descriptions, provide a comparison:

Image 1: {result1.content}

Image 2: {result2.content}

"""
        if aspect:
            comparison_prompt += f"Focus specifically on: {aspect}\n"

        comparison_prompt += "Describe the similarities and differences between these images."

        # Use a text-based comparison (no image needed)
        from core.providers.ollama import Message

        messages = [
            Message(role="user", content=comparison_prompt)
        ]

        response = await self.ollama.chat(messages)

        return VisionResult(
            content=response.content,
            analysis_type=AnalysisType.CUSTOM,
            model=response.model,
        )

    async def _process_image(
        self,
        image: Union[str, Path, bytes],
    ) -> ProcessedImage:
        """Process an image for analysis"""
        if isinstance(image, bytes):
            return self.processor.process_bytes(image)
        elif isinstance(image, (str, Path)):
            path = Path(image)
            if path.exists():
                return self.processor.process_file(path)
            else:
                # Assume it's base64
                return self.processor.process_base64(str(image))
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")


# =============================================================================
# Factory Function
# =============================================================================

_vision_service: Optional[VisionService] = None


def get_vision_service() -> VisionService:
    """Get or create the vision service singleton"""
    global _vision_service
    if _vision_service is None:
        _vision_service = VisionService()
    return _vision_service


# =============================================================================
# Convenience Functions
# =============================================================================

async def describe_image(image: Union[str, Path, bytes]) -> str:
    """
    Convenience function to describe an image.

    Args:
        image: Image to describe

    Returns:
        Description text
    """
    service = get_vision_service()
    result = await service.describe(image)
    return result.content


async def ask_about_image(
    image: Union[str, Path, bytes],
    question: str,
) -> str:
    """
    Convenience function to ask about an image.

    Args:
        image: Image to analyze
        question: Question to answer

    Returns:
        Answer text
    """
    service = get_vision_service()
    result = await service.ask(image, question)
    return result.content
