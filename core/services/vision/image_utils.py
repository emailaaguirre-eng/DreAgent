"""
=============================================================================
HUMMINGBIRD-LEA - Image Utilities
Powered by CoDre-X | B & D Servicing LLC
=============================================================================
Utilities for image processing, encoding, and validation.

Features:
- Image loading from files, URLs, and base64
- Automatic resizing for model compatibility
- Format validation and conversion
- Base64 encoding/decoding
=============================================================================
"""

import base64
import io
import logging
import mimetypes
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, Tuple, List
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class ImageFormat(Enum):
    """Supported image formats"""
    JPEG = "jpeg"
    PNG = "png"
    GIF = "gif"
    WEBP = "webp"
    BMP = "bmp"
    UNKNOWN = "unknown"

    @classmethod
    def from_mime(cls, mime_type: str) -> "ImageFormat":
        """Get format from MIME type"""
        mapping = {
            "image/jpeg": cls.JPEG,
            "image/jpg": cls.JPEG,
            "image/png": cls.PNG,
            "image/gif": cls.GIF,
            "image/webp": cls.WEBP,
            "image/bmp": cls.BMP,
        }
        return mapping.get(mime_type.lower(), cls.UNKNOWN)

    @classmethod
    def from_extension(cls, ext: str) -> "ImageFormat":
        """Get format from file extension"""
        ext = ext.lower().lstrip(".")
        mapping = {
            "jpg": cls.JPEG,
            "jpeg": cls.JPEG,
            "png": cls.PNG,
            "gif": cls.GIF,
            "webp": cls.WEBP,
            "bmp": cls.BMP,
        }
        return mapping.get(ext, cls.UNKNOWN)


# Recommended sizes for vision models
MAX_IMAGE_SIZE = (1344, 1344)  # llava max resolution
THUMBNAIL_SIZE = (512, 512)
MAX_FILE_SIZE_MB = 20


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ImageInfo:
    """Information about an image"""
    width: int
    height: int
    format: ImageFormat
    file_size_bytes: int
    mode: str  # RGB, RGBA, L, etc.
    has_alpha: bool
    source: str  # file path, URL, or "base64"

    @property
    def aspect_ratio(self) -> float:
        """Get aspect ratio (width/height)"""
        return self.width / self.height if self.height > 0 else 1.0

    @property
    def file_size_mb(self) -> float:
        """Get file size in MB"""
        return self.file_size_bytes / (1024 * 1024)


@dataclass
class ProcessedImage:
    """A processed image ready for analysis"""
    base64_data: str              # Base64 encoded image data
    info: ImageInfo               # Image information
    original_size: Tuple[int, int]  # Original (width, height)
    processed_size: Tuple[int, int]  # After resizing (width, height)
    was_resized: bool
    processing_time_ms: float
    created_at: datetime = field(default_factory=datetime.utcnow)

    def get_data_url(self) -> str:
        """Get data URL for embedding in HTML/CSS"""
        mime = f"image/{self.info.format.value}"
        return f"data:{mime};base64,{self.base64_data}"


# =============================================================================
# Image Processor
# =============================================================================

class ImageProcessor:
    """
    Processes images for use with vision models.

    Handles:
    - Loading from various sources (file, bytes, base64)
    - Automatic resizing to fit model requirements
    - Format conversion
    - Base64 encoding
    """

    def __init__(
        self,
        max_size: Tuple[int, int] = MAX_IMAGE_SIZE,
        quality: int = 85,
        output_format: ImageFormat = ImageFormat.JPEG,
    ):
        """
        Initialize the processor.

        Args:
            max_size: Maximum (width, height) for output
            quality: JPEG quality (1-100)
            output_format: Output format for processed images
        """
        self.max_size = max_size
        self.quality = quality
        self.output_format = output_format

    def process_file(self, file_path: Union[str, Path]) -> ProcessedImage:
        """
        Process an image from a file path.

        Args:
            file_path: Path to the image file

        Returns:
            ProcessedImage ready for analysis
        """
        import time
        start_time = time.time()

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        # Load image
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("Pillow is required for image processing. Install with: pip install Pillow")

        with Image.open(path) as img:
            original_size = img.size
            file_size = path.stat().st_size

            # Get format
            img_format = ImageFormat.from_extension(path.suffix)
            if img_format == ImageFormat.UNKNOWN:
                img_format = ImageFormat.JPEG

            # Process
            processed_img, was_resized = self._resize_if_needed(img)
            base64_data = self._encode_to_base64(processed_img)

            processing_time = (time.time() - start_time) * 1000

            info = ImageInfo(
                width=processed_img.size[0],
                height=processed_img.size[1],
                format=self.output_format,
                file_size_bytes=len(base64.b64decode(base64_data)),
                mode=processed_img.mode,
                has_alpha=processed_img.mode in ("RGBA", "LA", "PA"),
                source=str(path.absolute()),
            )

            return ProcessedImage(
                base64_data=base64_data,
                info=info,
                original_size=original_size,
                processed_size=processed_img.size,
                was_resized=was_resized,
                processing_time_ms=processing_time,
            )

    def process_bytes(
        self,
        data: bytes,
        source: str = "bytes",
    ) -> ProcessedImage:
        """
        Process an image from bytes.

        Args:
            data: Raw image bytes
            source: Source description for metadata

        Returns:
            ProcessedImage ready for analysis
        """
        import time
        start_time = time.time()

        try:
            from PIL import Image
        except ImportError:
            raise ImportError("Pillow is required. Install with: pip install Pillow")

        with Image.open(io.BytesIO(data)) as img:
            original_size = img.size

            # Detect format
            img_format = ImageFormat.from_mime(Image.MIME.get(img.format, "image/jpeg"))

            # Process
            processed_img, was_resized = self._resize_if_needed(img)
            base64_data = self._encode_to_base64(processed_img)

            processing_time = (time.time() - start_time) * 1000

            info = ImageInfo(
                width=processed_img.size[0],
                height=processed_img.size[1],
                format=self.output_format,
                file_size_bytes=len(base64.b64decode(base64_data)),
                mode=processed_img.mode,
                has_alpha=processed_img.mode in ("RGBA", "LA", "PA"),
                source=source,
            )

            return ProcessedImage(
                base64_data=base64_data,
                info=info,
                original_size=original_size,
                processed_size=processed_img.size,
                was_resized=was_resized,
                processing_time_ms=processing_time,
            )

    def process_base64(
        self,
        base64_data: str,
        source: str = "base64",
    ) -> ProcessedImage:
        """
        Process an image from base64 string.

        Args:
            base64_data: Base64 encoded image
            source: Source description

        Returns:
            ProcessedImage ready for analysis
        """
        # Remove data URL prefix if present
        if "," in base64_data:
            base64_data = base64_data.split(",", 1)[1]

        # Decode and process
        data = base64.b64decode(base64_data)
        return self.process_bytes(data, source)

    def _resize_if_needed(self, img) -> Tuple[any, bool]:
        """Resize image if it exceeds max size"""
        from PIL import Image

        width, height = img.size

        # Check if resize needed
        if width <= self.max_size[0] and height <= self.max_size[1]:
            # Convert to RGB if needed (for JPEG output)
            if self.output_format == ImageFormat.JPEG and img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")
            return img, False

        # Calculate new size maintaining aspect ratio
        ratio = min(self.max_size[0] / width, self.max_size[1] / height)
        new_size = (int(width * ratio), int(height * ratio))

        # Resize with high quality
        resized = img.resize(new_size, Image.Resampling.LANCZOS)

        # Convert to RGB if needed
        if self.output_format == ImageFormat.JPEG and resized.mode in ("RGBA", "LA", "P"):
            resized = resized.convert("RGB")

        logger.debug(f"Resized image from {img.size} to {new_size}")
        return resized, True

    def _encode_to_base64(self, img) -> str:
        """Encode PIL image to base64"""
        buffer = io.BytesIO()

        # Save to buffer
        format_str = self.output_format.value.upper()
        if format_str == "JPEG":
            img.save(buffer, format="JPEG", quality=self.quality, optimize=True)
        else:
            img.save(buffer, format=format_str)

        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")


# =============================================================================
# Validation Functions
# =============================================================================

def validate_image_file(file_path: Union[str, Path]) -> Tuple[bool, Optional[str]]:
    """
    Validate an image file.

    Args:
        file_path: Path to the image

    Returns:
        Tuple of (is_valid, error_message)
    """
    path = Path(file_path)

    # Check existence
    if not path.exists():
        return False, f"File not found: {path}"

    # Check extension
    valid_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
    if path.suffix.lower() not in valid_extensions:
        return False, f"Unsupported format: {path.suffix}. Supported: {valid_extensions}"

    # Check file size
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return False, f"File too large: {size_mb:.1f}MB. Maximum: {MAX_FILE_SIZE_MB}MB"

    # Try to open with PIL
    try:
        from PIL import Image
        with Image.open(path) as img:
            img.verify()
        return True, None
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"


def validate_base64_image(data: str) -> Tuple[bool, Optional[str]]:
    """
    Validate a base64 encoded image.

    Args:
        data: Base64 string (with or without data URL prefix)

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Remove data URL prefix if present
        if "," in data:
            data = data.split(",", 1)[1]

        # Decode
        decoded = base64.b64decode(data)

        # Check size
        size_mb = len(decoded) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            return False, f"Image too large: {size_mb:.1f}MB. Maximum: {MAX_FILE_SIZE_MB}MB"

        # Validate with PIL
        from PIL import Image
        with Image.open(io.BytesIO(decoded)) as img:
            img.verify()

        return True, None

    except base64.binascii.Error:
        return False, "Invalid base64 encoding"
    except Exception as e:
        return False, f"Invalid image: {str(e)}"


# =============================================================================
# Factory Function
# =============================================================================

_processor_instance: Optional[ImageProcessor] = None


def get_image_processor() -> ImageProcessor:
    """Get or create the image processor singleton"""
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = ImageProcessor()
    return _processor_instance
