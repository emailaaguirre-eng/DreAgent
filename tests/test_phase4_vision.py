"""
=============================================================================
HUMMINGBIRD-LEA - Phase 4 Vision/OCR Tests
Powered by CoDre-X | B & D Servicing LLC
=============================================================================
Tests for Phase 4 Vision/OCR components:
- Image processing utilities
- Vision service
- OCR service
- Image analysis engine
- Agent mixin
=============================================================================
"""

import pytest
import tempfile
import base64
from pathlib import Path

# Import Phase 4 components
from core.services.vision import (
    # Image utilities
    ImageFormat,
    ImageInfo,
    ProcessedImage,
    ImageProcessor,
    get_image_processor,
    validate_image_file,
    validate_base64_image,
    MAX_IMAGE_SIZE,
    # Vision service
    AnalysisType,
    VisionResult,
    VisionService,
    get_vision_service,
    # OCR service
    OCRLanguage,
    TextRegion,
    OCRResult,
    OCRService,
    get_ocr_service,
    # Analysis engine
    ImageCategory,
    AnalysisMode,
    ImageAnalysisResult,
    ImageAnalysisEngine,
    get_analysis_engine,
    # Agent mixin
    VisionMixin,
    should_use_vision,
    detect_image_intent,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_image_bytes():
    """Create a simple test image (1x1 white pixel PNG)"""
    # Minimal valid PNG image
    png_data = bytes([
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  # PNG signature
        0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,  # IHDR chunk
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,  # 1x1 dimensions
        0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,  # 8-bit RGB
        0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,  # IDAT chunk
        0x54, 0x08, 0xD7, 0x63, 0xF8, 0xFF, 0xFF, 0xFF,  # compressed data
        0x00, 0x05, 0xFE, 0x02, 0xFE, 0xDC, 0xCC, 0x59,  # checksum
        0xE7, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E,  # IEND chunk
        0x44, 0xAE, 0x42, 0x60, 0x82,
    ])
    return png_data


@pytest.fixture
def sample_image_base64(sample_image_bytes):
    """Get base64 encoded test image"""
    return base64.b64encode(sample_image_bytes).decode("utf-8")


@pytest.fixture
def sample_image_file(tmp_path, sample_image_bytes):
    """Create a test image file"""
    image_path = tmp_path / "test_image.png"
    image_path.write_bytes(sample_image_bytes)
    return image_path


# =============================================================================
# Image Format Tests
# =============================================================================

class TestImageFormat:
    """Tests for ImageFormat enum"""

    def test_from_mime_jpeg(self):
        """Test MIME type to format conversion"""
        assert ImageFormat.from_mime("image/jpeg") == ImageFormat.JPEG
        assert ImageFormat.from_mime("image/jpg") == ImageFormat.JPEG

    def test_from_mime_png(self):
        assert ImageFormat.from_mime("image/png") == ImageFormat.PNG

    def test_from_mime_unknown(self):
        assert ImageFormat.from_mime("image/unknown") == ImageFormat.UNKNOWN

    def test_from_extension(self):
        assert ImageFormat.from_extension(".jpg") == ImageFormat.JPEG
        assert ImageFormat.from_extension("jpeg") == ImageFormat.JPEG
        assert ImageFormat.from_extension(".png") == ImageFormat.PNG


# =============================================================================
# Image Processor Tests
# =============================================================================

class TestImageProcessor:
    """Tests for image processing utilities"""

    def test_processor_initialization(self):
        """Test processor initialization"""
        processor = ImageProcessor()
        assert processor.max_size == MAX_IMAGE_SIZE
        assert processor.quality == 85

    def test_processor_singleton(self):
        """Test processor singleton pattern"""
        processor1 = get_image_processor()
        processor2 = get_image_processor()
        assert processor1 is processor2

    def test_process_bytes(self, sample_image_bytes):
        """Test processing image from bytes"""
        processor = ImageProcessor()

        try:
            result = processor.process_bytes(sample_image_bytes)
            assert isinstance(result, ProcessedImage)
            assert result.base64_data
            assert result.info.width > 0
            assert result.info.height > 0
        except ImportError:
            pytest.skip("Pillow not installed")

    def test_process_base64(self, sample_image_base64):
        """Test processing image from base64"""
        processor = ImageProcessor()

        try:
            result = processor.process_base64(sample_image_base64)
            assert isinstance(result, ProcessedImage)
            assert result.base64_data
        except ImportError:
            pytest.skip("Pillow not installed")


# =============================================================================
# Validation Tests
# =============================================================================

class TestValidation:
    """Tests for image validation functions"""

    def test_validate_image_file_not_found(self, tmp_path):
        """Test validation of non-existent file"""
        fake_path = tmp_path / "nonexistent.jpg"
        is_valid, error = validate_image_file(fake_path)
        assert is_valid is False
        assert "not found" in error.lower()

    def test_validate_image_file_wrong_extension(self, tmp_path):
        """Test validation of wrong file extension"""
        bad_file = tmp_path / "test.xyz"
        bad_file.write_text("not an image")
        is_valid, error = validate_image_file(bad_file)
        assert is_valid is False
        assert "unsupported" in error.lower()

    def test_validate_image_file_valid(self, sample_image_file):
        """Test validation of valid image file"""
        try:
            is_valid, error = validate_image_file(sample_image_file)
            assert is_valid is True
            assert error is None
        except ImportError:
            pytest.skip("Pillow not installed")

    def test_validate_base64_invalid(self):
        """Test validation of invalid base64"""
        is_valid, error = validate_base64_image("not valid base64!!!")
        assert is_valid is False

    def test_validate_base64_valid(self, sample_image_base64):
        """Test validation of valid base64 image"""
        try:
            is_valid, error = validate_base64_image(sample_image_base64)
            assert is_valid is True
            assert error is None
        except ImportError:
            pytest.skip("Pillow not installed")


# =============================================================================
# Analysis Type Tests
# =============================================================================

class TestAnalysisTypes:
    """Tests for analysis type enums"""

    def test_analysis_type_values(self):
        """Test AnalysisType enum values"""
        assert AnalysisType.DESCRIBE.value == "describe"
        assert AnalysisType.DETAILED.value == "detailed"
        assert AnalysisType.TEXT.value == "text"
        assert AnalysisType.DOCUMENT.value == "document"

    def test_image_category_values(self):
        """Test ImageCategory enum values"""
        assert ImageCategory.DOCUMENT.value == "document"
        assert ImageCategory.PHOTO.value == "photo"
        assert ImageCategory.SCREENSHOT.value == "screenshot"

    def test_analysis_mode_values(self):
        """Test AnalysisMode enum values"""
        assert AnalysisMode.AUTO.value == "auto"
        assert AnalysisMode.VISION_ONLY.value == "vision"
        assert AnalysisMode.OCR_ONLY.value == "ocr"
        assert AnalysisMode.COMBINED.value == "combined"


# =============================================================================
# OCR Language Tests
# =============================================================================

class TestOCRLanguage:
    """Tests for OCR language support"""

    def test_ocr_language_values(self):
        """Test OCRLanguage enum values"""
        assert OCRLanguage.ENGLISH.value == "en"
        assert OCRLanguage.SPANISH.value == "es"
        assert OCRLanguage.CHINESE_SIMPLIFIED.value == "ch_sim"


# =============================================================================
# Vision Intent Detection Tests
# =============================================================================

class TestIntentDetection:
    """Tests for vision intent detection"""

    def test_should_use_vision_with_image(self):
        """Test vision detection when image is provided"""
        assert should_use_vision("hello", has_image=True) is True

    def test_should_use_vision_keywords(self):
        """Test vision detection with keywords"""
        assert should_use_vision("describe this image") is True
        assert should_use_vision("what's in this picture?") is True
        assert should_use_vision("extract text from this") is True
        assert should_use_vision("analyze this screenshot") is True

    def test_should_not_use_vision(self):
        """Test no vision for non-image queries"""
        assert should_use_vision("hello") is False
        assert should_use_vision("what is the weather?") is False
        assert should_use_vision("write some code") is False

    def test_detect_image_intent_extract(self):
        """Test extract text intent detection"""
        assert detect_image_intent("extract the text from this") == "extract_text"
        assert detect_image_intent("what does it say?") == "extract_text"
        assert detect_image_intent("OCR this image") == "extract_text"

    def test_detect_image_intent_document(self):
        """Test document intent detection"""
        assert detect_image_intent("analyze this document") == "analyze_document"
        assert detect_image_intent("scan this form") == "analyze_document"
        assert detect_image_intent("read this receipt") == "analyze_document"

    def test_detect_image_intent_question(self):
        """Test question intent detection"""
        assert detect_image_intent("what color is the car?") == "question"
        assert detect_image_intent("how many people are there?") == "question"
        assert detect_image_intent("is there a dog in the image?") == "question"

    def test_detect_image_intent_describe(self):
        """Test describe intent detection"""
        assert detect_image_intent("describe this") == "describe"
        assert detect_image_intent("tell me about this image") == "describe"

    def test_detect_image_intent_general(self):
        """Test general intent"""
        assert detect_image_intent("process this") == "general"


# =============================================================================
# Data Class Tests
# =============================================================================

class TestDataClasses:
    """Tests for data classes"""

    def test_vision_result_has_text(self):
        """Test VisionResult.has_text property"""
        result_with_text = VisionResult(
            content="This is some visible text in the image that is long enough.",
            analysis_type=AnalysisType.TEXT,
        )
        assert result_with_text.has_text is True

        result_no_text = VisionResult(
            content="No text detected in the image",
            analysis_type=AnalysisType.TEXT,
        )
        assert result_no_text.has_text is False

    def test_text_region_properties(self):
        """Test TextRegion bounding box properties"""
        region = TextRegion(
            text="Hello",
            confidence=0.95,
            bounding_box=[[10, 20], [100, 20], [100, 50], [10, 50]],
            position=(10, 20),
            size=(90, 30),
        )
        assert region.top_left == (10, 20)
        assert region.bottom_right == (100, 50)

    def test_ocr_result_high_confidence(self):
        """Test OCRResult high confidence text filtering"""
        regions = [
            TextRegion(
                text="High",
                confidence=0.9,
                bounding_box=[[0, 0], [10, 0], [10, 10], [0, 10]],
                position=(0, 0),
                size=(10, 10),
            ),
            TextRegion(
                text="Low",
                confidence=0.3,
                bounding_box=[[0, 0], [10, 0], [10, 10], [0, 10]],
                position=(0, 0),
                size=(10, 10),
            ),
        ]

        result = OCRResult(
            text="High Low",
            regions=regions,
            languages=["en"],
            total_confidence=0.6,
            word_count=2,
            processing_time_ms=100,
            image_size=(100, 100),
        )

        high_conf_text = result.get_high_confidence_text(threshold=0.7)
        assert "High" in high_conf_text
        assert "Low" not in high_conf_text

    def test_image_analysis_result_summary(self):
        """Test ImageAnalysisResult summary"""
        result = ImageAnalysisResult(
            description="A beautiful sunset",
            extracted_text="Sign: Exit",
            category=ImageCategory.PHOTO,
            analysis_mode=AnalysisMode.COMBINED,
            has_text=True,
        )

        summary = result.summary
        assert "Description" in summary
        assert "sunset" in summary
        assert "Extracted Text" in summary
        assert "Exit" in summary

    def test_image_analysis_result_to_dict(self):
        """Test ImageAnalysisResult serialization"""
        result = ImageAnalysisResult(
            description="Test",
            extracted_text="",
            category=ImageCategory.PHOTO,
            analysis_mode=AnalysisMode.VISION_ONLY,
        )

        data = result.to_dict()
        assert data["description"] == "Test"
        assert data["category"] == "photo"
        assert data["analysis_mode"] == "vision"


# =============================================================================
# Service Singleton Tests
# =============================================================================

class TestServiceSingletons:
    """Tests for service singleton patterns"""

    def test_vision_service_singleton(self):
        """Test VisionService singleton"""
        service1 = get_vision_service()
        service2 = get_vision_service()
        assert service1 is service2

    def test_ocr_service_singleton(self):
        """Test OCRService singleton"""
        service1 = get_ocr_service()
        service2 = get_ocr_service()
        assert service1 is service2

    def test_analysis_engine_singleton(self):
        """Test ImageAnalysisEngine singleton"""
        engine1 = get_analysis_engine()
        engine2 = get_analysis_engine()
        assert engine1 is engine2


# =============================================================================
# Integration Tests (require Ollama/EasyOCR)
# =============================================================================

class TestVisionIntegration:
    """Integration tests requiring Ollama"""

    @pytest.mark.skip(reason="Requires Ollama to be running with llava model")
    @pytest.mark.asyncio
    async def test_vision_describe(self, sample_image_file):
        """Test vision description"""
        service = get_vision_service()
        result = await service.describe(sample_image_file)
        assert result.content
        assert result.analysis_type == AnalysisType.DESCRIBE

    @pytest.mark.skip(reason="Requires Ollama to be running with llava model")
    @pytest.mark.asyncio
    async def test_vision_ask(self, sample_image_file):
        """Test visual question answering"""
        service = get_vision_service()
        result = await service.ask(sample_image_file, "What is in this image?")
        assert result.content

    @pytest.mark.skip(reason="Requires EasyOCR to be installed")
    @pytest.mark.asyncio
    async def test_ocr_extract(self, sample_image_file):
        """Test OCR text extraction"""
        service = get_ocr_service()
        result = await service.extract_text(sample_image_file)
        assert isinstance(result, OCRResult)

    @pytest.mark.skip(reason="Requires Ollama and EasyOCR")
    @pytest.mark.asyncio
    async def test_full_analysis_pipeline(self, sample_image_file):
        """Test full image analysis pipeline"""
        engine = get_analysis_engine()
        result = await engine.analyze(sample_image_file)

        assert isinstance(result, ImageAnalysisResult)
        assert result.category in ImageCategory


# =============================================================================
# VisionMixin Tests
# =============================================================================

class TestVisionMixin:
    """Tests for VisionMixin agent integration"""

    def test_mixin_properties(self):
        """Test VisionMixin lazy property initialization"""
        class MockAgent(VisionMixin):
            def __init__(self):
                super().__init__()

        agent = MockAgent()

        # Properties should be lazy-loaded
        assert agent._vision_engine is None
        assert agent._vision_service is None
        assert agent._ocr_service is None


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
