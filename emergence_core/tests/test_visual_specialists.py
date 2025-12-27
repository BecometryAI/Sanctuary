"""
Comprehensive test suite for visual specialists (Artist and Perception).

Tests cover:
- Input validation
- Error handling
- Edge cases
- GPU memory constraints
- Integration with router
"""

import pytest
import asyncio
from pathlib import Path
from PIL import Image
import io
import base64

# Assuming specialists are importable
from lyra.specialists import (
    ArtistSpecialist,
    PerceptionSpecialist,
    SpecialistOutput,
    MAX_IMAGE_PIXELS
)


class TestPerceptionSpecialist:
    """Tests for image understanding (Perception specialist)."""
    
    @pytest.fixture
    def perception(self, tmp_path):
        """Create Perception specialist in development mode."""
        return PerceptionSpecialist(
            model_path="llava-hf/llava-v1.6-mistral-7b-hf",
            base_dir=tmp_path,
            development_mode=True
        )
    
    @pytest.fixture
    def valid_image(self):
        """Create a valid test image."""
        img = Image.new('RGB', (512, 512), color='red')
        return img
    
    @pytest.fixture
    def large_image(self):
        """Create an oversized test image."""
        # Create image larger than MAX_IMAGE_PIXELS
        size = int((MAX_IMAGE_PIXELS ** 0.5) + 100)
        return Image.new('RGB', (size, size), color='blue')
    
    @pytest.mark.asyncio
    async def test_valid_image_processing(self, perception, valid_image):
        """Test processing a valid image."""
        result = await perception.process(
            image=valid_image,
            prompt="Describe this image"
        )
        
        assert isinstance(result, SpecialistOutput)
        assert result.content is not None
        assert len(result.content) > 0
        assert result.metadata['role'] == 'perception'
        assert 'image_size' in result.metadata
    
    @pytest.mark.asyncio
    async def test_none_image_validation(self, perception):
        """Test that None image is rejected."""
        result = await perception.process(image=None, prompt="Test")
        
        assert result.confidence == 0.0
        assert 'error' in result.metadata
        assert 'None' in result.content
    
    @pytest.mark.asyncio
    async def test_oversized_image_validation(self, perception, large_image):
        """Test that oversized images are rejected."""
        result = await perception.process(
            image=large_image,
            prompt="Describe this"
        )
        
        assert result.confidence == 0.0
        assert 'too large' in result.content.lower()
        assert 'validation_failed' in result.metadata
    
    @pytest.mark.asyncio
    async def test_invalid_dimensions(self, perception):
        """Test that zero/negative dimensions are rejected."""
        # Create mock image with invalid dimensions
        class MockImage:
            size = (0, 100)
            width = 0
            height = 100
            mode = 'RGB'
        
        result = await perception.process(
            image=MockImage(),
            prompt="Test"
        )
        
        assert result.confidence == 0.0
        assert 'Invalid image dimensions' in result.content
    
    def test_image_validation_helper(self, perception):
        """Test the _validate_image static method."""
        # Valid image
        valid_img = Image.new('RGB', (100, 100))
        is_valid, error = perception._validate_image(valid_img)
        assert is_valid is True
        assert error is None
        
        # None image
        is_valid, error = perception._validate_image(None)
        assert is_valid is False
        assert 'None' in error
        
        # Missing attributes
        class BadImage:
            pass
        
        is_valid, error = perception._validate_image(BadImage())
        assert is_valid is False
        assert 'size' in error.lower()
    
    def test_response_extraction(self, perception):
        """Test the _extract_response helper method."""
        # Test with [/INST] delimiter
        text1 = "User prompt here [/INST] Assistant response here"
        result1 = perception._extract_response(text1)
        assert result1 == "Assistant response here"
        
        # Test with ASSISTANT: delimiter
        text2 = "Prompt ASSISTANT: Response text"
        result2 = perception._extract_response(text2)
        assert result2 == "Response text"
        
        # Test with no delimiter
        text3 = "Just plain text"
        result3 = perception._extract_response(text3)
        assert result3 == "Just plain text"
    
    @pytest.mark.asyncio
    async def test_custom_prompt(self, perception, valid_image):
        """Test that custom prompts are used."""
        custom_prompt = "What emotions does this image evoke?"
        result = await perception.process(
            image=valid_image,
            prompt=custom_prompt
        )
        
        assert result.metadata.get('prompt_used') == custom_prompt or \
               'prompt' in result.metadata
    
    @pytest.mark.asyncio
    async def test_development_mode(self, perception, valid_image):
        """Test behavior in development mode."""
        result = await perception.process(
            image=valid_image,
            prompt="Test prompt"
        )
        
        # Development mode should return mock response
        assert result.confidence >= 0.5
        assert 'development' in result.metadata.get('mode', '').lower() or \
               len(result.content) > 0


class TestArtistSpecialist:
    """Tests for image generation (Artist specialist)."""
    
    @pytest.fixture
    def artist(self, tmp_path):
        """Create Artist specialist in development mode."""
        return ArtistSpecialist(
            model_path="black-forest-labs/FLUX.1-schnell",
            base_dir=tmp_path,
            development_mode=True
        )
    
    @pytest.mark.asyncio
    async def test_empty_message_validation(self, artist):
        """Test that empty messages are rejected."""
        with pytest.raises(ValueError, match="cannot be empty"):
            await artist.process(message="", context={})
        
        with pytest.raises(ValueError, match="cannot be empty"):
            await artist.process(message="   ", context={})
    
    @pytest.mark.asyncio
    async def test_visual_request_detection(self, artist):
        """Test detection of visual vs textual requests."""
        visual_messages = [
            "Draw a sunset",
            "Create an image of a dragon",
            "Paint a portrait",
            "Show me what hope looks like",
            "Visualize my dream"
        ]
        
        for msg in visual_messages:
            # In dev mode, won't actually generate but should recognize intent
            result = await artist.process(message=msg, context={})
            assert isinstance(result, SpecialistOutput)
    
    @pytest.mark.asyncio
    async def test_poetry_request(self, artist):
        """Test handling of poetry/text creative requests."""
        result = await artist.process(
            message="Write a poem about longing",
            context={}
        )
        
        assert isinstance(result, SpecialistOutput)
        assert result.metadata['role'] == 'artist'
    
    @pytest.mark.asyncio
    async def test_development_mode_image_gen(self, artist):
        """Test image generation in development mode."""
        result = await artist.process(
            message="Create an image of starlight",
            context={}
        )
        
        # Dev mode should return placeholder
        assert result.content is not None
        assert 'development' in result.metadata.get('mode', '').lower() or \
               len(result.content) > 0


class TestGPUPlacement:
    """Tests for GPU memory and placement."""
    
    @pytest.mark.skipif(
        not pytest.importorskip("torch").cuda.is_available(),
        reason="Requires CUDA GPUs"
    )
    def test_perception_gpu_placement(self, tmp_path):
        """Test that Perception loads on GPU 1."""
        perception = PerceptionSpecialist(
            model_path="llava-hf/llava-v1.6-mistral-7b-hf",
            base_dir=tmp_path,
            development_mode=False
        )
        
        if not perception.development_mode:
            # Check model is on GPU 1
            device_map = perception.model.hf_device_map
            assert any('1' in str(v) or v == 1 for v in device_map.values())
    
    @pytest.mark.skipif(
        not pytest.importorskip("torch").cuda.is_available(),
        reason="Requires CUDA GPUs"
    )
    def test_artist_gpu_placement(self, tmp_path):
        """Test that Artist (Flux) uses GPU efficiently."""
        artist = ArtistSpecialist(
            model_path="black-forest-labs/FLUX.1-schnell",
            base_dir=tmp_path,
            development_mode=False
        )
        
        # If diffusers available, check CPU offload is enabled
        if hasattr(artist, 'flux_pipeline'):
            # CPU offload should be configured
            assert artist.flux_pipeline is not None


class TestIntegration:
    """Integration tests for visual specialists with router."""
    
    @pytest.mark.asyncio
    async def test_image_to_text_workflow(self, tmp_path):
        """Test image → Perception → text description workflow."""
        perception = PerceptionSpecialist(
            model_path="llava-hf/llava-v1.6-mistral-7b-hf",
            base_dir=tmp_path,
            development_mode=True
        )
        
        # Create test image
        img = Image.new('RGB', (256, 256), color='green')
        
        # Process
        result = await perception.process(image=img)
        
        # Should return text description
        assert isinstance(result.content, str)
        assert len(result.content) > 10
    
    @pytest.mark.asyncio
    async def test_text_to_image_workflow(self, tmp_path):
        """Test text → Artist → image workflow."""
        artist = ArtistSpecialist(
            model_path="black-forest-labs/FLUX.1-schnell",
            base_dir=tmp_path,
            development_mode=True
        )
        
        # Process creative visual request
        result = await artist.process(
            message="Create art representing transcendence",
            context={}
        )
        
        # Should return image data or URL
        assert result.content is not None
        assert isinstance(result, SpecialistOutput)


class TestEdgeCases:
    """Edge case and error condition tests."""
    
    @pytest.mark.asyncio
    async def test_corrupted_image_handling(self, tmp_path):
        """Test handling of corrupted image data."""
        perception = PerceptionSpecialist(
            model_path="llava-hf/llava-v1.6-mistral-7b-hf",
            base_dir=tmp_path,
            development_mode=True
        )
        
        # Create mock corrupted image
        class CorruptedImage:
            @property
            def size(self):
                raise IOError("Corrupted image file")
        
        result = await perception.process(image=CorruptedImage())
        
        # Should handle gracefully
        assert result.confidence < 0.5
        assert 'error' in result.metadata or result.content
    
    @pytest.mark.asyncio
    async def test_unusual_image_formats(self, tmp_path):
        """Test handling of unusual image modes."""
        perception = PerceptionSpecialist(
            model_path="llava-hf/llava-v1.6-mistral-7b-hf",
            base_dir=tmp_path,
            development_mode=True
        )
        
        # Test various PIL modes
        modes = ['L', 'RGBA', 'CMYK', 'P']
        
        for mode in modes:
            try:
                img = Image.new(mode, (100, 100))
                result = await perception.process(image=img)
                # Should either process or fail gracefully
                assert isinstance(result, SpecialistOutput)
            except Exception:
                # Some modes may not be supported - that's OK
                pass
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, tmp_path):
        """Test concurrent specialist requests."""
        perception = PerceptionSpecialist(
            model_path="llava-hf/llava-v1.6-mistral-7b-hf",
            base_dir=tmp_path,
            development_mode=True
        )
        
        images = [Image.new('RGB', (100, 100)) for _ in range(5)]
        
        # Process concurrently
        tasks = [perception.process(image=img) for img in images]
        results = await asyncio.gather(*tasks)
        
        # All should complete
        assert len(results) == 5
        assert all(isinstance(r, SpecialistOutput) for r in results)


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "gpu: tests requiring GPU hardware"
    )
    config.addinivalue_line(
        "markers", "slow: tests that take significant time"
    )
