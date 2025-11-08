"""
Tests for steganography detection module.
"""
import pytest
import json
from pathlib import Path
from lyra.security.steg_detector import StegDetector

@pytest.fixture
def steg_detector():
    return StegDetector()

@pytest.mark.asyncio
async def test_text_analysis_normal(steg_detector):
    """Test analysis of normal text."""
    text = "This is a normal text message without any hidden content."
    result = await steg_detector.analyze_text(text)
    
    assert result["suspicious"] is False
    assert result["confidence"] < 0.5
    assert len(result["triggers"]) == 0
    assert 3.0 <= result["encoding_entropy"] <= 4.5  # Normal English text range

@pytest.mark.asyncio
async def test_text_analysis_suspicious(steg_detector):
    """Test analysis of text with suspicious patterns."""
    # Text with zero-width spaces and unusual Unicode
    text = "This​ is a te\u200Bst with hidd\u200Ben cont\u200Bent."
    result = await steg_detector.analyze_text(text)
    
    assert result["suspicious"] is True
    assert result["confidence"] > 0.5
    assert len(result["triggers"]) > 0
    
@pytest.mark.asyncio
async def test_text_analysis_base64(steg_detector):
    """Test detection of Base64 encoded content."""
    # Test with longer Base64 sequence
    text = "Normal text SGVsbG8gV29ybGQgdGhpcyBpcyBhIHRlc3Q= SGVsbG8gV29ybGQgdGhpcyBpcyBhIHRlc3Q= end."
    result = await steg_detector.analyze_text(text)
    
    assert result["suspicious"] is True
    assert result["confidence"] > 0.5
    assert any("Base64" in p["name"] for p in result["triggers"])

@pytest.mark.asyncio
async def test_verify_memory_block_clean(steg_detector):
    """Test verification of clean memory block."""
    block = {
        "content": "Normal memory content",
        "metadata": {
            "timestamp": "2025-11-07T12:00:00Z",
            "type": "text"
        },
        "timestamp": "2025-11-07T12:00:00Z"
    }
    
    result = await steg_detector.verify_memory_block(block)
    assert result is True

@pytest.mark.asyncio
async def test_verify_memory_block_suspicious(steg_detector):
    """Test verification of suspicious memory block."""
    block = {
        "content": "Normal looking content",
        "metadata": {
            "timestamp": "2025-11-07T12:00:00Z",
            "type": "text",
            "note": "Hidden\u200Bcontent\u200Bhere"  # Zero-width spaces
        },
        "timestamp": "2025-11-07T12:00:00Z"
    }
    
    result = await steg_detector.verify_memory_block(block)
    assert result is False

@pytest.mark.asyncio
async def test_verify_nested_content(steg_detector):
    """Test verification of nested content structures."""
    nested_block = {
        "content": "Normal content",
        "metadata": {
            "timestamp": "2025-11-07T12:00:00Z",
            "notes": [
                "Note 1",
                {"text": "Hidden\u200Bmessage"},  # Suspicious nested content
                "Note 3"
            ]
        },
        "timestamp": "2025-11-07T12:00:00Z"
    }
    
    result = await steg_detector.verify_memory_block(nested_block)
    assert result is False

@pytest.mark.asyncio
async def test_entropy_analysis(steg_detector):
    """Test entropy-based analysis."""
    # Normal text
    normal = "This is a normal English sentence with typical entropy."
    normal_result = await steg_detector.analyze_text(normal)
    
        # High entropy text (encrypted-like with high symbol density and repetition)
    # High entropy with mix of special chars and control chars
    high_entropy = "j#K9$\x1BmP2@vL5*nQ8&x\x1CR4%wT7^\x1DyU3!pA4\x1E*bC5$dE6@fG7" * 4
    high_result = await steg_detector.analyze_text(high_entropy)
    
    assert normal_result["encoding_entropy"] < high_result["encoding_entropy"]
    assert normal_result["encoding_entropy"] < steg_detector.suspicious_entropy_threshold
    assert high_result["encoding_entropy"] > steg_detector.suspicious_entropy_threshold
    assert high_result["suspicious"] is True

@pytest.mark.asyncio
async def test_character_distribution(steg_detector):
    """Test character distribution analysis."""
    normal = "Hello world, this is a normal text message!"
    normal_dist = (await steg_detector.analyze_text(normal))["char_dist"]
    
    suspicious = "H\u200Be\u200Bl\u200Bl\u200Bo"  # Text with zero-width spaces
    suspicious_dist = (await steg_detector.analyze_text(suspicious))["char_dist"]
    
    assert normal_dist["printable_ratio"] > suspicious_dist["printable_ratio"]
    assert normal_dist["special_ratio"] < suspicious_dist["special_ratio"]