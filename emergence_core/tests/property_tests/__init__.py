"""
Property-based tests for Lyra Emergence cognitive architecture.

This package contains property-based tests using Hypothesis to validate
invariants and robustness of the cognitive architecture under diverse inputs.

Modules:
    strategies: Custom Hypothesis strategies for generating test data
    test_workspace_properties: Tests for GlobalWorkspace invariants
    test_attention_properties: Tests for AttentionController invariants
    test_memory_properties: Tests for memory system invariants
    test_emotion_properties: Tests for emotional dynamics invariants
"""

__all__ = [
    "strategies",
    "test_workspace_properties",
    "test_attention_properties",
    "test_memory_properties",
    "test_emotion_properties",
]
