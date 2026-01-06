#!/usr/bin/env python3
"""
Standalone test for competitive attention dynamics without full dependencies.
This allows testing the core competitive dynamics logic without needing chromadb, etc.
"""

import sys
import os

# Add the emergence_core directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'emergence_core'))

# Mock the workspace module to avoid dependency issues
from unittest.mock import Mock
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class Percept:
    """Mock Percept for testing."""
    id: str
    modality: str
    raw: Any
    complexity: int = 1
    timestamp: datetime = None
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}
        if self.id is None:
            self.id = str(uuid.uuid4())

# Mock the workspace module before importing attention
sys.modules['lyra.cognitive_core.workspace'] = Mock()
sys.modules['lyra.cognitive_core.workspace'].Percept = Percept
sys.modules['lyra.cognitive_core.workspace'].GlobalWorkspace = Mock

# Now import just what we need from the attention module
import importlib.util
spec = importlib.util.spec_from_file_location(
    "attention",
    os.path.join(os.path.dirname(__file__), 'emergence_core', 'lyra', 'cognitive_core', 'attention.py')
)
attention = importlib.util.module_from_spec(spec)

# Inject our mock Percept before executing
attention.Percept = Percept
spec.loader.exec_module(attention)

def test_competitive_attention_basic():
    """Test basic CompetitiveAttention functionality."""
    print("Testing CompetitiveAttention initialization...")
    comp = attention.CompetitiveAttention(
        inhibition_strength=0.3,
        ignition_threshold=0.5,
        iterations=10,
    )
    assert comp.inhibition_strength == 0.3
    assert comp.ignition_threshold == 0.5
    print("✓ CompetitiveAttention initialization works")

def test_lateral_inhibition():
    """Test that lateral inhibition suppresses low-activation percepts."""
    print("\nTesting lateral inhibition...")
    comp = attention.CompetitiveAttention(
        inhibition_strength=0.4,
        iterations=10,
    )
    
    percepts = [
        Percept(id="p1", modality="text", raw="High priority", embedding=[1.0, 0.0, 0.0]),
        Percept(id="p2", modality="text", raw="Low priority", embedding=[0.0, 1.0, 0.0]),
    ]
    
    base_scores = {
        "p1": 0.9,  # High score
        "p2": 0.3,  # Low score
    }
    
    sorted_percepts, metrics = comp.compete(percepts, base_scores)
    
    high_activation = comp.activations["p1"]
    low_activation = comp.activations["p2"]
    
    print(f"  High activation: {high_activation:.3f}")
    print(f"  Low activation: {low_activation:.3f}")
    
    assert high_activation > low_activation, "High should suppress low"
    assert metrics.inhibition_events > 0, "Inhibition events should be tracked"
    print("✓ Lateral inhibition works correctly")

def test_ignition_threshold():
    """Test that ignition threshold filters low-activation percepts."""
    print("\nTesting ignition threshold...")
    comp = attention.CompetitiveAttention(
        inhibition_strength=0.5,
        ignition_threshold=0.6,
        iterations=10,
    )
    
    percepts = [
        Percept(id="p1", modality="text", raw="High", embedding=[1.0, 0.0, 0.0]),
        Percept(id="p2", modality="text", raw="Low", embedding=[0.0, 1.0, 0.0]),
    ]
    
    base_scores = {
        "p1": 0.9,  # Should exceed threshold
        "p2": 0.3,  # Should be suppressed
    }
    
    selected, metrics = comp.select_for_workspace(percepts, base_scores)
    
    print(f"  Selected: {len(selected)} out of {len(percepts)}")
    print(f"  Winners: {len(metrics.winner_ids)}")
    print(f"  Suppressed: {len(metrics.suppressed_percepts)}")
    
    assert len(selected) < len(percepts), "Should filter some percepts"
    assert len(metrics.suppressed_percepts) > 0, "Should track suppressed percepts"
    print("✓ Ignition threshold works correctly")

def test_coalition_formation():
    """Test coalition formation for related percepts."""
    print("\nTesting coalition formation...")
    comp = attention.CompetitiveAttention(coalition_boost=0.3)
    
    percepts = [
        Percept(id="p1", modality="text", raw="Topic A content", embedding=[1.0, 0.0, 0.0]),
        Percept(id="p2", modality="text", raw="More topic A", embedding=[0.95, 0.05, 0.0]),
        Percept(id="p3", modality="text", raw="Unrelated topic B", embedding=[0.0, 0.0, 1.0]),
    ]
    
    base_scores = {p.id: 0.5 for p in percepts}
    
    _, metrics = comp.compete(percepts, base_scores)
    
    print(f"  Coalition formations: {len(metrics.coalition_formations)}")
    total_partners = sum(len(partners) for partners in metrics.coalition_formations.values())
    print(f"  Total coalition links: {total_partners}")
    
    assert len(metrics.coalition_formations) == len(percepts), "All percepts tracked"
    print("✓ Coalition formation works")

def test_competition_metrics():
    """Test that competition metrics are tracked properly."""
    print("\nTesting competition metrics...")
    comp = attention.CompetitiveAttention()
    
    percepts = [
        Percept(id="p1", modality="text", raw="P1", embedding=[1.0, 0.0, 0.0]),
        Percept(id="p2", modality="text", raw="P2", embedding=[0.0, 1.0, 0.0]),
    ]
    
    base_scores = {p.id: 0.5 for p in percepts}
    
    _, metrics = comp.compete(percepts, base_scores)
    
    print(f"  Inhibition events: {metrics.inhibition_events}")
    print(f"  Activation spread before: {metrics.activation_spread_before:.3f}")
    print(f"  Activation spread after: {metrics.activation_spread_after:.3f}")
    
    assert isinstance(metrics.inhibition_events, int)
    assert isinstance(metrics.suppressed_percepts, list)
    assert isinstance(metrics.winner_ids, list)
    assert isinstance(metrics.coalition_formations, dict)
    print("✓ Competition metrics tracked correctly")

def main():
    """Run all tests."""
    print("=" * 60)
    print("Running Competitive Attention Standalone Tests")
    print("=" * 60)
    
    try:
        test_competitive_attention_basic()
        test_lateral_inhibition()
        test_ignition_threshold()
        test_coalition_formation()
        test_competition_metrics()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
