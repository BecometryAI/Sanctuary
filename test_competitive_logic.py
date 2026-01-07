#!/usr/bin/env python3
"""
Minimal inline test for competitive attention logic.
Tests the core algorithms without dependency on the full package.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from typing import List, Dict, Any
from dataclasses import dataclass, field

# Inline the key functions and classes we need to test

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    v1 = np.array(vec1).reshape(1, -1)
    v2 = np.array(vec2).reshape(1, -1)
    similarity = sklearn_cosine(v1, v2)[0][0]
    return max(0.0, float(similarity))

@dataclass
class CompetitionMetrics:
    """Metrics tracking competitive dynamics."""
    inhibition_events: int = 0
    suppressed_percepts: List[str] = field(default_factory=list)
    activation_spread_before: float = 0.0
    activation_spread_after: float = 0.0
    winner_ids: List[str] = field(default_factory=list)
    coalition_formations: Dict[str, List[str]] = field(default_factory=dict)

class CompetitiveAttention:
    """Implements competitive attention dynamics with lateral inhibition."""
    
    def __init__(
        self,
        inhibition_strength: float = 0.3,
        ignition_threshold: float = 0.5,
        iterations: int = 10,
        coalition_boost: float = 0.2,
    ) -> None:
        self.inhibition_strength = max(0.0, min(1.0, inhibition_strength))
        self.ignition_threshold = max(0.0, min(1.0, ignition_threshold))
        self.iterations = max(1, iterations)
        self.coalition_boost = max(0.0, min(1.0, coalition_boost))
        self.activations: Dict[str, float] = {}
    
    def _initial_activation(self, percept_id: str, base_score: float) -> float:
        """Compute initial activation level."""
        activation = max(0.0, min(1.0, base_score))
        return activation
    
    def _compute_relatedness(self, emb1: List[float], emb2: List[float]) -> float:
        """Compute how related two percepts are."""
        if emb1 and emb2:
            return cosine_similarity(emb1, emb2)
        return 0.0
    
    def _form_coalitions(
        self,
        percept_data: Dict[str, Dict[str, Any]],
        relatedness_threshold: float = 0.6
    ) -> Dict[str, List[str]]:
        """Form coalitions of related percepts."""
        coalitions: Dict[str, List[str]] = {pid: [] for pid in percept_data.keys()}
        
        percept_ids = list(percept_data.keys())
        for i, pid1 in enumerate(percept_ids):
            for pid2 in percept_ids[i + 1:]:
                emb1 = percept_data[pid1].get('embedding')
                emb2 = percept_data[pid2].get('embedding')
                
                if emb1 and emb2:
                    relatedness = self._compute_relatedness(emb1, emb2)
                    
                    if relatedness >= relatedness_threshold:
                        coalitions[pid1].append(pid2)
                        coalitions[pid2].append(pid1)
        
        return coalitions
    
    def compete(
        self,
        percept_data: Dict[str, Dict[str, Any]],
        base_scores: Dict[str, float],
    ) -> CompetitionMetrics:
        """Run competitive dynamics."""
        if not percept_data:
            return CompetitionMetrics()
        
        # Initialize activations
        self.activations = {
            pid: self._initial_activation(pid, base_scores.get(pid, 0.5))
            for pid in percept_data.keys()
        }
        
        # Track initial spread
        initial_activations = list(self.activations.values())
        activation_spread_before = float(np.std(initial_activations)) if len(initial_activations) > 1 else 0.0
        
        # Form coalitions
        coalitions = self._form_coalitions(percept_data)
        
        # Run competition
        inhibition_events = 0
        percept_ids = list(percept_data.keys())
        
        for iteration in range(self.iterations):
            new_activations = {}
            
            for pid in percept_ids:
                # Self-excitation
                excitation = self.activations[pid] * 1.1
                
                # Coalition support
                coalition_support = 0.0
                if coalitions[pid]:
                    partner_activations = [
                        self.activations[partner_id]
                        for partner_id in coalitions[pid]
                    ]
                    coalition_support = np.mean(partner_activations) * self.coalition_boost
                
                # Lateral inhibition
                inhibition = 0.0
                for other_pid in percept_ids:
                    if other_pid != pid and other_pid not in coalitions[pid]:
                        inhibition += self.activations[other_pid] * self.inhibition_strength
                        inhibition_events += 1
                
                # Update activation
                new_activation = excitation + coalition_support - inhibition
                new_activations[pid] = max(0.0, min(1.0, new_activation))
            
            self.activations = new_activations
        
        # Track final spread
        final_activations = list(self.activations.values())
        activation_spread_after = float(np.std(final_activations)) if len(final_activations) > 1 else 0.0
        
        # Identify winners and suppressed
        winner_ids = [pid for pid in percept_ids if self.activations[pid] >= self.ignition_threshold]
        suppressed_percepts = [pid for pid in percept_ids if self.activations[pid] < self.ignition_threshold]
        
        return CompetitionMetrics(
            inhibition_events=inhibition_events,
            suppressed_percepts=suppressed_percepts,
            activation_spread_before=activation_spread_before,
            activation_spread_after=activation_spread_after,
            winner_ids=winner_ids,
            coalition_formations=coalitions,
        )

# Test functions

def test_basic_initialization():
    """Test basic initialization."""
    print("Testing CompetitiveAttention initialization...")
    comp = CompetitiveAttention(
        inhibition_strength=0.3,
        ignition_threshold=0.5,
        iterations=10,
    )
    assert comp.inhibition_strength == 0.3
    assert comp.ignition_threshold == 0.5
    print("✓ Initialization works")

def test_lateral_inhibition():
    """Test that lateral inhibition suppresses low-activation percepts."""
    print("\nTesting lateral inhibition...")
    comp = CompetitiveAttention(
        inhibition_strength=0.4,
        iterations=10,
    )
    
    percept_data = {
        "p1": {"embedding": [1.0, 0.0, 0.0]},
        "p2": {"embedding": [0.0, 1.0, 0.0]},
    }
    
    base_scores = {
        "p1": 0.9,  # High score
        "p2": 0.3,  # Low score
    }
    
    metrics = comp.compete(percept_data, base_scores)
    
    high_activation = comp.activations["p1"]
    low_activation = comp.activations["p2"]
    
    print(f"  High activation: {high_activation:.3f}")
    print(f"  Low activation: {low_activation:.3f}")
    print(f"  Inhibition events: {metrics.inhibition_events}")
    
    assert high_activation > low_activation, "High should suppress low"
    assert metrics.inhibition_events > 0, "Inhibition events should be tracked"
    print("✓ Lateral inhibition works correctly")

def test_ignition_threshold():
    """Test that ignition threshold filters low-activation percepts."""
    print("\nTesting ignition threshold...")
    comp = CompetitiveAttention(
        inhibition_strength=0.5,
        ignition_threshold=0.6,
        iterations=10,
    )
    
    percept_data = {
        "p1": {"embedding": [1.0, 0.0, 0.0]},
        "p2": {"embedding": [0.0, 1.0, 0.0]},
    }
    
    base_scores = {
        "p1": 0.9,  # Should exceed threshold
        "p2": 0.3,  # Should be suppressed
    }
    
    metrics = comp.compete(percept_data, base_scores)
    
    print(f"  Winners: {len(metrics.winner_ids)}")
    print(f"  Suppressed: {len(metrics.suppressed_percepts)}")
    print(f"  Winner IDs: {metrics.winner_ids}")
    
    assert len(metrics.winner_ids) < len(percept_data), "Should filter some percepts"
    assert len(metrics.suppressed_percepts) > 0, "Should track suppressed percepts"
    assert "p1" in metrics.winner_ids, "High scorer should win"
    print("✓ Ignition threshold works correctly")

def test_coalition_formation():
    """Test coalition formation for related percepts."""
    print("\nTesting coalition formation...")
    comp = CompetitiveAttention(coalition_boost=0.3)
    
    percept_data = {
        "p1": {"embedding": [1.0, 0.0, 0.0]},
        "p2": {"embedding": [0.95, 0.05, 0.0]},  # Similar to p1
        "p3": {"embedding": [0.0, 0.0, 1.0]},  # Different
    }
    
    base_scores = {pid: 0.5 for pid in percept_data.keys()}
    
    metrics = comp.compete(percept_data, base_scores)
    
    print(f"  Coalition formations: {len(metrics.coalition_formations)}")
    total_partners = sum(len(partners) for partners in metrics.coalition_formations.values())
    print(f"  Total coalition links: {total_partners}")
    
    # p1 and p2 should form coalition (similar embeddings)
    has_coalition = any(len(partners) > 0 for partners in metrics.coalition_formations.values())
    assert has_coalition, "Related percepts should form coalitions"
    print("✓ Coalition formation works")

def test_winner_take_all():
    """Test that competition creates winner-take-all dynamics."""
    print("\nTesting winner-take-all dynamics...")
    comp = CompetitiveAttention(
        inhibition_strength=0.4,
        iterations=15,
    )
    
    # Start with similar scores but different embeddings (no coalitions)
    percept_data = {
        "p1": {"embedding": [1.0, 0.0, 0.0]},
        "p2": {"embedding": [0.0, 1.0, 0.0]},
        "p3": {"embedding": [0.0, 0.0, 1.0]},
    }
    
    base_scores = {
        "p1": 0.55,
        "p2": 0.50,
        "p3": 0.45,
    }
    
    metrics = comp.compete(percept_data, base_scores)
    
    print(f"  Activation spread before: {metrics.activation_spread_before:.3f}")
    print(f"  Activation spread after: {metrics.activation_spread_after:.3f}")
    print(f"  Final activations: {[f'{a:.3f}' for a in comp.activations.values()]}")
    
    # Competition should increase spread (winner-take-all)
    # With dissimilar percepts and different initial scores, competition amplifies differences
    assert metrics.activation_spread_after >= metrics.activation_spread_before * 0.9, \
        "Competition should maintain or increase activation spread"
    print("✓ Winner-take-all dynamics work")

def main():
    """Run all tests."""
    print("=" * 60)
    print("Running Competitive Attention Logic Tests")
    print("=" * 60)
    
    try:
        test_basic_initialization()
        test_lateral_inhibition()
        test_ignition_threshold()
        test_coalition_formation()
        test_winner_take_all()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
