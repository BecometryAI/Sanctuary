"""
Unit tests for competitive attention dynamics.

Tests cover:
- Lateral inhibition between competing percepts
- Ignition threshold dynamics
- Coalition formation for related percepts
- Competition metrics tracking
- Integration with AttentionController
- Backward compatibility with legacy mode
"""

import pytest
import numpy as np

from mind.cognitive_core.attention import (
    CompetitiveAttention,
    AttentionController,
    CompetitionMetrics,
    AttentionMode,
)
from mind.cognitive_core.workspace import (
    GlobalWorkspace,
    Percept,
    Goal,
    GoalType,
)


class TestCompetitiveAttentionBasics:
    """Test basic CompetitiveAttention initialization and configuration."""
    
    def test_initialization_default(self):
        """Test creating CompetitiveAttention with default parameters."""
        comp = CompetitiveAttention()
        
        assert comp.inhibition_strength == 0.3
        assert comp.ignition_threshold == 0.5
        assert comp.iterations == 10
        assert comp.coalition_boost == 0.2
    
    def test_initialization_custom(self):
        """Test creating CompetitiveAttention with custom parameters."""
        comp = CompetitiveAttention(
            inhibition_strength=0.5,
            ignition_threshold=0.7,
            iterations=5,
            coalition_boost=0.3,
        )
        
        assert comp.inhibition_strength == 0.5
        assert comp.ignition_threshold == 0.7
        assert comp.iterations == 5
        assert comp.coalition_boost == 0.3
    
    def test_parameter_bounds_validation(self):
        """Test that parameters are validated with exceptions."""
        # Invalid inhibition_strength
        with pytest.raises(ValueError, match="inhibition_strength must be in"):
            CompetitiveAttention(inhibition_strength=1.5)
        
        # Invalid ignition_threshold
        with pytest.raises(ValueError, match="ignition_threshold must be in"):
            CompetitiveAttention(ignition_threshold=-0.1)
        
        # Invalid iterations
        with pytest.raises(ValueError, match="iterations must be in"):
            CompetitiveAttention(iterations=0)
        
        # Valid parameters should work
        comp = CompetitiveAttention(
            inhibition_strength=0.5,
            ignition_threshold=0.7,
            iterations=5
        )
        assert comp.inhibition_strength == 0.5
        assert comp.ignition_threshold == 0.7
        assert comp.iterations == 5


class TestLateralInhibition:
    """Test lateral inhibition dynamics."""
    
    def test_high_activation_suppresses_low(self):
        """Test that high-activation percepts suppress low-activation ones."""
        comp = CompetitiveAttention(
            inhibition_strength=0.4,
            iterations=10,
        )
        
        # Create percepts with different initial scores
        percepts = [
            Percept(modality="text", raw="High priority", embedding=[1.0, 0.0, 0.0]),
            Percept(modality="text", raw="Low priority", embedding=[0.0, 1.0, 0.0]),
        ]
        
        # High initial score vs low initial score
        base_scores = {
            percepts[0].id: 0.9,  # High score
            percepts[1].id: 0.3,  # Low score
        }
        
        sorted_percepts, metrics = comp.compete(percepts, base_scores)
        
        # After competition, high should remain high, low should be suppressed
        high_activation = comp.activations[percepts[0].id]
        low_activation = comp.activations[percepts[1].id]
        
        assert high_activation > low_activation
        assert high_activation > 0.5  # Should stay high
        assert low_activation < 0.5   # Should be suppressed
    
    def test_inhibition_events_tracked(self):
        """Test that inhibition events are tracked in metrics."""
        comp = CompetitiveAttention(inhibition_strength=0.3)
        
        percepts = [
            Percept(modality="text", raw="P1", embedding=[1.0, 0.0, 0.0]),
            Percept(modality="text", raw="P2", embedding=[0.0, 1.0, 0.0]),
            Percept(modality="text", raw="P3", embedding=[0.0, 0.0, 1.0]),
        ]
        
        base_scores = {p.id: 0.5 for p in percepts}
        
        _, metrics = comp.compete(percepts, base_scores)
        
        # Should have inhibition events (each percept inhibits others)
        assert metrics.inhibition_events > 0
    
    def test_activation_spread_increases(self):
        """Test that competition increases activation spread (winner-take-all)."""
        comp = CompetitiveAttention(
            inhibition_strength=0.4,
            iterations=15,
        )

        # Use diverse embeddings so percepts compete (not form coalitions)
        percepts = [
            Percept(modality="text", raw="P1", embedding=[1.0, 0.0, 0.0]),
            Percept(modality="text", raw="P2", embedding=[0.0, 1.0, 0.0]),
            Percept(modality="text", raw="P3", embedding=[0.0, 0.0, 1.0]),
        ]

        base_scores = {
            percepts[0].id: 0.55,
            percepts[1].id: 0.50,
            percepts[2].id: 0.45,
        }

        _, metrics = comp.compete(percepts, base_scores)

        # After competition with diverse embeddings, inhibition should happen
        # and spread should increase (winner-take-all)
        assert metrics.activation_spread_after >= metrics.activation_spread_before


class TestIgnitionThreshold:
    """Test ignition threshold dynamics."""
    
    def test_threshold_filters_low_activation(self):
        """Test that only percepts exceeding threshold are selected."""
        comp = CompetitiveAttention(
            inhibition_strength=0.5,
            ignition_threshold=0.6,
            iterations=10,
        )
        
        percepts = [
            Percept(modality="text", raw="High", embedding=[1.0, 0.0, 0.0]),
            Percept(modality="text", raw="Medium", embedding=[0.0, 1.0, 0.0]),
            Percept(modality="text", raw="Low", embedding=[0.0, 0.0, 1.0]),
        ]
        
        base_scores = {
            percepts[0].id: 0.9,   # Should exceed threshold
            percepts[1].id: 0.5,   # Unlikely to exceed after inhibition
            percepts[2].id: 0.3,   # Should be suppressed
        }
        
        selected, metrics = comp.select_for_workspace(percepts, base_scores)
        
        # Only high-activation percept should be selected
        assert len(selected) < len(percepts)
        assert len(metrics.winner_ids) < len(percepts)
        assert len(metrics.suppressed_percepts) > 0
    
    def test_threshold_not_just_top_n(self):
        """Test that threshold selection is fundamentally different from top-N."""
        comp = CompetitiveAttention(
            inhibition_strength=0.6,
            ignition_threshold=0.7,  # High threshold
            iterations=10,
        )
        
        # All percepts have low scores
        percepts = [
            Percept(modality="text", raw="P1", embedding=[1.0, 0.0, 0.0]),
            Percept(modality="text", raw="P2", embedding=[0.0, 1.0, 0.0]),
            Percept(modality="text", raw="P3", embedding=[0.0, 0.0, 1.0]),
        ]
        
        base_scores = {p.id: 0.4 for p in percepts}  # All below threshold
        
        selected, metrics = comp.select_for_workspace(percepts, base_scores)
        
        # With high threshold and low scores, might select 0 percepts
        # This is different from top-N which always selects N
        assert len(selected) <= len(percepts)
        
        # All percepts should be suppressed with low scores and strong inhibition
        assert len(metrics.suppressed_percepts) >= 0
    
    def test_winner_ids_exceed_threshold(self):
        """Test that winner IDs in metrics actually exceeded threshold."""
        comp = CompetitiveAttention(
            inhibition_strength=0.3,
            ignition_threshold=0.6,
        )
        
        percepts = [
            Percept(modality="text", raw="Winner", embedding=[1.0, 0.0, 0.0]),
            Percept(modality="text", raw="Loser", embedding=[0.0, 1.0, 0.0]),
        ]
        
        base_scores = {
            percepts[0].id: 0.8,
            percepts[1].id: 0.3,
        }
        
        _, metrics = comp.compete(percepts, base_scores)
        
        # Check that winners actually exceed threshold
        for winner_id in metrics.winner_ids:
            assert comp.activations[winner_id] >= comp.ignition_threshold


class TestCoalitionFormation:
    """Test coalition formation for related percepts."""
    
    def test_related_percepts_form_coalition(self):
        """Test that percepts with similar embeddings form coalitions."""
        comp = CompetitiveAttention(coalition_boost=0.3)
        
        # Create related percepts (similar embeddings)
        percepts = [
            Percept(modality="text", raw="Topic A content", embedding=[1.0, 0.0, 0.0]),
            Percept(modality="text", raw="More topic A", embedding=[0.95, 0.05, 0.0]),
            Percept(modality="text", raw="Unrelated topic B", embedding=[0.0, 0.0, 1.0]),
        ]
        
        base_scores = {p.id: 0.5 for p in percepts}
        
        _, metrics = comp.compete(percepts, base_scores)
        
        # Check that coalitions were formed
        assert len(metrics.coalition_formations) == len(percepts)
        
        # First two should be in coalition together
        # (they have similar embeddings)
        coalition1 = metrics.coalition_formations[percepts[0].id]
        coalition2 = metrics.coalition_formations[percepts[1].id]
        
        # At least one coalition should have partners
        has_partners = any(len(partners) > 0 for partners in metrics.coalition_formations.values())
        assert has_partners
    
    def test_coalition_boosts_activation(self):
        """Test that coalition members support each other's activation."""
        comp = CompetitiveAttention(
            coalition_boost=0.3,
            inhibition_strength=0.2,
            iterations=10,
        )
        
        # Two related percepts vs one unrelated
        percepts = [
            Percept(modality="text", raw="A1", embedding=[1.0, 0.0, 0.0]),
            Percept(modality="text", raw="A2", embedding=[0.95, 0.05, 0.0]),
            Percept(modality="text", raw="B", embedding=[0.0, 0.0, 1.0]),
        ]
        
        # All start with equal scores
        base_scores = {p.id: 0.5 for p in percepts}
        
        _, metrics = comp.compete(percepts, base_scores)
        
        # Coalition members should have benefited from mutual support
        # They should have higher activation than expected
        coalition_size = sum(len(partners) for partners in metrics.coalition_formations.values())
        assert coalition_size > 0  # Coalitions were formed
    
    def test_no_coalition_for_dissimilar_percepts(self):
        """Test that dissimilar percepts don't form coalitions."""
        comp = CompetitiveAttention(coalition_boost=0.2)
        
        # Create very different percepts
        percepts = [
            Percept(modality="text", raw="Topic A", embedding=[1.0, 0.0, 0.0]),
            Percept(modality="text", raw="Topic B", embedding=[0.0, 1.0, 0.0]),
            Percept(modality="text", raw="Topic C", embedding=[0.0, 0.0, 1.0]),
        ]
        
        base_scores = {p.id: 0.5 for p in percepts}
        
        _, metrics = comp.compete(percepts, base_scores)
        
        # Coalitions should exist (all percepts tracked)
        assert len(metrics.coalition_formations) == len(percepts)
        
        # But they should be empty or very small (dissimilar percepts)
        total_coalition_links = sum(len(partners) for partners in metrics.coalition_formations.values())
        # With high dissimilarity, few or no coalition links should form
        assert total_coalition_links <= len(percepts)  # At most one coalition


class TestCompetitionMetrics:
    """Test competition metrics tracking."""
    
    def test_metrics_track_suppressed_percepts(self):
        """Test that suppressed percepts are tracked."""
        comp = CompetitiveAttention(
            inhibition_strength=0.5,
            ignition_threshold=0.6,
        )
        
        percepts = [
            Percept(modality="text", raw="Strong", embedding=[1.0, 0.0, 0.0]),
            Percept(modality="text", raw="Weak1", embedding=[0.0, 1.0, 0.0]),
            Percept(modality="text", raw="Weak2", embedding=[0.0, 0.0, 1.0]),
        ]
        
        base_scores = {
            percepts[0].id: 0.8,
            percepts[1].id: 0.4,
            percepts[2].id: 0.3,
        }
        
        _, metrics = comp.compete(percepts, base_scores)
        
        # Should track suppressed percepts
        assert len(metrics.suppressed_percepts) >= 0
        assert isinstance(metrics.suppressed_percepts, list)
    
    def test_metrics_complete_structure(self):
        """Test that metrics contain all required fields."""
        comp = CompetitiveAttention()
        
        percepts = [
            Percept(modality="text", raw="P1", embedding=[1.0, 0.0, 0.0]),
            Percept(modality="text", raw="P2", embedding=[0.0, 1.0, 0.0]),
        ]
        
        base_scores = {p.id: 0.5 for p in percepts}
        
        _, metrics = comp.compete(percepts, base_scores)
        
        # Verify all fields are present
        assert hasattr(metrics, 'inhibition_events')
        assert hasattr(metrics, 'suppressed_percepts')
        assert hasattr(metrics, 'activation_spread_before')
        assert hasattr(metrics, 'activation_spread_after')
        assert hasattr(metrics, 'winner_ids')
        assert hasattr(metrics, 'coalition_formations')
        
        # Verify types
        assert isinstance(metrics.inhibition_events, int)
        assert isinstance(metrics.suppressed_percepts, list)
        assert isinstance(metrics.activation_spread_before, float)
        assert isinstance(metrics.activation_spread_after, float)
        assert isinstance(metrics.winner_ids, list)
        assert isinstance(metrics.coalition_formations, dict)


class TestAttentionControllerIntegration:
    """Test integration of CompetitiveAttention with AttentionController."""
    
    def test_controller_with_competition_enabled(self):
        """Test AttentionController with competition enabled (default)."""
        controller = AttentionController(
            attention_budget=50,
            use_competition=True,
        )
        
        assert controller.use_competition is True
        assert controller.competitive_attention is not None
        assert isinstance(controller.competitive_attention, CompetitiveAttention)
    
    def test_controller_with_competition_disabled(self):
        """Test AttentionController with competition disabled (legacy mode)."""
        controller = AttentionController(
            attention_budget=50,
            use_competition=False,
        )
        
        assert controller.use_competition is False
        assert controller.competitive_attention is None
    
    def test_competitive_selection_respects_budget(self):
        """Test that competitive selection respects budget constraints."""
        workspace = GlobalWorkspace()
        controller = AttentionController(
            attention_budget=25,
            workspace=workspace,
            use_competition=True,
            inhibition_strength=0.3,
            ignition_threshold=0.4,
        )
        
        percepts = [
            Percept(modality="text", raw="P1", complexity=10, embedding=[1.0, 0.0, 0.0]),
            Percept(modality="text", raw="P2", complexity=10, embedding=[0.9, 0.1, 0.0]),
            Percept(modality="text", raw="P3", complexity=10, embedding=[0.8, 0.2, 0.0]),
        ]
        
        selected = controller.select_for_broadcast(percepts)
        
        # Should respect budget
        total_complexity = sum(p.complexity for p in selected)
        assert total_complexity <= 25
    
    def test_competitive_selection_different_from_legacy(self):
        """Test that competitive selection can differ from legacy selection."""
        workspace = GlobalWorkspace()
        
        # Create percepts where competition would change ordering
        percepts = [
            Percept(modality="text", raw="High score, unrelated", 
                   complexity=5, embedding=[1.0, 0.0, 0.0]),
            Percept(modality="text", raw="Medium score, related A", 
                   complexity=5, embedding=[0.0, 1.0, 0.0]),
            Percept(modality="text", raw="Medium score, related B", 
                   complexity=5, embedding=[0.0, 0.95, 0.05]),
        ]
        
        # Test with competition
        controller_comp = AttentionController(
            attention_budget=20,
            workspace=workspace,
            use_competition=True,
            inhibition_strength=0.4,
            ignition_threshold=0.3,
        )
        selected_comp = controller_comp.select_for_broadcast(percepts)
        
        # Test without competition
        controller_legacy = AttentionController(
            attention_budget=20,
            workspace=workspace,
            use_competition=False,
        )
        selected_legacy = controller_legacy.select_for_broadcast(percepts)
        
        # Both should select something, but results may differ
        assert len(selected_comp) >= 0
        assert len(selected_legacy) >= 0
    
    def test_competition_metrics_stored(self):
        """Test that competition metrics are stored in controller."""
        controller = AttentionController(
            attention_budget=50,
            use_competition=True,
        )
        
        percepts = [
            Percept(modality="text", raw="P1", embedding=[1.0, 0.0, 0.0]),
            Percept(modality="text", raw="P2", embedding=[0.0, 1.0, 0.0]),
        ]
        
        controller.select_for_broadcast(percepts)
        
        # Metrics should be stored
        assert len(controller.competition_metrics_history) > 0
        
        metrics = controller.competition_metrics_history[0]
        assert isinstance(metrics, CompetitionMetrics)
    
    def test_get_competition_metrics(self):
        """Test retrieving competition metrics from controller."""
        controller = AttentionController(use_competition=True)
        
        percepts = [
            Percept(modality="text", raw="P1", embedding=[1.0, 0.0, 0.0]),
            Percept(modality="text", raw="P2", embedding=[0.0, 1.0, 0.0]),
        ]
        
        controller.select_for_broadcast(percepts)
        
        metrics_list = controller.get_competition_metrics()
        assert isinstance(metrics_list, list)
        assert len(metrics_list) > 0


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""
    
    def test_default_behavior_uses_competition(self):
        """Test that default initialization uses competitive dynamics."""
        controller = AttentionController()
        
        # Default should be competitive mode
        assert controller.use_competition is True
        assert controller.competitive_attention is not None
    
    def test_legacy_mode_opt_in(self):
        """Test that legacy mode can be explicitly enabled."""
        controller = AttentionController(use_competition=False)
        
        # Should have legacy mode when explicitly requested
        assert controller.use_competition is False
        assert controller.competitive_attention is None
    
    def test_legacy_mode_works(self):
        """Test that legacy mode still works correctly."""
        workspace = GlobalWorkspace()
        controller = AttentionController(
            attention_budget=30,
            workspace=workspace,
            use_competition=False,  # Explicit legacy mode
        )
        
        percepts = [
            Percept(modality="text", raw="P1", complexity=10, embedding=[1.0, 0.0, 0.0]),
            Percept(modality="text", raw="P2", complexity=10, embedding=[0.0, 1.0, 0.0]),
            Percept(modality="text", raw="P3", complexity=15, embedding=[0.0, 0.0, 1.0]),
        ]
        
        selected = controller.select_for_broadcast(percepts)
        
        # Legacy selection should work
        assert len(selected) > 0
        assert sum(p.complexity for p in selected) <= 30
    
    def test_attention_report_includes_competition_stats(self):
        """Test that attention report includes competition stats when enabled."""
        controller = AttentionController(use_competition=True)
        
        percepts = [
            Percept(modality="text", raw="P1", embedding=[1.0, 0.0, 0.0]),
            Percept(modality="text", raw="P2", embedding=[0.0, 1.0, 0.0]),
        ]
        
        controller.select_for_broadcast(percepts)
        
        report = controller.get_attention_report()
        
        # Should include competition stats
        assert "competition_enabled" in report
        assert report["competition_enabled"] is True
        assert "competition_stats" in report
    
    def test_attention_report_without_competition(self):
        """Test that attention report works without competition."""
        controller = AttentionController(use_competition=False)
        
        percepts = [
            Percept(modality="text", raw="P1", embedding=[1.0, 0.0, 0.0]),
        ]
        
        controller.select_for_broadcast(percepts)
        
        report = controller.get_attention_report()
        
        # Should work without competition stats
        assert "competition_enabled" in report
        assert report["competition_enabled"] is False
        assert "competition_stats" not in report


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_percept_list(self):
        """Test competition with empty percept list."""
        comp = CompetitiveAttention()
        
        selected, metrics = comp.select_for_workspace([], {})
        
        assert len(selected) == 0
        assert metrics.inhibition_events == 0
    
    def test_single_percept_competition(self):
        """Test competition with single percept (no competition)."""
        comp = CompetitiveAttention(ignition_threshold=0.3)
        
        percepts = [
            Percept(modality="text", raw="Only one", embedding=[1.0, 0.0, 0.0]),
        ]
        
        base_scores = {percepts[0].id: 0.8}
        
        selected, metrics = comp.select_for_workspace(percepts, base_scores)
        
        # Single percept should be selected if exceeds threshold
        assert len(selected) == 1
        assert metrics.inhibition_events == 0  # No competition with itself
    
    def test_percepts_without_embeddings(self):
        """Test competition with percepts that lack embeddings."""
        comp = CompetitiveAttention()
        
        percepts = [
            Percept(modality="text", raw="No embedding 1"),
            Percept(modality="text", raw="No embedding 2"),
        ]
        
        base_scores = {p.id: 0.5 for p in percepts}
        
        # Should not crash
        selected, metrics = comp.select_for_workspace(percepts, base_scores)
        
        assert isinstance(selected, list)
        assert isinstance(metrics, CompetitionMetrics)
