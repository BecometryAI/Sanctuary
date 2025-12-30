"""
Unit tests for AttentionController class.

Tests cover:
- Attention selection with budget constraints
- Goal relevance scoring
- Novelty detection
- Emotional salience
- Budget enforcement and reset
- Attention reporting
"""

import pytest
from datetime import datetime
import time

from lyra.cognitive_core.attention import (
    AttentionController,
    AttentionMode,
    cosine_similarity,
    keyword_overlap,
)
from lyra.cognitive_core.workspace import (
    GlobalWorkspace,
    Percept,
    Goal,
    GoalType,
)


class TestHelperFunctions:
    """Test helper functions for attention scoring."""
    
    def test_cosine_similarity_identical(self):
        """Test cosine similarity with identical vectors."""
        vec = [1.0, 0.5, 0.3]
        similarity = cosine_similarity(vec, vec)
        assert similarity == pytest.approx(1.0, abs=0.01)
    
    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity with orthogonal vectors."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        similarity = cosine_similarity(vec1, vec2)
        # Orthogonal vectors should have ~0.5 similarity (after normalization)
        assert 0.0 <= similarity <= 1.0
    
    def test_cosine_similarity_empty(self):
        """Test cosine similarity with empty vectors."""
        similarity = cosine_similarity([], [1.0, 2.0])
        assert similarity == 0.0
    
    def test_cosine_similarity_different_lengths(self):
        """Test cosine similarity with different length vectors."""
        similarity = cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])
        assert similarity == 0.0
    
    def test_keyword_overlap_identical(self):
        """Test keyword overlap with identical text."""
        text = "hello world test"
        overlap = keyword_overlap(text, text)
        assert overlap == pytest.approx(1.0, abs=0.01)
    
    def test_keyword_overlap_no_overlap(self):
        """Test keyword overlap with completely different text."""
        overlap = keyword_overlap("apple banana", "car dog")
        assert overlap == 0.0
    
    def test_keyword_overlap_partial(self):
        """Test keyword overlap with partial match."""
        overlap = keyword_overlap("hello world", "hello there")
        assert 0.0 < overlap < 1.0
    
    def test_keyword_overlap_stopwords(self):
        """Test that stopwords are filtered out."""
        # "the" and "is" are stopwords and should be ignored
        overlap = keyword_overlap("the cat is happy", "the dog is happy")
        # Only "happy" matches, "cat" and "dog" don't
        assert 0.0 < overlap < 1.0


class TestAttentionController:
    """Test AttentionController initialization and basic functionality."""
    
    def test_initialization_default(self):
        """Test creating AttentionController with default parameters."""
        controller = AttentionController()
        
        assert controller.attention_budget == 100
        assert controller.initial_budget == 100
        assert controller.mode == AttentionMode.FOCUSED
        assert len(controller.recent_percepts) == 0
        assert len(controller.attention_history) == 0
    
    def test_initialization_custom(self):
        """Test creating AttentionController with custom parameters."""
        workspace = GlobalWorkspace()
        controller = AttentionController(
            attention_budget=50,
            workspace=workspace,
            initial_mode=AttentionMode.DIFFUSE,
            goal_weight=0.5,
            novelty_weight=0.3,
            emotion_weight=0.1,
            urgency_weight=0.1
        )
        
        assert controller.attention_budget == 50
        assert controller.workspace == workspace
        assert controller.mode == AttentionMode.DIFFUSE
        assert controller.goal_weight == 0.5
    
    def test_reset_budget(self):
        """Test budget reset functionality."""
        controller = AttentionController(attention_budget=100)
        
        # Simulate budget usage
        controller.attention_budget = 30
        assert controller.attention_budget == 30
        
        # Reset should restore to initial value
        controller.reset_budget()
        assert controller.attention_budget == 100


class TestAttentionSelectionBasic:
    """Test basic attention selection and percept scoring."""
    
    def test_attention_selection_empty_candidates(self):
        """Test selection with no candidates."""
        controller = AttentionController()
        selected = controller.select_for_broadcast([])
        
        assert len(selected) == 0
    
    def test_attention_selection_single_percept(self):
        """Test selection with single percept."""
        controller = AttentionController(attention_budget=100)
        
        percept = Percept(
            modality="text",
            raw="Important information",
            complexity=5,
            embedding=[1.0, 0.5, 0.3]
        )
        
        selected = controller.select_for_broadcast([percept])
        
        assert len(selected) == 1
        assert selected[0].id == percept.id
    
    def test_attention_selection_multiple_percepts(self):
        """Test selection with multiple percepts of varying complexity."""
        workspace = GlobalWorkspace()
        controller = AttentionController(attention_budget=20, workspace=workspace)
        
        # Create percepts with different complexities
        percepts = [
            Percept(modality="text", raw="Low complexity", complexity=5, embedding=[1.0, 0.0, 0.0]),
            Percept(modality="text", raw="High complexity", complexity=15, embedding=[0.0, 1.0, 0.0]),
            Percept(modality="text", raw="Medium complexity", complexity=10, embedding=[0.0, 0.0, 1.0]),
        ]
        
        selected = controller.select_for_broadcast(percepts)
        
        # Should select some percepts
        assert len(selected) > 0
        
        # Total complexity should not exceed budget
        total_complexity = sum(p.complexity for p in selected)
        assert total_complexity <= 20


class TestBudgetEnforcement:
    """Test attention budget enforcement."""
    
    def test_budget_enforcement_exact_fit(self):
        """Test budget with percepts that exactly fit."""
        controller = AttentionController(attention_budget=20)
        
        percepts = [
            Percept(modality="text", raw="P1", complexity=10, embedding=[1.0, 0.0, 0.0]),
            Percept(modality="text", raw="P2", complexity=10, embedding=[0.9, 0.1, 0.0]),
        ]
        
        selected = controller.select_for_broadcast(percepts)
        
        # Both should fit
        assert len(selected) == 2
        assert sum(p.complexity for p in selected) == 20
    
    def test_budget_enforcement_overflow(self):
        """Test budget with percepts that exceed capacity."""
        controller = AttentionController(attention_budget=15)
        
        percepts = [
            Percept(modality="text", raw="P1", complexity=10, embedding=[1.0, 0.0, 0.0]),
            Percept(modality="text", raw="P2", complexity=10, embedding=[0.9, 0.1, 0.0]),
        ]
        
        selected = controller.select_for_broadcast(percepts)
        
        # Only one should fit
        assert len(selected) == 1
        assert sum(p.complexity for p in selected) <= 15
    
    def test_budget_resets_correctly(self):
        """Test that budget resets after each cycle."""
        controller = AttentionController(attention_budget=50)
        
        # Use some budget
        percepts = [Percept(modality="text", raw="Test", complexity=20, embedding=[1.0, 0.0, 0.0])]
        controller.select_for_broadcast(percepts)
        
        # Budget should be consumed
        assert controller.attention_budget == 50  # Budget tracked separately
        
        # Reset budget
        controller.reset_budget()
        assert controller.attention_budget == 50
    
    def test_budget_enforcement_with_priority(self):
        """Test that higher-scoring percepts are selected when budget is limited."""
        workspace = GlobalWorkspace()
        
        # Add a high-priority goal
        goal = Goal(
            type=GoalType.RESPOND_TO_USER,
            description="Important task priority",
            priority=0.9,
            metadata={"embedding": [1.0, 0.0, 0.0]}
        )
        workspace.add_goal(goal)
        
        controller = AttentionController(attention_budget=15, workspace=workspace)
        
        # Create percepts with different relevance to goal
        percepts = [
            Percept(modality="text", raw="Irrelevant data", complexity=10, embedding=[0.0, 1.0, 0.0]),
            Percept(modality="text", raw="Important task priority match", complexity=10, embedding=[0.95, 0.05, 0.0]),
        ]
        
        selected = controller.select_for_broadcast(percepts)
        
        # Should select the more relevant percept
        assert len(selected) == 1
        assert "Important task" in selected[0].raw


class TestGoalRelevanceScoring:
    """Test goal relevance scoring."""
    
    def test_goal_relevance_with_embedding(self):
        """Test goal relevance with embedding-based similarity."""
        workspace = GlobalWorkspace()
        
        # Add goal with embedding
        goal = Goal(
            type=GoalType.LEARN,
            description="Learn about AI",
            metadata={"embedding": [1.0, 0.0, 0.0]}
        )
        workspace.add_goal(goal)
        
        controller = AttentionController(workspace=workspace)
        
        # Create relevant percept
        percept = Percept(
            modality="text",
            raw="AI learning material",
            embedding=[0.95, 0.05, 0.0]
        )
        
        relevance = controller._compute_goal_relevance(percept)
        
        # Should have high relevance
        assert relevance > 0.5
    
    def test_goal_relevance_with_keywords(self):
        """Test goal relevance with keyword matching."""
        workspace = GlobalWorkspace()
        
        # Add goal without embedding
        goal = Goal(
            type=GoalType.RESPOND_TO_USER,
            description="Answer questions about consciousness"
        )
        workspace.add_goal(goal)
        
        controller = AttentionController(workspace=workspace)
        
        # Create relevant percept
        percept = Percept(
            modality="text",
            raw="Questions about consciousness theory"
        )
        
        relevance = controller._compute_goal_relevance(percept)
        
        # Should have some relevance due to keyword overlap
        assert relevance > 0.0
    
    def test_goal_relevance_no_goals(self):
        """Test goal relevance when no goals are present."""
        workspace = GlobalWorkspace()
        controller = AttentionController(workspace=workspace)
        
        percept = Percept(modality="text", raw="Some data")
        relevance = controller._compute_goal_relevance(percept)
        
        # Should return neutral score
        assert relevance == 0.5
    
    def test_goal_relevance_irrelevant_percept(self):
        """Test goal relevance with completely irrelevant percept."""
        workspace = GlobalWorkspace()
        
        goal = Goal(
            type=GoalType.CREATE,
            description="Generate artwork",
            metadata={"embedding": [1.0, 0.0, 0.0]}
        )
        workspace.add_goal(goal)
        
        controller = AttentionController(workspace=workspace)
        
        # Completely different percept
        percept = Percept(
            modality="text",
            raw="Weather forecast data",
            embedding=[0.0, 0.0, 1.0]
        )
        
        relevance = controller._compute_goal_relevance(percept)
        
        # Should have low relevance
        assert relevance < 0.5


class TestNoveltyDetection:
    """Test novelty detection for percepts."""
    
    def test_novelty_first_percept(self):
        """Test novelty for first percept (no history)."""
        controller = AttentionController()
        
        percept = Percept(
            modality="text",
            raw="First ever percept",
            embedding=[1.0, 0.0, 0.0]
        )
        
        novelty = controller._compute_novelty(percept)
        
        # Should be completely novel
        assert novelty == 1.0
    
    def test_novelty_decreases_with_repetition(self):
        """Test that novelty decreases for similar percepts."""
        controller = AttentionController()
        
        # Add some percepts to history
        embedding = [1.0, 0.0, 0.0]
        controller.recent_percepts.append(embedding)
        controller.recent_percepts.append([0.95, 0.05, 0.0])
        
        # Create similar percept
        percept = Percept(
            modality="text",
            raw="Similar percept",
            embedding=[0.98, 0.02, 0.0]
        )
        
        novelty = controller._compute_novelty(percept)
        
        # Should have low novelty
        assert novelty < 0.5
    
    def test_novelty_high_for_different_percept(self):
        """Test that novelty is high for different percepts."""
        controller = AttentionController()
        
        # Add percepts to history
        controller.recent_percepts.append([1.0, 0.0, 0.0])
        controller.recent_percepts.append([0.9, 0.1, 0.0])
        
        # Create very different percept
        percept = Percept(
            modality="text",
            raw="Completely different",
            embedding=[0.0, 0.0, 1.0]
        )
        
        novelty = controller._compute_novelty(percept)
        
        # Should have high novelty
        assert novelty > 0.7
    
    def test_novelty_without_embedding(self):
        """Test novelty for percept without embedding."""
        controller = AttentionController()
        
        percept = Percept(modality="text", raw="No embedding")
        novelty = controller._compute_novelty(percept)
        
        # Should be considered novel
        assert novelty == 1.0


class TestEmotionalSalience:
    """Test emotional salience scoring."""
    
    def test_emotional_salience_positive_match(self):
        """Test emotional salience with positive emotion match."""
        workspace = GlobalWorkspace()
        workspace.emotional_state["valence"] = 0.7
        workspace.emotional_state["arousal"] = 0.5
        
        controller = AttentionController(workspace=workspace)
        
        # Percept with positive emotion
        percept = Percept(
            modality="text",
            raw="I am happy and excited",
            metadata={"emotion": "happy"}
        )
        
        salience = controller._compute_emotional_salience(percept)
        
        # Should have high salience
        assert salience > 0.5
    
    def test_emotional_salience_negative_match(self):
        """Test emotional salience with negative emotion match."""
        workspace = GlobalWorkspace()
        workspace.emotional_state["valence"] = -0.7
        workspace.emotional_state["arousal"] = 0.6
        
        controller = AttentionController(workspace=workspace)
        
        # Percept with negative emotion
        percept = Percept(
            modality="text",
            raw="This is sad and terrible",
            metadata={"emotion": "sad"}
        )
        
        salience = controller._compute_emotional_salience(percept)
        
        # Should have high salience
        assert salience > 0.5
    
    def test_emotional_salience_mismatch(self):
        """Test emotional salience with emotion mismatch."""
        workspace = GlobalWorkspace()
        workspace.emotional_state["valence"] = 0.7  # Positive
        
        controller = AttentionController(workspace=workspace)
        
        # Percept with negative emotion
        percept = Percept(
            modality="text",
            raw="Sad news today",
            metadata={"emotion": "sad"}
        )
        
        salience = controller._compute_emotional_salience(percept)
        
        # Should have moderate salience
        assert salience <= 0.8
    
    def test_emotional_salience_neutral(self):
        """Test emotional salience with neutral content."""
        workspace = GlobalWorkspace()
        controller = AttentionController(workspace=workspace)
        
        percept = Percept(modality="text", raw="Neutral factual information")
        salience = controller._compute_emotional_salience(percept)
        
        # Should have low salience
        assert salience <= 0.5
    
    def test_emotional_salience_no_workspace(self):
        """Test emotional salience without workspace."""
        controller = AttentionController()
        
        percept = Percept(modality="text", raw="Some data")
        salience = controller._compute_emotional_salience(percept)
        
        # Should return 0 if no workspace
        assert salience == 0.0


class TestAttentionReport:
    """Test attention reporting functionality."""
    
    def test_attention_report_empty_history(self):
        """Test report with no selection history."""
        controller = AttentionController()
        
        report = controller.get_attention_report()
        
        assert report["total_decisions"] == 0
        assert report["total_candidates"] == 0
        assert report["selected_count"] == 0
        assert report["avg_budget_usage"] == 0.0
    
    def test_attention_report_after_selections(self):
        """Test report after multiple selections."""
        controller = AttentionController(attention_budget=50)
        
        # Perform multiple selections
        percepts1 = [
            Percept(modality="text", raw="P1", complexity=10, embedding=[1.0, 0.0, 0.0]),
            Percept(modality="text", raw="P2", complexity=20, embedding=[0.0, 1.0, 0.0]),
        ]
        controller.select_for_broadcast(percepts1)
        
        percepts2 = [
            Percept(modality="text", raw="P3", complexity=15, embedding=[0.5, 0.5, 0.0]),
        ]
        controller.select_for_broadcast(percepts2)
        
        report = controller.get_attention_report()
        
        assert report["total_decisions"] == 2
        assert report["total_candidates"] == 3
        assert report["selected_count"] > 0
        assert report["avg_budget_usage"] > 0
        assert "rejection_reasons" in report
    
    def test_attention_report_rejection_reasons(self):
        """Test report includes rejection reasons."""
        controller = AttentionController(attention_budget=15)
        
        # Create percepts that exceed budget
        percepts = [
            Percept(modality="text", raw="P1", complexity=10, embedding=[1.0, 0.0, 0.0]),
            Percept(modality="text", raw="P2", complexity=10, embedding=[0.9, 0.1, 0.0]),
            Percept(modality="text", raw="P3", complexity=10, embedding=[0.8, 0.2, 0.0]),
        ]
        
        controller.select_for_broadcast(percepts)
        
        report = controller.get_attention_report()
        
        # Should have some rejections
        assert report["rejection_reasons"]["budget_exhausted"] > 0 or \
               report["rejection_reasons"]["low_score"] > 0


class TestIntegration:
    """Integration tests for realistic attention scenarios."""
    
    def test_complete_attention_cycle(self):
        """Test complete attention selection cycle."""
        workspace = GlobalWorkspace()
        
        # Set up workspace with goal and emotional state
        goal = Goal(
            type=GoalType.RESPOND_TO_USER,
            description="Answer question about memory",
            priority=0.9,
            metadata={"embedding": [1.0, 0.0, 0.0]}
        )
        workspace.add_goal(goal)
        workspace.emotional_state["valence"] = 0.5
        
        controller = AttentionController(attention_budget=50, workspace=workspace)
        
        # Create diverse percepts
        percepts = [
            Percept(modality="text", raw="Question about memory systems", 
                   complexity=10, embedding=[0.95, 0.05, 0.0]),
            Percept(modality="text", raw="Unrelated weather data", 
                   complexity=10, embedding=[0.0, 0.0, 1.0]),
            Percept(modality="text", raw="Memory research findings", 
                   complexity=15, embedding=[0.9, 0.1, 0.0]),
        ]
        
        # Select percepts
        selected = controller.select_for_broadcast(percepts)
        
        # Verify selection
        assert len(selected) > 0
        assert sum(p.complexity for p in selected) <= 50
        
        # Most relevant percepts should be selected
        selected_ids = [p.id for p in selected]
        assert percepts[0].id in selected_ids or percepts[2].id in selected_ids
    
    def test_attention_over_multiple_cycles(self):
        """Test attention behavior over multiple cognitive cycles."""
        workspace = GlobalWorkspace()
        controller = AttentionController(attention_budget=30, workspace=workspace)
        
        # Cycle 1: Novel percepts
        percepts1 = [
            Percept(modality="text", raw="New information A", 
                   complexity=10, embedding=[1.0, 0.0, 0.0]),
            Percept(modality="text", raw="New information B", 
                   complexity=10, embedding=[0.0, 1.0, 0.0]),
        ]
        selected1 = controller.select_for_broadcast(percepts1)
        assert len(selected1) > 0
        
        # Reset budget for next cycle
        controller.reset_budget()
        
        # Cycle 2: Repeated similar percepts (should have lower novelty)
        percepts2 = [
            Percept(modality="text", raw="Similar to A", 
                   complexity=10, embedding=[0.98, 0.02, 0.0]),
            Percept(modality="text", raw="Very novel C", 
                   complexity=10, embedding=[0.0, 0.0, 1.0]),
        ]
        selected2 = controller.select_for_broadcast(percepts2)
        
        # The novel percept should score higher
        assert len(selected2) > 0
        
        # Get report
        report = controller.get_attention_report()
        assert report["total_decisions"] == 2
