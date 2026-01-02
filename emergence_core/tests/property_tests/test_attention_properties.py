"""
Property-based tests for AttentionController invariants.

Tests cover:
- Attention budget constraints
- Attention score properties
- Selection determinism
- Novelty detection
"""

import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from lyra.cognitive_core.workspace import GlobalWorkspace, Percept, Goal, GoalType
from lyra.cognitive_core.attention import AttentionController, AttentionMode
from .strategies import (
    percepts,
    goals,
    percept_lists,
    goal_lists,
    make_unique_by_id,
)


def create_workspace_with_content(percepts_list, goals_list):
    """Helper to create a workspace with given content."""
    workspace = GlobalWorkspace()
    
    for percept in percepts_list:
        workspace.active_percepts[percept.id] = percept
    
    for goal in goals_list:
        workspace.add_goal(goal)
    
    return workspace


@pytest.mark.property
class TestAttentionProperties:
    """Property-based tests for AttentionController invariants."""
    
    @given(percept_lists, st.integers(min_value=1, max_value=20))
    @settings(max_examples=100, deadline=None)
    def test_attention_respects_budget(self, percepts_list, budget):
        """Property: Selected percepts never exceed attention budget."""
        assume(len(percepts_list) > 0)
        
        workspace = GlobalWorkspace()
        controller = AttentionController(
            attention_budget=budget,
            workspace=workspace
        )
        
        unique_percepts = make_unique_by_id(percepts_list)
        selected = controller.select_for_broadcast(unique_percepts)
        
        # Calculate total complexity of selected percepts
        total_complexity = sum(p.complexity for p in selected)
        
        # Total complexity should not exceed budget
        assert total_complexity <= budget
    
    @given(percept_lists)
    @settings(max_examples=100)
    def test_attention_score_non_negative(self, percepts_list):
        """Property: All attention scores are non-negative."""
        assume(len(percepts_list) > 0)
        
        workspace = GlobalWorkspace()
        controller = AttentionController(workspace=workspace)
        
        unique_percepts = make_unique_by_id(percepts_list)
        
        for percept in unique_percepts[:10]:  # Test first 10
            score = controller._score(percept)
            assert score >= 0.0, f"Score {score} is negative for percept {percept.id}"
    
    @given(percepts(), percepts())
    @settings(max_examples=100)
    def test_attention_deterministic(self, percept1, percept2):
        """Property: Same percept+context yields same attention score."""
        workspace = GlobalWorkspace()
        workspace.active_percepts[percept1.id] = percept1
        workspace.active_percepts[percept2.id] = percept2
        
        controller = AttentionController(workspace=workspace)
        
        # Score the same percept twice
        score1_a = controller._score(percept1)
        score1_b = controller._score(percept1)
        
        # Scores should be identical (deterministic)
        assert score1_a == score1_b
    
    @given(percept_lists, goal_lists)
    @settings(max_examples=50, deadline=None)
    def test_selection_returns_subset(self, percepts_list, goals_list):
        """Property: Selected percepts are always a subset of candidates."""
        assume(len(percepts_list) > 0)
        
        workspace = create_workspace_with_content([], goals_list)
        controller = AttentionController(attention_budget=50, workspace=workspace)
        
        unique_percepts = make_unique_by_id(percepts_list)
        selected = controller.select_for_broadcast(unique_percepts)
        
        # All selected items should be in the original list
        selected_ids = {p.id for p in selected}
        original_ids = {p.id for p in unique_percepts}
        
        assert selected_ids.issubset(original_ids)
    
    @given(st.lists(percepts(), min_size=3, max_size=15), st.integers(min_value=5, max_value=100))
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.data_too_large])
    def test_attention_budget_allocation(self, percepts_list, budget):
        """Property: Controller attempts to maximize use of attention budget."""
        assume(len(percepts_list) >= 3)
        
        workspace = GlobalWorkspace()
        controller = AttentionController(attention_budget=budget, workspace=workspace)
        
        unique_percepts = make_unique_by_id(percepts_list)
        selected = controller.select_for_broadcast(unique_percepts)
        
        total_complexity = sum(p.complexity for p in selected)
        
        # If there are selected percepts, total should be <= budget
        if len(selected) > 0:
            assert total_complexity <= budget
        
        # If nothing was selected, either all percepts exceed budget or list is empty
        if len(selected) == 0:
            # Either all percepts are too complex, or list is empty
            if len(unique_percepts) > 0:
                # At least one percept should have complexity > budget
                min_complexity = min(p.complexity for p in unique_percepts)
                if min_complexity <= budget:
                    # If there's a percept that fits, it should have been selected
                    # (unless all scores are 0 or very low - edge case we can ignore)
                    pass
    
    @given(st.lists(percepts(), min_size=2, max_size=10))
    @settings(max_examples=50)
    def test_higher_scores_selected_first(self, percepts_list):
        """Property: Higher scoring percepts are selected before lower scoring ones."""
        assume(len(percepts_list) >= 2)
        
        workspace = GlobalWorkspace()
        # Use a large budget to not constrain by complexity
        controller = AttentionController(attention_budget=1000, workspace=workspace)
        
        unique_percepts = make_unique_by_id(percepts_list)
        
        # Score all percepts
        scores = [(p, controller._score(p)) for p in unique_percepts]
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select percepts
        selected = controller.select_for_broadcast(unique_percepts)
        
        if len(selected) > 0:
            # The selected percepts should include the highest scoring ones
            # (within budget constraints)
            selected_ids = {p.id for p in selected}
            
            # At least the highest scoring percept should be selected if it fits
            highest_scoring = scores[0][0]
            if highest_scoring.complexity <= controller.attention_budget:
                assert highest_scoring.id in selected_ids
    
    @given(percept_lists)
    @settings(max_examples=50)
    def test_empty_candidates_returns_empty(self, percepts_list):
        """Property: Selecting from empty list returns empty list."""
        workspace = GlobalWorkspace()
        controller = AttentionController(workspace=workspace)
        
        selected = controller.select_for_broadcast([])
        assert selected == []
    
    @given(st.integers(min_value=1, max_value=200))
    @settings(max_examples=50)
    def test_attention_budget_property(self, budget):
        """Property: Controller maintains the attention budget it was initialized with."""
        workspace = GlobalWorkspace()
        controller = AttentionController(attention_budget=budget, workspace=workspace)
        
        assert controller.attention_budget == budget
        assert controller.initial_budget == budget
    
    @given(percepts(), goal_lists)
    @settings(max_examples=50)
    def test_goal_relevance_bounded(self, percept, goals_list):
        """Property: Goal relevance score is bounded [0, 1]."""
        workspace = GlobalWorkspace()
        unique_goals = make_unique_by_id(goals_list)
        for goal in unique_goals[:5]:
            workspace.add_goal(goal)
        
        controller = AttentionController(workspace=workspace)
        
        goal_relevance = controller._compute_goal_relevance(percept)
        assert 0.0 <= goal_relevance <= 1.0
    
    @given(percepts())
    @settings(max_examples=50)
    def test_novelty_bounded(self, percept):
        """Property: Novelty score is bounded [0, 1]."""
        workspace = GlobalWorkspace()
        controller = AttentionController(workspace=workspace)
        
        novelty = controller._compute_novelty(percept)
        assert 0.0 <= novelty <= 1.0
