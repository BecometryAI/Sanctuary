"""
Property-based tests for AttentionController.

Tests validate:
- Attention budget constraints
- Attention score bounds
- Selection behavior
"""

import pytest
from hypothesis import given, settings
from lyra.cognitive_core.workspace import GlobalWorkspace
from lyra.cognitive_core.attention import AttentionController
from .strategies import percept_lists, goal_lists, emotional_states


@pytest.mark.property
class TestAttentionProperties:
    
    @given(percept_lists)
    @settings(max_examples=50, deadline=500)
    def test_attention_budget_never_exceeded(self, percepts_list):
        """Property: Broadcast selection never exceeds budget limit."""
        budget = 5
        workspace = GlobalWorkspace()
        attention = AttentionController(attention_budget=budget, workspace=workspace)
        
        # Select percepts for broadcast
        selected = attention.select_for_broadcast(percepts_list)
        
        # Calculate total complexity
        total_complexity = sum(p.complexity for p in selected)
        
        # Budget constraint must be respected
        assert total_complexity <= budget
        assert len(selected) <= len(percepts_list)
    
    @given(percept_lists, goal_lists, emotional_states())
    @settings(max_examples=50, deadline=500)
    def test_attention_selection_subset(self, percepts_list, goals_list, emotions):
        """Property: Selected percepts are always a subset of candidates."""
        workspace = GlobalWorkspace()
        attention = AttentionController(attention_budget=10, workspace=workspace)
        
        # Add goals and emotions to workspace for context
        for goal in goals_list:
            workspace.add_goal(goal)
        workspace.emotional_state.update(emotions)
        
        # Select percepts
        selected = attention.select_for_broadcast(percepts_list)
        
        # All selected percepts must be from the original list
        selected_ids = {p.id for p in selected}
        candidate_ids = {p.id for p in percepts_list}
        
        assert selected_ids.issubset(candidate_ids)
    
    @given(percept_lists)
    @settings(max_examples=50)
    def test_empty_candidates_returns_empty_selection(self, percepts_list):
        """Property: Empty candidate list results in empty selection."""
        workspace = GlobalWorkspace()
        attention = AttentionController(attention_budget=10, workspace=workspace)
        
        # Test with empty list
        selected = attention.select_for_broadcast([])
        assert len(selected) == 0
        
    @given(percept_lists)
    @settings(max_examples=50)
    def test_attention_budget_reset_works(self, percepts_list):
        """Property: Budget reset restores initial budget."""
        initial_budget = 100
        workspace = GlobalWorkspace()
        attention = AttentionController(attention_budget=initial_budget, workspace=workspace)
        
        # Select some percepts
        attention.select_for_broadcast(percepts_list[:5] if len(percepts_list) > 5 else percepts_list)
        
        # Reset budget
        attention.reset_budget()
        
        # Budget should be restored
        assert attention.attention_budget == initial_budget
    
    @given(percept_lists)
    @settings(max_examples=50)
    def test_attention_selection_deterministic_with_same_input(self, percepts_list):
        """Property: Same input produces same output (deterministic behavior)."""
        if not percepts_list:
            return  # Skip empty lists
        
        workspace = GlobalWorkspace()
        attention = AttentionController(attention_budget=10, workspace=workspace)
        
        # Select twice with same input
        selected1 = attention.select_for_broadcast(percepts_list)
        
        # Reset attention state
        attention.recent_percepts.clear()
        attention.attention_history.clear()
        
        selected2 = attention.select_for_broadcast(percepts_list)
        
        # Same percepts should be selected
        assert len(selected1) == len(selected2)
        assert {p.id for p in selected1} == {p.id for p in selected2}
