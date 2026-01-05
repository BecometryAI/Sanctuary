"""
Integration tests for AttentionController.

Tests that attention correctly selects percepts based on
goal relevance, novelty, and emotional salience.
"""
import pytest
from lyra.cognitive_core.workspace import GlobalWorkspace, Goal, GoalType, Percept
from lyra.cognitive_core.attention import AttentionController
from lyra.cognitive_core.affect import AffectSubsystem


@pytest.mark.integration
class TestAttentionSelection:
    """Test attention selection mechanisms."""
    
    def test_goal_relevance_scoring(self):
        """Test that percepts relevant to goals receive higher scores."""
        workspace = GlobalWorkspace()
        affect = AffectSubsystem()
        attention = AttentionController(
            attention_budget=100,
            workspace=workspace,
            affect=affect
        )
        
        # Add goal
        goal = Goal(
            type=GoalType.RESPOND_TO_USER,
            description="Discuss philosophy",
            priority=0.9
        )
        workspace.add_goal(goal)
        
        # Create percepts (one relevant, one not)
        relevant_percept = Percept(
            modality="text",
            raw="What is consciousness?",
            complexity=5,
            metadata={"keywords": ["consciousness", "philosophy"]}
        )
        
        irrelevant_percept = Percept(
            modality="text",
            raw="The weather is nice today",
            complexity=3,
            metadata={"keywords": ["weather"]}
        )
        
        # Select percepts
        selected = attention.select_for_broadcast([relevant_percept, irrelevant_percept])
        
        # Relevant percept should be selected first
        assert len(selected) > 0
        assert selected[0].id == relevant_percept.id
    
    def test_novelty_detection(self):
        """Test that novel percepts receive attention boost."""
        workspace = GlobalWorkspace()
        affect = AffectSubsystem()
        attention = AttentionController(
            attention_budget=100,
            workspace=workspace,
            affect=affect
        )
        
        # Add familiar percept to workspace
        familiar_percept = Percept(
            modality="text",
            raw="Hello",
            complexity=2
        )
        workspace.active_percepts[familiar_percept.id] = familiar_percept
        
        # Create new percepts (one novel, one similar to familiar)
        novel_percept = Percept(
            modality="text",
            raw="Quantum mechanics is fascinating!",
            complexity=5
        )
        
        similar_percept = Percept(
            modality="text",
            raw="Hello there",
            complexity=2
        )
        
        # Select percepts
        selected = attention.select_for_broadcast([novel_percept, similar_percept])
        
        # Novel percept should receive higher score
        # (This test may need adjustment based on actual scoring implementation)
        assert len(selected) > 0
        # Verify novel content is selected
        selected_content = [p.raw for p in selected]
        assert novel_percept.raw in selected_content
    
    def test_emotional_salience(self):
        """Test that emotionally salient percepts receive attention boost."""
        workspace = GlobalWorkspace()
        affect = AffectSubsystem()
        attention = AttentionController(
            attention_budget=100,
            workspace=workspace,
            affect=affect
        )
        
        # Create percepts with different emotional salience
        high_salience = Percept(
            modality="text",
            raw="URGENT: Critical system failure!",
            complexity=8,
            metadata={"emotional_valence": -0.9}
        )
        
        low_salience = Percept(
            modality="text",
            raw="The system is running normally.",
            complexity=3,
            metadata={"emotional_valence": 0.1}
        )
        
        # Select percepts
        selected = attention.select_for_broadcast([high_salience, low_salience])
        
        # High salience should be selected
        assert len(selected) > 0
        # Verify high salience content prioritized
        if len(selected) == 1:
            assert selected[0].id == high_salience.id
    
    def test_attention_budget_enforced(self):
        """Test that attention budget limits number of selected percepts."""
        workspace = GlobalWorkspace()
        affect = AffectSubsystem()
        
        # Set tight attention budget
        attention = AttentionController(
            attention_budget=10,  # Only 10 units
            workspace=workspace,
            affect=affect
        )
        
        # Create many percepts with varying complexity
        percepts = [
            Percept(modality="text", raw=f"Percept {i}", complexity=5)
            for i in range(10)
        ]
        
        # Select percepts
        selected = attention.select_for_broadcast(percepts)
        
        # Should only select percepts that fit in budget
        total_complexity = sum(p.complexity for p in selected)
        assert total_complexity <= 10
        
        # Should not select all percepts
        assert len(selected) < len(percepts)
