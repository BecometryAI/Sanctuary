"""
Example integration of meta-cognitive capabilities with the cognitive loop.

This demonstrates how the new meta-cognitive monitoring, action learning,
and attention history can be integrated into the existing Sanctuary cognitive architecture.
"""

# NOTE: This is an example/reference implementation showing integration points.
# Actual integration would occur in the cognitive loop implementation.

from typing import Dict, Any, List
from datetime import datetime


class MetaCognitiveIntegration:
    """
    Example integration of meta-cognitive capabilities into the cognitive loop.
    
    This class demonstrates the integration points where the new meta-cognitive
    systems can be incorporated into the existing cognitive architecture.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize meta-cognitive integration."""
        from mind.cognitive_core.meta_cognition import MetaCognitiveSystem
        
        # Initialize the unified meta-cognitive system
        self.meta_system = MetaCognitiveSystem(
            config=config.get("meta_cognitive", {})
        )
        
        # Track current allocation for outcome recording
        self.current_allocation_id = None
    
    def observe_processing(self, process_type: str, complexity: float, 
                          execute_fn, *args, **kwargs):
        """
        Wrap a cognitive process with meta-cognitive observation.
        
        Args:
            process_type: Type of process (e.g., "reasoning", "memory_retrieval")
            complexity: Estimated input complexity (0.0-1.0)
            execute_fn: Function to execute
            *args, **kwargs: Arguments for execute_fn
            
        Returns:
            Result of execute_fn
        """
        with self.meta_system.monitor.observe(process_type) as ctx:
            ctx.set_complexity(complexity)
            
            # Execute the cognitive process
            result = execute_fn(*args, **kwargs)
            
            # Assess output quality
            quality = self._assess_output_quality(result)
            ctx.set_quality(quality)
            
            return result
    
    def record_action_outcome(self, action, intended_outcome: str, 
                             actual_outcome: str, context: Dict[str, Any]):
        """
        Record the outcome of an action.
        
        Args:
            action: The action that was executed
            intended_outcome: What the action was supposed to achieve
            actual_outcome: What actually happened
            context: Contextual information (emotions, workspace state, etc.)
        """
        action_id = getattr(action, 'id', str(id(action)))
        action_type = str(getattr(action, 'type', 'unknown'))
        
        self.meta_system.action_learner.record_outcome(
            action_id=action_id,
            action_type=action_type,
            intended=intended_outcome,
            actual=actual_outcome,
            context=context
        )
    
    def record_attention_allocation(self, allocation: Dict[str, float],
                                   trigger: str, workspace_state):
        """
        Record an attention allocation decision.
        
        Args:
            allocation: Dict mapping targets to attention amounts
            trigger: What caused this allocation
            workspace_state: Current workspace state
            
        Returns:
            Allocation ID for later outcome recording
        """
        self.current_allocation_id = self.meta_system.attention_history.record_allocation(
            allocation=allocation,
            trigger=trigger,
            workspace_state=workspace_state
        )
        return self.current_allocation_id
    
    def record_attention_outcome(self, goal_progress: Dict[str, float],
                                discoveries: List[str], missed: List[str]):
        """
        Record the outcome of an attention allocation.
        
        Args:
            goal_progress: Progress made on each goal (goal_id -> progress)
            discoveries: Things that were discovered/noticed
            missed: Things that were missed (known in retrospect)
        """
        if self.current_allocation_id:
            self.meta_system.attention_history.record_outcome(
                allocation_id=self.current_allocation_id,
                goal_progress=goal_progress,
                discoveries=discoveries,
                missed=missed
            )
            self.current_allocation_id = None
    
    def get_periodic_assessment(self) -> str:
        """
        Get a periodic self-assessment.
        
        Returns:
            Human-readable assessment string
        """
        assessment = self.meta_system.get_self_assessment()
        
        lines = ["=== Meta-Cognitive Self-Assessment ==="]
        
        if assessment.processing_patterns:
            lines.append(f"\nProcessing Patterns ({len(assessment.processing_patterns)}):")
            for pattern in assessment.processing_patterns[:3]:
                lines.append(f"  • {pattern.description}")
        
        if assessment.identified_strengths:
            lines.append(f"\nStrengths:")
            for strength in assessment.identified_strengths[:3]:
                lines.append(f"  • {strength}")
        
        if assessment.identified_weaknesses:
            lines.append(f"\nAreas for Improvement:")
            for weakness in assessment.identified_weaknesses[:3]:
                lines.append(f"  • {weakness}")
        
        if assessment.suggested_adaptations:
            lines.append(f"\nSuggested Adaptations:")
            for adaptation in assessment.suggested_adaptations[:3]:
                lines.append(f"  • {adaptation}")
        
        return "\n".join(lines)
    
    def introspect(self, query: str) -> str:
        """
        Answer meta-cognitive questions.
        
        Args:
            query: Question about cognitive patterns
            
        Returns:
            Answer to the question
        """
        return self.meta_system.introspect(query)
    
    def _assess_output_quality(self, result) -> float:
        """
        Assess the quality of a processing result.
        
        This is a placeholder - actual implementation would depend on
        the type of result and available quality metrics.
        
        Args:
            result: The processing result
            
        Returns:
            Quality score (0.0-1.0)
        """
        # Simple heuristic - actual implementation would be more sophisticated
        if result is None:
            return 0.0
        
        if isinstance(result, (list, dict)):
            return 0.7 if len(result) > 0 else 0.3
        
        if isinstance(result, str):
            return min(1.0, len(result) / 100)
        
        return 0.5  # Default


# Example usage in cognitive loop
def example_cognitive_cycle_with_metacognition():
    """
    Example of how to integrate meta-cognitive monitoring into a cognitive cycle.
    """
    
    # Initialize integration
    config = {
        "meta_cognitive": {
            "monitor": {"max_observations": 1000},
            "action_learner": {"max_outcomes": 1000},
            "attention_history": {"max_allocations": 1000}
        }
    }
    meta_integration = MetaCognitiveIntegration(config)
    
    # Simulate a cognitive cycle
    print("=== Cognitive Cycle with Meta-Cognition ===\n")
    
    # 1. Goal selection (with monitoring)
    print("1. Selecting goal...")
    def select_goal():
        # Simulate goal selection
        return {"id": "goal_1", "type": "respond", "priority": 0.8}
    
    selected_goal = meta_integration.observe_processing(
        process_type="goal_selection",
        complexity=0.5,
        execute_fn=select_goal
    )
    print(f"   Selected: {selected_goal['type']}\n")
    
    # 2. Attention allocation (with tracking)
    print("2. Allocating attention...")
    allocation_id = meta_integration.record_attention_allocation(
        allocation={"goal_1": 0.7, "goal_2": 0.3},
        trigger="new_percept",
        workspace_state={"cycle": 1}
    )
    print(f"   Allocation ID: {allocation_id[:8]}...\n")
    
    # 3. Reasoning (with monitoring)
    print("3. Performing reasoning...")
    def reason():
        # Simulate reasoning
        return "Concluded that greeting is appropriate"
    
    reasoning_result = meta_integration.observe_processing(
        process_type="reasoning",
        complexity=0.6,
        execute_fn=reason
    )
    print(f"   Result: {reasoning_result}\n")
    
    # 4. Action execution (with outcome tracking)
    print("4. Executing action...")
    class MockAction:
        def __init__(self):
            self.id = "action_1"
            self.type = "speak"
    
    action = MockAction()
    intended = "provide helpful greeting"
    actual = "provided friendly greeting with enthusiasm"
    
    meta_integration.record_action_outcome(
        action=action,
        intended_outcome=intended,
        actual_outcome=actual,
        context={"user_sentiment": "positive", "time_of_day": "morning"}
    )
    print(f"   Action completed: {action.type}\n")
    
    # 5. Record attention outcome
    print("5. Recording attention outcome...")
    meta_integration.record_attention_outcome(
        goal_progress={"goal_1": 0.5, "goal_2": 0.1},
        discoveries=["user prefers enthusiastic greetings"],
        missed=[]
    )
    print("   Outcome recorded\n")
    
    # 6. Periodic assessment (every N cycles)
    print("6. Self-assessment:")
    assessment = meta_integration.get_periodic_assessment()
    print(assessment)
    print()
    
    # 7. Introspection queries
    print("7. Introspection:")
    questions = [
        "What patterns have I identified?",
        "How reliable are my actions?",
        "What are my strengths?"
    ]
    
    for question in questions:
        response = meta_integration.introspect(question)
        print(f"   Q: {question}")
        print(f"   A: {response[:100]}...")
        print()


if __name__ == "__main__":
    # Run example
    example_cognitive_cycle_with_metacognition()
    
    print("\n" + "="*60)
    print("Example completed successfully!")
    print("="*60)
