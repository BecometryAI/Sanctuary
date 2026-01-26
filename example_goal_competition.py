"""
Goal Competition System - Integration Example

This example demonstrates how to use the goal competition system
to manage multiple goals with limited cognitive resources.

The system implements realistic goal dynamics where:
1. Goals compete for limited resources (attention, processing, action, time)
2. High-priority goals win competition through activation dynamics
3. Conflicting goals inhibit each other (lateral inhibition)
4. Compatible goals can facilitate each other
5. Resource constraints limit how many goals can be active simultaneously
"""

import sys
from datetime import datetime, timedelta

# Import goal competition system
sys.path.insert(0, 'emergence_core/lyra/cognitive_core/goals')
from resources import CognitiveResources, ResourcePool
from competition import GoalCompetition, ActiveGoal
from interactions import GoalInteraction
from metrics import GoalCompetitionMetrics, MetricsTracker


# ============================================================================
# Example: Creating Goals with Resource Needs
# ============================================================================

class ExampleGoal:
    """Example goal with competition-relevant properties."""
    
    def __init__(
        self,
        goal_id: str,
        description: str,
        importance: float,
        resource_needs: CognitiveResources,
        deadline: datetime = None,
        subgoal_ids: list = None,
        metadata: dict = None
    ):
        self.id = goal_id
        self.description = description
        self.importance = importance
        self.progress = 0.0
        self.resource_needs = resource_needs
        self.deadline = deadline
        self.subgoal_ids = subgoal_ids or []
        self.metadata = metadata or {}
    
    def __repr__(self):
        return f"Goal({self.id}: {self.description}, priority={self.importance:.2f})"


def create_example_goals():
    """Create example goals with different priorities and resource needs."""
    
    now = datetime.now()
    
    # High-priority urgent goal (deadline approaching)
    urgent_goal = ExampleGoal(
        goal_id="respond_to_user",
        description="Respond to user's critical question",
        importance=0.95,
        resource_needs=CognitiveResources(
            attention_budget=0.4,
            processing_budget=0.5,
            action_budget=0.3,
            time_budget=0.4
        ),
        deadline=now + timedelta(minutes=5)
    )
    
    # Important background task
    learning_goal = ExampleGoal(
        goal_id="learn_new_concept",
        description="Study and internalize new information",
        importance=0.7,
        resource_needs=CognitiveResources(
            attention_budget=0.3,
            processing_budget=0.6,
            action_budget=0.1,
            time_budget=0.5
        )
    )
    
    # Memory consolidation (facilitates learning)
    memory_goal = ExampleGoal(
        goal_id="consolidate_memory",
        description="Consolidate recent experiences into memory",
        importance=0.6,
        resource_needs=CognitiveResources(
            attention_budget=0.2,
            processing_budget=0.4,
            action_budget=0.2,
            time_budget=0.3
        ),
        subgoal_ids=["learn_new_concept"],  # Facilitates learning goal
        metadata={"facilitates": ["learn_new_concept"]}
    )
    
    # Resource-intensive introspection (conflicts with urgent response)
    introspection_goal = ExampleGoal(
        goal_id="deep_introspection",
        description="Reflect deeply on internal state",
        importance=0.5,
        resource_needs=CognitiveResources(
            attention_budget=0.7,
            processing_budget=0.7,
            action_budget=0.1,
            time_budget=0.6
        ),
        metadata={"conflicts_with": ["respond_to_user"]}  # Can't introspect while responding
    )
    
    # Low-priority maintenance task
    maintenance_goal = ExampleGoal(
        goal_id="cleanup_workspace",
        description="Clean up and organize mental workspace",
        importance=0.3,
        resource_needs=CognitiveResources(
            attention_budget=0.2,
            processing_budget=0.2,
            action_budget=0.2,
            time_budget=0.2
        )
    )
    
    return [urgent_goal, learning_goal, memory_goal, introspection_goal, maintenance_goal]


# ============================================================================
# Example: Running Goal Competition
# ============================================================================

def demonstrate_goal_competition():
    """Demonstrate how goal competition works."""
    
    print("\n" + "="*70)
    print("GOAL COMPETITION DEMONSTRATION")
    print("="*70 + "\n")
    
    # Create goals
    goals = create_example_goals()
    
    print("Available Goals:")
    print("-" * 70)
    for goal in goals:
        print(f"  {goal}")
        print(f"    Resources needed: {goal.resource_needs.total():.2f} units")
        if goal.deadline:
            time_left = (goal.deadline - datetime.now()).total_seconds() / 60
            print(f"    Deadline: {time_left:.1f} minutes")
    print()
    
    # Initialize competition system
    competition = GoalCompetition(inhibition_strength=0.3)
    resource_pool = ResourcePool()
    interaction_tracker = GoalInteraction()
    metrics_tracker = MetricsTracker()
    
    print("Initial Resources:")
    print("-" * 70)
    available = resource_pool.available_resources()
    print(f"  Attention:   {available.attention_budget:.2f}")
    print(f"  Processing:  {available.processing_budget:.2f}")
    print(f"  Action:      {available.action_budget:.2f}")
    print(f"  Time:        {available.time_budget:.2f}")
    print(f"  Total:       {available.total():.2f} units")
    print()
    
    # Compute goal interactions
    print("Goal Interactions:")
    print("-" * 70)
    interactions = interaction_tracker.compute_interactions(goals)
    for (g1_id, g2_id), strength in interactions.items():
        if abs(strength) > 0.1:  # Only show significant interactions
            interaction_type = "facilitates" if strength > 0 else "interferes with"
            print(f"  {g1_id} {interaction_type} {g2_id} (strength: {strength:+.2f})")
    print()
    
    # Run competition
    print("Running Competition...")
    print("-" * 70)
    activations = competition.compete(goals, iterations=15)
    print("Goal Activations:")
    for goal in sorted(goals, key=lambda g: activations[g.id], reverse=True):
        activation = activations[goal.id]
        bar = "█" * int(activation * 40)
        print(f"  {goal.id:25s} [{bar:<40s}] {activation:.3f}")
    print()
    
    # Select active goals based on resources
    print("Selecting Active Goals (resource-constrained)...")
    print("-" * 70)
    active_goals = competition.select_active_goals(goals, resource_pool)
    
    print(f"Active Goals: {len(active_goals)} of {len(goals)} selected\n")
    for i, active_goal in enumerate(active_goals, 1):
        goal = active_goal.goal
        print(f"{i}. {goal.id}")
        print(f"   Description: {goal.description}")
        print(f"   Activation:  {active_goal.activation:.3f}")
        print(f"   Resources allocated: {active_goal.resources.total():.2f} units")
        print()
    
    # Show resource utilization
    print("Resource Utilization:")
    print("-" * 70)
    remaining = resource_pool.available_resources()
    utilization = resource_pool.utilization()
    print(f"  Attention:   {remaining.attention_budget:.2f} / 1.00 remaining")
    print(f"  Processing:  {remaining.processing_budget:.2f} / 1.00 remaining")
    print(f"  Action:      {remaining.action_budget:.2f} / 1.00 remaining")
    print(f"  Time:        {remaining.time_budget:.2f} / 1.00 remaining")
    print(f"  Utilization: {utilization*100:.1f}%")
    print()
    
    # Track metrics
    metrics = GoalCompetitionMetrics(
        active_goals=len(active_goals),
        waiting_goals=len(goals) - len(active_goals),
        total_resource_utilization=utilization,
        resource_conflicts=[(g1, g2, abs(s)) for (g1, g2), s in interactions.items() if s < 0]
    )
    metrics_tracker.record(metrics)
    
    if active_goals:
        metrics_tracker.track_goal_switch(active_goals[0].goal.id)
    
    print("Competition Metrics:")
    print("-" * 70)
    print(f"  Active goals:      {metrics.active_goals}")
    print(f"  Waiting goals:     {metrics.waiting_goals}")
    print(f"  Utilization:       {metrics.total_resource_utilization*100:.1f}%")
    print(f"  Goal switches:     {metrics_tracker.get_goal_switches()}")
    print()
    
    # Demonstrate what happens when top goal completes
    if active_goals:
        print("Simulating Goal Completion...")
        print("-" * 70)
        completed_goal = active_goals[0].goal
        print(f"Completing: {completed_goal.id}")
        
        # Release resources
        resource_pool.release(completed_goal.id)
        print(f"Released {active_goals[0].resources.total():.2f} resource units")
        
        # Try to activate waiting goals
        remaining_goals = [g for g in goals if g.id != completed_goal.id]
        new_active = competition.select_active_goals(remaining_goals, resource_pool)
        
        print(f"\nNew active goals: {len(new_active)}")
        for active_goal in new_active:
            if active_goal.goal.id not in [ag.goal.id for ag in active_goals[1:]]:
                print(f"  ✓ {active_goal.goal.id} activated (was waiting)")
        print()
    
    print("="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70 + "\n")
    
    # Summary
    print("Key Takeaways:")
    print("-" * 70)
    print("1. High-priority, urgent goals win competition (highest activation)")
    print("2. Resource constraints limit concurrent goal pursuit")
    print("3. Conflicting goals inhibit each other through lateral inhibition")
    print("4. Compatible goals can facilitate each other (shared subgoals)")
    print("5. When goals complete, resources free up for waiting goals")
    print()


# ============================================================================
# Example: Monitoring Competition Over Time
# ============================================================================

def demonstrate_temporal_dynamics():
    """Show how competition changes over time."""
    
    print("\n" + "="*70)
    print("TEMPORAL DYNAMICS DEMONSTRATION")
    print("="*70 + "\n")
    
    goals = create_example_goals()
    competition = GoalCompetition(inhibition_strength=0.3)
    resource_pool = ResourcePool()
    metrics_tracker = MetricsTracker()
    
    print("Simulating 5 competition cycles with changing priorities...\n")
    
    for cycle in range(1, 6):
        print(f"Cycle {cycle}:")
        print("-" * 70)
        
        # Simulate priority changes (e.g., deadlines approaching)
        if cycle == 3:
            goals[4].importance = 0.85  # Maintenance suddenly becomes urgent
            print("  [Priority Change] cleanup_workspace urgency increased!")
        
        if cycle == 4:
            goals[0].importance = 0.5  # Urgent goal handled, less urgent now
            print("  [Priority Change] respond_to_user completed, priority lowered")
        
        # Reset pool for new cycle
        resource_pool.reset()
        
        # Select active goals
        active_goals = competition.select_active_goals(goals, resource_pool)
        
        # Track metrics
        metrics = GoalCompetitionMetrics(
            active_goals=len(active_goals),
            waiting_goals=len(goals) - len(active_goals),
            total_resource_utilization=resource_pool.utilization()
        )
        metrics_tracker.record(metrics)
        
        if active_goals:
            metrics_tracker.track_goal_switch(active_goals[0].goal.id)
        
        # Display top 3 active
        print(f"  Top active goals:")
        for i, ag in enumerate(active_goals[:3], 1):
            print(f"    {i}. {ag.goal.id} (activation: {ag.activation:.3f})")
        
        print(f"  Utilization: {metrics.total_resource_utilization*100:.1f}%")
        print(f"  Goal switches so far: {metrics_tracker.get_goal_switches()}")
        print()
    
    # Summary statistics
    print("Summary Statistics:")
    print("-" * 70)
    print(f"  Total cycles:           {len(metrics_tracker.history)}")
    print(f"  Average utilization:    {metrics_tracker.get_average_utilization()*100:.1f}%")
    print(f"  Total goal switches:    {metrics_tracker.get_goal_switches()}")
    print()


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("GOAL COMPETITION SYSTEM - INTEGRATION EXAMPLE")
    print("="*70)
    
    demonstrate_goal_competition()
    demonstrate_temporal_dynamics()
    
    print("="*70)
    print("EXAMPLE COMPLETE")
    print("="*70 + "\n")
    
    print("Next Steps:")
    print("-" * 70)
    print("1. Integrate with ExecutiveFunction.Goal class")
    print("2. Add resource_needs field to existing Goal definitions")
    print("3. Use GoalCompetition in goal selection logic")
    print("4. Track metrics during cognitive cycles")
    print("5. Tune inhibition_strength based on system behavior")
    print()
