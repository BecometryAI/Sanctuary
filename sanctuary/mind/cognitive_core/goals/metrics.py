"""
Goal Competition Metrics

Tracks metrics related to goal competition dynamics, including
resource utilization, inhibition events, and goal switching patterns.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
from datetime import datetime

# Valid utilization range
MIN_UTILIZATION = 0.0
MAX_UTILIZATION = 1.0


@dataclass
class GoalCompetitionMetrics:
    """
    Metrics for tracking goal competition dynamics.
    
    Attributes:
        timestamp: When metrics were captured
        active_goals: Number of goals currently active (with resources)
        waiting_goals: Number of goals waiting for resources
        total_resource_utilization: Fraction of resources in use (0.0 to 1.0)
        inhibition_events: Number of times goals inhibited each other
        facilitation_events: Number of times goals facilitated each other
        goal_switches: Number of times the top-priority goal changed
        resource_conflicts: List of (goal1_id, goal2_id, conflict_level) tuples
    """
    timestamp: datetime = field(default_factory=datetime.now)
    active_goals: int = 0
    waiting_goals: int = 0
    total_resource_utilization: float = 0.0
    inhibition_events: int = 0
    facilitation_events: int = 0
    goal_switches: int = 0
    resource_conflicts: List[Tuple[str, str, float]] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate metrics."""
        if self.active_goals < 0:
            raise ValueError(f"active_goals must be >= 0, got {self.active_goals}")
        if self.waiting_goals < 0:
            raise ValueError(f"waiting_goals must be >= 0, got {self.waiting_goals}")
        if not MIN_UTILIZATION <= self.total_resource_utilization <= MAX_UTILIZATION:
            raise ValueError(
                f"total_resource_utilization must be in [{MIN_UTILIZATION}, {MAX_UTILIZATION}], "
                f"got {self.total_resource_utilization}"
            )
    
    def to_dict(self):
        """Convert metrics to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'active_goals': self.active_goals,
            'waiting_goals': self.waiting_goals,
            'total_resource_utilization': self.total_resource_utilization,
            'inhibition_events': self.inhibition_events,
            'facilitation_events': self.facilitation_events,
            'goal_switches': self.goal_switches,
            'resource_conflicts': [
                {'goal1': g1, 'goal2': g2, 'level': level}
                for g1, g2, level in self.resource_conflicts
            ]
        }
    
    @staticmethod
    def from_dict(data: dict) -> 'GoalCompetitionMetrics':
        """Create metrics from dictionary."""
        conflicts = [
            (c['goal1'], c['goal2'], c['level'])
            for c in data.get('resource_conflicts', [])
        ]
        
        return GoalCompetitionMetrics(
            timestamp=datetime.fromisoformat(data['timestamp']),
            active_goals=data['active_goals'],
            waiting_goals=data['waiting_goals'],
            total_resource_utilization=data['total_resource_utilization'],
            inhibition_events=data.get('inhibition_events', 0),
            facilitation_events=data.get('facilitation_events', 0),
            goal_switches=data.get('goal_switches', 0),
            resource_conflicts=conflicts
        )


class MetricsTracker:
    """
    Tracks goal competition metrics over time.
    
    Maintains a history of metrics snapshots to analyze
    competition dynamics and resource allocation patterns.
    """
    
    def __init__(self, max_history: int = 100):
        """
        Initialize metrics tracker.
        
        Args:
            max_history: Maximum number of metrics snapshots to retain
        """
        self.max_history = max_history
        self.history: List[GoalCompetitionMetrics] = []
        self._last_top_goal: str = ""
        self._goal_switches = 0
    
    def record(self, metrics: GoalCompetitionMetrics):
        """
        Record a metrics snapshot.
        
        Args:
            metrics: Metrics to record
        """
        self.history.append(metrics)
        
        # Trim history if needed
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def track_goal_switch(self, current_top_goal_id: str):
        """
        Track if the top-priority goal has changed.
        
        Args:
            current_top_goal_id: ID of current highest-priority goal
        """
        if self._last_top_goal and current_top_goal_id != self._last_top_goal:
            self._goal_switches += 1
        self._last_top_goal = current_top_goal_id
    
    def get_goal_switches(self) -> int:
        """Get total number of goal switches tracked."""
        return self._goal_switches
    
    def get_average_utilization(self) -> float:
        """
        Calculate average resource utilization across history.
        
        Returns:
            Average utilization (0.0 to 1.0)
        """
        if not self.history:
            return 0.0
        
        total = sum(m.total_resource_utilization for m in self.history)
        return total / len(self.history)
    
    def get_total_inhibition_events(self) -> int:
        """Get total inhibition events across history."""
        return sum(m.inhibition_events for m in self.history)
    
    def get_total_facilitation_events(self) -> int:
        """Get total facilitation events across history."""
        return sum(m.facilitation_events for m in self.history)
    
    def get_latest(self) -> GoalCompetitionMetrics:
        """
        Get most recent metrics snapshot.
        
        Returns:
            Latest metrics
            
        Raises:
            IndexError: If no metrics recorded yet
        """
        if not self.history:
            raise IndexError("No metrics recorded yet")
        return self.history[-1]
    
    def clear(self):
        """Clear all tracked metrics."""
        self.history.clear()
        self._last_top_goal = ""
        self._goal_switches = 0
