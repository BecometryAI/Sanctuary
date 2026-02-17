"""
Goal Priority Dynamics

Adjusts goal priorities over time based on:
- Staleness/frustration: goals that stall gain urgency
- Deadline proximity: goals approaching deadline escalate
- Emotional congruence: emotion-relevant goals get boosted
"""

from __future__ import annotations

import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Tuning constants
STALL_CYCLE_THRESHOLD = 30        # Cycles before a goal is considered "stalled"
STALL_BOOST_PER_CYCLE = 0.002     # Priority boost per cycle while stalled
MAX_STALL_BOOST = 0.15            # Maximum cumulative stall boost
DEADLINE_BOOST_MAX = 0.20         # Maximum deadline urgency boost
EMOTION_CONGRUENCE_BOOST = 0.08   # Boost when emotion matches goal
PROGRESS_DECAY = 0.02             # Priority reduction when goal is progressing well
ADJUSTMENT_CAP = 0.25             # Maximum total adjustment per cycle


@dataclass
class GoalAdjustment:
    """Record of a priority adjustment made to a goal."""
    goal_id: str
    old_priority: float
    new_priority: float
    reason: str
    adjustment: float


@dataclass
class GoalDynamicsState:
    """Tracks per-goal dynamics state across cycles."""
    # goal_id -> cycle count when goal was first seen
    first_seen_cycle: Dict[str, int] = field(default_factory=dict)
    # goal_id -> last progress value observed
    last_progress: Dict[str, float] = field(default_factory=dict)
    # goal_id -> cycles since progress changed
    cycles_since_progress: Dict[str, int] = field(default_factory=dict)


class GoalDynamics:
    """
    Adjusts goal priorities dynamically based on staleness,
    deadlines, and emotional congruence.

    Called once per cognitive cycle (after affect update, before action).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.stall_threshold = config.get("stall_cycle_threshold", STALL_CYCLE_THRESHOLD)
        self.stall_boost_rate = config.get("stall_boost_per_cycle", STALL_BOOST_PER_CYCLE)
        self.max_stall_boost = config.get("max_stall_boost", MAX_STALL_BOOST)
        self.deadline_boost_max = config.get("deadline_boost_max", DEADLINE_BOOST_MAX)
        self.emotion_boost = config.get("emotion_congruence_boost", EMOTION_CONGRUENCE_BOOST)
        self.progress_decay = config.get("progress_decay", PROGRESS_DECAY)
        self.adjustment_cap = config.get("adjustment_cap", ADJUSTMENT_CAP)

        self._state = GoalDynamicsState()
        logger.info("GoalDynamics initialized")

    def adjust_priorities(
        self,
        goals: List[Any],
        cycle_count: int,
        emotional_state: Optional[Dict[str, Any]] = None,
    ) -> List[GoalAdjustment]:
        """
        Compute and return priority adjustments for all active goals.

        Does NOT mutate the goals — returns a list of adjustments
        that the caller applies via workspace.update_goal_priority().

        Args:
            goals: Current active Goal objects
            cycle_count: Current cognitive cycle number
            emotional_state: Current affect state dict (valence, arousal, label, etc.)

        Returns:
            List of GoalAdjustment records for goals whose priority changed
        """
        adjustments = []
        emotional_state = emotional_state or {}

        for goal in goals:
            goal_id = goal.id
            old_priority = goal.priority

            # Track first-seen cycle
            if goal_id not in self._state.first_seen_cycle:
                self._state.first_seen_cycle[goal_id] = cycle_count
                self._state.last_progress[goal_id] = goal.progress
                self._state.cycles_since_progress[goal_id] = 0

            # Update progress tracking
            prev_progress = self._state.last_progress.get(goal_id, 0.0)
            if goal.progress > prev_progress + 0.01:
                # Progress was made — reset stall counter
                self._state.cycles_since_progress[goal_id] = 0
                self._state.last_progress[goal_id] = goal.progress
            else:
                self._state.cycles_since_progress[goal_id] = (
                    self._state.cycles_since_progress.get(goal_id, 0) + 1
                )

            total_adjustment = 0.0

            # 1. Staleness / frustration boost
            stall_cycles = self._state.cycles_since_progress.get(goal_id, 0)
            if stall_cycles > self.stall_threshold and goal.progress < 0.9:
                excess = stall_cycles - self.stall_threshold
                stall_boost = min(excess * self.stall_boost_rate, self.max_stall_boost)
                total_adjustment += stall_boost

            # 2. Deadline urgency boost
            deadline_boost = self._compute_deadline_boost(goal)
            total_adjustment += deadline_boost

            # 3. Emotional congruence boost
            emotion_adj = self._compute_emotion_adjustment(goal, emotional_state)
            total_adjustment += emotion_adj

            # 4. Progress decay — actively progressing goals slightly lower priority
            #    (resources are already being used efficiently)
            if goal.progress > 0.5 and stall_cycles == 0:
                total_adjustment -= self.progress_decay

            # Cap total adjustment
            total_adjustment = max(-self.adjustment_cap, min(self.adjustment_cap, total_adjustment))

            if abs(total_adjustment) > 0.005:
                new_priority = max(0.0, min(1.0, old_priority + total_adjustment))
                if abs(new_priority - old_priority) > 0.005:
                    reason = self._build_reason(stall_cycles, deadline_boost, emotion_adj)
                    adjustments.append(GoalAdjustment(
                        goal_id=goal_id,
                        old_priority=old_priority,
                        new_priority=new_priority,
                        reason=reason,
                        adjustment=total_adjustment,
                    ))

        # Clean up state for goals that no longer exist
        active_ids = {g.id for g in goals}
        stale_ids = set(self._state.first_seen_cycle.keys()) - active_ids
        for stale_id in stale_ids:
            self._state.first_seen_cycle.pop(stale_id, None)
            self._state.last_progress.pop(stale_id, None)
            self._state.cycles_since_progress.pop(stale_id, None)

        return adjustments

    def _compute_deadline_boost(self, goal) -> float:
        """Compute priority boost based on deadline proximity."""
        deadline = getattr(goal, 'deadline', None)
        if deadline is None:
            return 0.0

        now = datetime.now()
        if deadline <= now:
            return self.deadline_boost_max  # Overdue — max boost

        remaining = (deadline - now).total_seconds()
        if remaining <= 0:
            return self.deadline_boost_max

        # Exponential urgency curve: boost increases as deadline approaches
        # At 1 hour remaining: ~90% of max boost
        # At 1 day remaining: ~30% of max boost
        # At 1 week remaining: ~5% of max boost
        hours_remaining = remaining / 3600.0
        if hours_remaining < 1.0:
            fraction = 0.9
        elif hours_remaining < 24.0:
            fraction = 0.3 + 0.6 * (1.0 - hours_remaining / 24.0)
        elif hours_remaining < 168.0:  # 1 week
            fraction = 0.05 + 0.25 * (1.0 - hours_remaining / 168.0)
        else:
            fraction = 0.0

        return self.deadline_boost_max * fraction

    def _compute_emotion_adjustment(
        self, goal, emotional_state: Dict[str, Any]
    ) -> float:
        """Compute priority adjustment based on emotional congruence."""
        valence = emotional_state.get("valence", 0.0)
        arousal = emotional_state.get("arousal", 0.0)
        label = emotional_state.get("label", "neutral")

        goal_metadata = getattr(goal, 'metadata', {}) or {}
        goal_type = getattr(goal, 'type', None)
        goal_type_value = goal_type.value if hasattr(goal_type, 'value') else str(goal_type)

        # High arousal boosts action-oriented goals
        if arousal > 0.7 and goal_type_value in ("respond_to_user", "speak_autonomous"):
            return self.emotion_boost * 0.5

        # Negative valence boosts introspective / safety goals
        if valence < -0.4 and goal_type_value in ("introspect", "maintain_value"):
            return self.emotion_boost

        # Strong positive emotion boosts creative / learning goals
        if valence > 0.4 and goal_type_value in ("learn", "create"):
            return self.emotion_boost * 0.7

        # Emotional valence stored in goal metadata
        goal_valence = goal_metadata.get("emotional_valence", 0.0)
        if goal_valence != 0.0 and abs(valence) > 0.3:
            # Same-sign valence = congruent
            if (goal_valence > 0) == (valence > 0):
                return self.emotion_boost * 0.5
            else:
                return -self.emotion_boost * 0.3

        return 0.0

    @staticmethod
    def _build_reason(stall_cycles: int, deadline_boost: float, emotion_adj: float) -> str:
        parts = []
        if stall_cycles > STALL_CYCLE_THRESHOLD:
            parts.append(f"stalled {stall_cycles} cycles")
        if deadline_boost > 0.01:
            parts.append(f"deadline urgency +{deadline_boost:.3f}")
        if abs(emotion_adj) > 0.005:
            parts.append(f"emotion {'boost' if emotion_adj > 0 else 'decay'} {emotion_adj:+.3f}")
        return "; ".join(parts) if parts else "minor adjustment"
