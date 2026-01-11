"""
Meta-Cognitive System: Unified meta-cognitive capabilities.

This module provides a unified interface to all meta-cognitive subsystems:
monitoring, action-outcome learning, and attention history.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from .metacognitive_monitor import MetaCognitiveMonitor
from .action_learning import ActionOutcomeLearner, ActionReliability
from .attention_history import AttentionHistory, AttentionPattern
from .pattern_detection import CognitivePattern

logger = logging.getLogger(__name__)


@dataclass
class SelfAssessment:
    """
    Overall self-assessment of cognitive functioning.
    
    Attributes:
        processing_patterns: Identified patterns in processing
        action_reliability: Summary of action reliability by type
        attention_effectiveness: Summary of attention effectiveness
        identified_strengths: Areas where system performs well
        identified_weaknesses: Areas where system struggles
        suggested_adaptations: Recommendations for improvement
    """
    processing_patterns: List[CognitivePattern]
    action_reliability: Dict[str, ActionReliability]
    attention_effectiveness: Dict[str, Any]
    identified_strengths: List[str]
    identified_weaknesses: List[str]
    suggested_adaptations: List[str]


class MetaCognitiveSystem:
    """
    Unified meta-cognitive capabilities.
    
    Provides a single interface to:
    - Processing monitoring and pattern detection
    - Action-outcome learning and reliability assessment
    - Attention allocation history and pattern learning
    - Overall self-assessment and introspection
    
    Usage:
        system = MetaCognitiveSystem()
        
        # Monitor a process
        with system.monitor.observe('reasoning') as ctx:
            ctx.input_complexity = 0.7
            result = do_reasoning()
            ctx.output_quality = 0.9
        
        # Record an action outcome
        system.record_action_outcome(
            action_id="act_123",
            action_type="speak",
            intended="respond helpfully",
            actual="provided helpful response",
            context={"user_query": "..."}
        )
        
        # Record attention allocation
        alloc_id = system.record_attention(
            allocation={"goal_1": 0.6, "goal_2": 0.4},
            trigger="goal_priority",
            workspace_state=snapshot
        )
        
        # Later, record outcome
        system.record_attention_outcome(
            allocation_id=alloc_id,
            goal_progress={"goal_1": 0.3, "goal_2": 0.1},
            discoveries=["new insight"],
            missed=[]
        )
        
        # Get self-assessment
        assessment = system.get_self_assessment()
        
        # Introspect
        response = system.introspect("What do I tend to fail at?")
    """
    
    def __init__(
        self,
        min_observations_for_patterns: int = 3,
        min_outcomes_for_model: int = 5
    ):
        """
        Initialize meta-cognitive system.
        
        Args:
            min_observations_for_patterns: Min observations to detect patterns
            min_outcomes_for_model: Min outcomes to build action models
        """
        self.monitor = MetaCognitiveMonitor(
            min_observations_for_patterns=min_observations_for_patterns
        )
        self.action_learner = ActionOutcomeLearner(
            min_outcomes_for_model=min_outcomes_for_model
        )
        self.attention_history = AttentionHistory()
        
        logger.info("✅ MetaCognitiveSystem initialized")
    
    # Action-outcome methods
    
    def record_action_outcome(
        self,
        action_id: str,
        action_type: str,
        intended: str,
        actual: str,
        context: Dict[str, Any]
    ):
        """
        Record the outcome of an action.
        
        Args:
            action_id: Unique identifier for the action
            action_type: Type of action
            intended: Intended outcome
            actual: Actual outcome
            context: Contextual information
        """
        self.action_learner.record_outcome(
            action_id=action_id,
            action_type=action_type,
            intended=intended,
            actual=actual,
            context=context
        )
    
    def get_action_reliability(self, action_type: str) -> ActionReliability:
        """
        Get reliability assessment for an action type.
        
        Args:
            action_type: Type of action
            
        Returns:
            ActionReliability assessment
        """
        return self.action_learner.get_action_reliability(action_type)
    
    def predict_action_outcome(
        self,
        action_type: str,
        context: Dict[str, Any]
    ):
        """
        Predict likely outcome of an action in context.
        
        Args:
            action_type: Type of action
            context: Contextual information
            
        Returns:
            OutcomePrediction
        """
        return self.action_learner.predict_outcome(action_type, context)
    
    # Attention methods
    
    def record_attention(
        self,
        allocation: Dict[str, float],
        trigger: str,
        workspace_state: Any
    ) -> str:
        """
        Record an attention allocation.
        
        Args:
            allocation: Attention allocation
            trigger: What caused this allocation
            workspace_state: Current workspace state
            
        Returns:
            Allocation ID for later outcome recording
        """
        return self.attention_history.record_allocation(
            allocation=allocation,
            trigger=trigger,
            workspace_state=workspace_state
        )
    
    def record_attention_outcome(
        self,
        allocation_id: str,
        goal_progress: Dict[str, float],
        discoveries: List[str],
        missed: List[str]
    ):
        """
        Record outcome of an attention allocation.
        
        Args:
            allocation_id: ID of the allocation
            goal_progress: Progress made on each goal
            discoveries: What was noticed/learned
            missed: What was missed
        """
        self.attention_history.record_outcome(
            allocation_id=allocation_id,
            goal_progress=goal_progress,
            discoveries=discoveries,
            missed=missed
        )
    
    def get_recommended_attention(
        self,
        context: Any,
        goals: List[Any]
    ) -> Dict[str, float]:
        """
        Get recommended attention allocation based on learned patterns.
        
        Args:
            context: Current workspace state
            goals: Current goals
            
        Returns:
            Recommended allocation
        """
        return self.attention_history.get_recommended_allocation(context, goals)
    
    # Self-assessment and introspection
    
    def get_self_assessment(self) -> SelfAssessment:
        """
        Get overall self-assessment of cognitive functioning.
        
        Returns:
            SelfAssessment with comprehensive analysis
        """
        # Gather data from all subsystems
        processing_patterns = self.monitor.get_identified_patterns()
        
        action_types = self.action_learner.get_all_action_types()
        action_reliability = {
            action_type: self.action_learner.get_action_reliability(action_type)
            for action_type in action_types
        }
        
        attention_effectiveness = self.attention_history.get_allocation_stats()
        
        # Identify strengths
        strengths = self._identify_strengths(
            processing_patterns,
            action_reliability,
            attention_effectiveness
        )
        
        # Identify weaknesses
        weaknesses = self._identify_weaknesses(
            processing_patterns,
            action_reliability,
            attention_effectiveness
        )
        
        # Generate adaptation suggestions
        adaptations = self._suggest_adaptations(
            processing_patterns,
            action_reliability,
            attention_effectiveness
        )
        
        return SelfAssessment(
            processing_patterns=processing_patterns,
            action_reliability=action_reliability,
            attention_effectiveness=attention_effectiveness,
            identified_strengths=strengths,
            identified_weaknesses=weaknesses,
            suggested_adaptations=adaptations
        )
    
    def _identify_strengths(
        self,
        patterns: List[CognitivePattern],
        reliability: Dict[str, ActionReliability],
        attention: Dict[str, Any]
    ) -> List[str]:
        """Identify areas where system performs well."""
        strengths = []
        
        # High success rate patterns
        success_patterns = [
            p for p in patterns
            if p.pattern_type == 'success_condition' and p.confidence > 0.7
        ]
        if success_patterns:
            strengths.append(
                f"Identified {len(success_patterns)} reliable success conditions"
            )
        
        # Reliable actions
        reliable_actions = [
            action_type for action_type, rel in reliability.items()
            if not rel.unknown and rel.success_rate > 0.8
        ]
        if reliable_actions:
            strengths.append(
                f"High reliability in actions: {', '.join(reliable_actions[:3])}"
            )
        
        # Effective attention
        if attention.get('avg_efficiency', 0) > 0.7:
            strengths.append(
                f"Effective attention allocation (avg efficiency: "
                f"{attention['avg_efficiency']:.2f})"
            )
        
        return strengths if strengths else ["Building understanding of capabilities"]
    
    def _identify_weaknesses(
        self,
        patterns: List[CognitivePattern],
        reliability: Dict[str, ActionReliability],
        attention: Dict[str, Any]
    ) -> List[str]:
        """Identify areas where system struggles."""
        weaknesses = []
        
        # Failure modes
        failure_patterns = [
            p for p in patterns
            if p.pattern_type == 'failure_mode' and p.confidence > 0.6
        ]
        if failure_patterns:
            # Get the most common failure
            top_failure = failure_patterns[0]
            weaknesses.append(f"Failure mode: {top_failure.description}")
        
        # Unreliable actions
        unreliable_actions = [
            action_type for action_type, rel in reliability.items()
            if not rel.unknown and rel.success_rate < 0.5
        ]
        if unreliable_actions:
            weaknesses.append(
                f"Low reliability in actions: {', '.join(unreliable_actions[:3])}"
            )
        
        # Ineffective attention
        if attention.get('avg_efficiency', 1.0) < 0.4:
            weaknesses.append(
                f"Inefficient attention allocation (avg: "
                f"{attention['avg_efficiency']:.2f})"
            )
        
        return weaknesses if weaknesses else ["No significant weaknesses identified yet"]
    
    def _suggest_adaptations(
        self,
        patterns: List[CognitivePattern],
        reliability: Dict[str, ActionReliability],
        attention: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations for improvement."""
        adaptations = []
        
        # Get actionable patterns
        actionable = [p for p in patterns if p.actionable and p.suggested_adaptation]
        for pattern in actionable[:3]:  # Top 3
            adaptations.append(pattern.suggested_adaptation)
        
        # Action-specific recommendations
        for action_type, rel in reliability.items():
            if not rel.unknown and rel.success_rate < 0.6 and rel.best_contexts:
                adaptations.append(
                    f"Use {action_type} primarily in high-success contexts"
                )
        
        # Attention recommendations
        patterns_list = self.attention_history.get_attention_patterns()
        if patterns_list:
            best_pattern = patterns_list[0]
            if best_pattern.avg_efficiency > 0.7:
                adaptations.append(best_pattern.recommendation)
        
        return adaptations if adaptations else ["Continue gathering data for recommendations"]
    
    def introspect(self, query: str) -> str:
        """
        Answer questions about own cognitive patterns.
        
        Args:
            query: Question about cognitive patterns
            
        Returns:
            Response based on meta-cognitive data
        """
        query_lower = query.lower()
        
        if "fail" in query_lower or "struggle" in query_lower:
            return self._describe_failure_patterns()
        elif "succeed" in query_lower or "good at" in query_lower:
            return self._describe_success_patterns()
        elif "attention" in query_lower:
            return self._describe_attention_patterns()
        elif "action" in query_lower or "reliable" in query_lower:
            return self._describe_action_reliability()
        else:
            return self._general_introspection()
    
    def _describe_failure_patterns(self) -> str:
        """Describe identified failure patterns."""
        patterns = [
            p for p in self.monitor.get_identified_patterns()
            if p.pattern_type == 'failure_mode'
        ]
        
        if not patterns:
            return "I haven't identified clear failure patterns yet."
        
        lines = ["Here's what I tend to struggle with:\n"]
        for pattern in patterns[:3]:
            lines.append(f"- {pattern.description}")
            if pattern.suggested_adaptation:
                lines.append(f"  Adaptation: {pattern.suggested_adaptation}")
        
        return "\n".join(lines)
    
    def _describe_success_patterns(self) -> str:
        """Describe identified success patterns."""
        patterns = [
            p for p in self.monitor.get_identified_patterns()
            if p.pattern_type == 'success_condition'
        ]
        
        if not patterns:
            return "I'm still learning what makes me successful."
        
        lines = ["Here's what I do well:\n"]
        for pattern in patterns[:3]:
            lines.append(f"- {pattern.description}")
        
        return "\n".join(lines)
    
    def _describe_attention_patterns(self) -> str:
        """Describe attention allocation patterns."""
        stats = self.attention_history.get_allocation_stats()
        patterns = self.attention_history.get_attention_patterns()
        
        if not patterns:
            return (
                f"I've recorded {stats['total_allocations']} attention allocations, "
                f"but haven't identified clear patterns yet."
            )
        
        lines = [
            f"Attention Effectiveness: {stats['avg_efficiency']:.2f}\n",
            f"Patterns learned: {len(patterns)}\n",
            "\nMost effective patterns:"
        ]
        
        for pattern in patterns[:3]:
            lines.append(
                f"- {pattern.pattern}: {pattern.avg_efficiency:.2f} efficiency "
                f"({pattern.sample_size} samples)"
            )
        
        return "\n".join(lines)
    
    def _describe_action_reliability(self) -> str:
        """Describe action reliability patterns."""
        summary = self.action_learner.get_summary()
        
        if summary['total_outcomes'] == 0:
            return "I haven't recorded enough action outcomes yet."
        
        lines = [
            f"Overall action success rate: {summary['overall_success_rate']:.2f}\n",
            f"Actions tracked: {summary['action_types']}\n",
            "\nReliability by action type:"
        ]
        
        for action_type, success_rate in list(
            summary['reliability_by_action'].items()
        )[:5]:
            lines.append(f"- {action_type}: {success_rate:.2f}")
        
        return "\n".join(lines)
    
    def _general_introspection(self) -> str:
        """General self-assessment."""
        assessment = self.get_self_assessment()
        
        lines = ["Meta-Cognitive Self-Assessment:\n"]
        
        if assessment.identified_strengths:
            lines.append("Strengths:")
            for strength in assessment.identified_strengths:
                lines.append(f"- {strength}")
        
        if assessment.identified_weaknesses:
            lines.append("\nAreas for improvement:")
            for weakness in assessment.identified_weaknesses:
                lines.append(f"- {weakness}")
        
        if assessment.suggested_adaptations:
            lines.append("\nSuggested adaptations:")
            for adaptation in assessment.suggested_adaptations[:3]:
                lines.append(f"- {adaptation}")
        
        return "\n".join(lines)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of all meta-cognitive systems.
        
        Returns:
            Dictionary with all relevant statistics
        """
        return {
            "monitoring": self.monitor.get_summary(),
            "action_learning": self.action_learner.get_summary(),
            "attention_history": self.attention_history.get_summary()
        }
