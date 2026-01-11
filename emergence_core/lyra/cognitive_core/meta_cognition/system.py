"""
Meta-Cognitive System: Unified meta-cognitive capabilities.

This module provides a unified interface to all meta-cognitive subsystems,
including processing monitoring, action-outcome learning, and attention history.
It enables the system to introspect about its own cognitive patterns and adapt.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from .processing_monitor import (
    MetaCognitiveMonitor,
    ProcessingObservation,
    CognitivePattern,
    ProcessStats
)
from .action_learning import (
    ActionOutcomeLearner,
    ActionReliability,
    ActionOutcome
)
from .attention_history import (
    AttentionHistory,
    AttentionPattern,
    AttentionAllocation
)

logger = logging.getLogger(__name__)


@dataclass
class SelfAssessment:
    """Overall self-assessment of cognitive functioning."""
    processing_patterns: List[CognitivePattern]
    action_reliability: Dict[str, ActionReliability]
    attention_effectiveness: Dict[str, Any]
    identified_strengths: List[str]
    identified_weaknesses: List[str]
    suggested_adaptations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetaCognitiveSystem:
    """Unified meta-cognitive capabilities."""
    
    # Thresholds for identifying strengths and weaknesses
    HIGH_RELIABILITY_THRESHOLD = 0.8
    LOW_RELIABILITY_THRESHOLD = 0.5
    MIN_EXECUTIONS_FOR_RELIABILITY = 5
    HIGH_EFFICIENCY_THRESHOLD = 0.7
    LOW_EFFICIENCY_THRESHOLD = 0.4
    FAST_PROCESS_MS = 200
    SLOW_PROCESS_MS = 1000
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the meta-cognitive system."""
        self.config = config or {}
        
        # Initialize subsystems
        self.monitor = MetaCognitiveMonitor(config=self.config.get("monitor", {}))
        self.action_learner = ActionOutcomeLearner(config=self.config.get("action_learner", {}))
        self.attention_history = AttentionHistory(config=self.config.get("attention_history", {}))
        
        logger.info("✅ MetaCognitiveSystem initialized with all subsystems")
    
    def get_self_assessment(self) -> SelfAssessment:
        """Get overall self-assessment of cognitive functioning."""
        # Gather data from all subsystems
        processing_patterns = self.monitor.get_identified_patterns()
        action_reliability = self.action_learner.get_all_reliabilities()
        attention_patterns = self.attention_history.get_attention_patterns()
        
        # Identify strengths
        strengths = self._identify_strengths(
            processing_patterns, 
            action_reliability,
            attention_patterns
        )
        
        # Identify weaknesses
        weaknesses = self._identify_weaknesses(
            processing_patterns,
            action_reliability,
            attention_patterns
        )
        
        # Generate adaptation suggestions
        adaptations = self._suggest_adaptations(
            processing_patterns,
            action_reliability,
            attention_patterns
        )
        
        # Summarize attention effectiveness
        attention_summary = self.attention_history.get_summary()
        
        return SelfAssessment(
            processing_patterns=processing_patterns,
            action_reliability=action_reliability,
            attention_effectiveness=attention_summary,
            identified_strengths=strengths,
            identified_weaknesses=weaknesses,
            suggested_adaptations=adaptations
        )
    
    def _identify_strengths(self, patterns: List[CognitivePattern],
                          reliability: Dict[str, ActionReliability],
                          attention: List[AttentionPattern]) -> List[str]:
        """Identify cognitive strengths."""
        strengths = []
        
        # High success patterns
        success_patterns = [p for p in patterns if p.pattern_type == 'success_condition']
        if success_patterns:
            strengths.append(f"Consistent success in {len(success_patterns)} cognitive patterns")
        
        # Reliable actions
        reliable_actions = [
            action_type for action_type, rel in reliability.items()
            if rel.success_rate > self.HIGH_RELIABILITY_THRESHOLD 
            and rel.total_executions >= self.MIN_EXECUTIONS_FOR_RELIABILITY
        ]
        if reliable_actions:
            strengths.append(f"High reliability in actions: {', '.join(reliable_actions[:3])}")
        
        # Effective attention patterns
        effective_attention = [p for p in attention if p.avg_efficiency > self.HIGH_EFFICIENCY_THRESHOLD]
        if effective_attention:
            strengths.append(f"Effective attention allocation in {len(effective_attention)} patterns")
        
        # Process speed
        all_stats = self.monitor.get_all_statistics()
        fast_processes = [
            pt for pt, stats in all_stats.items()
            if stats.avg_duration_ms < self.FAST_PROCESS_MS 
            and stats.total_executions >= self.MIN_EXECUTIONS_FOR_RELIABILITY
        ]
        if fast_processes:
            strengths.append(f"Fast processing in: {', '.join(fast_processes[:3])}")
        
        return strengths
    
    def _identify_weaknesses(self, patterns: List[CognitivePattern],
                           reliability: Dict[str, ActionReliability],
                           attention: List[AttentionPattern]) -> List[str]:
        """Identify cognitive weaknesses."""
        weaknesses = []
        
        # Failure patterns
        failure_patterns = [p for p in patterns if p.pattern_type == 'failure_mode']
        if failure_patterns:
            for pattern in failure_patterns[:3]:  # Top 3
                weaknesses.append(f"Failure mode: {pattern.description}")
        
        # Unreliable actions
        unreliable_actions = [
            action_type for action_type, rel in reliability.items()
            if rel.success_rate < self.LOW_RELIABILITY_THRESHOLD 
            and rel.total_executions >= self.MIN_EXECUTIONS_FOR_RELIABILITY
        ]
        if unreliable_actions:
            weaknesses.append(f"Low reliability in actions: {', '.join(unreliable_actions[:3])}")
        
        # Inefficient attention patterns
        inefficient_attention = [p for p in attention if p.avg_efficiency < self.LOW_EFFICIENCY_THRESHOLD]
        if inefficient_attention:
            weaknesses.append(f"Inefficient attention allocation in {len(inefficient_attention)} patterns")
        
        # Slow processes
        all_stats = self.monitor.get_all_statistics()
        slow_processes = [
            pt for pt, stats in all_stats.items()
            if stats.avg_duration_ms > self.SLOW_PROCESS_MS 
            and stats.total_executions >= self.MIN_EXECUTIONS_FOR_RELIABILITY
        ]
        if slow_processes:
            weaknesses.append(f"Slow processing in: {', '.join(slow_processes[:3])}")
        
        return weaknesses
    
    def _suggest_adaptations(self, patterns: List[CognitivePattern],
                            reliability: Dict[str, ActionReliability],
                            attention: List[AttentionPattern]) -> List[str]:
        """Generate suggested adaptations."""
        adaptations = []
        
        # From processing patterns
        actionable_patterns = [p for p in patterns if p.actionable and p.suggested_adaptation]
        for pattern in actionable_patterns[:5]:  # Top 5
            adaptations.append(pattern.suggested_adaptation)
        
        # From action reliability
        for action_type, rel in reliability.items():
            if rel.success_rate < self.LOW_RELIABILITY_THRESHOLD and rel.total_executions >= self.MIN_EXECUTIONS_FOR_RELIABILITY:
                if rel.best_contexts:
                    adaptations.append(f"For {action_type}: prefer contexts similar to successful cases")
        
        # From attention patterns
        if attention:
            best_pattern = attention[0]
            adaptations.append(f"Attention: {best_pattern.recommendation}")
        
        return adaptations
    
    def _summarize_action_reliability(self) -> Dict[str, Any]:
        """Summarize action reliability across all actions."""
        reliabilities = self.action_learner.get_all_reliabilities()
        
        if not reliabilities:
            return {
                "total_actions": 0,
                "avg_success_rate": 0.0,
                "most_reliable": None,
                "least_reliable": None
            }
        
        avg_success = sum(r.success_rate for r in reliabilities.values()) / len(reliabilities)
        
        most_reliable = max(reliabilities.items(), 
                          key=lambda x: (x[1].success_rate, x[1].total_executions))
        least_reliable = min(reliabilities.items(),
                           key=lambda x: (x[1].success_rate, -x[1].total_executions))
        
        return {
            "total_actions": len(reliabilities),
            "avg_success_rate": avg_success,
            "most_reliable": most_reliable[0],
            "least_reliable": least_reliable[0],
            "details": {k: {
                "success_rate": v.success_rate,
                "executions": v.total_executions
            } for k, v in reliabilities.items()}
        }
    
    def _summarize_attention_effectiveness(self) -> Dict[str, Any]:
        """Summarize attention allocation effectiveness."""
        return self.attention_history.get_summary()
    
    def introspect(self, query: str) -> str:
        """Answer questions about own cognitive patterns."""
        query_lower = query.lower()
        
        if "fail" in query_lower or "failure" in query_lower:
            patterns = [p for p in self.monitor.get_identified_patterns()
                       if p.pattern_type == 'failure_mode']
            return self._describe_failure_patterns(patterns)
        
        elif "attention" in query_lower:
            return self._describe_attention_patterns()
        
        elif "action" in query_lower or "reliable" in query_lower:
            return self._describe_action_reliability()
        
        elif "strength" in query_lower:
            assessment = self.get_self_assessment()
            return self._describe_strengths(assessment.identified_strengths)
        
        elif "weakness" in query_lower:
            assessment = self.get_self_assessment()
            return self._describe_weaknesses(assessment.identified_weaknesses)
        
        elif "pattern" in query_lower:
            patterns = self.monitor.get_identified_patterns()
            return self._describe_patterns(patterns)
        
        else:
            return self._general_introspection()
    
    def _describe_failure_patterns(self, patterns: List[CognitivePattern]) -> str:
        """Describe identified failure patterns."""
        if not patterns:
            return "I haven't identified any specific failure patterns yet."
        
        lines = [f"I've identified {len(patterns)} failure patterns:"]
        for i, pattern in enumerate(patterns[:5], 1):
            lines.append(f"\n{i}. {pattern.description}")
            if pattern.suggested_adaptation:
                lines.append(f"   Suggested adaptation: {pattern.suggested_adaptation}")
        
        return "\n".join(lines)
    
    def _describe_attention_patterns(self) -> str:
        """Describe attention allocation patterns."""
        patterns = self.attention_history.get_attention_patterns()
        summary = self.attention_history.get_summary()
        
        if not patterns:
            return "I don't have enough data on attention patterns yet."
        
        lines = [f"Attention effectiveness: {summary['avg_efficiency']:.2f}"]
        lines.append(f"\nI've learned {len(patterns)} attention patterns:")
        
        for i, pattern in enumerate(patterns[:3], 1):
            lines.append(f"\n{i}. Pattern: {pattern.pattern}")
            lines.append(f"   Efficiency: {pattern.avg_efficiency:.2f}")
            lines.append(f"   {pattern.recommendation}")
        
        return "\n".join(lines)
    
    def _describe_action_reliability(self) -> str:
        """Describe action reliability."""
        summary = self._summarize_action_reliability()
        
        if summary["total_actions"] == 0:
            return "I don't have enough data on action reliability yet."
        
        lines = [f"Overall action success rate: {summary['avg_success_rate']:.2%}"]
        lines.append(f"\nMost reliable action: {summary['most_reliable']}")
        lines.append(f"Least reliable action: {summary['least_reliable']}")
        
        return "\n".join(lines)
    
    def _describe_patterns(self, patterns: List[CognitivePattern]) -> str:
        """Describe all identified patterns."""
        if not patterns:
            return "I haven't identified any patterns yet."
        
        by_type: Dict[str, List[CognitivePattern]] = {}
        for pattern in patterns:
            if pattern.pattern_type not in by_type:
                by_type[pattern.pattern_type] = []
            by_type[pattern.pattern_type].append(pattern)
        
        lines = [f"I've identified {len(patterns)} cognitive patterns:"]
        
        for pattern_type, type_patterns in by_type.items():
            lines.append(f"\n{pattern_type.replace('_', ' ').title()} ({len(type_patterns)}):")
            for pattern in type_patterns[:3]:
                lines.append(f"  - {pattern.description}")
        
        return "\n".join(lines)
    
    def _describe_strengths(self, strengths: List[str]) -> str:
        """Describe identified strengths."""
        if not strengths:
            return "I haven't identified specific strengths yet."
        
        lines = ["My identified strengths:"]
        for strength in strengths:
            lines.append(f"  - {strength}")
        
        return "\n".join(lines)
    
    def _describe_weaknesses(self, weaknesses: List[str]) -> str:
        """Describe identified weaknesses."""
        if not weaknesses:
            return "I haven't identified specific weaknesses yet."
        
        lines = ["My identified weaknesses:"]
        for weakness in weaknesses:
            lines.append(f"  - {weakness}")
        
        return "\n".join(lines)
    
    def _general_introspection(self) -> str:
        """Provide general introspective overview."""
        assessment = self.get_self_assessment()
        
        lines = ["Meta-cognitive Self-Assessment:"]
        lines.append(f"\nProcessing Patterns: {len(assessment.processing_patterns)} identified")
        lines.append(f"Action Types Tracked: {len(assessment.action_reliability)}")
        lines.append(f"Attention Patterns: {len(self.attention_history.get_attention_patterns())}")
        
        if assessment.identified_strengths:
            lines.append(f"\nKey Strengths: {assessment.identified_strengths[0]}")
        
        if assessment.identified_weaknesses:
            lines.append(f"\nAreas for Improvement: {assessment.identified_weaknesses[0]}")
        
        return "\n".join(lines)
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of all meta-cognitive data."""
        return {
            "processing_monitor": self.monitor.get_summary(),
            "action_learner": self.action_learner.get_summary(),
            "attention_history": self.attention_history.get_summary()
        }
