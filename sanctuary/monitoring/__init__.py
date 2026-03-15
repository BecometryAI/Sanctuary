"""Visualization & monitoring for Phase 6.4.

Backend data providers for monitoring the cognitive system:
- Dashboard data provider: Real-time workspace state (goals, percepts, emotions, metrics)
- Attention heatmap tracker: What content receives attention over time
- Consciousness trace recorder: Full state replay for cognitive cycles
- Communication decision logger: Visualize speak/silence decisions and reasons
"""

from sanctuary.monitoring.dashboard import DashboardDataProvider
from sanctuary.monitoring.attention_heatmap import AttentionHeatmapTracker
from sanctuary.monitoring.consciousness_trace import ConsciousnessTraceRecorder
from sanctuary.monitoring.communication_log import CommunicationDecisionLogger

__all__ = [
    "DashboardDataProvider",
    "AttentionHeatmapTracker",
    "ConsciousnessTraceRecorder",
    "CommunicationDecisionLogger",
]
