"""
Attention Controller: Selective attention mechanism.

This module implements the AttentionController class, which decides what information
gains access to the limited-capacity GlobalWorkspace. It implements selective attention
based on goal relevance, novelty, emotional salience, and resource constraints.

The attention mechanism is crucial for:
- Creating the selective nature of consciousness
- Managing cognitive resource allocation
- Prioritizing information based on multiple factors
- Implementing both top-down (goal-driven) and bottom-up (stimulus-driven) attention
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum


class AttentionMode(Enum):
    """
    Different modes of attention allocation.

    FOCUSED: Narrow, goal-driven attention on specific targets
    DIFFUSE: Broad, exploratory attention across multiple inputs
    VIGILANT: Heightened alertness for threat or novelty detection
    RELAXED: Low-intensity monitoring during low-demand periods
    """
    FOCUSED = "focused"
    DIFFUSE = "diffuse"
    VIGILANT = "vigilant"
    RELAXED = "relaxed"


@dataclass
class AttentionScore:
    """
    Scores for different factors contributing to attention.

    Attributes:
        goal_relevance: How relevant to current goals (0.0-1.0)
        novelty: How novel or unexpected (0.0-1.0)
        emotional_salience: Emotional importance (0.0-1.0)
        urgency: Time-sensitivity of the information (0.0-1.0)
        total: Weighted sum of all factors
    """
    goal_relevance: float
    novelty: float
    emotional_salience: float
    urgency: float
    total: float


class AttentionController:
    """
    Decides what information enters the GlobalWorkspace based on salience.

    The AttentionController implements selective attention, acting as a gatekeeper
    for the limited-capacity GlobalWorkspace. It evaluates incoming information
    from multiple sources (percepts, memories, emotions, internal states) and
    assigns attention scores based on multiple factors.

    Key Responsibilities:
    - Evaluate attention worthiness of candidate information
    - Implement both top-down (goal-driven) and bottom-up (stimulus-driven) attention
    - Manage attention resources and prevent cognitive overload
    - Track attention history to detect novelty and habituation
    - Dynamically adjust attention mode based on context

    Integration Points:
    - GlobalWorkspace: Selects what content enters the workspace
    - PerceptionSubsystem: Evaluates salience of incoming percepts
    - AffectSubsystem: Uses emotional state to modulate attention
    - CognitiveCore: Receives attention mode adjustments based on system state
    - SelfMonitor: Can redirect attention to internal states when needed

    Attention Mechanisms:
    1. Goal-Driven (Top-Down): Prioritizes information relevant to active goals
    2. Stimulus-Driven (Bottom-Up): Responds to novel, intense, or unexpected stimuli
    3. Emotional Salience: Amplifies attention for emotionally significant content
    4. Habituation: Reduces attention to repeated, non-threatening stimuli
    5. Resource Management: Prevents overload by limiting concurrent attention targets

    The controller uses a weighted scoring system that can be dynamically adjusted
    based on current context, emotional state, and cognitive load. Different attention
    modes shift these weights to support different cognitive strategies.

    Attributes:
        mode: Current attention allocation strategy
        goal_weight: Weight for goal-relevance in scoring (0.0-1.0)
        novelty_weight: Weight for novelty in scoring (0.0-1.0)
        emotion_weight: Weight for emotional salience in scoring (0.0-1.0)
        urgency_weight: Weight for urgency in scoring (0.0-1.0)
    """

    def __init__(
        self,
        initial_mode: AttentionMode = AttentionMode.FOCUSED,
        goal_weight: float = 0.4,
        novelty_weight: float = 0.3,
        emotion_weight: float = 0.2,
        urgency_weight: float = 0.1,
    ) -> None:
        """
        Initialize the attention controller.

        Args:
            initial_mode: Starting attention mode (focused, diffuse, vigilant, relaxed)
            goal_weight: Importance of goal-relevance in attention (0.0-1.0)
            novelty_weight: Importance of novelty in attention (0.0-1.0)
            emotion_weight: Importance of emotional salience in attention (0.0-1.0)
            urgency_weight: Importance of urgency in attention (0.0-1.0)

        Note: Weights should sum to approximately 1.0 for balanced scoring.
        """
        pass
