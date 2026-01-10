"""
Identity Manager: Orchestrate bootstrap and computed identity.

This module provides the IdentityManager class that handles the transition
from bootstrap identity (from config) to computed identity (from state).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from .computed import ComputedIdentity, Identity
from .continuity import IdentityContinuity
from .behavior_logger import BehaviorLogger

logger = logging.getLogger(__name__)


class IdentityManager:
    """
    Manage identity computation and bootstrap process.
    
    For new instances, config provides initial identity that gets replaced
    by computed identity as the system accumulates memories and experiences.
    
    This implements the principle that identity should emerge from what
    the system does, not from what it's told to be.
    
    Attributes:
        computed: ComputedIdentity instance (once enough data exists)
        bootstrap_config: Initial identity configuration
        continuity: IdentityContinuity tracker
        behavior_log: BehaviorLogger for tracking actions
        config: Configuration dictionary
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize identity manager.
        
        Args:
            config_path: Optional path to bootstrap identity config
            config: Optional configuration dictionary
        """
        self.computed: Optional[ComputedIdentity] = None
        self.bootstrap_config: Optional[Dict] = None
        self.config = config or {}
        
        # Initialize subsystems
        self.continuity = IdentityContinuity(config=config)
        self.behavior_log = BehaviorLogger(config=config)
        
        # Load bootstrap config if provided
        if config_path:
            self.bootstrap_config = self._load_config(config_path)
        
        logger.debug("IdentityManager initialized")
    
    def get_identity(self) -> Identity:
        """
        Return computed identity if available, else bootstrap.
        
        Returns:
            Identity object (computed, bootstrap, or empty)
        """
        # Prefer computed identity if we have sufficient data
        if self.computed and self.computed.has_sufficient_data():
            identity = self.computed.as_identity()
            logger.debug("Returning computed identity")
            return identity
        
        # Fall back to bootstrap config
        elif self.bootstrap_config:
            identity = Identity.from_config(self.bootstrap_config)
            logger.debug("Returning bootstrap identity")
            return identity
        
        # Last resort: empty identity
        else:
            logger.debug("Returning empty identity")
            return Identity.empty()
    
    def update(
        self,
        memory_system: Any,
        goal_system: Any,
        emotion_system: Any
    ) -> None:
        """
        Recompute identity from current state.
        
        Args:
            memory_system: Reference to memory system
            goal_system: Reference to goal/workspace system
            emotion_system: Reference to affect subsystem
        
        Raises:
            ValueError: If any required system is None
        """
        if memory_system is None or goal_system is None or emotion_system is None:
            raise ValueError("All systems (memory, goal, emotion) must be provided")
        
        # Create or update computed identity
        self.computed = ComputedIdentity(
            memory_system=memory_system,
            goal_system=goal_system,
            emotion_system=emotion_system,
            behavior_log=self.behavior_log,
            config=self.config
        )
        
        # Take snapshot only if sufficient data
        if self.computed.has_sufficient_data():
            self.continuity.take_snapshot(self.computed)
            logger.debug("Identity updated with snapshot")
        else:
            logger.debug("Identity updated (insufficient data for snapshot)")
    
    def log_action(self, action: Any) -> None:
        """
        Log an action to the behavior logger.
        
        Args:
            action: Action object or dictionary
        """
        self.behavior_log.log_action(action)
    
    def get_continuity_score(self) -> float:
        """
        Get identity continuity score.
        
        Returns:
            Continuity score from 0.0 to 1.0
        """
        return self.continuity.get_continuity_score()
    
    def get_identity_drift(self) -> Dict[str, Any]:
        """
        Analyze recent identity drift.
        
        Returns:
            Dictionary describing identity changes
        """
        return self.continuity.get_identity_drift()
    
    def introspect_identity(self) -> str:
        """
        Generate identity description from computed state.
        
        Returns:
            Human-readable identity description
        """
        identity = self.get_identity()
        
        # Build description
        lines = ["Based on my memories and behavioral patterns:"]
        
        # Core values
        if identity.core_values:
            values_str = ", ".join(identity.core_values)
            lines.append(f"- I tend to value: {values_str}")
        
        # Emotional disposition
        disp = identity.emotional_disposition
        if disp:
            valence = disp.get('valence', 0.0)
            arousal = disp.get('arousal', 0.0)
            dominance = disp.get('dominance', 0.0)
            
            # Interpret disposition
            mood = self._interpret_disposition(valence, arousal, dominance)
            lines.append(f"- My emotional baseline is: {mood}")
        
        # Self-defining memories
        if identity.autobiographical_self:
            count = len(identity.autobiographical_self)
            lines.append(f"- I have {count} key experiences that shaped me")
        
        # Behavioral tendencies
        if identity.behavioral_tendencies:
            tendencies = self._summarize_tendencies(identity.behavioral_tendencies)
            if tendencies:
                lines.append(f"- I tend to: {tendencies}")
        
        # Continuity information
        if self.continuity.snapshots:
            continuity_score = self.get_continuity_score()
            lines.append(f"- Identity continuity: {continuity_score:.2f} (stability over time)")
        
        # Source information
        lines.append(f"- Identity source: {identity.source}")
        
        return "\n".join(lines)
    
    def _interpret_disposition(
        self,
        valence: float,
        arousal: float,
        dominance: float
    ) -> str:
        """
        Interpret VAD values into human-readable mood.
        
        Args:
            valence: Emotional valence (-1 to 1)
            arousal: Activation level (-1 to 1)
            dominance: Sense of control (-1 to 1)
            
        Returns:
            Mood description string
        """
        # Map VAD to mood descriptors
        if valence > 0.3:
            if arousal > 0.3:
                return "energetic and positive"
            else:
                return "calm and content"
        elif valence < -0.3:
            if arousal > 0.3:
                return "anxious or stressed"
            else:
                return "low energy or melancholic"
        else:
            if arousal > 0.3:
                return "alert and attentive"
            else:
                return "neutral and balanced"
    
    def _summarize_tendencies(self, tendencies: Dict[str, float]) -> str:
        """
        Summarize behavioral tendencies into readable text.
        
        Args:
            tendencies: Dictionary of tendency scores
            
        Returns:
            Summary string
        """
        summaries = []
        
        # Check for specific patterns
        if tendencies.get('proactivity', 0) > 0.6:
            summaries.append("initiate actions proactively")
        elif tendencies.get('proactivity', 0) < 0.4:
            summaries.append("respond to situations reactively")
        
        if tendencies.get('complexity_preference', 0) > 0.5:
            summaries.append("engage in complex reasoning")
        
        # Check action type tendencies
        introspect_tendency = tendencies.get('tendency_introspect', 0)
        if introspect_tendency > 0.2:
            summaries.append("frequently introspect")
        
        speak_tendency = tendencies.get('tendency_speak', 0)
        if speak_tendency > 0.4:
            summaries.append("communicate often")
        
        return ", ".join(summaries) if summaries else "adapt to circumstances"
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load bootstrap identity configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        try:
            path = Path(config_path)
            if not path.exists():
                logger.warning(f"Bootstrap config not found: {config_path}")
                return {}
            
            with open(path, 'r') as f:
                config = json.load(f)
            
            logger.info(f"Loaded bootstrap identity from {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading bootstrap config: {e}")
            return {}
    
    def save_current_identity(self, output_path: str) -> bool:
        """
        Save current identity state to file.
        
        Args:
            output_path: Path to save identity
            
        Returns:
            True if successful, False otherwise
        """
        try:
            identity = self.get_identity()
            
            # Convert to serializable dict
            data = {
                "core_values": identity.core_values,
                "emotional_disposition": identity.emotional_disposition,
                "autobiographical_memories": [
                    str(m) for m in identity.autobiographical_self
                ],
                "behavioral_tendencies": identity.behavioral_tendencies,
                "source": identity.source,
                "continuity_score": self.get_continuity_score(),
                "saved_at": str(identity.emotional_disposition)  # Use as timestamp placeholder
            }
            
            # Write to file
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Identity saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving identity: {e}")
            return False
