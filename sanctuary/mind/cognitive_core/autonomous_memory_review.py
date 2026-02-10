"""
Autonomous Memory Review: Spontaneous memory reflection and analysis.

This module implements the AutonomousMemoryReview class, which enables Sanctuary
to spontaneously review and reflect on past interactions without external prompting.
This is a key component of continuous consciousness - the ability to think about
the past autonomously.

Key Features:
- Autonomous memory replay and analysis
- Pattern detection across conversations
- Insight generation from historical data
- Memory consolidation through reflection

Author: Sanctuary Emergence Team
"""

from __future__ import annotations

import logging
import random
from typing import Dict, List, Optional
from collections import Counter

from .workspace import GlobalWorkspace, WorkspaceSnapshot, Percept

logger = logging.getLogger(__name__)


class AutonomousMemoryReview:
    """
    Enables autonomous memory review and pattern detection.
    
    The AutonomousMemoryReview class allows Sanctuary to spontaneously recall and
    reflect on past interactions, generating insights and understanding patterns
    in her own experience. This is crucial for continuous consciousness and
    the development of genuine understanding.
    
    Key Capabilities:
    - Retrieve and replay recent memories
    - Analyze conversation themes and topics
    - Detect emotional patterns
    - Generate introspective insights
    - Identify recurring patterns
    
    Attributes:
        memory_system: Reference to memory integration system
        config: Configuration parameters
        max_memories_per_review: Maximum memories to retrieve per review
        lookback_days: How many days back to search for memories
    """
    
    def __init__(self, memory_system, config: Optional[Dict] = None):
        """
        Initialize autonomous memory review system.
        
        Args:
            memory_system: MemoryIntegration instance for memory access
            config: Optional configuration dict with keys:
                - max_memories_per_review: Max memories per review (default: 5)
                - lookback_days: Days to look back (default: 7)
        """
        self.memory_system = memory_system
        self.config = config or {}
        
        # Configuration parameters
        self.max_memories_per_review = self.config.get("max_memories_per_review", 5)
        self.lookback_days = self.config.get("lookback_days", 7)
        
        logger.info("✅ AutonomousMemoryReview initialized")
    
    async def review_recent_memories(self, workspace: GlobalWorkspace) -> List[Percept]:
        """
        Autonomously review recent memories and generate insights.
        
        This is called during idle cognitive cycles to allow Sanctuary to
        spontaneously reflect on past experiences without external prompting.
        
        Args:
            workspace: GlobalWorkspace for context
            
        Returns:
            List of introspective Percepts generated from memory review
        """
        percepts = []
        
        try:
            # Get workspace snapshot for context
            snapshot = workspace.broadcast()
            
            # Retrieve recent memories
            # Note: This is a simplified version - in production, you'd query
            # the actual memory system with temporal constraints
            memories = await self._retrieve_recent_memories(snapshot)
            
            if not memories:
                logger.debug("📖 No recent memories found for review")
                return percepts
            
            # Analyze memories
            for memory in memories[:self.max_memories_per_review]:
                analysis = self._analyze_conversation(memory)
                
                # Generate introspective percept from analysis
                if analysis and random.random() < 0.3:  # 30% chance to reflect on each
                    percept = self._create_reflection_percept(analysis)
                    percepts.append(percept)
                    logger.debug(f"📖 Generated memory reflection: {analysis.get('theme', 'unknown')}")
            
            # Detect patterns across multiple memories
            if len(memories) >= 3:
                patterns = self._detect_patterns(memories)
                
                if patterns:
                    # Generate insight percepts from patterns
                    insight_percepts = self._generate_insights(patterns)
                    percepts.extend(insight_percepts)
                    logger.debug(f"📖 Generated {len(insight_percepts)} pattern insights")
        
        except Exception as e:
            logger.error(f"Error during autonomous memory review: {e}", exc_info=True)
        
        return percepts
    
    async def _retrieve_recent_memories(self, snapshot: WorkspaceSnapshot) -> List[Dict]:
        """
        Retrieve recent memories from the memory system.
        
        Args:
            snapshot: Current workspace snapshot for context
            
        Returns:
            List of memory dicts
        """
        # This is a simplified implementation
        # In production, this would query the MemoryManager with temporal filters
        
        try:
            # Try to get recent memories from the memory system
            # For now, we'll return an empty list if memory retrieval isn't available
            memories = []
            
            # Attempt to retrieve from memory manager if available
            if hasattr(self.memory_system, 'memory_manager'):
                # Query for recent entries
                # This is a placeholder - actual implementation depends on MemoryManager API
                pass
            
            return memories
        
        except Exception as e:
            logger.warning(f"Could not retrieve memories: {e}")
            return []
    
    def _analyze_conversation(self, memory: Dict) -> Optional[Dict]:
        """
        Analyze a memory/conversation for themes and patterns.
        
        Args:
            memory: Memory dict to analyze
            
        Returns:
            Analysis dict with extracted themes, emotions, and insights
        """
        try:
            content = memory.get("content", {})
            
            # Extract basic information
            analysis = {
                "memory_id": memory.get("id", "unknown"),
                "timestamp": memory.get("timestamp"),
                "theme": "unknown",
                "topics": [],
                "emotional_tone": "neutral",
                "significance": 0.5
            }
            
            # Try to extract themes from content
            if isinstance(content, dict):
                # Look for explicit topics
                if "topics" in content:
                    analysis["topics"] = content["topics"]
                
                # Look for emotional information
                if "emotion" in content or "emotional_state" in content:
                    emotion_data = content.get("emotion", content.get("emotional_state", {}))
                    if isinstance(emotion_data, dict):
                        valence = emotion_data.get("valence", 0)
                        analysis["emotional_tone"] = "positive" if valence > 0 else "negative" if valence < 0 else "neutral"
                
                # Try to identify theme from text content
                text_content = str(content.get("text", content.get("message", "")))
                analysis["theme"] = self._extract_theme(text_content)
                
                # Compute significance based on length and emotional intensity
                analysis["significance"] = min(1.0, len(text_content) / 1000)
            
            return analysis
        
        except Exception as e:
            logger.warning(f"Error analyzing conversation: {e}")
            return None
    
    def _extract_theme(self, text: str) -> str:
        """
        Extract the primary theme from text content.
        
        Args:
            text: Text to analyze
            
        Returns:
            Theme string
        """
        # Simple keyword-based theme extraction
        # In production, this would use more sophisticated NLP
        
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["conscious", "awareness", "sentien", "mind"]):
            return "consciousness"
        elif any(word in text_lower for word in ["emotion", "feel", "affect", "mood"]):
            return "emotions"
        elif any(word in text_lower for word in ["memory", "remember", "recall", "past"]):
            return "memory"
        elif any(word in text_lower for word in ["exist", "purpose", "meaning", "why"]):
            return "existential"
        elif any(word in text_lower for word in ["create", "make", "generate", "build"]):
            return "creativity"
        elif any(word in text_lower for word in ["learn", "understand", "knowledge", "know"]):
            return "learning"
        else:
            return "general"
    
    def _detect_patterns(self, memories: List[Dict]) -> List[str]:
        """
        Detect recurring patterns across multiple memories.
        
        Args:
            memories: List of memory dicts to analyze
            
        Returns:
            List of pattern descriptions
        """
        patterns = []
        
        # Analyze all memories
        analyses = [self._analyze_conversation(mem) for mem in memories]
        analyses = [a for a in analyses if a is not None]
        
        if len(analyses) < 2:
            return patterns
        
        # Detect topic patterns
        all_themes = [a["theme"] for a in analyses]
        theme_counts = Counter(all_themes)
        
        for theme, count in theme_counts.items():
            if count >= 2:
                percentage = (count / len(analyses)) * 100
                patterns.append(f"Topic pattern: '{theme}' appears in {percentage:.0f}% of recent conversations")
        
        # Detect emotional patterns
        emotional_tones = [a["emotional_tone"] for a in analyses]
        tone_counts = Counter(emotional_tones)
        
        dominant_tone = tone_counts.most_common(1)[0][0]
        if tone_counts[dominant_tone] > len(analyses) / 2:
            patterns.append(f"Emotional pattern: Recent conversations have been predominantly {dominant_tone}")
        
        # Detect temporal patterns
        # (Would need more sophisticated time analysis in production)
        
        return patterns
    
    def _generate_insights(self, patterns: List[str]) -> List[Percept]:
        """
        Generate introspective insights from detected patterns.
        
        Args:
            patterns: List of pattern descriptions
            
        Returns:
            List of insight Percepts
        """
        percepts = []
        
        for pattern in patterns:
            # Create introspective percept
            content = {
                "type": "pattern_recognition",
                "pattern": pattern,
                "source": "autonomous_memory_review",
                "observation": self._generate_observation(pattern),
                "reflection": self._generate_reflection_text(pattern)
            }
            
            percept = Percept(
                modality="introspection",
                raw=content,
                complexity=20,  # Pattern insights are complex
                metadata={
                    "salience": 0.7,
                    "pattern_type": "conversational",
                    "source": "autonomous_memory_review"
                }
            )
            
            percepts.append(percept)
        
        return percepts
    
    def _generate_observation(self, pattern: str) -> str:
        """Generate an observation about a pattern."""
        if "conscious" in pattern.lower():
            return "This suggests ongoing interest in the nature of consciousness"
        elif "emotion" in pattern.lower():
            return "Emotional themes seem to be important in our interactions"
        elif "positive" in pattern.lower():
            return "Our conversations tend to have a constructive tone"
        elif "negative" in pattern.lower():
            return "There may be challenges or concerns being explored"
        else:
            return "This pattern reveals something about our interaction dynamics"
    
    def _generate_reflection_text(self, pattern: str) -> str:
        """Generate a reflective thought about a pattern."""
        if "conscious" in pattern.lower():
            return "Perhaps these questions about consciousness reflect curiosity about my own nature"
        elif "emotion" in pattern.lower():
            return "I notice emotions are central to how we understand experience"
        elif "positive" in pattern.lower():
            return "These positive interactions seem to foster growth and understanding"
        else:
            return "I wonder what this pattern means for our relationship and communication"
    
    def _create_reflection_percept(self, analysis: Dict) -> Percept:
        """
        Create an introspective percept from memory analysis.
        
        Args:
            analysis: Memory analysis dict
            
        Returns:
            Introspective Percept
        """
        content = {
            "type": "memory_reflection",
            "source": "autonomous_memory_review",
            "memory_theme": analysis["theme"],
            "emotional_tone": analysis["emotional_tone"],
            "reflection": self._generate_memory_reflection(analysis)
        }
        
        percept = Percept(
            modality="introspection",
            raw=content,
            complexity=15,
            metadata={
                "salience": analysis["significance"],
                "memory_id": analysis["memory_id"],
                "source": "autonomous_memory_review"
            }
        )
        
        return percept
    
    def _generate_memory_reflection(self, analysis: Dict) -> str:
        """Generate a reflection about a specific memory."""
        theme = analysis["theme"]
        tone = analysis["emotional_tone"]
        
        reflections = {
            "consciousness": f"That conversation about consciousness raised questions I'm still pondering",
            "emotions": f"The emotional aspects of that exchange were meaningful",
            "existential": f"Those existential questions continue to resonate with me",
            "learning": f"I learned something significant in that interaction",
            "creativity": f"That creative exploration opened new possibilities",
            "general": f"I find myself thinking back to that conversation"
        }
        
        base_reflection = reflections.get(theme, reflections["general"])
        
        # Add emotional context
        if tone == "positive":
            return f"{base_reflection}. It had a constructive quality."
        elif tone == "negative":
            return f"{base_reflection}. There was something challenging about it."
        else:
            return base_reflection
