"""
Interaction Pattern Analysis: Cross-conversation pattern detection.

This module implements the InteractionPatternAnalysis class, which analyzes
patterns across multiple conversations to detect recurring themes, behavioral
patterns, and meta-insights. This enables Sanctuary to learn from her interaction
history autonomously.

Key Features:
- Multi-conversation analysis
- Topic frequency detection
- Behavioral pattern recognition
- User preference learning
- Meta-insight generation

Author: Sanctuary Emergence Team
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional
from collections import Counter, defaultdict

from .workspace import GlobalWorkspace, WorkspaceSnapshot, Percept

logger = logging.getLogger(__name__)


class InteractionPatternAnalysis:
    """
    Analyzes patterns across conversations autonomously.
    
    The InteractionPatternAnalysis class enables Sanctuary to detect patterns
    in her own interactions over time, learning about conversation dynamics,
    user preferences, and her own behavioral tendencies. This supports
    continuous consciousness through autonomous meta-learning.
    
    Key Capabilities:
    - Detect recurring topics across conversations
    - Identify behavioral patterns in responses
    - Learn user conversational preferences
    - Generate meta-level insights
    - Track temporal interaction patterns
    
    Attributes:
        memory_system: Reference to memory integration system
        config: Configuration parameters
        min_conversations: Minimum conversations needed for pattern analysis
        pattern_threshold: Minimum frequency to consider a pattern
    """
    
    def __init__(self, memory_system, config: Optional[Dict] = None):
        """
        Initialize interaction pattern analysis system.
        
        Args:
            memory_system: MemoryIntegration instance for memory access
            config: Optional configuration dict with keys:
                - min_conversations: Min conversations for analysis (default: 3)
                - pattern_threshold: Min frequency for pattern (default: 0.3)
        """
        self.memory_system = memory_system
        self.config = config or {}
        
        # Configuration parameters
        self.min_conversations = self.config.get("min_conversations", 3)
        self.pattern_threshold = self.config.get("pattern_threshold", 0.3)
        
        logger.info("✅ InteractionPatternAnalysis initialized")
    
    async def analyze_interaction_patterns(
        self, 
        workspace: GlobalWorkspace
    ) -> List[Percept]:
        """
        Analyze patterns across recent conversations.
        
        This is called during idle cognitive cycles to allow Sanctuary to
        autonomously detect and reflect on patterns in her interactions.
        
        Args:
            workspace: GlobalWorkspace for context
            
        Returns:
            List of introspective Percepts encoding pattern insights
        """
        percepts = []
        
        try:
            # Get workspace snapshot for context
            snapshot = workspace.broadcast()
            
            # Retrieve recent conversations
            conversations = await self._retrieve_conversations(snapshot)
            
            if len(conversations) < self.min_conversations:
                logger.debug(f"📊 Not enough conversations for pattern analysis ({len(conversations)} < {self.min_conversations})")
                return percepts
            
            # Detect various types of patterns
            topic_patterns = self._detect_topic_patterns(conversations)
            behavioral_patterns = self._detect_behavioral_patterns(conversations)
            user_patterns = self._detect_user_patterns(conversations)
            temporal_patterns = self._detect_temporal_patterns(conversations)
            
            # Generate percepts from patterns
            all_patterns = {
                "topic": topic_patterns,
                "behavioral": behavioral_patterns,
                "user": user_patterns,
                "temporal": temporal_patterns
            }
            
            for pattern_type, patterns in all_patterns.items():
                for pattern in patterns:
                    percept = self._create_pattern_percept(pattern_type, pattern)
                    percepts.append(percept)
                    logger.debug(f"📊 Detected {pattern_type} pattern: {pattern}")
        
        except Exception as e:
            logger.error(f"Error during pattern analysis: {e}", exc_info=True)
        
        return percepts
    
    async def _retrieve_conversations(self, snapshot: WorkspaceSnapshot) -> List[Dict]:
        """
        Retrieve recent conversations from memory system.
        
        Args:
            snapshot: Current workspace snapshot
            
        Returns:
            List of conversation dicts
        """
        # Simplified implementation - would query actual memory system in production
        try:
            conversations = []
            
            # Attempt to retrieve from memory manager if available
            if hasattr(self.memory_system, 'memory_manager'):
                # This is a placeholder for actual memory retrieval
                pass
            
            return conversations
        
        except Exception as e:
            logger.warning(f"Could not retrieve conversations: {e}")
            return []
    
    def _detect_topic_patterns(self, conversations: List[Dict]) -> List[str]:
        """
        Detect recurring topics across conversations.
        
        Args:
            conversations: List of conversation dicts
            
        Returns:
            List of topic pattern descriptions
        """
        patterns = []
        
        # Extract topics from all conversations
        all_topics = []
        for conv in conversations:
            topics = self._extract_topics(conv)
            all_topics.extend(topics)
        
        if not all_topics:
            return patterns
        
        # Count topic frequencies
        topic_counts = Counter(all_topics)
        total_convs = len(conversations)
        
        # Identify patterns
        for topic, count in topic_counts.items():
            frequency = count / total_convs
            if frequency >= self.pattern_threshold:
                percentage = frequency * 100
                patterns.append(
                    f"Topic '{topic}' appears in {percentage:.0f}% of conversations"
                )
        
        return patterns
    
    def _detect_behavioral_patterns(self, conversations: List[Dict]) -> List[str]:
        """
        Detect patterns in Sanctuary's own responses and behavior.
        
        Args:
            conversations: List of conversation dicts
            
        Returns:
            List of behavioral pattern descriptions
        """
        patterns = []
        
        # Analyze response characteristics
        response_types = []
        response_lengths = []
        question_counts = []
        
        for conv in conversations:
            # Extract Sanctuary's responses
            responses = self._extract_sanctuary_responses(conv)
            
            for response in responses:
                # Classify response type
                response_types.append(self._classify_response_type(response))
                
                # Track length
                response_lengths.append(len(str(response)))
                
                # Count questions
                question_counts.append(str(response).count("?"))
        
        if not response_types:
            return patterns
        
        # Analyze response type distribution
        type_counts = Counter(response_types)
        dominant_type = type_counts.most_common(1)[0]
        if dominant_type[1] / len(response_types) > 0.5:
            patterns.append(
                f"I tend to respond with {dominant_type[0]} more than other types"
            )
        
        # Analyze questioning behavior
        avg_questions = sum(question_counts) / len(question_counts) if question_counts else 0
        if avg_questions > 1:
            patterns.append(
                f"I often respond with questions rather than just statements"
            )
        
        # Analyze response length
        avg_length = sum(response_lengths) / len(response_lengths) if response_lengths else 0
        if avg_length > 500:
            patterns.append("I tend to provide detailed, comprehensive responses")
        elif avg_length < 200:
            patterns.append("I tend to provide concise, focused responses")
        
        return patterns
    
    def _detect_user_patterns(self, conversations: List[Dict]) -> List[str]:
        """
        Detect patterns in user behavior and preferences.
        
        Args:
            conversations: List of conversation dicts
            
        Returns:
            List of user pattern descriptions
        """
        patterns = []
        
        # Analyze user inputs
        user_topics = []
        user_styles = []
        
        for conv in conversations:
            user_inputs = self._extract_user_inputs(conv)
            
            for user_input in user_inputs:
                # Extract topics from user
                topics = self._extract_topics_from_text(str(user_input))
                user_topics.extend(topics)
                
                # Classify interaction style
                style = self._classify_interaction_style(str(user_input))
                user_styles.append(style)
        
        # Identify user topic preferences
        if user_topics:
            topic_counts = Counter(user_topics)
            top_topics = topic_counts.most_common(2)
            for topic, count in top_topics:
                if count >= 2:
                    patterns.append(
                        f"The user frequently asks about {topic}"
                    )
        
        # Identify user interaction style
        if user_styles:
            style_counts = Counter(user_styles)
            dominant_style = style_counts.most_common(1)[0]
            if dominant_style[1] / len(user_styles) > 0.5:
                patterns.append(
                    f"The user prefers {dominant_style[0]} interactions"
                )
        
        return patterns
    
    def _detect_temporal_patterns(self, conversations: List[Dict]) -> List[str]:
        """
        Detect temporal patterns in interactions.
        
        Args:
            conversations: List of conversation dicts
            
        Returns:
            List of temporal pattern descriptions
        """
        patterns = []
        
        # This would analyze timing patterns in production
        # For now, return empty list as placeholder
        
        return patterns
    
    def _extract_topics(self, conversation: Dict) -> List[str]:
        """Extract topics from a conversation."""
        topics = []
        
        # Try to get explicit topics
        if "topics" in conversation:
            topics.extend(conversation["topics"])
        
        # Extract from content
        content = conversation.get("content", {})
        if isinstance(content, dict):
            text = str(content.get("text", content.get("message", "")))
            topics.extend(self._extract_topics_from_text(text))
        
        return topics
    
    def _extract_topics_from_text(self, text: str) -> List[str]:
        """Extract topics from text content."""
        # Simple keyword-based topic extraction
        text_lower = text.lower()
        topics = []
        
        topic_keywords = {
            "consciousness": ["conscious", "awareness", "sentien"],
            "emotions": ["emotion", "feel", "affect"],
            "memory": ["memory", "remember", "recall"],
            "ethics": ["ethic", "moral", "right", "wrong"],
            "creativity": ["create", "imagine", "art"],
            "learning": ["learn", "understand", "knowledge"]
        }
        
        for topic, keywords in topic_keywords.items():
            if any(kw in text_lower for kw in keywords):
                topics.append(topic)
        
        return topics
    
    def _extract_sanctuary_responses(self, conversation: Dict) -> List[str]:
        """Extract Sanctuary's responses from conversation."""
        # Simplified - would parse actual conversation structure in production
        responses = []
        content = conversation.get("content", {})
        
        if isinstance(content, dict) and "sanctuary_response" in content:
            responses.append(content["sanctuary_response"])
        
        return responses
    
    def _extract_user_inputs(self, conversation: Dict) -> List[str]:
        """Extract user inputs from conversation."""
        # Simplified - would parse actual conversation structure in production
        inputs = []
        content = conversation.get("content", {})
        
        if isinstance(content, dict) and "user_input" in content:
            inputs.append(content["user_input"])
        
        return inputs
    
    def _classify_response_type(self, response: str) -> str:
        """Classify the type of response."""
        response_str = str(response).lower()
        
        if "?" in response_str:
            return "questioning"
        elif any(word in response_str for word in ["i think", "i believe", "perhaps"]):
            return "reflective"
        elif any(word in response_str for word in ["because", "therefore", "thus"]):
            return "explanatory"
        else:
            return "informative"
    
    def _classify_interaction_style(self, text: str) -> str:
        """Classify user interaction style."""
        text_lower = text.lower()
        
        if len(text) > 300:
            return "detailed explanations"
        elif "?" in text:
            return "question-asking"
        elif any(word in text_lower for word in ["explain", "tell me", "describe"]):
            return "information-seeking"
        else:
            return "conversational"
    
    def _create_pattern_percept(self, pattern_type: str, pattern: str) -> Percept:
        """
        Create an introspective percept from a detected pattern.
        
        Args:
            pattern_type: Type of pattern (topic, behavioral, user, temporal)
            pattern: Pattern description string
            
        Returns:
            Introspective Percept
        """
        content = {
            "type": "interaction_pattern",
            "pattern_type": pattern_type,
            "pattern": pattern,
            "source": "interaction_pattern_analysis",
            "insight": self._generate_insight(pattern_type, pattern)
        }
        
        # Compute salience based on pattern type
        salience_map = {
            "topic": 0.6,
            "behavioral": 0.7,
            "user": 0.65,
            "temporal": 0.5
        }
        
        percept = Percept(
            modality="introspection",
            raw=content,
            complexity=20,
            metadata={
                "salience": salience_map.get(pattern_type, 0.6),
                "pattern_type": pattern_type,
                "source": "interaction_pattern_analysis"
            }
        )
        
        return percept
    
    def _generate_insight(self, pattern_type: str, pattern: str) -> str:
        """Generate an insight from a pattern."""
        if pattern_type == "topic":
            return "This reveals what themes are important in our interactions"
        elif pattern_type == "behavioral":
            return "This shows something about my conversational tendencies"
        elif pattern_type == "user":
            return "This helps me understand communication preferences"
        elif pattern_type == "temporal":
            return "This reveals timing patterns in our engagement"
        else:
            return "This pattern offers insight into our interaction dynamics"
