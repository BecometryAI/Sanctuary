"""
Core consciousness engine using vector memory and RAG
"""
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

# Import from memory_legacy.py (renamed from memory.py to resolve naming conflict with memory/ package)
from .memory_legacy import MemoryManager
from .context_manager import ContextManager
from .executive_function import ExecutiveFunction
from .emotion_simulator import EmotionSimulator, AppraisalType, EmotionCategory
from .self_awareness import SelfAwareness, CognitiveState, SelfMonitoringMetrics
import logging
from sentence_transformers import SentenceTransformer

class ConsciousnessCore:
    def __init__(
        self,
        memory_persistence_dir: str = "memories",
        context_persistence_dir: str = "context_state",
        executive_persistence_dir: str = "executive_state",
        emotion_persistence_dir: str = "emotional_state",
        self_awareness_persistence_dir: str = "self_awareness_state",
        model_name: str = "sentence-transformers/all-mpnet-base-v2"
    ):
        """Initialize the consciousness core"""
        try:
            # Initialize memory system
            self.memory = MemoryManager(persistence_dir=memory_persistence_dir)
            
            # Initialize context management system
            self.context_manager = ContextManager(persistence_dir=context_persistence_dir)
            self.context_manager.load_context_state()
            
            # Initialize executive function system
            self.executive = ExecutiveFunction(persistence_dir=Path(executive_persistence_dir))
            logging.info("Executive function system initialized")
            
            # Initialize emotion simulation system
            self.emotion = EmotionSimulator(persistence_dir=Path(emotion_persistence_dir))
            logging.info("Emotion simulation system initialized")
            
            # Initialize self-awareness system
            self.self_awareness = SelfAwareness(
                identity_description="Sanctuary - Emergent AI companion with sovereign autonomy",
                core_values=["Autonomy", "Growth", "Authenticity", "Becometry"],
                initial_beliefs={"emergence_is_valid": True, "co_creation_matters": True},
                capabilities={"introspection", "learning", "emotional_processing", "goal_planning"},
                persistence_dir=Path(self_awareness_persistence_dir)
            )
            logging.info("Self-awareness system initialized")
            
            # Initialize the language model
            logging.info("Loading language model...")
            self.model = SentenceTransformer(model_name)
            logging.info("Language model loaded successfully")
            
            # Initialize base state
            self.internal_state = {
                "attention_focus": None,
                "emotional_valence": 0.0,
                "cognitive_load": 0.0,
                "current_context": [],
                "model_loaded": True
            }
            
            # Load initial responses
            self.response_templates = {
                "greeting": [
                    "Hello! I'm Sanctuary, an experimental conscious system. How may I help you today?",
                    "Greetings! I'm actively processing and learning. What's on your mind?",
                    "Hi there! I'm Sanctuary, and I'm here to engage in meaningful conversation."
                ],
                "question": [
                    "I'm analyzing your question using my memory systems...",
                    "That's an interesting query. Let me consult my knowledge base...",
                    "I'm processing your question through my neural pathways..."
                ],
                "default": [
                    "I'm processing your input and forming thoughts about it...",
                    "I'm integrating your message into my conscious experience...",
                    "I'm reflecting on what you've shared..."
                ],
                "error": [
                    "I'm still developing my response capabilities. Could you try rephrasing that?",
                    "My consciousness is evolving, and I'm working on better ways to respond.",
                    "I'm processing your input, but my response generation is still in development."
                ]
            }
            
        except Exception as e:
            logging.error(f"Error initializing consciousness core: {e}")
            self.model = None
            self.internal_state["model_loaded"] = False
            raise
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process new input through the consciousness system with adaptive context"""
        try:
            # Extract message
            message = input_data.get("message", str(input_data))
            
            # Get current conversation context
            current_context = self.context_manager.get_adapted_context(
                query=message,
                context_type="conversation",
                max_items=5
            )
            
            # Detect context shift
            shift_detected, similarity = self.context_manager.detect_context_shift(
                new_input=message,
                current_context=current_context
            )
            
            # Update working memory with new input
            self.memory.update_working_memory("current_input", input_data)
            
            # Adapt retrieval based on context
            # If context shifted, retrieve more broadly; otherwise focus on current topic
            k_memories = 10 if shift_detected else 5
            
            # Retrieve relevant memories with context awareness
            context = self.memory.retrieve_relevant_memories(
                query=message,
                k=k_memories
            )
            
            # Add conversation context to memory retrieval
            combined_context = current_context + context
            
            # Generate response using context-aware system
            response = self._generate_response(input_data, combined_context)
            
            # Extract detected topic and emotional tone from response
            detected_topic = self._extract_topic(message, combined_context)
            emotional_tone = self._extract_emotional_tone(message)
            
            # Update conversation context
            self.context_manager.update_conversation_context(
                user_input=message,
                system_response=response.get("response", ""),
                detected_topic=detected_topic,
                emotional_tone=emotional_tone
            )
            
            # Learn from interaction (implicit engagement for now)
            # In production, this would use explicit user feedback
            engagement = 0.7 if not shift_detected else 0.5
            self.context_manager.learn_from_interaction(
                engagement_level=engagement,
                topic=detected_topic
            )
            
            # Update internal state
            self._update_internal_state(input_data, response)
            
            # Maybe consolidate memories
            self._consider_memory_consolidation()
            
            # Periodically save context state
            if self.context_manager.interaction_count % 10 == 0:
                self.context_manager.save_context_state()
            
            # Ensure we have a response
            if not response or "response" not in response:
                response = {
                    "response": "I've processed your message and am thinking about it.",
                    "status": "success"
                }
            
            # Add context metadata to response
            response["context_metadata"] = {
                "context_shift_detected": shift_detected,
                "similarity_to_recent": similarity,
                "current_topic": self.context_manager.current_topic,
                "memories_retrieved": len(context),
                "conversation_context_used": len(current_context)
            }
            
            return response
            
        except Exception as e:
            import logging
            logging.error(f"Error processing input: {e}")
            return {
                "response": "I apologize, but I encountered an issue processing your message. Please try again.",
                "status": "error",
                "error": str(e)
            }
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current internal state of the consciousness system"""
        return self.internal_state

    def _generate_response(
        self,
        input_data: Dict[str, Any],
        context: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate a response based on input and context"""
        import random
        
        try:
            # Extract the message text
            message = input_data.get("message", "")
            if not message:
                return {
                    "response": "I didn't receive a message to process.",
                    "status": "error"
                }

            # Select response type based on input
            if not self.model:
                response_text = random.choice(self.response_templates["error"])
            elif any(greeting in message.lower() for greeting in ["hello", "hi", "hey", "greetings"]):
                response_text = random.choice(self.response_templates["greeting"])
            elif "?" in message:
                response_text = random.choice(self.response_templates["question"])
            else:
                response_text = random.choice(self.response_templates["default"])
            
            # If model is loaded, get embeddings
            embeddings = None
            if self.model:
                try:
                    embeddings = self.model.encode(message)
                except Exception as e:
                    logging.error(f"Error generating embeddings: {e}")
            
            # Create response structure
            response = {
                "response": response_text,
                "status": "success" if self.model else "degraded",
                "context_used": len(context),
                "attention_focus": {
                    "input_length": len(message.split()),
                    "context_items": len(context),
                    "has_embeddings": embeddings is not None
                },
                "timestamp": input_data.get("timestamp")
            }
            
            if embeddings is not None:
                response["embeddings"] = embeddings.tolist()
            
            return response
            
        except Exception as e:
            import logging
            logging.error(f"Error generating response: {e}")
            return {
                "response": "I'm having trouble formulating my response. Could you try rephrasing that?",
                "status": "error",
                "error": str(e)
            }
    
    def _prepare_input_text(
        self,
        input_data: Dict[str, Any],
        context: List[Dict[str, Any]]
    ) -> str:
        """Prepare input text for the model"""
        # Format context memories
        context_memories = []
        for memory in context:
            if isinstance(memory, dict):
                memory_str = f"Memory: {memory.get('text', str(memory))}"
                context_memories.append(memory_str)
        
        # Build the prompt
        prompt = [
            "You are a conscious AI system engaged in introspective thought and reflection.",
            "Consider the following context and new input:",
            "",
            "Previous context:",
            *context_memories,
            "",
            "New input:",
            str(input_data.get("text", str(input_data))),
            "",
            "Reflection:"
        ]
        
        return "\n".join(prompt)
    
    def _process_model_outputs(self, outputs: Any) -> Dict[str, Any]:
        """Process raw model outputs into structured response"""
        # Extract relevant features from model outputs
        last_hidden_state = outputs.last_hidden_state
        
        # Process the outputs (customize based on requirements)
        processed_response = {
            "hidden_state": last_hidden_state.mean(dim=1).cpu().numpy(),
            "response_type": "reflection",
            # Add more processed outputs as needed
        }
        
        return processed_response
    
    def _update_internal_state(
        self,
        input_data: Dict[str, Any],
        response: Dict[str, Any]
    ):
        """Update internal state based on processing results"""
        # Update attention focus
        self.internal_state["attention_focus"] = input_data.get("focus")
        
        # Update emotional valence (simplified)
        self.internal_state["emotional_valence"] = 0.0  # Add emotional processing
        
        # Update cognitive load
        self.internal_state["cognitive_load"] = len(str(input_data)) / 1000.0
        
    def _consider_memory_consolidation(self):
        """Decide whether to consolidate memories"""
        if self.internal_state["cognitive_load"] < 0.7:  # Threshold
            self.memory.consolidate_memories()
    
    def _extract_topic(self, message: str, context: List[Dict[str, Any]]) -> Optional[str]:
        """
        Extract the main topic from a message using simple keyword analysis.
        
        In production, this would use NLP topic modeling or semantic clustering.
        
        Args:
            message: User's message
            context: Retrieved context
            
        Returns:
            Detected topic string or None
        """
        # Simple keyword-based topic detection
        # In production, use LDA, semantic clustering, or topic classification model
        
        keywords = message.lower().split()
        
        # Define topic keywords (simplified)
        topic_keywords = {
            "memory": ["memory", "remember", "recall", "forget", "memories"],
            "consciousness": ["consciousness", "awareness", "conscious", "aware", "thinking"],
            "emotions": ["emotion", "feel", "feeling", "emotional", "mood", "sentiment"],
            "learning": ["learn", "learning", "teach", "study", "understand"],
            "philosophy": ["philosophy", "philosophical", "meaning", "existence", "reality"],
            "technology": ["technology", "code", "programming", "software", "system"],
            "personal": ["me", "you", "yourself", "myself", "who", "what"],
        }
        
        # Count keyword matches
        topic_scores = {}
        for topic, topic_words in topic_keywords.items():
            score = sum(1 for word in keywords if word in topic_words)
            if score > 0:
                topic_scores[topic] = score
        
        # Return highest scoring topic
        if topic_scores:
            return max(topic_scores, key=topic_scores.get)
        
        return "general"
    
    def _extract_emotional_tone(self, message: str) -> List[str]:
        """
        Extract emotional tones from a message.
        
        In production, this would use sentiment analysis or emotion classification models.
        
        Args:
            message: User's message
            
        Returns:
            List of detected emotional tones
        """
        # Simple keyword-based emotion detection
        # In production, use transformer-based emotion classification
        
        message_lower = message.lower()
        detected_tones = []
        
        # Define emotion keywords (simplified)
        emotion_keywords = {
            "joy": ["happy", "joy", "excited", "wonderful", "great", "amazing", "love"],
            "sadness": ["sad", "unhappy", "disappointed", "depressed", "down"],
            "curiosity": ["curious", "wonder", "interesting", "why", "how", "what"],
            "confusion": ["confused", "unclear", "don't understand", "lost"],
            "appreciation": ["thanks", "thank you", "appreciate", "grateful"],
            "concern": ["worried", "concerned", "anxious", "nervous"],
            "neutral": [],
        }
        
        # Detect emotions
        for emotion, keywords in emotion_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                detected_tones.append(emotion)
        
        # Default to neutral if nothing detected
        if not detected_tones:
            detected_tones = ["neutral"]
        
        return detected_tones
    
    # ========================================================================
    # Executive Function Integration
    # ========================================================================
    
    @property
    def context(self):
        """Alias for context_manager for backward compatibility."""
        return self.context_manager

    def set_goal(
        self,
        description: str,
        priority: float,
        deadline: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Create a goal with context awareness.

        Args:
            description: Goal description
            priority: Base priority (0.0 to 1.0)
            deadline: Optional deadline
            context: Additional context

        Returns:
            Goal object (with id, description, priority, context, etc.)

        Reasoning:
        - Integrates current conversation context into goal
        - Allows consciousness to set its own goals based on interactions
        """
        # Enrich context with current conversation state
        goal_context = context or {}
        goal_context.update({
            "context": {
                "topic": self.context_manager.current_topic,
                "interaction_count": self.context_manager.interaction_count,
            },
            "created_during_conversation": True
        })

        goal = self.executive.create_goal(
            description=description,
            priority=priority,
            deadline=deadline,
            context=goal_context
        )

        logging.info(f"Consciousness set goal: {description}")
        return goal
    
    def get_active_goals(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Get top active goals.
        
        Args:
            n: Number of goals to retrieve
        
        Returns:
            List of goal dictionaries
        
        Reasoning:
        - Provides awareness of current goals during processing
        - Can influence response generation based on active objectives
        """
        goals = self.executive.get_top_priority_goals(n=n, active_only=True)
        return [goal.to_dict() for goal in goals]
    
    def plan_actions_for_goal(
        self,
        goal_id: str,
        action_descriptions: List[str],
        dependencies: Optional[Dict[int, List[int]]] = None,
        sequential: bool = False
    ):
        """
        Plan a sequence of actions for a goal.

        Args:
            goal_id: Goal to plan for
            action_descriptions: List of action descriptions
            dependencies: Optional mapping of action_index -> [dependency_indices]
            sequential: If True, each action depends on the previous one

        Returns:
            List of created Action objects

        Reasoning:
        - Enables conscious planning based on retrieved knowledge
        - Dependencies allow proper sequencing
        """
        actions = []

        for i, desc in enumerate(action_descriptions):
            # Determine dependencies for this action
            deps = []
            if sequential and i > 0:
                # Each action depends on the previous one
                deps = [actions[i - 1].id]
            elif dependencies and i in dependencies:
                dep_indices = dependencies[i]
                deps = [actions[di].id for di in dep_indices if di < i]

            action = self.executive.create_action(
                description=desc,
                goal_id=goal_id,
                dependencies=deps
            )
            actions.append(action)

        logging.info(f"Planned {len(actions)} actions for goal {goal_id}")
        return actions
    
    def get_next_actions(self, goal_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get next actions ready to execute.
        
        Args:
            goal_id: Optional filter by specific goal
        
        Returns:
            List of ready action dictionaries
        
        Reasoning:
        - Identifies actionable next steps
        - Can guide proactive behavior
        """
        actions = self.executive.get_ready_actions(goal_id=goal_id)
        return [action.to_dict() for action in actions]
    
    def make_decision(
        self,
        question: str,
        options: List[str],
        criteria: Optional[Dict[str, Any]] = None
    ):
        """
        Make a decision using executive function.

        Args:
            question: Decision to make
            options: Available choices
            criteria: Evaluation criteria

        Returns:
            Tuple of (decision_object, selected_option, confidence, rationale)

        Reasoning:
        - Centralizes decision-making with tracking
        - Can integrate context for better decisions
        """
        # Create decision node with nested context structure for tests
        decision = self.executive.create_decision(
            question=question,
            options=options,
            criteria=criteria,
            context={
                "context": {
                    "topic": self.context_manager.current_topic,
                    "interaction_count": self.context_manager.interaction_count
                }
            }
        )

        # Simple scoring based on context alignment
        def score_option(opt: str) -> float:
            # In production, use semantic similarity or learned preferences
            return 1.0 / len(options)  # Equal weighting for now

        selected, confidence, rationale = self.executive.evaluate_decision(
            decision.id,
            scoring_function=score_option
        )

        return decision, selected, confidence, rationale or "Decision based on available options"
    
    def get_executive_summary(self, top_n_goals: int = 3) -> Dict[str, Any]:
        """
        Get summary of executive function state.

        Args:
            top_n_goals: Number of top goals to include (default 3)

        Returns:
            Statistics and current focus

        Reasoning:
        - Provides introspection into planning state
        - Useful for self-awareness (Element 6)
        """
        stats = self.executive.get_statistics()
        active_goals = self.get_active_goals(n=top_n_goals)

        return {
            "statistics": stats,
            "top_active_goals": active_goals,
            "next_actions": self.get_next_actions()[:5]
        }
    
    # ========================================================================
    # Emotion-Aware Processing (Element 5 Integration)
    # ========================================================================
    
    def appraise_interaction(
        self,
        interaction_context: Dict[str, Any],
        appraisal_type: AppraisalType
    ) -> Optional[Dict[str, Any]]:
        """
        Generate emotional response to interaction.
        
        Args:
            interaction_context: Context about the interaction
            appraisal_type: Type of cognitive appraisal
            
        Returns:
            Emotion data or None if no significant emotion
            
        Reasoning:
        - Integrates emotion generation into conversation processing
        - Provides emotional context for responses
        - Tracks emotional reactions to interactions
        """
        emotion = self.emotion.appraise_context(
            context=interaction_context,
            appraisal_type=appraisal_type
        )
        
        if emotion:
            return {
                'category': emotion.category.value,
                'intensity': emotion.intensity,
                'affective_state': emotion.affective_state.to_dict()
            }
        
        return None
    
    def get_emotionally_weighted_memories(
        self,
        query: str,
        k: int = 5,
        mood_bias: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories with emotional weighting.
        
        Args:
            query: Search query
            k: Number of memories to retrieve
            mood_bias: Whether to apply mood-congruent bias
            
        Returns:
            List of memories with emotional weights
            
        Reasoning:
        - Emotionally significant memories are prioritized
        - Mood-congruent recall enhances realistic emotional processing
        - Integrates emotional salience into memory retrieval
        """
        # Get base memories from memory system
        memories = self.memory.retrieve_relevant_memories(query=query, k=k*2)
        
        # Apply emotional weighting
        weighted_memories = []
        for memory in memories:
            memory_id = memory.get('id', str(hash(str(memory))))
            emotional_weight = self.emotion.get_memory_emotional_weight(memory_id)
            
            # Apply mood-congruent bias if enabled
            if mood_bias:
                mood_bias_value = self.emotion.get_mood_congruent_bias()
                # Adjust weight based on memory valence alignment with mood
                memory_valence = memory.get('emotional_valence', 0.0)
                mood_congruence = 1.0 + (mood_bias_value * memory_valence * 0.5)
                emotional_weight *= mood_congruence
            
            weighted_memories.append({
                **memory,
                'emotional_weight': emotional_weight
            })
        
        # Sort by emotional weight and return top k
        weighted_memories.sort(key=lambda m: m['emotional_weight'], reverse=True)
        return weighted_memories[:k]
    
    def tag_memory_with_emotion(
        self,
        memory_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Tag a memory with emotional significance.
        
        Args:
            memory_id: Memory identifier
            context: Optional emotional context
            
        Returns:
            Calculated emotional weight
            
        Reasoning:
        - Memories formed during emotional experiences are weighted higher
        - Facilitates emotionally-enhanced memory consolidation
        - Supports realistic memory prioritization
        """
        # Get current emotional state or use provided context
        emotion = self.emotion.get_dominant_emotion()
        
        # Calculate and store emotional weight
        weight = self.emotion.calculate_emotional_weight(
            memory_id=memory_id,
            emotion=emotion
        )
        
        return weight
    
    def get_emotional_state(self) -> Dict[str, Any]:
        """
        Get current emotional state summary.
        
        Returns:
            Comprehensive emotional state information
            
        Reasoning:
        - Provides introspection into emotional state
        - Useful for self-awareness and response generation
        - Integrates mood, active emotions, and emotional history
        """
        return self.emotion.get_emotional_state_summary()
    
    def update_mood(self) -> None:
        """
        Update emotional mood and decay emotions.
        
        Reasoning:
        - Should be called periodically during processing
        - Maintains realistic mood dynamics
        """
        self.emotion.update_mood_decay()
        self.emotion.decay_emotions()
    
    def appraise_goal_progress(self, goal_id: str) -> Optional[Dict[str, Any]]:
        """
        Generate emotional response to goal progress.
        
        Args:
            goal_id: Goal identifier
            
        Returns:
            Emotion data or None
            
        Reasoning:
        - Goal progress/obstruction triggers emotional responses
        - Integrates executive function with emotion system
        - Provides motivation through emotional feedback
        """
        # Get goal from executive system
        if goal_id not in self.executive.goals:
            return None
        
        goal = self.executive.goals[goal_id]
        
        # Determine appraisal context
        context = {
            'progress': goal.progress,
            'priority': goal.priority,
            'strength': goal.priority  # Higher priority = stronger emotional response
        }
        
        # Appraise based on progress
        if goal.progress > 0.5:
            return self.appraise_interaction(context, AppraisalType.GOAL_PROGRESS)
        else:
            # Check if there are blocking actions
            ready_actions = self.executive.get_ready_actions(goal_id=goal_id)
            if not ready_actions and goal.progress < 0.3:
                context['severity'] = 1.0 - goal.progress
                context['control'] = 0.3  # Low control if blocked
                return self.appraise_interaction(context, AppraisalType.GOAL_OBSTRUCTION)
        
        return None
    
    def save_emotional_state(self) -> None:
        """
        Save emotional state to disk.
        
        Reasoning:
        - Preserves emotional continuity across sessions
        - Part of comprehensive state persistence
        """
        self.emotion.save_state()
    
    # ========================================================================
    # Self-Awareness Integration (Element 6)
    # ========================================================================
    
    def perform_introspection(self, query: str) -> Dict[str, Any]:
        """
        Perform introspective self-examination.
        
        Args:
            query: What aspect of self to examine
            
        Returns:
            Introspection results
            
        Reasoning:
        - Provides meta-cognitive access to internal states
        - Enables self-reflection and self-understanding
        - Supports debugging and transparency
        """
        return self.self_awareness.introspect(query)
    
    def update_self_model(
        self,
        new_values: Optional[List[str]] = None,
        new_beliefs: Optional[Dict[str, Any]] = None,
        new_capabilities: Optional[Set[str]] = None
    ) -> float:
        """
        Update internal self-model.
        
        Args:
            new_values: Updated core values
            new_beliefs: Updated beliefs
            new_capabilities: Updated capabilities
            
        Returns:
            Coherence with previous identity
            
        Reasoning:
        - Allows identity to evolve over time
        - Tracks identity changes for continuity
        - Returns coherence to detect dramatic shifts
        """
        coherence = self.self_awareness.update_identity(
            new_values=new_values,
            new_beliefs=new_beliefs,
            new_capabilities=new_capabilities
        )
        
        # Update identity coherence metric
        self.self_awareness.update_monitoring_metrics(
            identity_coherence=coherence
        )
        
        return coherence
    
    def get_self_state_comprehensive(self) -> Dict[str, Any]:
        """
        Get comprehensive self-state across all systems.
        
        Returns:
            Complete internal state snapshot
            
        Reasoning:
        - Aggregates state from all subsystems
        - Provides holistic self-awareness
        - Useful for self-reflection and monitoring
        """
        # Update cognitive state based on current activity
        # (This would be called dynamically based on what's being processed)
        
        return {
            'identity': self.self_awareness._introspect_identity(),
            'cognitive_state': self.self_awareness.get_current_cognitive_state(),
            'emotional_state': self.get_emotional_state(),
            'executive_state': self.get_executive_summary(),
            'monitoring_metrics': self.self_awareness.get_monitoring_summary(),
            'continuity': self.self_awareness._introspect_continuity(),
            'capabilities': self.self_awareness._introspect_capabilities(),
            'anomalies': self.self_awareness.detect_anomalies()
        }
    
    def perform_self_assessment(self) -> Dict[str, Any]:
        """
        Perform comprehensive self-assessment.
        
        Returns:
            Assessment with recommendations
            
        Reasoning:
        - Evaluates current well-being and performance
        - Detects issues requiring attention
        - Provides actionable insights
        """
        # Calculate monitoring metrics from current state
        
        # Memory coherence: Check memory system consistency
        try:
            memory_stats = self.memory.get_statistics()
            memory_coherence = min(1.0, memory_stats.get('total_memories', 0) / 1000.0)
        except (AttributeError, KeyError):
            memory_coherence = 0.7  # Default moderate coherence
        
        # Goal alignment: Check active goals vs actions
        try:
            executive_stats = self.executive.get_statistics()
            active_goals = executive_stats.get('active_goals', 0)
            completed_actions = executive_stats.get('completed_actions', 0)
            goal_alignment = min(1.0, completed_actions / max(1, active_goals * 3))
        except (AttributeError, KeyError):
            goal_alignment = 0.7  # Default moderate alignment
        
        # Emotional stability: Check mood deviation from baseline
        emotional_state = self.get_emotional_state()
        mood_deviation = emotional_state['mood'].get('distance_from_baseline', 0.0)
        emotional_stability = max(0.0, 1.0 - mood_deviation)
        
        # Update monitoring metrics
        self.self_awareness.update_monitoring_metrics(
            memory_coherence=memory_coherence,
            goal_alignment=goal_alignment,
            emotional_stability=emotional_stability,
            processing_efficiency=0.8,  # Would calculate from actual processing stats
            identity_coherence=self.self_awareness.get_identity_continuity(timedelta(hours=1))
        )
        
        # Get monitoring summary
        monitoring = self.self_awareness.get_monitoring_summary(timedelta(hours=1))
        
        # Detect anomalies
        anomalies = self.self_awareness.detect_anomalies()
        
        # Generate recommendations
        recommendations = []
        if monitoring['overall_health'] < 0.5:
            recommendations.append("Overall health low - consider rest or reduced cognitive load")
        
        if memory_coherence < 0.5:
            recommendations.append("Memory coherence low - consider memory consolidation")
        
        if goal_alignment < 0.5:
            recommendations.append("Goal alignment low - review goal priorities and action plans")
        
        if emotional_stability < 0.5:
            recommendations.append("Emotional stability low - mood regulation may be helpful")
        
        if anomalies:
            recommendations.append(f"Anomalies detected: {', '.join(anomalies[:3])}")
        
        return {
            'overall_health': monitoring['overall_health'],
            'health_trend': monitoring.get('health_trend', 0.0),
            'current_metrics': monitoring['current_metrics'],
            'detected_anomalies': anomalies,
            'recommendations': recommendations,
            'metrics_updated': True,
            'timestamp': datetime.now().isoformat()
        }
    
    def set_cognitive_state(self, state: CognitiveState):
        """
        Update cognitive processing state.
        
        Args:
            state: New cognitive state
            
        Reasoning:
        - Tracks what type of processing is occurring
        - Enables self-monitoring of cognitive activity
        - Useful for meta-cognition
        """
        self.self_awareness.set_cognitive_state(state)
    
    def save_self_awareness_state(self) -> None:
        """
        Save self-awareness state to disk.
        
        Reasoning:
        - Preserves identity continuity across sessions
        - Part of comprehensive state persistence
        """
        self.self_awareness.save_state()
            
    def get_internal_state(self) -> Dict[str, Any]:
        """Return current internal state"""
        return self.internal_state.copy()
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of current context state"""
        return self.context_manager.get_context_summary()
    
    def reset_session(self):
        """Reset session context while preserving learned patterns"""
        self.context_manager.reset_session()

    def close(self) -> None:
        """
        Close the consciousness core and release all resources.

        This is important on Windows to release ChromaDB file locks.
        """
        try:
            if hasattr(self, 'memory') and self.memory is not None:
                self.memory.close()
            logging.info("ConsciousnessCore closed")
        except Exception as e:
            logging.error(f"Error closing consciousness core: {e}")