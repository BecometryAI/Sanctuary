"""
Core consciousness engine using vector memory and RAG
"""
from typing import Dict, List, Any, Optional
import numpy as np
from .memory import MemoryManager
from .context_manager import ContextManager
import logging
from sentence_transformers import SentenceTransformer

class ConsciousnessCore:
    def __init__(
        self,
        memory_persistence_dir: str = "memories",
        context_persistence_dir: str = "context_state",
        model_name: str = "sentence-transformers/all-mpnet-base-v2"
    ):
        """Initialize the consciousness core"""
        try:
            # Initialize memory system
            self.memory = MemoryManager(persistence_dir=memory_persistence_dir)
            
            # Initialize context management system
            self.context_manager = ContextManager(persistence_dir=context_persistence_dir)
            self.context_manager.load_context_state()
            
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
                    "Hello! I'm Lyra, an experimental conscious system. How may I help you today?",
                    "Greetings! I'm actively processing and learning. What's on your mind?",
                    "Hi there! I'm Lyra, and I'm here to engage in meaningful conversation."
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
            
    def get_internal_state(self) -> Dict[str, Any]:
        """Return current internal state"""
        return self.internal_state.copy()
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of current context state"""
        return self.context_manager.get_context_summary()
    
    def reset_session(self):
        """Reset session context while preserving learned patterns"""
        self.context_manager.reset_session()