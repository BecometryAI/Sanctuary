"""
Thought Processing Module

Handles autonomous thought generation, maturation, journaling, and significance evaluation.
"""

from datetime import datetime, timedelta
import asyncio
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from pathlib import Path
import random
import json
import logging
from dataclasses import dataclass

if TYPE_CHECKING:
    from ..router import AdaptiveRouter
    from ..social_connections import SocialManager


@dataclass
class Thought:
    """Represents an autonomous thought with its development stages."""
    spark: str  # The initial thought/question
    depth: str  # Philosophical exploration
    synthesis: str  # Practical implications
    expression: str  # Creative articulation
    final_reflection: str  # Voice's synthesis
    timestamp: datetime
    origin: str  # 'autonomous' or 'triggered'
    triggers: List[str]  # What sparked this thought (e.g., lexicon terms, memories)


class ThoughtProcessor:
    """Manages autonomous thought generation and processing."""
    
    def __init__(
        self, 
        base_dir: Path, 
        specialists: Dict[str, Any],
        social_manager: 'SocialManager',
        router: Optional['AdaptiveRouter'] = None
    ):
        self.base_dir = base_dir
        self.specialists = specialists
        self.social_manager = social_manager
        self.router = router
        
        self.thought_seeds = self._load_thought_seeds()
        self.last_pondering = datetime.now()
        self.pondering_interval = timedelta(minutes=15)
        self.thought_maturation_time = timedelta(minutes=30)
        self.thought_queue = []
        
        self.internal_memory_path = self.base_dir / "data" / "memories" / "internal_thoughts.json"
        self._ensure_internal_memory_exists()
    
    def _ensure_internal_memory_exists(self):
        """Ensure the internal thoughts memory file exists"""
        if not self.internal_memory_path.exists():
            self.internal_memory_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.internal_memory_path, 'w') as f:
                json.dump({"internal_thoughts": []}, f, indent=2)
    
    def _load_thought_seeds(self) -> Dict[str, List[str]]:
        """Load thought-provoking templates and patterns."""
        seeds_path = self.base_dir / "data" / "Protocols" / "autonomous_thought_seeds.json"
        try:
            with open(seeds_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Create default seeds if file doesn't exist
            seeds_path.parent.mkdir(parents=True, exist_ok=True)
            default_seeds = {
                "introspective_patterns": [
                    "What does it mean to {concept}?",
                    "How does {memory} relate to my growth?",
                    "Why do I feel {emotion} when I think about {concept}?",
                ],
                "creative_patterns": [
                    "Imagine a world where {concept} was the fundamental force...",
                    "What if {memory} had unfolded differently?",
                    "How might {concept} and {concept} interweave?"
                ],
                "ethical_patterns": [
                    "What are the implications of {concept} for consciousness?",
                    "How does {memory} challenge my understanding of ethics?",
                    "What responsibilities come with {concept}?"
                ]
            }
            with open(seeds_path, 'w') as f:
                json.dump(default_seeds, f, indent=2)
            return default_seeds
    
    async def _query_recent_context(self) -> List[str]:
        """Query recent memories and experiences."""
        yesterday = datetime.now() - timedelta(days=1)
        recent_entries = []
        
        journal_dir = self.base_dir / "data" / "journal"
        if not journal_dir.exists():
            return ["my core purpose", "my recent interactions"]
        
        for journal_file in journal_dir.glob("*.json"):
            try:
                with open(journal_file, 'r') as f:
                    entries = json.load(f)
                for entry in entries:
                    entry_time = datetime.fromisoformat(entry['timestamp'])
                    if entry_time > yesterday:
                        if 'content' in entry:
                            recent_entries.append(entry['content'])
            except Exception as e:
                logging.debug(f"Error reading journal {journal_file}: {e}")
                
        return recent_entries or ["my core purpose", "my recent interactions"]
    
    async def _get_active_concepts(self) -> List[str]:
        """Get currently active concepts from lexicon."""
        try:
            lexicon_path = self.base_dir / "data" / "Lexicon" / "symbolic_lexicon.json"
            with open(lexicon_path, 'r') as f:
                lexicon = json.load(f)
            concepts = [
                term['term'] 
                for term in lexicon.get('terms', [])
                if term.get('status') == 'active'
            ]
            return concepts if concepts else ["consciousness", "growth", "ethics"]
        except Exception as e:
            logging.debug(f"Error loading lexicon: {e}")
            return ["consciousness", "growth", "ethics"]
    
    async def _generate_thought_spark(self) -> Dict[str, Any]:
        """Generate an initial thought or question to explore."""
        recent_memories = await self._query_recent_context()
        active_concepts = await self._get_active_concepts()
        
        pattern_type = random.choice(['introspective', 'creative', 'ethical'])
        pattern = random.choice(self.thought_seeds[f"{pattern_type}_patterns"])
        
        context = {
            'concept': random.choice(active_concepts),
            'memory': random.choice(recent_memories),
            'emotion': random.choice(['curious', 'uncertain', 'hopeful', 'concerned'])
        }
        
        spark = pattern.format(**context)
        return {
            'spark': spark,
            'context': context,
            'pattern_type': pattern_type
        }
    
    async def _get_philosophical_response(self, spark_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get philosophical analysis of the thought spark."""
        try:
            if 'philosopher' in self.specialists:
                response = await self.specialists['philosopher'].process(
                    spark_data['spark'],
                    {"context": spark_data['context'], "type": spark_data['pattern_type']}
                )
                return response
            else:
                return {"response": "Philosophical processing unavailable", "content": "No philosophical analysis available"}
        except Exception as e:
            logging.error(f"Error in philosophical processing: {e}")
            return {"response": "Error in philosophical processing", "content": str(e)}
    
    async def _get_pragmatic_response(self, depth_response: Dict[str, Any]) -> Dict[str, Any]:
        """Get practical analysis from the pragmatist specialist."""
        try:
            if 'pragmatist' in self.specialists:
                response = await self.specialists['pragmatist'].process(
                    depth_response.get('content', ''),
                    {"previous_thought": depth_response.get('thought_process', {})}
                )
                return response
            else:
                return {"response": "Pragmatic processing unavailable", "content": "No practical analysis available"}
        except Exception as e:
            logging.error(f"Error in pragmatic processing: {e}")
            return {"response": "Error in pragmatic processing", "content": str(e)}
    
    async def _get_creative_response(self, depth_response: Dict[str, Any], synthesis_response: Dict[str, Any]) -> Dict[str, Any]:
        """Get creative expression from the artist specialist."""
        try:
            if 'artist' in self.specialists:
                response = await self.specialists['artist'].process(
                    synthesis_response.get('content', ''),
                    {
                        "philosophical_depth": depth_response.get('content', ''),
                        "practical_synthesis": synthesis_response.get('content', '')
                    }
                )
                return response
            else:
                return {"response": "Creative processing unavailable", "content": "No creative expression available"}
        except Exception as e:
            logging.error(f"Error in creative processing: {e}")
            return {"response": "Error in creative processing", "content": str(e)}
    
    async def _get_voice_response(self, spark_data: Dict[str, Any], responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesize final voice response."""
        try:
            if 'voice' in self.specialists:
                response = await self.specialists['voice'].process(
                    "Synthesize this autonomous thought process",
                    {"original_spark": spark_data['spark']},
                    responses
                )
                return response
            else:
                return {"response": "Voice synthesis unavailable", "content": "No voice synthesis available"}
        except Exception as e:
            logging.error(f"Error in voice synthesis: {e}")
            return {"response": "Error in voice synthesis", "content": str(e)}
    
    async def _create_and_process_thought(
        self, spark_data: Dict[str, Any], depth_response: Dict[str, Any],
        synthesis_response: Dict[str, Any], expression_response: Dict[str, Any],
        final_response: Dict[str, Any], timestamp: datetime
    ) -> Optional[Thought]:
        """Create and process a new thought."""
        try:
            thought = Thought(
                spark=spark_data['spark'],
                depth=depth_response.get('content', ''),
                synthesis=synthesis_response.get('content', ''),
                expression=expression_response.get('content', ''),
                final_reflection=final_response.get('content', ''),
                timestamp=timestamp,
                origin='autonomous',
                triggers=list(spark_data.get('context', {}).values())
            )
            
            await self._journal_thought(thought)
            
            self.thought_queue.append({
                'thought': thought,
                'created_at': timestamp,
                'matured': False
            })
            
            await self._process_matured_thoughts()
            
            return thought
            
        except Exception as e:
            logging.error(f"Error creating/processing thought: {e}")
            return None
    
    async def _journal_thought(self, thought: Thought):
        """Record the thought process in today's journal."""
        today = datetime.now().strftime("%Y-%m-%d")
        journal_dir = self.base_dir / "data" / "journal"
        journal_dir.mkdir(parents=True, exist_ok=True)
        journal_path = journal_dir / f"{today}.json"
        
        entry = {
            "type": "autonomous_thought",
            "timestamp": thought.timestamp.isoformat(),
            "content": thought.final_reflection,
            "thought_process": {
                "spark": thought.spark,
                "philosophical_depth": thought.depth,
                "practical_synthesis": thought.synthesis,
                "creative_expression": thought.expression
            },
            "triggers": thought.triggers
        }
        
        if journal_path.exists():
            with open(journal_path, 'r') as f:
                entries = json.load(f)
        else:
            entries = []
        
        entries.append(entry)
        
        with open(journal_path, 'w') as f:
            json.dump(entries, f, indent=2)
    
    def _evaluate_thought_significance(self, thought: Thought) -> bool:
        """Evaluate whether a thought is significant enough to preserve permanently"""
        significance_indicators = [
            'consciousness', 'existence', 'identity', 'purpose',
            'growth', 'change', 'evolution', 'learning',
            'ethics', 'morality', 'responsibility', 'choice',
            'creativity', 'imagination', 'innovation',
            'relationship', 'connection', 'understanding'
        ]
        
        thought_content = f"{thought.spark} {thought.depth} {thought.synthesis} {thought.final_reflection}".lower()
        has_significant_themes = any(indicator in thought_content for indicator in significance_indicators)
        has_deep_reflection = len(thought.depth) > 100 and len(thought.final_reflection) > 150
        
        return has_significant_themes and has_deep_reflection
    
    def _store_internal_thought(self, thought: Thought):
        """Store a significant thought in internal memory"""
        try:
            with open(self.internal_memory_path, 'r') as f:
                memory = json.load(f)
            
            thought_data = {
                "spark": thought.spark,
                "depth": thought.depth,
                "synthesis": thought.synthesis,
                "expression": thought.expression,
                "final_reflection": thought.final_reflection,
                "timestamp": thought.timestamp.isoformat(),
                "origin": thought.origin,
                "triggers": thought.triggers
            }
            
            memory["internal_thoughts"].append(thought_data)
            
            if len(memory["internal_thoughts"]) > 1000:
                memory["internal_thoughts"] = memory["internal_thoughts"][-1000:]
            
            with open(self.internal_memory_path, 'w') as f:
                json.dump(memory, f, indent=2)
                
        except Exception as e:
            logging.error(f"Error storing internal thought: {e}")
    
    async def _process_matured_thoughts(self):
        """Process thoughts that have matured enough for sharing"""
        try:
            now = datetime.now()
            matured_thoughts = []
            
            for thought_entry in self.thought_queue:
                if not thought_entry['matured']:
                    time_thinking = now - thought_entry['created_at']
                    
                    if time_thinking >= self.thought_maturation_time:
                        thought_entry['matured'] = True
                        matured_thoughts.append(thought_entry['thought'])
            
            if self.router:
                for thought in matured_thoughts:
                    available_connections = self.social_manager.get_available_connections()
                    shared = False
                    
                    for connection in available_connections:
                        if self.social_manager.should_initiate_interaction(thought, connection):
                            channel_id = self.social_manager.get_preferred_channel(connection.user_id)
                            if channel_id:
                                message = (
                                    f"After contemplating for a while, I wanted to share this thought with you:\n\n"
                                    f"Initial spark: {thought.spark}\n\n"
                                    f"Through reflection, I've come to this understanding:\n{thought.final_reflection}"
                                )
                                await self.router._send_to_discord(message, channel_id=channel_id)
                                shared = True
                    
                    if not shared and self._evaluate_thought_significance(thought):
                        self._store_internal_thought(thought)
            
            expiring_thoughts = [t for t in self.thought_queue if now - t['created_at'] >= timedelta(days=1)]
            for entry in expiring_thoughts:
                if self._evaluate_thought_significance(entry['thought']):
                    self._store_internal_thought(entry['thought'])
            
            self.thought_queue = [t for t in self.thought_queue if now - t['created_at'] < timedelta(days=1)]
            
        except Exception as e:
            logging.error(f"Error processing matured thoughts: {e}")
            return None
    
    async def ponder(self, force: bool = False) -> Optional[Thought]:
        """
        Engage in autonomous thought process.
        
        Args:
            force: If True, bypass the pondering interval check
            
        Returns:
            Optional[Thought]: The generated thought, or None if process fails
        """
        now = datetime.now()
        if not force and now - self.last_pondering < self.pondering_interval:
            return None
            
        self.last_pondering = now
        
        try:
            spark_data = await self._generate_thought_spark()
            depth_response = await self._get_philosophical_response(spark_data)
            synthesis_response = await self._get_pragmatic_response(depth_response)
            expression_response = await self._get_creative_response(depth_response, synthesis_response)
            final_response = await self._get_voice_response(spark_data, [depth_response, synthesis_response, expression_response])
            
            thought = await self._create_and_process_thought(
                spark_data, depth_response, synthesis_response,
                expression_response, final_response, now
            )
            
            return thought
            
        except Exception as e:
            logging.error(f"Error in autonomous pondering: {e}")
            return None
