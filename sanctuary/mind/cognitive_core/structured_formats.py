"""
Structured Formats: Pydantic schemas for LLM I/O.

This module defines all structured data formats used for LLM input/output
operations. It uses Pydantic for validation, serialization, and type safety.

The structured formats provide:
- Type-safe data schemas for LLM requests/responses
- Automatic validation of LLM outputs
- Clear contracts between cognitive components
- Version control for format evolution
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field, validator


class IntentType(str, Enum):
    """Types of user intent detected in natural language input."""
    QUESTION = "question"
    REQUEST = "request"
    STATEMENT = "statement"
    GREETING = "greeting"
    INTROSPECTION_REQUEST = "introspection_request"
    MEMORY_REQUEST = "memory_request"
    UNKNOWN = "unknown"


class GoalTypeEnum(str, Enum):
    """Types of goals that can be generated from user input."""
    RESPOND_TO_USER = "respond_to_user"
    RETRIEVE_MEMORY = "retrieve_memory"
    INTROSPECT = "introspect"
    LEARN = "learn"
    CREATE = "create"
    OTHER = "other"


class Intent(BaseModel):
    """Represents classified user intent from LLM parsing."""
    type: IntentType
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


class Goal(BaseModel):
    """Represents a goal extracted from user input."""
    type: GoalTypeEnum
    description: str = Field(min_length=1, max_length=500)
    priority: float = Field(ge=0.0, le=1.0, description="Priority between 0 and 1")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


class Entities(BaseModel):
    """Extracted entities from user input."""
    topics: List[str] = Field(default_factory=list, description="Main topics mentioned")
    temporal: List[str] = Field(default_factory=list, description="Time references")
    emotional_tone: Optional[str] = Field(None, description="Detected emotional tone")
    names: List[str] = Field(default_factory=list, description="Named entities")
    other: Dict[str, Any] = Field(default_factory=dict, description="Other extracted entities")


class ConversationContext(BaseModel):
    """Context information for conversation tracking."""
    turn_count: int = Field(ge=0)
    recent_topics: List[str] = Field(default_factory=list)
    user_name: Optional[str] = None
    conversation_phase: Optional[str] = None
    additional_context: Dict[str, Any] = Field(default_factory=dict)


class LLMInputParseRequest(BaseModel):
    """
    Structured request for input parsing via LLM.
    
    This is sent to the LLM to request parsing of user input
    into structured cognitive components.
    """
    user_text: str = Field(min_length=1, description="The user's input text to parse")
    conversation_context: Optional[ConversationContext] = None
    current_workspace_state: Optional[Dict[str, Any]] = None
    parse_options: Dict[str, Any] = Field(default_factory=dict)


class LLMInputParseResponse(BaseModel):
    """
    Structured response from LLM input parser.
    
    This is the expected output format from the LLM when parsing
    user input. It contains all extracted cognitive structures.
    """
    intent: Intent
    goals: List[Goal] = Field(default_factory=list)
    entities: Entities = Field(default_factory=Entities)
    context_updates: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    
    @validator('goals')
    def validate_goals(cls, v):
        """Ensure at least one goal is generated."""
        if not v:
            # Default goal if none provided
            return [Goal(
                type=GoalTypeEnum.RESPOND_TO_USER,
                description="Respond to user",
                priority=0.9
            )]
        return v


class EmotionalState(BaseModel):
    """Current emotional state for output generation."""
    valence: float = Field(ge=-1.0, le=1.0, description="Positive/negative dimension")
    arousal: float = Field(ge=0.0, le=1.0, description="Activation/energy dimension")
    dominance: float = Field(ge=0.0, le=1.0, description="Control/agency dimension")
    label: Optional[str] = Field(None, description="Human-readable emotion label")


class WorkspaceStateSnapshot(BaseModel):
    """Simplified workspace state for output generation."""
    emotions: EmotionalState
    active_goals: List[Dict[str, Any]] = Field(default_factory=list)
    attended_percepts: List[Dict[str, Any]] = Field(default_factory=list)
    recalled_memories: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OutputGenerationRequest(BaseModel):
    """
    Structured request for output generation via LLM.
    
    Contains all context needed to generate an identity-aligned,
    emotion-influenced response.
    """
    user_input: str
    workspace_state: WorkspaceStateSnapshot
    conversation_history: List[str] = Field(default_factory=list)
    identity_context: Optional[Dict[str, str]] = Field(
        None, 
        description="Charter and protocols excerpts"
    )
    generation_options: Dict[str, Any] = Field(default_factory=dict)


class OutputGenerationResponse(BaseModel):
    """
    Structured response from LLM output generator.
    
    Contains the generated response and metadata.
    """
    response_text: str = Field(min_length=1)
    emotional_alignment: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=1.0,
        description="How well response aligns with emotional state"
    )
    goal_alignment: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0, 
        description="How well response addresses goals"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LLMErrorResponse(BaseModel):
    """Response format for LLM errors."""
    error_type: str
    error_message: str
    fallback_used: bool = False
    timestamp: datetime = Field(default_factory=datetime.now)


# Conversion utilities

def workspace_snapshot_to_dict(snapshot) -> Dict[str, Any]:
    """
    Convert cognitive core WorkspaceSnapshot to serializable dict.
    
    Args:
        snapshot: WorkspaceSnapshot from cognitive_core.workspace
        
    Returns:
        Dictionary suitable for inclusion in LLM requests
    """
    return {
        "emotions": {
            "valence": snapshot.emotions.get("valence", 0.0),
            "arousal": snapshot.emotions.get("arousal", 0.0),
            "dominance": snapshot.emotions.get("dominance", 0.0),
            "label": snapshot.metadata.get("emotion_label", "neutral")
        },
        "active_goals": [
            {
                "type": str(g.type),
                "description": g.description,
                "priority": g.priority,
                "progress": g.progress
            }
            for g in sorted(snapshot.goals, key=lambda g: g.priority, reverse=True)[:5]
        ],
        "attended_percepts": [
            {
                "modality": p.modality,
                "content": str(p.raw)[:200] if hasattr(p, 'raw') else "",
                "attention_score": p.metadata.get("attention_score", 0)
            }
            for p in sorted(
                snapshot.percepts.values(),
                key=lambda p: p.metadata.get("attention_score", 0),
                reverse=True
            )[:5]
        ],
        "recalled_memories": [
            {
                "content": p.raw.get("content", "")[:200] if isinstance(p.raw, dict) else str(p.raw)[:200]
            }
            for p in snapshot.percepts.values()
            if p.modality == "memory"
        ][:3],
        "metadata": snapshot.metadata
    }


def parse_response_to_goals(parse_response: LLMInputParseResponse, workspace_goal_type):
    """
    Convert structured parse response to workspace Goal objects.
    
    Args:
        parse_response: LLMInputParseResponse from LLM
        workspace_goal_type: GoalType enum from workspace module
        
    Returns:
        List of workspace Goal objects
    """
    from .workspace import Goal as WorkspaceGoal
    
    goal_type_mapping = {
        "respond_to_user": "RESPOND_TO_USER",
        "retrieve_memory": "RETRIEVE_MEMORY",
        "introspect": "INTROSPECT",
        "learn": "LEARN",
        "create": "CREATE",
        "other": "OTHER"
    }
    
    workspace_goals = []
    for goal in parse_response.goals:
        goal_type_str = goal_type_mapping.get(goal.type, "OTHER")
        workspace_goals.append(WorkspaceGoal(
            type=getattr(workspace_goal_type, goal_type_str),
            description=goal.description,
            priority=goal.priority,
            progress=0.0,
            metadata=goal.metadata
        ))
    
    return workspace_goals
