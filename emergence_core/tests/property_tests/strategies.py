"""
Custom Hypothesis strategies for generating test data.

This module defines reusable strategies for generating valid instances
of cognitive architecture data types (Percepts, Goals, Memories, etc.)
for property-based testing.
"""

from hypothesis import strategies as st
from datetime import datetime, timedelta
import numpy as np
from typing import List

from lyra.cognitive_core.workspace import (
    Percept,
    Goal,
    GoalType,
    Memory,
)
from lyra.cognitive_core.affect import EmotionalState


# Strategy for generating valid embeddings
@st.composite
def embeddings(draw, dim=384):
    """
    Generate normalized embedding vectors.
    
    Args:
        draw: Hypothesis draw function
        dim: Dimension of the embedding vector (default 384)
        
    Returns:
        List[float]: Normalized embedding vector
    """
    vec = draw(st.lists(
        st.floats(min_value=-1, max_value=1, allow_nan=False, allow_infinity=False),
        min_size=dim,
        max_size=dim
    ))
    norm = np.linalg.norm(vec)
    if norm > 0:
        return [float(v / norm) for v in vec]
    return vec


# Strategy for generating Percepts
@st.composite
def percepts(draw):
    """
    Generate valid Percept objects.
    
    Args:
        draw: Hypothesis draw function
        
    Returns:
        Percept: A valid Percept instance
    """
    return Percept(
        id=draw(st.uuids()).hex,
        modality=draw(st.sampled_from(["text", "image", "audio", "introspection"])),
        raw=draw(st.one_of(
            st.text(min_size=1, max_size=500),
            st.dictionaries(st.text(max_size=20), st.text(max_size=100), max_size=5)
        )),
        embedding=draw(st.one_of(st.none(), embeddings())),
        complexity=draw(st.integers(min_value=1, max_value=10)),
        metadata=draw(st.dictionaries(st.text(max_size=20), st.text(max_size=100), max_size=5))
    )


# Strategy for generating Goals
@st.composite
def goals(draw):
    """
    Generate valid Goal objects.
    
    Args:
        draw: Hypothesis draw function
        
    Returns:
        Goal: A valid Goal instance
    """
    return Goal(
        id=draw(st.uuids()).hex,
        type=draw(st.sampled_from(list(GoalType))),
        description=draw(st.text(min_size=1, max_size=200)),
        priority=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        progress=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        metadata=draw(st.dictionaries(st.text(max_size=20), st.text(max_size=100), max_size=5))
    )


# Strategy for generating Memories
@st.composite
def memories(draw):
    """
    Generate valid Memory objects.
    
    Args:
        draw: Hypothesis draw function
        
    Returns:
        Memory: A valid Memory instance
    """
    # Generate timestamp within a reasonable range (past year)
    now = datetime.now()
    timestamp = draw(st.datetimes(
        min_value=now - timedelta(days=365),
        max_value=now
    ))
    
    return Memory(
        id=draw(st.uuids()).hex,
        content=draw(st.text(min_size=1, max_size=1000)),
        embedding=draw(st.one_of(st.none(), embeddings())),
        timestamp=timestamp,
        significance=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        tags=draw(st.lists(st.text(min_size=1, max_size=20), max_size=5)),
        metadata=draw(st.dictionaries(st.text(max_size=20), st.text(max_size=100), max_size=5))
    )


# Strategy for generating EmotionalState
@st.composite
def emotional_states(draw):
    """
    Generate valid EmotionalState objects with VAD values in [-1, 1].
    
    Args:
        draw: Hypothesis draw function
        
    Returns:
        EmotionalState: A valid EmotionalState instance
    """
    return EmotionalState(
        valence=draw(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        arousal=draw(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        dominance=draw(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    )


# Strategy for generating lists of percepts/goals with realistic sizes
percept_lists = st.lists(percepts(), min_size=0, max_size=50)
goal_lists = st.lists(goals(), min_size=0, max_size=10)
memory_lists = st.lists(memories(), min_size=0, max_size=20)


# Helper function for creating unique items
def make_unique_by_id(items: List) -> List:
    """
    Remove duplicates from a list based on ID attribute.
    
    Args:
        items: List of objects with 'id' attribute
        
    Returns:
        List with unique items (by ID)
    """
    seen_ids = set()
    unique_items = []
    for item in items:
        if item.id not in seen_ids:
            seen_ids.add(item.id)
            unique_items.append(item)
    return unique_items
