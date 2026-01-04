"""
Custom Hypothesis strategies for generating test data.

This module defines reusable test data generators compatible with the
existing workspace types (Percept, Goal, Memory, EmotionalState).
"""

from hypothesis import strategies as st
from lyra.cognitive_core.workspace import (
    Percept, Goal, Memory, GoalType
)
import uuid
from datetime import datetime, timedelta


# ===== EMBEDDING STRATEGY =====
@st.composite
def embeddings(draw, dim=384):
    """Generate normalized embedding vectors."""
    vec = draw(st.lists(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=dim,
        max_size=dim
    ))
    # Simple normalization
    norm = sum(x**2 for x in vec) ** 0.5
    if norm > 0:
        return [v / norm for v in vec]
    return vec


# ===== PERCEPT STRATEGY =====
@st.composite
def percepts(draw):
    """Generate valid Percept objects using EXISTING workspace structure."""
    return Percept(
        id=str(uuid.uuid4()),
        modality=draw(st.sampled_from(["text", "image", "audio", "introspection"])),
        embedding=draw(st.one_of(st.none(), embeddings())),
        raw=draw(st.text(min_size=1, max_size=200, alphabet=st.characters(blacklist_categories=('Cs',)))),
        complexity=draw(st.integers(min_value=1, max_value=10)),
        timestamp=draw(st.datetimes(min_value=datetime(2020, 1, 1), max_value=datetime(2030, 1, 1))),
        metadata=draw(st.dictionaries(
            st.text(min_size=1, max_size=20, alphabet=st.characters(blacklist_categories=('Cs',))),
            st.text(min_size=1, max_size=50, alphabet=st.characters(blacklist_categories=('Cs',))),
            max_size=5
        ))
    )


# ===== GOAL STRATEGY =====
@st.composite
def goals(draw):
    """Generate valid Goal objects."""
    return Goal(
        id=str(uuid.uuid4()),
        type=draw(st.sampled_from(list(GoalType))),
        description=draw(st.text(min_size=1, max_size=200, alphabet=st.characters(blacklist_categories=('Cs',)))),
        priority=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        progress=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        metadata=draw(st.dictionaries(
            st.text(min_size=1, max_size=20, alphabet=st.characters(blacklist_categories=('Cs',))),
            st.text(min_size=1, max_size=50, alphabet=st.characters(blacklist_categories=('Cs',))),
            max_size=5
        ))
    )


# ===== EMOTIONAL STATE STRATEGY =====
@st.composite
def emotional_states(draw):
    """Generate valid EmotionalState as Dict with VAD values in [-1, 1]."""
    return {
        'valence': draw(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False)),
        'arousal': draw(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False)),
        'dominance': draw(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False))
    }


# ===== MEMORY STRATEGY =====
@st.composite
def memories(draw):
    """Generate valid Memory objects."""
    return Memory(
        id=str(uuid.uuid4()),
        content=draw(st.text(min_size=1, max_size=200, alphabet=st.characters(blacklist_categories=('Cs',)))),
        timestamp=draw(st.datetimes(min_value=datetime(2020, 1, 1), max_value=datetime(2030, 1, 1))),
        significance=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        tags=draw(st.lists(st.text(min_size=1, max_size=20), max_size=5)),
        embedding=draw(st.one_of(st.none(), embeddings())),
        metadata=draw(st.dictionaries(
            st.text(min_size=1, max_size=20, alphabet=st.characters(blacklist_categories=('Cs',))),
            st.text(min_size=1, max_size=50, alphabet=st.characters(blacklist_categories=('Cs',))),
            max_size=5
        ))
    )


# ===== LIST STRATEGIES =====
percept_lists = st.lists(percepts(), min_size=0, max_size=15)  # Reduced for performance
goal_lists = st.lists(goals(), min_size=0, max_size=8)
memory_lists = st.lists(memories(), min_size=0, max_size=10)
