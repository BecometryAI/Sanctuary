#!/usr/bin/env python3
"""
Demonstration of Cue-Dependent Memory Retrieval

This script demonstrates how the cue-dependent retrieval system works:
1. Retrieval based on workspace state as cues
2. Spreading activation to associated memories
3. Emotional congruence biasing retrieval
4. Recency weighting with decay
5. Competitive retrieval with interference
6. Retrieval strengthening (use it or lose it)

Usage:
    python demo_cue_dependent_retrieval.py
"""

import sys
import math
from datetime import datetime, timedelta
from pathlib import Path

print("="*70)
print("CUE-DEPENDENT MEMORY RETRIEVAL DEMONSTRATION")
print("="*70)
print()

# ============================================================================
# PART 1: Emotional Congruence
# ============================================================================
print("PART 1: Emotional Congruence (PAD-based)")
print("-" * 70)
print()
print("Memories encoded in similar emotional states are easier to retrieve.")
print("Using Euclidean distance in PAD (Pleasure-Arousal-Dominance) space.")
print()

def emotional_congruence_pad(current_state, memory_state):
    """Calculate emotional congruence using PAD model."""
    if not memory_state or not current_state:
        return 0.5
    
    current_pleasure = current_state.get('valence', 0.0)
    current_arousal = current_state.get('arousal', 0.0)
    current_dominance = current_state.get('dominance', 0.0)
    
    memory_pleasure = memory_state.get('valence', 0.0)
    memory_arousal = memory_state.get('arousal', 0.0)
    memory_dominance = memory_state.get('dominance', 0.0)
    
    distance = math.sqrt(
        (current_pleasure - memory_pleasure) ** 2 +
        (current_arousal - memory_arousal) ** 2 +
        (current_dominance - memory_dominance) ** 2
    )
    
    max_distance = math.sqrt(6.0)
    congruence = 1.0 - (distance / max_distance)
    return max(0.0, min(1.0, congruence))

# Current emotional state: Happy and excited (joy)
current_state = {
    'valence': 0.8,    # High pleasure
    'arousal': 0.7,    # High arousal
    'dominance': 0.7   # High control
}

print("Current emotional state (JOY):")
print(f"  Valence (pleasure):   {current_state['valence']:.2f}")
print(f"  Arousal (activation): {current_state['arousal']:.2f}")
print(f"  Dominance (control):  {current_state['dominance']:.2f}")
print()

# Test memories with different emotional states
test_memories = [
    {
        'content': 'Had a great conversation with a friend',
        'emotional_state': {'valence': 0.85, 'arousal': 0.65, 'dominance': 0.75},
        'label': 'Similar Joy'
    },
    {
        'content': 'Felt anxious about the presentation',
        'emotional_state': {'valence': -0.6, 'arousal': 0.8, 'dominance': 0.2},
        'label': 'Fear/Anxiety'
    },
    {
        'content': 'Relaxed evening reading a book',
        'emotional_state': {'valence': 0.5, 'arousal': 0.2, 'dominance': 0.6},
        'label': 'Contentment'
    },
    {
        'content': 'Feeling sad after hearing bad news',
        'emotional_state': {'valence': -0.7, 'arousal': 0.3, 'dominance': 0.3},
        'label': 'Sadness'
    }
]

print("Emotional congruence with different memories:")
for memory in test_memories:
    congruence = emotional_congruence_pad(current_state, memory['emotional_state'])
    print(f"  {memory['label']:20} → {congruence:.3f} ({'HIGH' if congruence > 0.8 else 'MEDIUM' if congruence > 0.5 else 'LOW':6})")

print()
print("✓ Memories with similar emotional states have higher congruence!")
print()

# ============================================================================
# PART 2: Recency Weighting with Decay
# ============================================================================
print("PART 2: Recency Weighting with Exponential Decay")
print("-" * 70)
print()
print("More recent memories are easier to retrieve.")
print("Using exponential decay: weight = e^(-λ * age)")
print()

def recency_weight(last_accessed_str):
    """Calculate recency weight with exponential decay."""
    if not last_accessed_str:
        return 0.3
    
    try:
        last_accessed = datetime.fromisoformat(last_accessed_str)
        now = datetime.now()
        age_seconds = (now - last_accessed).total_seconds()
        age_hours = age_seconds / 3600.0
        decay_rate = 0.01  # Half-life ≈ 69 hours
        recency = math.exp(-decay_rate * age_hours)
        return recency
    except:
        return 0.3

# Test memories with different ages
age_tests = [
    (timedelta(minutes=30), "30 minutes ago"),
    (timedelta(hours=2), "2 hours ago"),
    (timedelta(hours=24), "1 day ago"),
    (timedelta(days=7), "1 week ago"),
    (timedelta(days=30), "1 month ago"),
    (timedelta(days=90), "3 months ago")
]

print("Recency weights for memories of different ages:")
for age_delta, label in age_tests:
    timestamp = (datetime.now() - age_delta).isoformat()
    weight = recency_weight(timestamp)
    print(f"  {label:20} → {weight:.4f} {'█' * int(weight * 40)}")

print()
print("✓ Recent memories have higher weights and decay exponentially!")
print()

# ============================================================================
# PART 3: Combined Activation
# ============================================================================
print("PART 3: Combined Activation")
print("-" * 70)
print()
print("Memory activation combines:")
print("  - Embedding similarity (cue match):   50%")
print("  - Recency (when accessed):            20%")
print("  - Emotional congruence (mood match):  30%")
print()

# Simulate a memory retrieval scenario
workspace_cue = "discussing emotions and feelings with someone"
current_emotion = {'valence': 0.7, 'arousal': 0.6, 'dominance': 0.7}

candidate_memories = [
    {
        'content': 'Had a deep conversation about emotions',
        'similarity': 0.85,  # High semantic similarity to cue
        'timestamp': (datetime.now() - timedelta(hours=2)).isoformat(),
        'emotional_state': {'valence': 0.75, 'arousal': 0.6, 'dominance': 0.65}
    },
    {
        'content': 'Read an article about emotional intelligence',
        'similarity': 0.65,  # Medium similarity
        'timestamp': (datetime.now() - timedelta(days=7)).isoformat(),
        'emotional_state': {'valence': 0.5, 'arousal': 0.4, 'dominance': 0.6}
    },
    {
        'content': 'Watched a movie that made me emotional',
        'similarity': 0.45,  # Lower similarity
        'timestamp': (datetime.now() - timedelta(hours=12)).isoformat(),
        'emotional_state': {'valence': 0.8, 'arousal': 0.7, 'dominance': 0.6}
    }
]

print(f"Workspace cue: '{workspace_cue}'")
print(f"Current emotion: Valence={current_emotion['valence']:.2f}, "
      f"Arousal={current_emotion['arousal']:.2f}, Dominance={current_emotion['dominance']:.2f}")
print()

print("Computing activations for candidate memories:")
activations = []
for i, memory in enumerate(candidate_memories, 1):
    similarity = memory['similarity']
    recency = recency_weight(memory['timestamp'])
    emotional_match = emotional_congruence_pad(current_emotion, memory['emotional_state'])
    
    activation = (
        similarity * 0.5 +
        recency * 0.2 +
        emotional_match * 0.3
    )
    
    activations.append(activation)
    
    print(f"\nMemory {i}: '{memory['content'][:50]}...'")
    print(f"  Similarity:   {similarity:.3f} × 0.5 = {similarity * 0.5:.3f}")
    print(f"  Recency:      {recency:.3f} × 0.2 = {recency * 0.2:.3f}")
    print(f"  Emotional:    {emotional_match:.3f} × 0.3 = {emotional_match * 0.3:.3f}")
    print(f"  → ACTIVATION: {activation:.3f} {'█' * int(activation * 40)}")

print()
print("✓ Activation combines multiple factors for realistic retrieval!")
print()

# ============================================================================
# PART 4: Spreading Activation
# ============================================================================
print("PART 4: Spreading Activation")
print("-" * 70)
print()
print("Highly activated memories spread activation to associated memories.")
print("Association strength determines how much activation spreads.")
print()

def spread_activation(initial_activations, associations, spread_factor=0.3, iterations=2):
    """Simulate spreading activation."""
    activations = initial_activations.copy()
    
    for iteration in range(iterations):
        new_activations = activations.copy()
        
        for mem_id, activation in activations.items():
            if activation < 0.3:  # Threshold
                continue
            
            # Get associations
            if mem_id in associations:
                for assoc_id, strength in associations[mem_id]:
                    spread = activation * strength * spread_factor
                    new_activations[assoc_id] = max(
                        new_activations.get(assoc_id, 0.0),
                        new_activations.get(assoc_id, 0.0) + spread
                    )
        
        activations = new_activations
    
    return activations

# Initial activations
initial = {
    'mem1': 0.85,  # Highly activated
    'mem2': 0.65,
    'mem3': 0.45,
    'mem4': 0.0,   # Not initially activated
    'mem5': 0.0
}

# Associations (id -> [(associated_id, strength)])
associations = {
    'mem1': [('mem4', 0.8)],    # mem1 strongly associated with mem4
    'mem2': [('mem5', 0.5)],    # mem2 moderately associated with mem5
    'mem3': []                   # mem3 has no associations
}

print("Initial activations:")
for mem_id, activation in sorted(initial.items()):
    print(f"  {mem_id}: {activation:.3f} {'█' * int(activation * 40)}")

print("\nAssociations:")
for mem_id, assocs in associations.items():
    if assocs:
        for assoc_id, strength in assocs:
            print(f"  {mem_id} → {assoc_id} (strength={strength:.2f})")

print("\nAfter spreading activation:")
spread = spread_activation(initial, associations)
for mem_id, activation in sorted(spread.items()):
    change = activation - initial.get(mem_id, 0.0)
    change_str = f"(+{change:.3f})" if change > 0 else ""
    print(f"  {mem_id}: {activation:.3f} {'█' * int(activation * 40)} {change_str}")

print()
print("✓ Activation spreads to associated memories!")
print()

# ============================================================================
# PART 5: Competitive Retrieval
# ============================================================================
print("PART 5: Competitive Retrieval with Interference")
print("-" * 70)
print()
print("Similar memories compete for limited retrieval slots.")
print("Once a memory is retrieved, it inhibits similar memories.")
print()

def competitive_retrieval(activations, similarities, limit=3, threshold=0.3, inhibition=0.4):
    """Simulate competitive retrieval."""
    retrieved = []
    remaining = dict(activations)
    
    while len(retrieved) < limit and remaining:
        # Get highest activation
        if not remaining:
            break
        
        best_id = max(remaining, key=remaining.get)
        best_activation = remaining[best_id]
        
        if best_activation < threshold:
            break
        
        retrieved.append((best_id, best_activation))
        del remaining[best_id]
        
        # Inhibit similar memories
        for mem_id in list(remaining.keys()):
            similarity = similarities.get((best_id, mem_id), 0.2)
            remaining[mem_id] -= similarity * inhibition
    
    return retrieved

# Test competitive retrieval
test_activations = {
    'mem_a': 0.85,
    'mem_b': 0.80,  # Very similar to mem_a
    'mem_c': 0.75,  # Similar to mem_a
    'mem_d': 0.55,  # Different topic
    'mem_e': 0.25   # Below threshold
}

# Similarity between memories
test_similarities = {
    ('mem_a', 'mem_b'): 0.9,  # Very similar
    ('mem_a', 'mem_c'): 0.7,  # Similar
    ('mem_a', 'mem_d'): 0.2,  # Different
    ('mem_b', 'mem_c'): 0.8,  # Similar
    ('mem_b', 'mem_d'): 0.2,
    ('mem_c', 'mem_d'): 0.3,
}

print("Memory activations (before competition):")
for mem_id, activation in sorted(test_activations.items(), key=lambda x: x[1], reverse=True):
    print(f"  {mem_id}: {activation:.3f} {'█' * int(activation * 40)}")

print("\nMemory similarities:")
print("  mem_a ↔ mem_b: 0.9 (very similar)")
print("  mem_a ↔ mem_c: 0.7 (similar)")
print("  mem_a ↔ mem_d: 0.2 (different)")

print("\nAfter competitive retrieval (limit=3):")
retrieved = competitive_retrieval(test_activations, test_similarities, limit=3)
for i, (mem_id, activation) in enumerate(retrieved, 1):
    print(f"  #{i}: {mem_id} (activation={activation:.3f})")

print()
print("✓ Similar memories compete - diverse memories are retrieved!")
print()

# ============================================================================
# PART 6: Retrieval Strengthening
# ============================================================================
print("PART 6: Retrieval Strengthening (Use It or Lose It)")
print("-" * 70)
print()
print("Successfully retrieved memories become easier to retrieve next time.")
print("This simulates the 'use it or lose it' principle of memory.")
print()

# Simulate multiple retrieval cycles
memory_state = {
    'mem1': {'retrieval_count': 0, 'base_activation': 0.5},
    'mem2': {'retrieval_count': 0, 'base_activation': 0.5},
    'mem3': {'retrieval_count': 0, 'base_activation': 0.5}
}

strengthening_factor = 0.05

print("Initial state (all memories equal):")
for mem_id, state in memory_state.items():
    print(f"  {mem_id}: retrieval_count={state['retrieval_count']}, "
          f"base_activation={state['base_activation']:.2f}")

print("\nSimulating 5 retrieval cycles where mem1 is often retrieved...")
for cycle in range(1, 6):
    # In this simulation, mem1 is retrieved more often
    if cycle <= 4:
        retrieved = ['mem1']
        if cycle % 2 == 0:
            retrieved.append('mem2')
    else:
        retrieved = ['mem2']
    
    # Strengthen retrieved memories
    for mem_id in retrieved:
        memory_state[mem_id]['retrieval_count'] += 1
        memory_state[mem_id]['base_activation'] += strengthening_factor
    
    print(f"\nCycle {cycle} - Retrieved: {', '.join(retrieved)}")

print("\nFinal state:")
for mem_id, state in sorted(memory_state.items(), 
                            key=lambda x: x[1]['base_activation'], 
                            reverse=True):
    print(f"  {mem_id}: retrieval_count={state['retrieval_count']}, "
          f"base_activation={state['base_activation']:.2f} "
          f"{'█' * int(state['base_activation'] * 40)}")

print()
print("✓ Frequently retrieved memories get stronger over time!")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*70)
print("SUMMARY: Cue-Dependent Memory Retrieval")
print("="*70)
print()
print("✓ Emotional Congruence: Memories matching current mood are easier to retrieve")
print("✓ Recency Weighting: Recent memories have higher activation")
print("✓ Combined Activation: Multiple factors contribute to retrieval")
print("✓ Spreading Activation: Activation spreads to associated memories")
print("✓ Competitive Retrieval: Similar memories compete, diverse ones retrieved")
print("✓ Retrieval Strengthening: Used memories become stronger")
print()
print("This creates a cognitively realistic memory system where retrieval")
print("depends on current context, not just semantic similarity!")
print("="*70)
