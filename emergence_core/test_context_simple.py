"""
Simplified test for context adaptation (ASCII-only output).
"""

import sys
import os
from pathlib import Path

# Disable ChromaDB telemetry to reduce noise
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lyra.consciousness import ConsciousnessCore


def print_separator(title=""):
    """Print a visual separator"""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'-'*60}\n")


def test_basic_context():
    """Test basic context adaptation"""
    print_separator("Context Adaptation - Basic Test")
    
    print("Initializing ConsciousnessCore...")
    core = ConsciousnessCore(
        memory_persistence_dir="test_mem_simple",
        context_persistence_dir="test_ctx_simple"
    )
    print("[OK] Initialized\n")
    
    # Test conversation
    messages = [
        "Hello! Tell me about memory.",
        "How does episodic memory work?",
        "Let's talk about something else - do you like pizza?",
        "What toppings do you recommend?"
    ]
    
    for i, msg in enumerate(messages, 1):
        print_separator(f"Turn {i}")
        print(f"User: {msg}\n")
        
        response = core.process_input({"message": msg})
        
        print(f"Lyra: {response.get('response', 'No response')}\n")
        
        meta = response.get("context_metadata", {})
        print("Context Info:")
        print(f"  Current Topic: {meta.get('current_topic', 'N/A')}")
        print(f"  Context Shift: {meta.get('context_shift_detected', 'N/A')}")
        print(f"  Similarity: {meta.get('similarity_to_recent', 0):.2f}")
        print(f"  Memories Used: {meta.get('memories_retrieved', 0)}")
        
    print_separator("Session Summary")
    
    summary = core.get_context_summary()
    print(f"Total Interactions: {summary['interaction_count']}")
    print(f"Topic Transitions: {summary['topic_transitions']}")
    print(f"Session Duration: {summary['session_duration_minutes']:.1f} min")
    print(f"Conversation Context Size: {summary['conversation_context_size']}")
    
    print_separator()
    print("[OK] Test Complete!")


if __name__ == "__main__":
    test_basic_context()
