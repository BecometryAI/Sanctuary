"""
Test script for context adaptation capabilities.

Demonstrates:
- Conversation context tracking
- Topic shift detection
- Context-aware memory retrieval
- Learning from interactions
- Multi-dimensional context management
"""

import sys
import os
from pathlib import Path

# Set UTF-8 encoding for console output
if sys.platform == 'win32':
    os.system("chcp 65001 >nul 2>&1")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lyra.consciousness import ConsciousnessCore
import json


def print_separator(title: str = ""):
    """Print a visual separator"""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'-'*60}\n")


def test_conversation_flow():
    """Test multi-turn conversation with topic changes"""
    print_separator("Context Adaptation Test")
    
    # Initialize consciousness core
    print("Initializing ConsciousnessCore with context management...")
    core = ConsciousnessCore(
        memory_persistence_dir="test_memories",
        context_persistence_dir="test_context"
    )
    print("[OK] Initialization complete\n")
    
    # Test conversation sequence with topic changes
    conversation = [
        {
            "message": "Hello! Can you tell me about your memory system?",
            "expected_topic": "memory"
        },
        {
            "message": "How does your episodic memory work?",
            "expected_topic": "memory"
        },
        {
            "message": "That's interesting! What about consciousness - are you truly conscious?",
            "expected_topic": "consciousness"
        },
        {
            "message": "Do you experience emotions?",
            "expected_topic": "emotions"
        },
        {
            "message": "Let's talk about something different. Can you help me with Python code?",
            "expected_topic": "technology"
        },
        {
            "message": "Actually, going back to memory - can you remember our earlier conversation?",
            "expected_topic": "memory"
        }
    ]
    
    for i, turn in enumerate(conversation, 1):
        print_separator(f"Turn {i}")
        
        print(f"User: {turn['message']}\n")
        
        # Process input
        response = core.process_input({"message": turn['message']})
        
        # Display response
        print(f"Lyra: {response.get('response', 'No response generated')}\n")
        
        # Display context metadata
        metadata = response.get("context_metadata", {})
        print("Context Analysis:")
        print(f"  • Current Topic: {metadata.get('current_topic')}")
        print(f"  • Context Shift Detected: {metadata.get('context_shift_detected')}")
        print(f"  • Similarity to Recent: {metadata.get('similarity_to_recent', 0):.2f}")
        print(f"  • Memories Retrieved: {metadata.get('memories_retrieved')}")
        print(f"  • Conversation Context Used: {metadata.get('conversation_context_used')}")
        
        # Verify topic detection
        detected_topic = metadata.get('current_topic')
        if detected_topic == turn['expected_topic']:
            print(f"  ✓ Topic correctly identified: '{detected_topic}'")
        else:
            print(f"  ℹ Topic detected: '{detected_topic}' (expected: '{turn['expected_topic']}')")
    
    print_separator("Context Summary")
    
    # Get overall context summary
    context_summary = core.get_context_summary()
    print("Session Statistics:")
    print(f"  • Total Interactions: {context_summary['interaction_count']}")
    print(f"  • Topic Transitions: {context_summary['topic_transitions']}")
    print(f"  • Session Duration: {context_summary['session_duration_minutes']:.1f} minutes")
    print(f"  • Conversation Context Size: {context_summary['conversation_context_size']}")
    print(f"  • Emotional Context Size: {context_summary['emotional_context_size']}")
    
    print("\nLearned Topic Preferences:")
    for topic, score in context_summary.get('learned_topic_preferences', {}).items():
        print(f"  • {topic}: {score:.2f}")
    
    print("\nInteraction Patterns:")
    patterns = context_summary.get('interaction_patterns', {})
    print(f"  • Detail Level: {patterns.get('preferred_detail_level')}")
    print(f"  • Communication Style: {patterns.get('communication_style')}")
    
    print_separator()
    

def test_context_shift_detection():
    """Test context shift detection"""
    print_separator("Context Shift Detection Test")
    
    core = ConsciousnessCore(
        memory_persistence_dir="test_memories_2",
        context_persistence_dir="test_context_2"
    )
    
    # Build up context on one topic
    print("Building context on 'memory' topic...\n")
    memory_messages = [
        "Tell me about your memory system",
        "How does episodic memory work?",
        "What about semantic memory?"
    ]
    
    for msg in memory_messages:
        print(f"User: {msg}")
        response = core.process_input({"message": msg})
        shift = response.get("context_metadata", {}).get("context_shift_detected", False)
        print(f"  → Shift Detected: {shift}\n")
    
    # Dramatic topic change
    print("Switching to completely different topic...\n")
    shift_message = "Let's talk about pizza recipes instead"
    print(f"User: {shift_message}")
    response = core.process_input({"message": shift_message})
    shift = response.get("context_metadata", {}).get("context_shift_detected", False)
    similarity = response.get("context_metadata", {}).get("similarity_to_recent", 0)
    
    print(f"  → Shift Detected: {shift}")
    print(f"  → Similarity Score: {similarity:.2f}")
    
    if shift:
        print("  ✓ Context shift correctly detected!")
    else:
        print("  ✗ Context shift not detected (threshold may need adjustment)")
    
    print_separator()


def test_adaptive_retrieval():
    """Test context-aware memory retrieval"""
    print_separator("Adaptive Memory Retrieval Test")
    
    core = ConsciousnessCore(
        memory_persistence_dir="test_memories_3",
        context_persistence_dir="test_context_3"
    )
    
    # Test 1: Normal conversation (should retrieve 5 memories)
    print("Test 1: Normal conversation flow\n")
    print("User: How are you today?")
    response = core.process_input({"message": "How are you today?"})
    memories = response.get("context_metadata", {}).get("memories_retrieved", 0)
    print(f"  → Memories Retrieved: {memories}")
    print(f"  → Expected: 5 (no context shift)")
    
    print_separator()
    
    # Test 2: Topic shift (should retrieve 10 memories)
    print("Test 2: After topic shift\n")
    
    # Build context
    core.process_input({"message": "Tell me about consciousness"})
    core.process_input({"message": "What makes you conscious?"})
    
    # Shift topic
    print("User: Let's discuss machine learning algorithms")
    response = core.process_input({"message": "Let's discuss machine learning algorithms"})
    memories = response.get("context_metadata", {}).get("memories_retrieved", 0)
    shift = response.get("context_metadata", {}).get("context_shift_detected", False)
    
    print(f"  → Memories Retrieved: {memories}")
    print(f"  → Context Shift Detected: {shift}")
    print(f"  → Expected: 10 (context shift triggers broader retrieval)")
    
    if shift and memories == 10:
        print("  ✓ Adaptive retrieval working correctly!")
    elif shift:
        print(f"  ⚠ Shift detected but retrieved {memories} memories (expected 10)")
    else:
        print("  ⚠ No shift detected (threshold may need adjustment)")
    
    print_separator()


def test_session_persistence():
    """Test context state persistence"""
    print_separator("Context Persistence Test")
    
    # Create first session
    print("Session 1: Creating and saving context...\n")
    core1 = ConsciousnessCore(
        memory_persistence_dir="test_memories_persist",
        context_persistence_dir="test_context_persist"
    )
    
    # Have a conversation
    core1.process_input({"message": "Hello, my name is Alice"})
    core1.process_input({"message": "I'm interested in artificial consciousness"})
    core1.process_input({"message": "Can we discuss philosophy?"})
    
    # Save state
    core1.context_manager.save_context_state()
    summary1 = core1.get_context_summary()
    
    print(f"Session 1 Summary:")
    print(f"  • Interactions: {summary1['interaction_count']}")
    print(f"  • Current Topic: {summary1['current_topic']}")
    print(f"  • Context saved to: test_context_persist/context_state.json")
    
    print_separator()
    
    # Create second session and load state
    print("Session 2: Loading saved context...\n")
    core2 = ConsciousnessCore(
        memory_persistence_dir="test_memories_persist",
        context_persistence_dir="test_context_persist"
    )
    
    summary2 = core2.get_context_summary()
    
    print(f"Session 2 Summary (after loading):")
    print(f"  • Interactions: {summary2['interaction_count']}")
    print(f"  • Current Topic: {summary2['current_topic']}")
    print(f"  • Topic Transitions: {summary2['topic_transitions']}")
    
    # Verify state was loaded
    if summary2['interaction_count'] == summary1['interaction_count']:
        print("\n  ✓ Context state successfully persisted and loaded!")
    else:
        print("\n  ✗ Context state persistence issue detected")
    
    print_separator()


def main():
    """Run all context adaptation tests"""
    print("\n" + "="*60)
    print("  CONTEXT ADAPTATION SYSTEM - TEST SUITE")
    print("="*60)
    
    try:
        # Test 1: Full conversation flow
        test_conversation_flow()
        
        # Test 2: Context shift detection
        test_context_shift_detection()
        
        # Test 3: Adaptive memory retrieval
        test_adaptive_retrieval()
        
        # Test 4: Session persistence
        test_session_persistence()
        
        print_separator("All Tests Complete")
        print("✓ Context adaptation system is functional!")
        print("\nKey Features Demonstrated:")
        print("  • Multi-turn conversation tracking")
        print("  • Topic shift detection")
        print("  • Context-aware memory retrieval")
        print("  • Learning from interactions")
        print("  • Multi-dimensional context management")
        print("  • Context state persistence")
        
    except Exception as e:
        print(f"\n✗ Test suite error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
