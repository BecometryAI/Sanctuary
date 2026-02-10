"""
Manual verification script for ConversationManager.

This script performs basic verification of the ConversationManager implementation
without requiring a full test environment. It checks imports, instantiation,
and basic structure.
"""

import sys
from pathlib import Path

# Add sanctuary to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from mind.cognitive_core.conversation import ConversationManager, ConversationTurn
        print("✅ ConversationManager imported")
        print("✅ ConversationTurn imported")
        
        from mind.client import SanctuaryAPI, Sanctuary
        print("✅ SanctuaryAPI imported")
        print("✅ Sanctuary imported")

        from sanctuary import SanctuaryAPI as API2, Sanctuary as Sanctuary2
        print("✅ Imports from sanctuary package work")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_conversation_turn_structure():
    """Test ConversationTurn structure."""
    print("\nTesting ConversationTurn structure...")
    
    try:
        from mind.cognitive_core.conversation import ConversationTurn
        from datetime import datetime
        
        # Create a sample turn
        turn = ConversationTurn(
            user_input="Hello",
            system_response="Hi there!",
            timestamp=datetime.now(),
            response_time=0.5,
            emotional_state={"valence": 0.7, "arousal": 0.5},
            metadata={"turn_number": 1}
        )
        
        # Verify attributes
        assert turn.user_input == "Hello"
        assert turn.system_response == "Hi there!"
        assert isinstance(turn.timestamp, datetime)
        assert turn.response_time == 0.5
        assert turn.emotional_state["valence"] == 0.7
        assert turn.metadata["turn_number"] == 1
        
        print("✅ ConversationTurn structure is correct")
        return True
    except Exception as e:
        print(f"❌ ConversationTurn test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_conversation_manager_instantiation():
    """Test ConversationManager can be instantiated."""
    print("\nTesting ConversationManager instantiation...")
    
    try:
        from mind.cognitive_core.core import CognitiveCore
        from mind.cognitive_core.conversation import ConversationManager
        
        # Create core
        core = CognitiveCore()
        print("✅ CognitiveCore created")
        
        # Create manager
        manager = ConversationManager(core)
        print("✅ ConversationManager created")
        
        # Verify initial state
        assert manager.turn_count == 0
        assert len(manager.conversation_history) == 0
        assert len(manager.current_topics) == 0
        assert manager.response_timeout == 10.0
        print("✅ Initial state is correct")
        
        # Test with custom config
        config = {"response_timeout": 15.0, "max_history_size": 50}
        manager2 = ConversationManager(core, config)
        assert manager2.response_timeout == 15.0
        assert manager2.conversation_history.maxlen == 50
        print("✅ Custom config works")
        
        return True
    except Exception as e:
        print(f"❌ ConversationManager instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_topic_extraction():
    """Test topic extraction helper method."""
    print("\nTesting topic extraction...")
    
    try:
        from mind.cognitive_core.core import CognitiveCore
        from mind.cognitive_core.conversation import ConversationManager
        
        core = CognitiveCore()
        manager = ConversationManager(core)
        
        # Test basic extraction
        topics = manager._extract_topics("Let's discuss quantum physics and relativity")
        print(f"   Extracted topics: {topics}")
        assert isinstance(topics, list)
        assert len(topics) <= 3
        
        # Test stopword filtering
        topics = manager._extract_topics("the cat and the dog")
        assert "the" not in topics
        assert "and" not in topics
        print(f"   Stopwords filtered: {topics}")
        
        print("✅ Topic extraction works correctly")
        return True
    except Exception as e:
        print(f"❌ Topic extraction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_metrics_structure():
    """Test metrics structure and methods."""
    print("\nTesting metrics...")
    
    try:
        from mind.cognitive_core.core import CognitiveCore
        from mind.cognitive_core.conversation import ConversationManager
        
        core = CognitiveCore()
        manager = ConversationManager(core)
        
        # Get initial metrics
        metrics = manager.get_metrics()
        
        # Verify structure
        assert "total_turns" in metrics
        assert "avg_response_time" in metrics
        assert "timeouts" in metrics
        assert "errors" in metrics
        assert "turn_count" in metrics
        assert "topics_tracked" in metrics
        assert "history_size" in metrics
        
        # Verify initial values
        assert metrics["total_turns"] == 0
        assert metrics["turn_count"] == 0
        assert metrics["history_size"] == 0
        
        print("✅ Metrics structure is correct")
        return True
    except Exception as e:
        print(f"❌ Metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_classes():
    """Test SanctuaryAPI and Sanctuary classes."""
    print("\nTesting API classes...")

    try:
        from mind.client import SanctuaryAPI, Sanctuary

        # Test SanctuaryAPI instantiation
        api = SanctuaryAPI()
        assert api is not None
        assert hasattr(api, 'core')
        assert hasattr(api, 'conversation')
        assert hasattr(api, 'start')
        assert hasattr(api, 'stop')
        assert hasattr(api, 'chat')
        print("✅ SanctuaryAPI instantiation works")

        # Test Sanctuary instantiation
        sanctuary = Sanctuary()
        assert sanctuary is not None
        assert hasattr(sanctuary, 'api')
        assert hasattr(sanctuary, 'loop')
        assert hasattr(sanctuary, 'start')
        assert hasattr(sanctuary, 'stop')
        assert hasattr(sanctuary, 'chat')
        print("✅ Sanctuary instantiation works")
        
        return True
    except Exception as e:
        print(f"❌ API classes test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cli_file():
    """Test that CLI file exists and has correct structure."""
    print("\nTesting CLI file...")
    
    try:
        cli_path = Path(__file__).parent / "mind" / "cli.py"
        assert cli_path.exists()
        print(f"✅ CLI file exists at {cli_path}")
        
        # Read and check for key components
        content = cli_path.read_text()
        assert "async def main()" in content
        assert "SanctuaryAPI" in content
        assert "quit" in content
        print("✅ CLI file has correct structure")
        
        return True
    except Exception as e:
        print(f"❌ CLI file test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all verification tests."""
    print("=" * 60)
    print("ConversationManager Verification Script")
    print("=" * 60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("ConversationTurn Structure", test_conversation_turn_structure()))
    results.append(("ConversationManager Instantiation", test_conversation_manager_instantiation()))
    results.append(("Topic Extraction", test_topic_extraction()))
    results.append(("Metrics Structure", test_metrics_structure()))
    results.append(("API Classes", test_api_classes()))
    results.append(("CLI File", test_cli_file()))
    
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All verification tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
