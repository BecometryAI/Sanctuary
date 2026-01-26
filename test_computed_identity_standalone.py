#!/usr/bin/env python3
"""
Standalone test for computed identity system.
Tests identity modules without requiring full framework dependencies.
"""

import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

# Direct imports avoiding full package initialization
import importlib.util

def load_module(module_name, file_path):
    """Load a module directly from file."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Load identity modules directly
computed = load_module(
    "computed",
    "emergence_core/lyra/cognitive_core/identity/computed.py"
)
behavior_logger = load_module(
    "behavior_logger",
    "emergence_core/lyra/cognitive_core/identity/behavior_logger.py"
)
continuity = load_module(
    "continuity",
    "emergence_core/lyra/cognitive_core/identity/continuity.py"
)
manager = load_module(
    "manager",
    "emergence_core/lyra/cognitive_core/identity/manager.py"
)

Identity = computed.Identity
BehaviorLogger = behavior_logger.BehaviorLogger
IdentityContinuity = continuity.IdentityContinuity
IdentityManager = manager.IdentityManager

def test_identity_creation():
    """Test creating an Identity object."""
    identity = Identity(
        core_values=['Truthfulness', 'Curiosity'],
        emotional_disposition={'valence': 0.5, 'arousal': 0.3, 'dominance': 0.6},
        autobiographical_self=[{'id': 'mem1', 'content': 'test'}],
        behavioral_tendencies={'tendency_speak': 0.6},
        source='computed'
    )
    
    assert len(identity.core_values) == 2
    assert identity.source == 'computed'
    print('✓ Identity creation test passed')

def test_empty_identity():
    """Test creating an empty identity."""
    identity = Identity.empty()
    
    assert identity.core_values == []
    assert identity.source == 'empty'
    print('✓ Empty identity test passed')

def test_behavior_logger():
    """Test BehaviorLogger."""
    logger = BehaviorLogger(max_history=100)
    
    action = {'type': 'speak', 'priority': 0.8, 'reason': 'respond to user'}
    logger.log_action(action)
    
    assert len(logger.action_history) == 1
    assert logger.action_history[0]['type'] == 'speak'
    print('✓ BehaviorLogger test passed')

def test_behavior_tendencies():
    """Test behavioral tendency analysis."""
    logger = BehaviorLogger()
    
    # Log mixed actions
    for _ in range(5):
        logger.log_action({'type': 'speak', 'priority': 0.7})
    for _ in range(3):
        logger.log_action({'type': 'introspect', 'priority': 0.5})
    for _ in range(2):
        logger.log_action({'type': 'wait', 'priority': 0.3})
    
    tendencies = logger.analyze_tendencies()
    
    assert 'tendency_speak' in tendencies
    assert tendencies['tendency_speak'] == 0.5  # 5 out of 10
    assert 'average_urgency' in tendencies
    print('✓ Behavioral tendencies test passed')

def test_identity_continuity():
    """Test IdentityContinuity tracking."""
    cont = IdentityContinuity(max_snapshots=50)
    
    identity = Identity(
        core_values=['Truthfulness'],
        emotional_disposition={'valence': 0.5, 'arousal': 0.3, 'dominance': 0.6},
        autobiographical_self=[],
        behavioral_tendencies={}
    )
    
    cont.take_snapshot(identity)
    
    assert len(cont.snapshots) == 1
    assert cont.snapshots[0].core_values == ['Truthfulness']
    print('✓ IdentityContinuity test passed')

def test_continuity_score_stable():
    """Test continuity score with stable identity."""
    cont = IdentityContinuity()
    
    # Take multiple snapshots with same values
    for _ in range(5):
        identity = Identity(
            core_values=['Truthfulness', 'Curiosity'],
            emotional_disposition={'valence': 0.5, 'arousal': 0.3, 'dominance': 0.6},
            autobiographical_self=[],
            behavioral_tendencies={}
        )
        cont.take_snapshot(identity)
    
    score = cont.get_continuity_score()
    
    # Should be very high (near 1.0) for stable identity
    assert score > 0.9
    print(f'✓ Continuity score test passed (score: {score:.3f})')

def test_identity_manager():
    """Test IdentityManager."""
    mgr = IdentityManager()
    
    assert mgr.computed is None
    assert isinstance(mgr.continuity, IdentityContinuity)
    assert isinstance(mgr.behavior_log, BehaviorLogger)
    
    # Log an action
    mgr.log_action({'type': 'speak', 'priority': 0.7})
    history = mgr.behavior_log.get_action_history()
    assert len(history) == 1
    
    print('✓ IdentityManager test passed')

def test_identity_introspection():
    """Test identity introspection."""
    mgr = IdentityManager()
    
    description = mgr.introspect_identity()
    
    assert isinstance(description, str)
    assert 'memories' in description.lower() or 'behavioral' in description.lower()
    print('✓ Identity introspection test passed')

def main():
    """Run all tests."""
    print("Testing Computed Identity System\n")
    print("=" * 50)
    
    try:
        test_identity_creation()
        test_empty_identity()
        test_behavior_logger()
        test_behavior_tendencies()
        test_identity_continuity()
        test_continuity_score_stable()
        test_identity_manager()
        test_identity_introspection()
        
        print("=" * 50)
        print("\n✅ All tests passed!")
        return 0
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
