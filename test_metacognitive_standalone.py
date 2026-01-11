#!/usr/bin/env python3
"""
Minimal standalone test for meta-cognitive capabilities.
This test doesn't require full Lyra dependencies.
"""

import sys
from pathlib import Path

# Add emergence_core to path
sys.path.insert(0, str(Path(__file__).parent / "emergence_core"))

# Test 1: Import modules
print("Test 1: Importing modules...")
try:
    from lyra.cognitive_core.meta_cognition.processing_monitor import (
        MetaCognitiveMonitor,
        ProcessingObservation,
        CognitiveResources,
    )
    from lyra.cognitive_core.meta_cognition.action_learning import (
        ActionOutcomeLearner,
        ActionOutcome,
    )
    from lyra.cognitive_core.meta_cognition.attention_history import (
        AttentionHistory,
        AttentionAllocation,
    )
    from lyra.cognitive_core.meta_cognition.system import (
        MetaCognitiveSystem,
        SelfAssessment,
    )
    print("✅ All imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Create instances
print("\nTest 2: Creating instances...")
try:
    monitor = MetaCognitiveMonitor()
    learner = ActionOutcomeLearner()
    history = AttentionHistory()
    system = MetaCognitiveSystem()
    print("✅ All instances created successfully")
except Exception as e:
    print(f"❌ Instance creation failed: {e}")
    sys.exit(1)

# Test 3: Basic monitoring
print("\nTest 3: Testing processing monitor...")
try:
    with monitor.observe("test_process") as ctx:
        ctx.set_complexity(0.7)
        ctx.set_quality(0.8)
        # Simulate some work
        result = 2 + 2
    
    assert len(monitor.observations) == 1
    obs = monitor.observations[0]
    assert obs.process_type == "test_process"
    assert obs.success is True
    assert obs.input_complexity == 0.7
    assert obs.output_quality == 0.8
    print(f"✅ Monitor test passed: {obs.duration_ms:.2f}ms")
except Exception as e:
    print(f"❌ Monitor test failed: {e}")
    sys.exit(1)

# Test 4: Action learning
print("\nTest 4: Testing action learner...")
try:
    learner.record_outcome(
        action_id="test_action_1",
        action_type="speak",
        intended="say hello",
        actual="said hello",
        context={"test": True}
    )
    
    assert len(learner.outcomes) == 1
    outcome = learner.outcomes[0]
    assert outcome.action_type == "speak"
    assert outcome.success is True
    
    reliability = learner.get_action_reliability("speak")
    assert reliability.success_rate == 1.0
    print(f"✅ Action learner test passed: success_rate={reliability.success_rate}")
except Exception as e:
    print(f"❌ Action learner test failed: {e}")
    sys.exit(1)

# Test 5: Attention history
print("\nTest 5: Testing attention history...")
try:
    class MockState:
        pass
    
    allocation_id = history.record_allocation(
        allocation={"goal1": 0.6, "goal2": 0.4},
        trigger="test_trigger",
        workspace_state=MockState()
    )
    
    assert len(history.allocations) == 1
    
    history.record_outcome(
        allocation_id=allocation_id,
        goal_progress={"goal1": 0.5},
        discoveries=["insight1"],
        missed=[]
    )
    
    assert allocation_id in history.outcomes
    outcome = history.outcomes[allocation_id]
    print(f"✅ Attention history test passed: efficiency={outcome.efficiency:.2f}")
except Exception as e:
    print(f"❌ Attention history test failed: {e}")
    sys.exit(1)

# Test 6: Unified system
print("\nTest 6: Testing unified meta-cognitive system...")
try:
    # Add some data
    with system.monitor.observe("reasoning") as ctx:
        ctx.set_complexity(0.5)
        ctx.set_quality(0.7)
    
    system.action_learner.record_outcome(
        action_id="action1",
        action_type="test",
        intended="test",
        actual="test done",
        context={}
    )
    
    class MockGoal:
        def __init__(self):
            self.id = "goal1"
            self.priority = 0.8
    
    allocation_id = system.attention_history.record_allocation(
        allocation={"goal1": 1.0},
        trigger="test",
        workspace_state=MockState()
    )
    system.attention_history.record_outcome(
        allocation_id=allocation_id,
        goal_progress={"goal1": 0.8},
        discoveries=[],
        missed=[]
    )
    
    # Get self-assessment
    assessment = system.get_self_assessment()
    assert isinstance(assessment, SelfAssessment)
    
    # Test introspection
    response = system.introspect("What patterns have I identified?")
    assert len(response) > 0
    
    print(f"✅ Unified system test passed")
    print(f"   - Processing patterns: {len(assessment.processing_patterns)}")
    print(f"   - Action types: {len(assessment.action_reliability)}")
    print(f"   - Strengths: {len(assessment.identified_strengths)}")
    print(f"   - Weaknesses: {len(assessment.identified_weaknesses)}")
except Exception as e:
    print(f"❌ Unified system test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Pattern detection
print("\nTest 7: Testing pattern detection...")

# Pattern detection constants
PATTERN_TEST_SUCCESS_COUNT = 5
PATTERN_TEST_FAILURE_COUNT = 5

try:
    # Create patterns by recording multiple observations
    for i in range(PATTERN_TEST_SUCCESS_COUNT + PATTERN_TEST_FAILURE_COUNT):
        with monitor.observe("pattern_test") as ctx:
            # Vary complexity and quality
            ctx.set_complexity(0.3 if i < PATTERN_TEST_SUCCESS_COUNT else 0.9)
            ctx.set_quality(0.8)
            if i >= PATTERN_TEST_SUCCESS_COUNT:
                # Simulate failures on high complexity
                raise Exception("High complexity failure")
except Exception:
    pass

try:
    patterns = monitor.get_identified_patterns()
    print(f"✅ Pattern detection test passed: {len(patterns)} patterns identified")
    for pattern in patterns[:3]:
        print(f"   - {pattern.pattern_type}: {pattern.description[:60]}...")
except Exception as e:
    print(f"❌ Pattern detection test failed: {e}")
    sys.exit(1)

# Test 8: Summaries
print("\nTest 8: Testing summary generation...")
try:
    monitor_summary = monitor.get_summary()
    learner_summary = learner.get_summary()
    history_summary = history.get_summary()
    system_summary = system.get_monitoring_summary()
    
    print(f"✅ Summary generation test passed")
    print(f"   - Monitor observations: {monitor_summary['total_observations']}")
    print(f"   - Action outcomes: {learner_summary['total_outcomes']}")
    print(f"   - Attention allocations: {history_summary['total_allocations']}")
except Exception as e:
    print(f"❌ Summary generation test failed: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("✅ ALL TESTS PASSED!")
print("="*60)
