"""
System validation script - checks all subsystems.

Usage:
    python scripts/validate_system.py
"""

import asyncio
from sanctuary import SanctuaryAPI


async def validate_system():
    """Run system validation checks."""
    print("🔬 Sanctuary System Validation\n")
    print("=" * 60)
    
    results = []
    
    # Test 1: Basic initialization
    print("\n1. Testing system initialization...")
    try:
        api = SanctuaryAPI()
        await api.start()
        print("   ✅ System initialized successfully")
        results.append(("Initialization", True))
    except Exception as e:
        print(f"   ❌ Initialization failed: {e}")
        results.append(("Initialization", False))
        return results
    
    # Test 2: Single conversation turn
    print("\n2. Testing single conversation turn...")
    try:
        turn = await api.chat("Hello!")
        assert len(turn.system_response) > 0
        print(f"   ✅ Response: {turn.system_response[:50]}...")
        results.append(("Single Turn", True))
    except Exception as e:
        print(f"   ❌ Single turn failed: {e}")
        results.append(("Single Turn", False))
    
    # Test 3: Emotional state
    print("\n3. Testing emotional state...")
    try:
        turn = await api.chat("How are you feeling?")
        assert "valence" in turn.emotional_state
        print(f"   ✅ Emotion: V={turn.emotional_state['valence']:.2f}")
        results.append(("Emotional State", True))
    except Exception as e:
        print(f"   ❌ Emotional state check failed: {e}")
        results.append(("Emotional State", False))
    
    # Test 4: Multi-turn coherence
    print("\n4. Testing multi-turn coherence...")
    try:
        await api.chat("My name is Test User.")
        turn = await api.chat("What's my name?")
        print(f"   ✅ Response: {turn.system_response[:50]}...")
        results.append(("Multi-Turn", True))
    except Exception as e:
        print(f"   ❌ Multi-turn failed: {e}")
        results.append(("Multi-Turn", False))
    
    # Test 5: Metrics collection
    print("\n5. Testing metrics collection...")
    try:
        metrics = api.get_metrics()
        assert "conversation" in metrics
        print(f"   ✅ Metrics: {metrics['conversation']['total_turns']} turns")
        results.append(("Metrics", True))
    except Exception as e:
        print(f"   ❌ Metrics failed: {e}")
        results.append(("Metrics", False))
    
    await api.stop()
    
    # Summary
    print("\n" + "=" * 60)
    print("Validation Summary:")
    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"\nPassed: {passed}/{total}")
    
    for test, result in results:
        status = "✅" if result else "❌"
        print(f"  {status} {test}")
    
    return results


if __name__ == "__main__":
    asyncio.run(validate_system())
