#!/usr/bin/env python
"""
Standalone test script for identity loader.
Tests the identity loader without requiring full framework dependencies.
"""

from pathlib import Path

from mind.cognitive_core.identity_loader import IdentityLoader, CharterDocument, ProtocolDocument


def test_identity_loader():
    """Test identity loader functionality."""
    print("=" * 60)
    print("Testing IdentityLoader")
    print("=" * 60)
    
    # Test 1: Initialize loader
    print("\n1. Testing initialization...")
    loader = IdentityLoader(Path('data/identity'))
    assert loader.identity_dir == Path('data/identity')
    assert loader.charter is None
    assert loader.protocols == []
    print("   ✅ Initialization successful")
    
    # Test 2: Load charter
    print("\n2. Testing charter loading...")
    charter = loader.load_charter()
    assert isinstance(charter, CharterDocument)
    assert len(charter.core_values) > 0
    assert charter.purpose_statement != ""
    assert len(charter.behavioral_guidelines) > 0
    print(f"   ✅ Charter loaded with {len(charter.core_values)} core values")
    print(f"   ✅ Charter has {len(charter.behavioral_guidelines)} behavioral guidelines")
    
    # Test 3: Load protocols
    print("\n3. Testing protocol loading...")
    protocols = loader.load_protocols()
    assert isinstance(protocols, list)
    assert len(protocols) > 0
    assert all(isinstance(p, ProtocolDocument) for p in protocols)
    print(f"   ✅ Loaded {len(protocols)} protocols")
    for proto in protocols[:3]:
        print(f"      - {proto.name} (priority: {proto.priority})")
    
    # Test 4: Load all
    print("\n4. Testing load_all()...")
    loader2 = IdentityLoader(Path('data/identity'))
    loader2.load_all()
    assert loader2.charter is not None
    assert len(loader2.protocols) > 0
    print("   ✅ load_all() successful")
    
    # Test 5: Get relevant protocols
    print("\n5. Testing get_relevant_protocols()...")
    context = {"emotion": "uncertain"}
    relevant = loader2.get_relevant_protocols(context)
    assert isinstance(relevant, list)
    assert len(relevant) > 0
    # Should be sorted by priority
    for i in range(len(relevant) - 1):
        assert relevant[i].priority >= relevant[i+1].priority
    print(f"   ✅ Got {len(relevant)} relevant protocols, sorted by priority")
    
    # Test 6: Charter structure
    print("\n6. Testing charter structure...")
    print(f"   Purpose: {charter.purpose_statement[:60]}...")
    print(f"   First core value: {charter.core_values[0][:60]}...")
    print(f"   First guideline: {charter.behavioral_guidelines[0][:60]}...")
    print("   ✅ Charter structure valid")
    
    # Test 7: Protocol structure
    print("\n7. Testing protocol structure...")
    if protocols:
        proto = protocols[0]
        print(f"   Name: {proto.name}")
        print(f"   Description: {proto.description[:60]}...")
        print(f"   Trigger conditions: {len(proto.trigger_conditions)}")
        print(f"   Actions: {len(proto.actions)}")
        print(f"   Priority: {proto.priority}")
        assert 0.0 <= proto.priority <= 1.0
        print("   ✅ Protocol structure valid")
    
    print("\n" + "=" * 60)
    print("✅ All identity loader tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_identity_loader()
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
