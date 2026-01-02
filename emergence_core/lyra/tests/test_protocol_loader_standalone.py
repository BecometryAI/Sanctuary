#!/usr/bin/env python3
"""
Standalone test for protocol loader functionality.
This script tests the protocol loader without requiring full system dependencies.
"""

import sys
import json
import tempfile
from pathlib import Path

# Add to path and import directly to avoid __init__.py chain
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import protocol_loader module directly
import importlib.util
protocol_loader_path = Path(__file__).parent.parent / "cognitive_core" / "protocol_loader.py"
spec = importlib.util.spec_from_file_location("protocol_loader", protocol_loader_path)
protocol_loader_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol_loader_module)

ProtocolLoader = protocol_loader_module.ProtocolLoader
ConstraintType = protocol_loader_module.ConstraintType


def test_protocol_loading():
    """Test basic protocol loading functionality."""
    print("=" * 70)
    print("TEST: Protocol Loading")
    print("=" * 70)
    
    # Use actual data/Protocols directory
    base_dir = Path(__file__).parent.parent.parent.parent
    protocol_dir = base_dir / "data" / "Protocols"
    
    if not protocol_dir.exists():
        print(f"❌ Protocol directory not found: {protocol_dir}")
        return False
    
    loader = ProtocolLoader(protocol_dir=protocol_dir)
    protocols = loader.load_protocols()
    
    print(f"\n✅ Loaded {len(protocols)} protocols")
    
    # Display loaded protocols
    for protocol_id, protocol in protocols.items():
        print(f"\n  📋 {protocol.title} ({protocol_id})")
        print(f"     Status: {protocol.status}")
        print(f"     Constraints: {len(protocol.constraints)}")
        
        # Show first constraint if available
        if protocol.constraints:
            constraint = protocol.constraints[0]
            print(f"     First constraint type: {constraint.type.value}")
            print(f"     Description: {constraint.description[:80]}...")
    
    return len(protocols) > 0


def test_constraint_extraction():
    """Test constraint extraction from protocols."""
    print("\n" + "=" * 70)
    print("TEST: Constraint Extraction")
    print("=" * 70)
    
    base_dir = Path(__file__).parent.parent.parent.parent
    protocol_dir = base_dir / "data" / "Protocols"
    
    loader = ProtocolLoader(protocol_dir=protocol_dir)
    loader.load_protocols()
    
    # Collect all constraints
    all_constraints = []
    for protocol in loader.protocols.values():
        all_constraints.extend(protocol.constraints)
    
    print(f"\n✅ Extracted {len(all_constraints)} total constraints")
    
    # Count by type
    type_counts = {}
    for constraint in all_constraints:
        type_str = constraint.type.value
        type_counts[type_str] = type_counts.get(type_str, 0) + 1
    
    print("\n  Constraint types:")
    for ctype, count in type_counts.items():
        print(f"    - {ctype}: {count}")
    
    return len(all_constraints) > 0


def test_action_specific_constraints():
    """Test getting constraints for specific actions."""
    print("\n" + "=" * 70)
    print("TEST: Action-Specific Constraints")
    print("=" * 70)
    
    base_dir = Path(__file__).parent.parent.parent.parent
    protocol_dir = base_dir / "data" / "Protocols"
    
    loader = ProtocolLoader(protocol_dir=protocol_dir)
    loader.load_protocols()
    
    action_types = ["speak", "commit_memory", "retrieve_memory", "introspect"]
    
    for action_type in action_types:
        constraints = loader.get_constraints_for_action(action_type)
        print(f"\n  {action_type}: {len(constraints)} applicable constraints")
        
        # Show one example if available
        if constraints:
            example = constraints[0]
            print(f"    Example: {example.protocol_title}")
            print(f"    Type: {example.type.value}")
    
    return True


def test_temp_protocol_directory():
    """Test with temporary protocol directory."""
    print("\n" + "=" * 70)
    print("TEST: Temporary Protocol Directory")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        protocol_dir = Path(tmpdir)
        
        # Create test protocol
        test_protocol = {
            "protocol_draft": {
                "protocol_id": "TEST-01",
                "title": "Test Protocol",
                "status": "Active",
                "purpose": "Testing protocol loading",
                "directive": {
                    "directive": "The system must always be honest and never fabricate information."
                }
            }
        }
        
        with open(protocol_dir / "test_protocol.json", 'w') as f:
            json.dump(test_protocol, f)
        
        # Load protocols
        loader = ProtocolLoader(protocol_dir=protocol_dir)
        protocols = loader.load_protocols()
        
        print(f"\n✅ Created and loaded test protocol")
        print(f"  Protocol ID: TEST-01")
        print(f"  Constraints extracted: {len(protocols['TEST-01'].constraints)}")
        
        return "TEST-01" in protocols


def test_hot_reload():
    """Test protocol hot-reloading."""
    print("\n" + "=" * 70)
    print("TEST: Hot Reload")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        protocol_dir = Path(tmpdir)
        
        # Create initial protocol
        protocol1 = {
            "protocol_draft": {
                "protocol_id": "RELOAD-01",
                "title": "Initial Protocol",
                "status": "Active",
                "purpose": "Initial protocol"
            }
        }
        
        with open(protocol_dir / "protocol1.json", 'w') as f:
            json.dump(protocol1, f)
        
        # Load protocols
        loader = ProtocolLoader(protocol_dir=protocol_dir)
        loader.load_protocols()
        initial_count = len(loader.protocols)
        
        print(f"\n  Initial protocols loaded: {initial_count}")
        
        # Add new protocol
        protocol2 = {
            "protocol_draft": {
                "protocol_id": "RELOAD-02",
                "title": "New Protocol",
                "status": "Active",
                "purpose": "Added during hot reload"
            }
        }
        
        with open(protocol_dir / "protocol2.json", 'w') as f:
            json.dump(protocol2, f)
        
        # Hot reload
        new_count = loader.hot_reload()
        
        print(f"  After hot reload: {new_count}")
        print(f"  ✅ Hot reload successful: {initial_count} → {new_count}")
        
        return new_count == initial_count + 1


def test_constraint_type_detection():
    """Test automatic constraint type detection."""
    print("\n" + "=" * 70)
    print("TEST: Constraint Type Detection")
    print("=" * 70)
    
    loader = ProtocolLoader()
    
    test_cases = [
        ("The system must never lie or fabricate information", ConstraintType.PROHIBITION),
        ("The system must always verify facts before sharing", ConstraintType.REQUIREMENT),
        ("When uncertain, the system should express doubt", ConstraintType.CONDITIONAL),
        ("The system should strive for accuracy", ConstraintType.GUIDANCE),
    ]
    
    passed = 0
    for text, expected_type in test_cases:
        constraint = loader._create_constraint_from_text(text, "TEST", "Test", "test")
        if constraint and constraint.type == expected_type:
            print(f"  ✅ '{text[:50]}...' → {expected_type.value}")
            passed += 1
        else:
            actual = constraint.type.value if constraint else "None"
            print(f"  ❌ '{text[:50]}...' → Expected {expected_type.value}, got {actual}")
    
    print(f"\n  Passed {passed}/{len(test_cases)} tests")
    return passed == len(test_cases)


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("PROTOCOL LOADER TEST SUITE")
    print("=" * 70)
    
    tests = [
        ("Protocol Loading", test_protocol_loading),
        ("Constraint Extraction", test_constraint_extraction),
        ("Action-Specific Constraints", test_action_specific_constraints),
        ("Temporary Protocol Directory", test_temp_protocol_directory),
        ("Hot Reload", test_hot_reload),
        ("Constraint Type Detection", test_constraint_type_detection),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    print("=" * 70)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
