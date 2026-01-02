#!/usr/bin/env python
"""
Simple validation script for identity system.
Validates that the identity files can be loaded and parsed correctly.
"""

import sys
import yaml
import re
from pathlib import Path


def validate_charter():
    """Validate charter.md structure."""
    print("Validating charter.md...")
    
    charter_path = Path("data/identity/charter.md")
    if not charter_path.exists():
        print("   ❌ charter.md not found")
        return False
    
    text = charter_path.read_text()
    
    # Check for required sections
    required_sections = ["Core Values", "Purpose Statement", "Behavioral"]
    missing = []
    for section in required_sections:
        if section not in text:
            missing.append(section)
    
    if missing:
        print(f"   ❌ Missing sections: {missing}")
        return False
    
    # Extract core values
    core_values_count = len([line for line in text.split('\n') 
                            if line.strip().startswith('-') or line.strip().startswith('*')])
    
    print(f"   ✅ Charter valid")
    print(f"   ✅ Found {core_values_count} bullet points total")
    return True


def validate_protocols():
    """Validate protocols.md structure and YAML."""
    print("\nValidating protocols.md...")
    
    protocols_path = Path("data/identity/protocols.md")
    if not protocols_path.exists():
        print("   ❌ protocols.md not found")
        return False
    
    text = protocols_path.read_text()
    
    # Extract YAML block
    pattern = r'```yaml\n(.*?)\n```'
    match = re.search(pattern, text, re.DOTALL)
    
    if not match:
        print("   ❌ No YAML block found")
        return False
    
    yaml_content = match.group(1)
    
    # Try to parse YAML
    try:
        protocols = yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        print(f"   ❌ YAML parsing error: {e}")
        return False
    
    if not isinstance(protocols, list):
        print("   ❌ Protocols should be a list")
        return False
    
    if len(protocols) == 0:
        print("   ❌ No protocols defined")
        return False
    
    # Validate protocol structure
    required_fields = ['name', 'description', 'priority']
    for i, proto in enumerate(protocols):
        for field in required_fields:
            if field not in proto:
                print(f"   ❌ Protocol {i} missing field: {field}")
                return False
        
        # Validate priority range
        if not (0.0 <= proto['priority'] <= 1.0):
            print(f"   ❌ Protocol {i} has invalid priority: {proto['priority']}")
            return False
    
    print(f"   ✅ Protocols valid")
    print(f"   ✅ Found {len(protocols)} protocols")
    for proto in protocols:
        print(f"      - {proto['name']} (priority: {proto['priority']})")
    
    return True


def main():
    """Run all validations."""
    print("=" * 60)
    print("Identity System Validation")
    print("=" * 60)
    
    results = []
    
    # Validate charter
    results.append(validate_charter())
    
    # Validate protocols
    results.append(validate_protocols())
    
    # Summary
    print("\n" + "=" * 60)
    if all(results):
        print("✅ All identity files validated successfully!")
        print("=" * 60)
        return 0
    else:
        print("❌ Some validations failed")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
