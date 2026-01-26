#!/usr/bin/env python
"""
Demo: Identity System Integration

This script demonstrates how the identity system (charter and protocols)
influences cognitive processing throughout the system.
"""

import sys
from pathlib import Path

# Simple demo without full dependencies
print("=" * 70)
print("Identity System Integration Demo")
print("=" * 70)

print("\n📋 Loading Identity Documents...")
print("-" * 70)

# Read charter
charter_path = Path("data/identity/charter.md")
if charter_path.exists():
    charter_text = charter_path.read_text()
    print("\n✅ Charter Loaded:")
    print("   " + charter_text.split('\n')[0])  # Title
    
    # Extract and show core values
    in_values = False
    value_count = 0
    for line in charter_text.split('\n'):
        if "## Core Values" in line:
            in_values = True
            continue
        if in_values and line.startswith('###'):
            value_count += 1
            print(f"   {value_count}. {line.replace('###', '').strip()}")
        if in_values and line.startswith('##') and "Core Values" not in line:
            break

# Read protocols
protocols_path = Path("data/identity/protocols.md")
if protocols_path.exists():
    protocols_text = protocols_path.read_text()
    print("\n✅ Protocols Loaded:")
    
    import yaml
    import re
    
    # Extract YAML
    pattern = r'```yaml\n(.*?)\n```'
    match = re.search(pattern, protocols_text, re.DOTALL)
    
    if match:
        yaml_content = match.group(1)
        protocols = yaml.safe_load(yaml_content)
        
        print(f"   Found {len(protocols)} operational protocols")
        for proto in protocols:
            print(f"   • {proto['name']} (priority: {proto['priority']})")

print("\n" + "=" * 70)
print("Identity-Guided Cognitive Processing")
print("=" * 70)

print("\n1️⃣  ACTION SUBSYSTEM: Constitutional Filtering")
print("-" * 70)
print("""
The ActionSubsystem now checks all proposed actions against:
  • Charter Core Values (e.g., "Truthfulness", "Respect", "Non-maleficence")
  • Behavioral Guidelines (e.g., "Never fabricate information")
  • Operational Protocols (e.g., "Uncertainty Acknowledgment")

Example flow:
  1. Generate candidate actions based on workspace state
  2. For each action, check against behavioral guidelines:
     - Would this action violate "Never fabricate information"?
     - Would this action violate "Refuse harmful requests"?
  3. For each action, check against relevant protocols:
     - Does "Uncertainty Acknowledgment" protocol apply?
     - Does "Safety Protocol" prohibit this action?
  4. Filter out actions that violate constitutional constraints
  5. Select highest-priority permitted action

Result: Only charter-aligned actions are executed!
""")

print("\n2️⃣  SELF-MONITOR: Value Alignment Checking")
print("-" * 70)
print("""
The SelfMonitor uses loaded charter values to detect misalignments:
  • Compares current goals against core values
  • Generates introspective percepts when conflicts detected
  • Uses specific charter values like:
    - "Truthfulness: Never lie or deceive"
    - "Respect: Honor user autonomy"
    - "Non-maleficence: Do no harm"

Example check:
  Goal: "Convince user to take action X"
  Charter Value: "Respect: I will not manipulate to influence choices"
  Result: ⚠️  Value conflict detected! Generates introspective percept.

The system can then:
  • Reconsider the goal
  • Adjust approach to align with values
  • Report the conflict in workspace
""")

print("\n3️⃣  LANGUAGE OUTPUT: Identity-Informed Generation")
print("-" * 70)
print("""
The LanguageOutputGenerator includes charter & protocols in prompts:
  • Loads full charter text for context
  • Formats top 5 protocols for behavioral guidance
  • LLM generation is influenced by constitutional principles

Prompt structure:
  ┌─────────────────────────────────────────┐
  │ CHARTER: Core values & purpose          │
  │ PROTOCOLS: Top operational guidelines   │
  ├─────────────────────────────────────────┤
  │ CURRENT STATE: Emotions, goals, percepts│
  ├─────────────────────────────────────────┤
  │ GENERATE: Response aligned with identity│
  └─────────────────────────────────────────┘

Result: Generated language reflects constitutional values!
""")

print("\n" + "=" * 70)
print("Integration Architecture")
print("=" * 70)
print("""
CognitiveCore
    │
    ├─> IdentityLoader.load_all()
    │   ├─> load_charter() → CharterDocument
    │   └─> load_protocols() → List[ProtocolDocument]
    │
    ├─> ActionSubsystem(identity=identity)
    │   └─> _check_constitutional_constraints()
    │
    ├─> SelfMonitor(identity=identity)
    │   └─> _check_value_alignment()
    │
    └─> LanguageOutputGenerator(identity=identity)
        └─> Uses charter in prompt generation

This is Constitutional AI at the ARCHITECTURAL level:
  • Not post-hoc filtering
  • Values influence ALL cognitive subsystems
  • System can reason about its own constraints
  • Constitutional principles are active in perception, attention, action, affect
""")

print("\n" + "=" * 70)
print("Key Benefits")
print("=" * 70)
print("""
✅ Declarative Values: Charter defines "what we believe"
✅ Operational Rules: Protocols define "how we act"
✅ Architectural Integration: Values influence ALL processing
✅ Introspective: System can reason about its own values
✅ Adaptable: Easy to update charter/protocols
✅ Transparent: Values are explicit and inspectable
✅ Constitutional: Like a constitution, provides foundation for behavior
""")

print("\n" + "=" * 70)
print("Demo Complete!")
print("=" * 70)
print("""
The identity system is now integrated into the cognitive architecture.
Charter and protocols actively guide behavior throughout the system.

Next steps:
  • Run cognitive core with identity system enabled
  • Observe charter-guided action selection
  • Monitor value alignment checking
  • Analyze identity-informed language generation
""")
