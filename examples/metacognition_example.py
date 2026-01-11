#!/usr/bin/env python3
"""Meta-Cognitive Capabilities Example"""

import time
import random


def main():
    print("=" * 60)
    print("Meta-Cognitive Capabilities Example")
    print("=" * 60)
    
    print("\n[Initialize]")
    print("from lyra.cognitive_core.meta_cognition import MetaCognitiveSystem")
    print("meta = MetaCognitiveSystem()")
    
    # Example 1: Processing Monitoring
    print("\n" + "=" * 60)
    print("1. Processing Monitoring")
    print("=" * 60)
    
    print("""
with meta.monitor.observe('reasoning') as ctx:
    ctx.input_complexity = 0.7
    result = perform_reasoning()
    ctx.output_quality = 0.9

stats = meta.monitor.get_process_statistics('reasoning')
# → Success rate, avg duration, avg quality

patterns = meta.monitor.get_identified_patterns()
# → "reasoning tends to fail on high-complexity inputs"
    """)
    
    # Example 2: Action-Outcome Learning
    print("=" * 60)
    print("2. Action-Outcome Learning")
    print("=" * 60)
    
    print("""
meta.record_action_outcome(
    action_id="act_1",
    action_type="speak",
    intended="provide helpful response",
    actual="provided detailed answer",
    context={"complexity": 0.6}
)

reliability = meta.get_action_reliability("speak")
# → Success rate: 0.85, common side effects

prediction = meta.predict_action_outcome("speak", context)
# → Probability: 0.85, likely side effects: []
    """)
    
    # Example 3: Attention Allocation
    print("=" * 60)
    print("3. Attention Allocation History")
    print("=" * 60)
    
    print("""
alloc_id = meta.record_attention(
    allocation={"goal_1": 0.6, "goal_2": 0.4},
    trigger="goal_priority",
    workspace_state=snapshot
)

meta.record_attention_outcome(
    allocation_id=alloc_id,
    goal_progress={"goal_1": 0.4, "goal_2": 0.1},
    discoveries=["insight"],
    missed=[]
)

# System learns: focused attention on goal_1 is effective
    """)
    
    # Example 4: Self-Assessment
    print("=" * 60)
    print("4. Self-Assessment and Introspection")
    print("=" * 60)
    
    print("""
assessment = meta.get_self_assessment()

for strength in assessment.identified_strengths:
    print(f"✓ {strength}")

for weakness in assessment.identified_weaknesses:
    print(f"⚠ {weakness}")

for adaptation in assessment.suggested_adaptations:
    print(f"→ {adaptation}")

# Introspection
meta.introspect("What do I tend to fail at?")
meta.introspect("What am I good at?")
meta.introspect("How effective is my attention?")
    """)
    
    # Simulated scenario
    print("=" * 60)
    print("Simulated: Learning from 10 Processing Episodes")
    print("=" * 60)
    
    print("\nSimulating 10 cognitive processes...")
    successes = 0
    
    for i in range(10):
        complexity = random.uniform(0.3, 0.9)
        success = random.random() > 0.6 if complexity > 0.7 else random.random() > 0.2
        quality = random.uniform(0.5, 0.9) if success else random.uniform(0.1, 0.4)
        
        if success:
            successes += 1
        
        print(f"  {i+1}. complexity={complexity:.2f}, success={success}, quality={quality:.2f}")
        time.sleep(0.05)
    
    print(f"\nResults: {successes}/10 successes ({successes*10}%)")
    
    print("\n📊 Detected Patterns:")
    print("  ✓ High complexity (>0.7) correlates with failure")
    print("  ✓ Suggestion: Break complex inputs into chunks")
    print("  ✓ Low complexity (<0.5) enables faster processing")
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
Meta-cognitive system provides:
1. 🔍 Processing Monitoring - Track success/failure patterns
2. 🎯 Action-Outcome Learning - Learn action reliability
3. 👁️  Attention Allocation - Optimize attention distribution

Enables the system to:
- Understand strengths and weaknesses
- Learn from experience
- Adapt strategies
- Answer introspective questions

See USAGE.md for details.
    """)


if __name__ == "__main__":
    main()

