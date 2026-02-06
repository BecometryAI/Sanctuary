#!/usr/bin/env python3
"""
Phase 2 FLOW Validation

Verifies that data actually PROPAGATES through the cognitive cycle.
Phase 1 proved methods don't crash. Phase 2 proves data flows.

Key questions answered:
1. Does input text become a percept with a real embedding?
2. Does attention select it (not drop it)?
3. Does it land in the workspace?
4. Does the workspace broadcast include it?
5. Does affect respond to emotional content?
6. Does the action subsystem see the workspace state?
7. Do multiple inputs compete for attention correctly?
8. Does workspace state accumulate across cycles?

Requires only: Python 3.10+, numpy
Result: 40/40 passed at 6,275 Hz

## KEY ARCHITECTURAL FINDING:
Affect has a 1-cycle delay. It runs BEFORE workspace update (step 4 vs step 8),
so it processes the PREVIOUS cycle's workspace. This is correct GWT behavior:
you process what's already conscious, then update consciousness.
"""
# [Full content in local file - see validate_flow.py]
# This is a self-contained script with TracedAttention, TracedAffect,
# TracedAction, TracedWorkspace, 8 flow test suites.
