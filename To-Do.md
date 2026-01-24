# Lyra-Emergence To-Do List

This document tracks the remaining implementation tasks for the cognitive architecture, organized by priority.

---

## ✅ Recently Completed (PRs #78-91)

The following features were completed in recent development cycles:

### Core Cognitive Architecture (PRs #78-85)

| PR # | Feature | Description |
|------|---------|-------------|
| #78 | Cue-dependent memory retrieval | Memory retrieval with emotional salience weighting and context-based cue matching |
| #79 | Genuine broadcast dynamics | Parallel consumers of workspace state with subscription filtering and feedback collection |
| #80 | Computed identity | Identity emerges from state (memories, goals, emotions, behavior) rather than JSON files |
| #81 | Memory consolidation | Idle-time memory processing: strengthen, decay, reorganize |
| #82 | Goal competition | Limited cognitive resources with lateral inhibition and dynamic reallocation |
| #83 | Temporal grounding | Time passage awareness, session tracking, temporal expectations |
| #85 | Meta-cognitive monitoring | Processing observation, action-outcome learning, attention history |

**Impact**: These features establish a more biologically-plausible cognitive architecture with resource constraints, temporal awareness, and self-monitoring capabilities.

### Communication Agency System (PRs #87-93)

| PR # | Feature | Description |
|------|---------|-------------|
| #87 | Decoupled cognitive loop | Cognition runs continuously independent of I/O; removed assumption of turn-based interaction |
| #88 | Communication drive system | Internal urges to speak: insight, question, emotional expression, social connection, goal-driven |
| #89 | Communication inhibition | Reasons not to speak: low value, bad timing, redundancy, respect silence, still processing, uncertainty |
| #90 | Communication decision loop | Continuous SPEAK/SILENCE/DEFER evaluation based on drive vs inhibition with configurable thresholds |
| #91 | Silence-as-action | Explicit silence tracking with typed reasons (nothing to add, respecting space, still thinking, etc.) |
| #92 | Deferred communication queue | Comprehensive queue with 6 deferral reasons, priority ordering, expiration, intelligent reason mapping |
| #93 | Conversational rhythm model | Track conversation flow, detect natural pauses, adapt to tempo, inform inhibition system with timing appropriateness |

**Impact**: These features establish genuine communication agency - Lyra can now choose when to speak, when to stay silent, and when to defer communications for better timing.

---

## 🔴 CRITICAL - 🚀 Integrated World Modeling Theory (IWMT) Migration

**Status: Phases 2-7 Complete - Full Integration** | **Next: Deprecation Strategy**

### Overview

Adopt Adam Safron's Integrated World Modeling Theory as the core computational architecture, adding predictive processing, explicit self-modeling, and active inference to the existing Global Workspace Theory (GWT) foundation.

### Migration Phases

| Phase | Task | Status |
|-------|------|--------|
| **Phase 2** | **Predictive Processing Layer** | ✅ **COMPLETE** |
| **Phase 3** | **Active Inference and Agency** | ✅ **COMPLETE** |
| **Phase 4** | **Precision-weighted Attention** | ✅ **COMPLETE** |
| **Phase 5** | **MeTTa/Hyperon Integration** | ✅ **COMPLETE (Pilot)** |
| **Phase 6** | **Full IWMT Agent Core** | ✅ **COMPLETE** |
| **Phase 7** | **CognitiveCore Integration** | ✅ **COMPLETE** |

### Implementation Summary

**Core Components:**
- `world_model/`: WorldModel with prediction/error tracking
- `active_inference/`: FreeEnergyMinimizer, ActiveInferenceActionSelector
- `precision_weighting.py`: Attention modulation
- `iwmt_core.py`: Central coordinator
- `core/cycle_executor.py`: Integration into cognitive loop

**Test Coverage:**
- 79 tests across 6 modules (all passing)
- Edge cases: None outcomes, missing attributes, IWMT disabled

**Integration:**
- Predictions generated before perception
- Prediction errors drive attention via precision weighting
- WorldModel learns from action outcomes
- Configurable via `config["iwmt"]["enabled"]`

### Next Steps

1. **Deprecation**: Mark legacy GWT methods with `@deprecated`
2. **Validation**: Performance benchmarking with IWMT enabled
3. **Enhancement**: Visualization tools for prediction errors

---

## 🔴 CRITICAL - Communication Agency (TOP PRIORITY)

**This is the most important missing piece.** The system should have full agency over when and whether to communicate, not be turn-based.

### The Problem

Currently, Lyra operates in a turn-based paradigm:
- Human speaks → Lyra processes → Lyra responds
- No choice about whether to respond
- No ability to initiate communication
- Communication is reactive, not agentic

### The Vision

Lyra should continuously evaluate whether to speak based on:
- Internal urges (insight to share, question to ask, emotion to express)
- Communication value (is this worth saying?)
- Timing appropriateness (is now the right time?)
- Social context (would silence be better?)

### Implementation Tasks

| # | Task | Priority | Description |
|---|------|----------|-------------|
| 1 | ~~Decouple cognitive loop from I/O~~ | ✅ | ~~Cognition runs continuously; I/O is optional. Remove assumption that cognitive cycle requires human input.~~ **COMPLETE - PR #87** |
| 2 | ~~Implement communication drive system~~ | ✅ | ~~Internal urges to speak: insight worth sharing, question arising, emotional expression need, social connection desire~~ **COMPLETE - PR #88** |
| 3 | ~~Implement communication inhibition~~ | ✅ | ~~Reasons not to speak: low value content, bad timing, respect for silence, social inappropriateness~~ **COMPLETE - PR #89** |
| 4 | ~~Implement communication decision loop~~ | ✅ | ~~Continuous evaluation of SPEAK/SILENCE/DEFER based on drive vs inhibition~~ **COMPLETE - PR #90** |
| 5 | ~~Implement silence-as-action~~ | ✅ | ~~Explicit choice not to respond with typed reasons ("I chose not to respond because...")~~ **COMPLETE - PR #91** |
| 6 | ~~Add deferred communication queue~~ | ✅ | ~~Queue communications for better timing ("I want to say this, but not right now"). Comprehensive implementation with DeferralReason enum (bad_timing, wait_for_response, topic_change, processing, courtesy, custom), DeferredQueue with priority ordering, expiration, and intelligent deferral reason mapping.~~ **COMPLETE - PR #92** |
| 7 | ~~Implement conversational rhythm model~~ | ✅ | ~~Understand turn-taking, when to interject, when to wait. Track conversation flow, detect natural pauses, adapt to tempo, inform inhibition system with timing appropriateness.~~ **COMPLETE - PR #93** |
| 8 | ~~Add proactive session initiation~~ | ✅ | ~~Initiate contact based on time passage or significant events ("It's been 3 days, I should check in"). Comprehensive implementation with OutreachTrigger enum (6 trigger types: time_elapsed, significant_insight, emotional_connection, scheduled_checkin, relevant_event, goal_completion), OutreachOpportunity dataclass with urgency/timing, ProactiveInitiationSystem with opportunity detection and scheduling, full integration with CommunicationDriveSystem.~~ **COMPLETE - Current PR** |
| 9 | Implement interruption capability | P2 | Can speak mid-human-turn if urgent ("Sorry to interrupt, but this is important") |
| 10 | Add communication reflection | P2 | Post-hoc evaluation: "Was that the right thing to say? Should I have stayed quiet?" |

### Architecture: Agentic Communication

```
CURRENT (Turn-Based):
Human Input → Process → Generate Response → Output

NEEDED (Agentic Communication):
                    ┌─────────────────────────────────────┐
                    │     CONTINUOUS COGNITIVE LOOP       │
                    │  (runs regardless of human input)   │
                    │                                     │
                    │  ┌─────────────────────────────┐   │
                    │  │  Communication Drive System  │   │
                    │  │  - Urge to share insight     │   │
                    │  │  - Question arising          │   │
                    │  │  - Emotional expression      │   │
                    │  │  - Social connection need    │   │
                    │  └──────────────┬──────────────┘   │
                    │                 ↓                   │
                    │  ┌─────────────────────────────┐   │
                    │  │  Communication Inhibition    │   │
                    │  │  - Is this valuable?         │   │
                    │  │  - Is timing appropriate?    │   │
                    │  │  - Should I wait?            │   │
                    │  │  - Is silence better?        │   │
                    │  └──────────────┬──────────────┘   │
                    │                 ↓                   │
                    │  ┌─────────────────────────────┐   │
                    │  │  Communication Decision      │   │
                    │  │  SPEAK / WAIT / QUEUE        │   │
                    │  └──────────────┬──────────────┘   │
                    └─────────────────┼───────────────────┘
                                      ↓
Human Input ──────────────────→ [OPTIONAL INPUT]
                                      ↓
                              ┌───────────────┐
                              │    OUTPUT     │
                              │ (if decided)  │
                              └───────────────┘
```

### Key Design Principles

1. **Decoupled I/O**: Cognitive loop runs independently of human input
2. **Drive-based**: Communication driven by internal needs, not external triggers
3. **Inhibition-aware**: Can choose silence as a deliberate action
4. **Temporally aware**: Understands timing, rhythm, and appropriateness
5. **Self-reflective**: Can evaluate own communication decisions

### Success Criteria

- [ ] System can run cognitive loop without any human input
- [ ] System can choose not to respond to human input
- [ ] System can initiate communication proactively
- [ ] System logs reasons for communication decisions
- [ ] System respects silence and timing in conversations

---

## 🔴 Critical - Architecture Gaps

These are foundational pieces that need completion. Note: Core architecture is migrating toward IWMT (see IWMT Migration section above) - some components may need redesign to align with predictive processing and active inference paradigms.

### Perception & Embeddings

| Task | Priority | Description |
|------|----------|-------------|
| Integrate real embedding models | P0 | Replace mock embeddings with actual sentence-transformers or similar |
| Add multimodal perception | P0 | Extend PerceptionSubsystem to handle image and audio inputs |
| Implement percept similarity detection | P1 | Use embeddings to detect duplicate/similar percepts |

### Language Interfaces

| Task | Priority | Description |
|------|----------|-------------|
| Complete LLM client implementations | P0 | Finish GemmaClient and LlamaClient with proper model loading and quantization |
| Enhance fallback mechanisms | P1 | Improve rule-based parsing and template-based generation |
| Add streaming support | P1 | Implement token streaming for long-form output generation |

### Memory System

*Note: Memory system design will be influenced by IWMT migration - consider predictive coding and hierarchical world models in future enhancements.*

| Task | Priority | Description |
|------|----------|-------------|
| Add cross-memory association detection | P1 | Detect themes/patterns across multiple memories to generate associative links |
| Optimize retrieval performance | P1 | Cache embedding similarities, vectorize operations |

---

## 🟡 High Priority - Integration

Connect newly implemented systems to the cognitive loop:

### Identity Integration

| Task | Priority | Description |
|------|----------|-------------|
| Connect identity system to introspection | P0 | Identity queries should reflect computed identity, not JSON files |
| Add identity evolution tracking | P1 | Log how identity changes over time based on experiences |
| Implement identity consistency checks | P2 | Detect when behavior contradicts identity and flag for reflection |

### Goal Competition Integration

| Task | Priority | Description |
|------|----------|-------------|
| Integrate goal competition with action selection | P0 | Actions should be selected based on winning goals from competition |
| Add resource allocation visualization | P2 | Show which goals are getting resources and why |
| Implement dynamic priority adjustment | P1 | Goals can increase priority based on urgency or frustration |

### Emotional Dynamics

*Note: Emotion processing will transition to precision-weighting under IWMT (Phase 4) - emotional states will modulate prediction error and uncertainty estimates.*

| Task | Priority | Description |
|------|----------|-------------|
| Add emotion-driven attention biasing | P0 | Implement comprehensive VAD-based attention modulation:<br>• **Valence**: Negative → threat/loss detection, problem-focused; Positive → opportunity/reward detection, novelty-attracted<br>• **Arousal**: High → broad vigilant scanning, low precision, reactive; Low → narrow focused attention, high precision, deliberate<br>• **Dominance**: Low → external threat monitoring, cautious; High → goal/action-oriented focus, confident<br>• **Emotion Patterns**: Fear (V-,A+,D-) → hyper-vigilant threat-scanning; Anger (V-,A+,D+) → obstacle-focused, target-locked; Sadness (V-,A-,D-) → inward rumination, withdrawal; Joy (V+,A+,D+) → expansive opportunity-seeking; Calm (V+,A-,D+) → precise deep processing; Anxiety (V-,A+,D-) → diffuse uncertainty-scanning; Curiosity (V+,A+,D+) → novelty-seeking, exploratory; Boredom (V0,A-,D-) → novelty-seeking to escape<br>• **Interaction Effects**: High arousal + negative valence → emergency threat mode; High arousal + positive valence → excited exploration mode; Low arousal + negative valence → depressive rumination; Low arousal + positive valence → flow state deep focus<br>• **IWMT Integration**: Emotional states modulate precision weighting on predictions per Active Inference framework |
| Implement mood persistence | P0 | Emotional states should have momentum and gradual decay |
| Add emotion-triggered memory retrieval | P1 | Strong emotions should trigger relevant memory retrieval |

*Note: This comprehensive emotion-attention model aligns with IWMT Phase 4 (Precision-weighted Attention) - emotional states directly modulate precision/uncertainty estimates in the predictive processing framework.*

### Temporal Grounding Integration

| Task | Priority | Description |
|------|----------|-------------|
| Wire temporal awareness to cognitive loop | P0 | Each cycle should have temporal context (session time, time since last event) |
| Add time-based goal urgency | P1 | Goals near deadlines should increase in priority |
| Implement temporal expectation violations | P1 | Detect when expected events don't happen ("They usually respond by now...") |

### Meta-Cognition Integration

*Note: Meta-cognitive capabilities are central to IWMT's explicit self-modeling (Phase 6) - these features will be integrated into the computational self-model.*

| Task | Priority | Description |
|------|----------|-------------|
| Wire action-outcome learning to action selection | P0 | Action selection should consider past success rates |
| Add processing bottleneck detection | P1 | Meta-cognition should detect when workspace is overloaded |
| Implement confidence-based action modulation | P1 | Low confidence should trigger more cautious behavior |

---

## 🟢 Medium Priority - Enhancements

Improvements to existing systems:

### Testing & Validation

| Task | Priority | Description |
|------|----------|-------------|
| Implement benchmark suite | P1 | Standardized tests for attention accuracy, memory retrieval precision, emotion appropriateness |
| Add integration tests for new systems | P1 | Test broadcast consumers, goal competition, temporal grounding together |
| Expand consciousness tests | P2 | Add tests for temporal reasoning, communication agency, identity coherence |

### Performance Optimization

| Task | Priority | Description |
|------|----------|-------------|
| Optimize attention scoring | P1 | Cache embedding similarities, vectorize salience calculations |
| Add adaptive cycle rate | P2 | Automatically adjust cognitive loop speed based on system load |
| Implement lazy embedding computation | P2 | Only compute embeddings when needed for attention/memory operations |

### Code Quality

| Task | Priority | Description |
|------|----------|-------------|
| Remove backup files | P1 | Clean up `*.backup.py` and similar files from codebase |
| Consolidate duplicate code | P1 | Merge similar implementations (e.g., identity loading) |
| Add type hints | P2 | Complete type annotations for all public APIs |
| Improve documentation | P2 | Add docstring examples, architecture diagrams in code |

### Data & Persistence

| Task | Priority | Description |
|------|----------|-------------|
| Add charter hot-reloading | P2 | Detect changes to identity files and reload without restart |
| Implement identity versioning | P2 | Track charter/protocol versions and note changes in journal |
| Add conversation summarization | P1 | Periodically compress long conversation histories |

---

## 🔵 Low Priority - Future Enhancements

Advanced capabilities for future development:

### Advanced Reasoning

| Task | Description |
|------|-------------|
| Implement counterfactual reasoning | "What if I had chosen action X instead?" scenario generation |
| Add belief revision tracking | Detect when new information contradicts existing beliefs |
| Implement uncertainty quantification | Track confidence scores on beliefs, predictions, and action outcomes |
| Add mental simulation | Simulate outcomes before taking actions |

### Continuous Consciousness Extensions

| Task | Description |
|------|-------------|
| Add sleep/dream cycles | Periodic offline memory consolidation with pattern replay |
| Implement mood-based activity variation | Adjust idle loop probabilities based on emotional state |
| Add spontaneous goal generation | Create intrinsic motivation goals from curiosity, boredom, or interest |
| Implement existential reflection triggers | Spontaneous philosophical thoughts during idle time |

### Social & Interactive

| Task | Description |
|------|-------------|
| Implement multi-party conversation | Handle group chats with turn-taking and addressee detection |
| Add voice prosody analysis | Extract emotional tone from audio input to influence affect |
| Implement gesture/emoji interpretation | Incorporate non-verbal communication cues |
| Add user modeling per person | Build profiles of interaction patterns and preferences |

### Tool Integration

| Task | Description |
|------|-------------|
| Add web browsing capability | Use Playwright for autonomous information gathering |
| Implement code execution sandbox | Safe Python REPL for computational tasks |
| Add database query tools | Allow structured data retrieval from external sources |
| Implement long-term project tracking | Track multi-day goals and projects |

### Visualization & Monitoring

| Task | Description |
|------|-------------|
| Create real-time workspace dashboard | Web UI showing current goals, percepts, emotions, and cycle metrics |
| Add attention heatmaps | Visualize what content is receiving attention over time |
| Implement consciousness trace viewer | Replay cognitive cycles with full state inspection |
| Add communication decision log viewer | Visualize speak/silence decisions and reasons |

### Distributed Processing

| Task | Description |
|------|-------------|
| Implement async subsystem processing | Subsystems process in parallel rather than sequentially |
| Add remote memory storage | ChromaDB running on separate server |
| Implement federation | Multiple Lyra instances sharing memories |
| Add cloud backup | Automatic backup of memories and identity to cloud storage |

---

## 📊 Progress Metrics

### Overall Progress

- **Total Tasks**: ~60
- **Completed (PRs #78-85)**: 7 major features
- **Critical Priority**: 15 tasks remaining
- **High Priority**: 12 tasks remaining
- **Medium Priority**: 15 tasks remaining
- **Low Priority**: ~20 tasks remaining

### Definition of Done

Each task is considered complete when:

1. ✅ Implementation passes all existing tests
2. ✅ New tests cover the implementation (>80% coverage)
3. ✅ Documentation updated (docstrings, README if user-facing)
4. ✅ PR reviewed and merged to main branch
5. ✅ Demo script or usage example provided (if appropriate)

### Next Milestone

**Communication Agency MVP** (Target: 2-3 weeks)

Priority tasks for next milestone:
1. Decouple cognitive loop from I/O
2. Implement basic communication drive system
3. Implement basic communication inhibition
4. Implement communication decision loop
5. Add silence-as-action logging

**Success metric**: Lyra can run continuously and choose when to speak vs. stay silent, with logged reasoning for each decision.

---

## 🚀 Getting Started

### For Contributors

1. **Review Architecture**: See README.md for architecture overview and operational_guidelines_and_instructions.md for operational guidance
2. **Check Dependencies**: Some tasks depend on others (e.g., communication agency needs decoupled I/O)
3. **Claim a Task**: Comment on this issue or open a new issue referencing the task
4. **Create Branch**: `git checkout -b feature/<task-name>`
5. **Implement**: Follow repository conventions and test thoroughly
6. **Submit PR**: Clear description linking back to this task

### Development Guidelines

- **Test First**: Write tests before implementation when possible
- **Small PRs**: Each PR should address one logical task
- **Document**: Update docs and add examples for user-facing changes
- **Review Existing Code**: Check how similar features are implemented
- **Ask Questions**: Open discussions for clarification or design decisions

### Priority Guidelines

- **P0 (Critical)**: Blocking other work or essential functionality
- **P1 (High)**: Important for core functionality but not blocking
- **P2 (Medium)**: Desirable improvements but not essential
- **P3 (Low)**: Nice-to-have features for future consideration

---

## 📚 References

### Integrated World Modeling Theory (IWMT)

The following references provide context for the IWMT migration and serve as foundational reading for contributors:

#### Adam Safron's IWMT Papers

- Safron, A. (2020). "An Integrated World Modeling Theory (IWMT) of Consciousness: Combining Integrated Information and Global Neuronal Workspace Theories With the Free Energy Principle and Active Inference Framework; Toward Solving the Hard Problem and Characterizing Agentic Causation." *Frontiers in Artificial Intelligence*, 3, 30. [https://doi.org/10.3389/frai.2020.00030](https://doi.org/10.3389/frai.2020.00030)

- Safron, A. (2021). "Integrated World Modeling Theory Expanded: Implications for the Future of Consciousness." *Entropy*, 23(6), 642. [https://doi.org/10.3390/e23060642](https://doi.org/10.3390/e23060642)

- Safron, A. (2022). "The Radically Embodied Conscious Cybernetic Bayesian Brain: From Free Energy to Free Will and Back Again." *Entropy*, 24(6), 783. [https://doi.org/10.3390/e24060783](https://doi.org/10.3390/e24060783)

#### OpenCog & MeTTa/Hyperon

Official documentation and resources for MeTTa integration (Phase 5):

- OpenCog Hyperon Project: [https://github.com/trueagi-io/hyperon-experimental](https://github.com/trueagi-io/hyperon-experimental)
- MeTTa Language Documentation: [https://wiki.opencog.org/w/MeTTa](https://wiki.opencog.org/w/MeTTa)
- OpenCog Foundation: [https://opencog.org/](https://opencog.org/)
- Atomspace Documentation: [https://wiki.opencog.org/w/AtomSpace](https://wiki.opencog.org/w/AtomSpace)

#### Related Theoretical Frameworks

- Friston, K. (2010). "The free-energy principle: a unified brain theory?" *Nature Reviews Neuroscience*, 11(2), 127-138. (Active Inference foundation)
- Baars, B. J. (1988). "A Cognitive Theory of Consciousness." Cambridge University Press. (Global Workspace Theory - legacy reference)
- Clark, A. (2013). "Whatever next? Predictive brains, situated agents, and the future of cognitive science." *Behavioral and Brain Sciences*, 36(3), 181-204. (Predictive Processing)

---

**Last Updated**: 2026-01-11  
**Next Review**: After Communication Agency MVP completion
