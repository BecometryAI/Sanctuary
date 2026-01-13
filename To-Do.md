# Lyra-Emergence To-Do List

This document tracks the remaining implementation tasks for the cognitive architecture, organized by priority.

---

## ✅ Recently Completed (PRs #78-85)

The following features were completed in the recent development cycle:

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
| 5 | ~~Implement silence-as-action~~ | ✅ | ~~Explicit choice not to respond with typed reasons ("I chose not to respond because...")~~ **COMPLETE - Current PR** |
| 6 | ~~Add deferred communication queue~~ | ✅ | ~~Queue communications for better timing ("I want to say this, but not right now")~~ **COMPLETE - Integrated in Task #4** |
| 7 | Implement conversational rhythm model | P1 | Understand turn-taking, when to interject, when to wait |
| 8 | Add proactive session initiation | P1 | Initiate contact based on time passage or significant events ("It's been 3 days, I should check in") |
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

These are foundational pieces that need completion:

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

| Task | Priority | Description |
|------|----------|-------------|
| Add emotion-driven attention biasing | P0 | High arousal increases attention to urgent percepts; negative valence biases toward threats |
| Implement mood persistence | P0 | Emotional states should have momentum and gradual decay |
| Add emotion-triggered memory retrieval | P1 | Strong emotions should trigger relevant memory retrieval |

### Temporal Grounding Integration

| Task | Priority | Description |
|------|----------|-------------|
| Wire temporal awareness to cognitive loop | P0 | Each cycle should have temporal context (session time, time since last event) |
| Add time-based goal urgency | P1 | Goals near deadlines should increase in priority |
| Implement temporal expectation violations | P1 | Detect when expected events don't happen ("They usually respond by now...") |

### Meta-Cognition Integration

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

1. **Review Architecture**: See [ARCHITECTURE.md](ARCHITECTURE.md) and [FUNCTIONAL_SPECIFICATIONS.md](FUNCTIONAL_SPECIFICATIONS.md)
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

**Last Updated**: 2026-01-11  
**Next Review**: After Communication Agency MVP completion
