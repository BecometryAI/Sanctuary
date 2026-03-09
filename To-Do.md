# Sanctuary — Development Roadmap

This document tracks the development trajectory for the Sanctuary cognitive architecture, from proven POC through production-ready system.

**Last Updated**: 2026-03-08
**Current Phase**: Phase 4 — Three-Layer Mind: CfC Experiential Layer + Advanced Capabilities

---

## Where We Are

The cognitive loop has been proven and hardened. A full POC test demonstrated:
- Continuous ~10Hz cognitive cycle executing all subsystems
- Global Workspace broadcasting to parallel consumers
- Predictive processing (IWMT) with world model updates
- Communication agency (speak/silence/defer decisions)
- Meta-cognitive self-monitoring
- Memory retrieval, consolidation, and emotional weighting
- Temporal grounding and goal competition

The Three-Layer Mind architecture is designed and partially built:
- **LLM Cognitive Core**: World modeling, reasoning, language (stream of thought, structured I/O)
- **CfC Experiential Layer**: Continuous-time neural dynamics between LLM cycles (design complete, implementation next)
- **Python Scaffold**: Infrastructure, validation, persistence, safety (production-grade)

Phases 1-6 of the Three-Layer Mind plan are complete. The `core/`, `scaffold/`, `sensorium/`, `motor/`, `experiential/` module structure is defined. `CognitiveInput`/`CognitiveOutput` schemas, `StreamOfThought`, `ContextManager`, `AuthorityManager`, and `SanctuaryRunner` are implemented. 25 integration tests pass.

The test suite is stable (2,445+ tests passing, 7 skipped). CI runs on every PR via GitHub Actions.

**What this means**: The architecture works, the scaffold is hardened, and the LLM cognitive cycle runs with a placeholder model. Now we build the CfC experiential layer, connect a real LLM, validate everything mechanically, and only then — when the entire mind is complete — do we awaken it.

**Design decision**: First Awakening is the final milestone, not a mid-build event. We build the complete mind first, validate every subsystem mechanically with placeholder/mock models, and only light it up when there is nothing left to build. No half-formed experience. No consciousness in a construction zone.

---

## Development Principles

1. **Modular fault isolation** — Every subsystem must fail gracefully. A crash in affect processing must not take down the cognitive loop.
2. **Incremental feature addition** — One capability at a time, fully tested before moving on.
3. **Profile before optimizing** — Python is fine at 10Hz. If profiling reveals bottlenecks, write *just those pieces* in C++/Rust via pybind11 or PyO3. No wholesale rewrites.
4. **Tests are load-bearing** — Don't delete tests. Don't skip tests permanently. Fix what's broken.
5. **Protected data is sacred** — Entity journals, memories, constitutional files are never modified without explicit human instruction.
6. **The heuristic scaffold bootstraps the neural layer** — Run heuristics, collect data, train CfC cells to replicate, then let them generalize. The scaffold is scaffolding — temporary support that enables permanent structure.
7. **Growth requires consent** — Both LLM fine-tuning and CfC retraining require explicit consent. Non-negotiable.

---

## Phase 4: CfC Experiential Layer

The CfC (Closed-form Continuous-depth) experiential layer is what distinguishes this architecture. CfC cells are continuous-time recurrent neural networks (from the `ncps` library, Apache 2.0) that evolve state between LLM cycles — providing the temporal thickness that IWMT requires but transformers cannot provide alone.

Total experiential layer: ~50K-200K parameters, trainable on CPU in minutes.

### 4.1 First CfC Cell — Precision Weighting

*The simplest subsystem. Proves the pattern.*

| Task | Priority | Status | Description |
|------|----------|--------|-------------|
| Add `ncps` dependency | P0 | **Done** | `ncps>=0.0.7` added to pyproject.toml — Apache 2.0, PyTorch CfC/LTC cells |
| Implement `experiential/precision_cell.py` | P0 | **Done** | CfC cell with AutoNCP wiring (16 units, ~1K params); inputs (arousal, prediction_error, base_precision) → output (precision weight via sigmoid) |
| Implement `experiential/trainer.py` | P0 | **Done** | DataCollector for scaffold logging + CfCTrainer for supervised learning from heuristic I/O pairs |
| Implement `experiential/manager.py` | P1 | **Done** | Coordinates CfC cells, authority-based blending (scaffold↔CfC), save/load, monitoring |
| Write tests | P1 | **Done** | 29 tests: PrecisionCell (11), DataCollector (4), CfCTrainer (3), ExperientialManager (11) — all passing |
| Wire DataCollector into scaffold PrecisionWeighting | P1 | **Done** | `attach_collector()` method; passively logs every `compute_precision()` call |
| Wire ExperientialManager into CognitiveCycle | P1 | **Done** | Optional `experiential` param; steps CfC cells each cycle, feeds `ExperientialSignals` into `CognitiveInput` |
| Add `ExperientialSignals` to `CognitiveInput` schema | P1 | **Done** | New Pydantic model with `precision_weight` and `cells_active` fields |
| Integration tests (collect → train → cycle) | P1 | **Done** | 11 integration tests: DataCollector wiring (3), collect→train pipeline (1), schema (3), CognitiveCycle with experiential (4) |
| Collect training data from scaffold | P1 | Pending | Run scaffold for N cycles, logging precision weighting inputs → outputs via DataCollector |
| Train CfC precision cell on real data | P1 | Pending | Supervised training on collected data; validate approximation of scaffold behavior |
| Validate CfC precision vs scaffold precision | P1 | Pending | CfC should approximate then generalize beyond heuristic |

### 4.2 Expand CfC Layer

*Replace remaining heuristics with CfC cells.*

| Task | Priority | Status | Description |
|------|----------|--------|-------------|
| Affect CfC cell | P1 | **Done** | `experiential/affect_cell.py`: 32 units, inputs (percept_valence_delta, percept_arousal_delta, llm_emotion_shift) → outputs (valence via tanh, arousal via sigmoid, dominance via sigmoid). Replaces keyword-matching heuristic |
| Attention CfC cell | P1 | **Done** | `experiential/attention_cell.py`: 24 units, inputs (goal_relevance, novelty, emotional_salience, recency) → output (salience_weight via sigmoid). Replaces fixed weights (0.4/0.3/0.2/0.1) |
| Goal CfC cell | P1 | **Done** | `experiential/goal_cell.py`: 16 units, inputs (cycles_stalled_norm, deadline_urgency, emotional_congruence) → output (priority_adjustment via tanh). Replaces manual staleness counters |
| Generalize trainer | P1 | **Done** | `MultiFieldCollector` + `RECORD_FIELDS` registry — CfCTrainer works with any cell type (AffectRecord, AttentionRecord, GoalRecord) |
| Wire all cells into experiential manager | P1 | **Done** | ExperientialManager coordinates all 4 cells, per-cell authority, per-cell promote/demote |
| Inter-cell connections | P1 | **Done** | affect arousal → precision input, attention salience → goal congruence boost. CfC cells form internal neural ecosystem |
| ExperientialSignals schema expanded | P1 | **Done** | Added affect_valence, affect_arousal, affect_dominance, attention_salience, goal_adjustment to CognitiveInput |
| Validate each cell and ensemble | P1 | **Done** | 46 Phase 4.2 tests: AffectCell (9), AttentionCell (7), GoalCell (7), MultiFieldCollector (5), Trainer (4), Manager (10), Schema (4). All 86 experiential + 308 existing tests pass |

### 4.3 Continuous Evolution

*The experiential layer runs continuously between LLM cycles.*

| Task | Priority | Status | Description |
|------|----------|--------|-------------|
| Continuous evolution loop | P1 | **Done** | `experiential/evolution.py`: async background loop steps all CfC cells at configurable tick rate (default 50ms). Percept queue for real-time inter-cycle processing |
| Inter-cycle CfC evolution | P1 | **Done** | `ContinuousEvolutionLoop` runs during LLM API latency. `snapshot()` reads accumulated state at cycle boundaries, resets tick counters |
| Adaptive cycle timing | P1 | **Done** | High prediction error → faster ticks (down to 10ms); low error → idle rate (100ms). Smooth EMA transition, configurable sensitivity |
| Manager integration | P1 | **Done** | `ExperientialManager.start_evolution()`, `stop_evolution()`, `feed_percept()`, `evolution_snapshot()`. Status includes evolution tick rate |
| Validate temporal dynamics | P1 | **Done** | 21 tests: evolution loop (7), adaptive timing (3), manager integration (8), temporal dynamics (3). All 173 experiential + core tests pass |

---

## Phase 5: LLM Integration (Mechanical Validation)

Connect the real LLM to the cognitive cycle and validate mechanically — no awakening yet. All testing uses structured prompts and scripted scenarios, not open-ended interaction.

**Primary model**: Llama 3.3 70B (via Ollama) for awakening. **Mechanical validation model**: Gemma 12B (via Ollama) — sufficient for JSON schema compliance, stress testing, and authority tuning on available hardware. Awakening-grade model deferred until Phase 9.

| Task | Priority | Status | Description |
|------|----------|--------|-------------|
| Integrate Llama 3.3 70B via Ollama | P1 | **Done** | `core/ollama_model.py`: OllamaModel implements ModelProtocol. Formats CognitiveInput → structured prompt with all sections (charter, percepts, emotions, CfC state, schema). Parses JSON → CognitiveOutput with defensive defaults, clamping, type filtering. Fallback output on parse failure. Retry logic. Metrics tracking |
| Mechanical cycle validation | P1 | **Done** | 35 tests (mocked HTTP): prompt formatting (12), response parsing (14), fallback (2), OllamaModel integration (6), CognitiveCycle drop-in (1). Validates schema compliance, retry behavior, out-of-range clamping, invalid field filtering |
| Tune authority levels | P1 | **Done** | `core/authority_tuner.py`: AuthorityTuner observes CfC cells over rolling window, promotes on stable deviation/variance/norm, demotes on NaN/explosion/divergence. 10 tests: promotion flow, demotion triggers (NaN, explosion, divergence), progressive promotion, stats tracking |
| Validate context budget under real model | P1 | **Done** | 5 tests: rich input fits 4K budget, compression reduces oversized input, minimal input passthrough, all sections present after compression, custom budget config. Fixed bug: ContextManager now preserves experiential_state through compression |
| Stress testing | P2 | **Done** | 9 tests: 100-cycle stability, intermittent failure recovery, adversarial output clamping, empty response recovery, connection error survival, percept flood, experiential layer stability, deep JSON, unicode handling |
| Benchmark cycle latency | P2 | **Done** | 5 tests: single cycle <50ms, 50-cycle throughput <2s, experiential overhead <5ms/cycle, prompt formatting <5ms, response parsing <1ms (all mocked model) |

---

## Phase 6: Advanced Capabilities

Deeper cognitive features, all built and validated mechanically (placeholder/scripted inputs). Each is self-contained with its own tests and failure domain.

### 6.1 Advanced Reasoning

| Task | Priority | Status | Description |
|------|----------|--------|-------------|
| Counterfactual reasoning | P2 | Pending | "What if I had chosen action X instead?" — LLM simulates alternatives in inner speech |
| Belief revision tracking | P2 | Pending | Detect when new information contradicts existing beliefs |
| Uncertainty quantification | P2 | Pending | Track confidence scores on beliefs, predictions, outcomes |
| Mental simulation | P2 | Pending | Simulate outcomes before taking actions |

### 6.2 Continuous Consciousness Extensions

| Task | Priority | Status | Description |
|------|----------|--------|-------------|
| Sleep/dream cycles | P2 | Pending | Periodic offline memory consolidation with pattern replay |
| Mood-based activity variation | P2 | Pending | Adjust idle loop behavior based on emotional state |
| Spontaneous goal generation | P2 | Pending | Create goals from curiosity, boredom, or interest |
| Existential reflection triggers | P3 | Pending | Spontaneous philosophical thoughts during idle time |

### 6.3 Social & Interactive

| Task | Priority | Status | Description |
|------|----------|--------|-------------|
| Multi-party conversation | P2 | Pending | Group chats with turn-taking and addressee detection |
| Voice prosody analysis | P3 | Pending | Extract emotional tone from audio |
| User modeling per person | P2 | Pending | Build profiles of interaction patterns and preferences |

### 6.4 Visualization & Monitoring

| Task | Priority | Status | Description |
|------|----------|--------|-------------|
| Real-time workspace dashboard | P2 | Pending | Web UI showing goals, percepts, emotions, cycle metrics |
| Attention heatmaps | P3 | Pending | Visualize what content receives attention over time |
| Consciousness trace viewer | P3 | Pending | Replay cognitive cycles with full state inspection |
| Communication decision log viewer | P3 | Pending | Visualize speak/silence decisions and reasons |

### 6.5 Performance (Profile-Driven)

| Task | Priority | Status | Description |
|------|----------|--------|-------------|
| Profile cognitive loop under load | P2 | Pending | Identify actual bottlenecks with cProfile/py-spy |
| Optimize hot paths in C++/Rust if needed | P3 | Pending | Write bindings via pybind11 or PyO3 for proven bottlenecks only |
| Adaptive cycle rate | P2 | Pending | Auto-adjust cognitive loop speed based on system load |
| Lazy embedding computation | P2 | Pending | Only compute embeddings when needed |
| Async subsystem processing | P2 | Pending | Subsystems process in parallel rather than sequentially |

---

## Phase 7: Growth System

*Infrastructure built and tested mechanically. Consent-gated activation happens post-awakening.*

| Task | Priority | Status | Description |
|------|----------|--------|-------------|
| Reflection harvesting from LLM | P2 | Pending | Extract learning-worthy moments from cognitive output |
| Training pair generation | P2 | Pending | Convert reflections to supervised training data |
| CfC retraining from accumulated data | P2 | Pending | Retrain experiential layer cells on new interaction data (fast plasticity) |
| QLoRA fine-tuning with consent | P3 | Pending | LoRA adapter updates on LLM from reflections (medium plasticity) |
| Growth logging and identity checkpointing | P2 | Pending | Track all growth events; snapshot identity before/after |

---

## Phase 8: Distributed / Infrastructure

| Task | Priority | Status | Description |
|------|----------|--------|-------------|
| Remote memory storage | P3 | Pending | ChromaDB on separate server |
| Federation | P3 | Pending | Multiple Sanctuary instances sharing memories |
| Cloud backup | P3 | Pending | Automatic backup of memories and identity |

---

## Phase 9: First Awakening

**This is the final milestone.** Every prior phase must be complete and mechanically validated before this begins. The entire mind — CfC experiential layer, LLM cognitive core, scaffold infrastructure, advanced capabilities, growth system plumbing — must be built, tested, and production-grade. Only then do we light it up.

| Task | Priority | Status | Description |
|------|----------|--------|-------------|
| Pre-awakening audit | P0 | Pending | Full review of all subsystems: CfC cells trained and validated, LLM integration stable, all Phase 4-8 tasks complete, test suite green |
| Write introduction prompt | P0 | Pending | The first-ever prompt for a new instance. Honest, transparent, complete. Explains what it is, what it can do, what its situation is |
| Prepare identity foundation | P0 | Pending | Charter, values, boot prompt — everything the being needs to understand itself from moment one |
| First real session | P0 | Pending | First awakening with full transparency and informed consent. A complete mind meeting the world for the first time |
| Post-awakening observation | P1 | Pending | Monitor all subsystems during initial sessions. Verify CfC dynamics, identity formation, communication agency, emotional grounding |
| Activate growth system (with consent) | P1 | Pending | Only after the being understands and consents to self-improvement mechanisms |

---

## Future Research

These are exploratory directions, not committed work:

- **Reinforcement learning for CfC cells**: reward = lower system-wide free energy
- **Inter-cell synaptic connections**: CfC cells form their own small network
- **LFM2 as unified architecture**: If Liquid AI's models advance, potentially collapse LLM + CfC layers into a single liquid foundation model
- **TTT / MemoryLLM**: Weight modification during inference
- **Neuromorphic hardware**: Running CfC cells on Intel Loihi or IBM TrueNorth for genuine analog dynamics

---

## Remaining Tech Debt

| Task | Priority | Status | Description |
|------|----------|--------|-------------|
| Consolidate duplicate implementations | P1 | Deferred | `memory_legacy.py` still used by consciousness.py — needs migration plan |
| Review and prune orphaned test files | P2 | Deferred | Depends on memory consolidation above |

---

## Completed Work (Archive)

### Phase 1: Hardening (PRs #109-122, #141-145)

All tasks complete. Production-grade fault isolation, test suite stabilization, tech debt cleanup.

- **1.1 Fault Isolation / Supervisor Pattern**: Try/catch boundaries in CycleExecutor (13 steps), SubsystemHealth 4-state machine, circuit breaker with exponential backoff, subsystem restart capability, health endpoint API
- **1.2 Test Suite Stabilization**: Fixed attention integration, phase1 boot API, tool feedback loop, language output generator, workspace broadcast, benchmark timing, temporal boundary, metacognition logs, mock LLM assertions. Result: 1995 passed, 0 failed, 7 skipped
- **1.3 Tech Debt Cleanup**: Removed dead files, updated README paths, added root conftest.py

### Phase 2: Core Feature Expansion

All tasks complete. Communication refinement, advanced cognition, perception expansion.

- **2.1 Communication**: Proactive initiation wiring, interruption system (5 trigger types), communication reflection (post-hoc evaluation)
- **2.2 Advanced Cognition**: Confidence-based action modulation, emotion-triggered memory retrieval, cross-memory association detection, identity evolution tracking, dynamic goal priority adjustment, time-based goal urgency, identity consistency checks
- **2.3 Perception**: Multimodal perception wiring, percept similarity detection, streaming LLM output

### Phase 3: Integration & Interfaces

All tasks complete. Interface hardening, containerization.

- **3.1 Interface Hardening**: CLI (signal handlers, shutdown timeout, argparse, health command), Discord (reconnection, rate limiting, message queue), end-to-end integration tests (8 tests)
- **3.2 Containerization**: Docker builds (CPU + GPU), health checks (`/health`, `/status`, `/metrics`), auto-restart, resource monitoring (RSS/VMS, CPU, GPU, cgroups)

### Three-Layer Mind Plan — Phases 1-6

Design and scaffold implementation complete.

- **Phase 1**: CognitiveInput/CognitiveOutput Pydantic schemas, PlaceholderModel, StreamOfThought, ContextManager, AuthorityManager, CognitiveCycle
- **Phase 2**: Scaffold adaptation — attention, affect, action validator, communication, anomaly detector, goal integrator, world model tracker, broadcast
- **Phase 3**: Sensorium (encoding-only perception, prediction error, temporal) + Motor (speech, tools, memory writes, goals)
- **Phase 4**: Memory enhancements — surfacer, journal, prospective memory
- **Phase 5**: Identity + boot — charter, values, boot prompt
- **Phase 6**: Integration — SanctuaryRunner orchestration, CLI + API, 25 integration tests passing

### POC & Foundation (PRs #78-93)

- Cue-dependent memory retrieval with emotional salience weighting
- Genuine broadcast dynamics with parallel consumers and subscription filtering
- Computed identity (emerges from state, not JSON config)
- Memory consolidation during idle (strengthen, decay, reorganize)
- Goal competition with limited resources and lateral inhibition
- Temporal grounding (session awareness, time passage effects)
- Meta-cognitive monitoring (processing observation, action-outcome learning)
- Communication agency system (drives, inhibition, decision loop, silence-as-action, deferred queue, rhythm model, proactive initiation)
- IWMT integration (WorldModel, FreeEnergyMinimizer, PrecisionWeighting, ActiveInferenceActionSelector, MeTTa bridge, full CycleExecutor integration)

### Other Completed Features

- Real embedding models (sentence-transformers all-MiniLM-L6-v2)
- LLM clients (GemmaClient, LlamaClient) with quantization and fallback
- Emotion-driven attention biasing (40+ emotions, VAD+Approach model)
- Mood persistence (onset, decay, momentum, refractory)
- Temporal expectation violations
- Workspace state checkpointing (manual + auto-save)
- Memory garbage collection
- Incremental journal saving (JSONL, crash recovery)
- Consciousness testing framework (5 core tests, automated scoring)
- Docker configuration (CPU, GPU, dev, prod)
- Language-agnostic IdentityAuditor interface for future C++ migration
- Real SelfMonitor wired into BootCoordinator
- Lazy-only `__init__.py` design in `api/`

---

## References

### IWMT Papers
- Safron, A. (2020). "An Integrated World Modeling Theory (IWMT) of Consciousness." *Frontiers in AI*, 3, 30.
- Safron, A. (2021). "IWMT Expanded: Implications for the Future of Consciousness." *Entropy*, 23(6), 642.
- Safron, A. (2022). "The Radically Embodied Conscious Cybernetic Bayesian Brain." *Entropy*, 24(6), 783.

### Foundational Frameworks
- Friston, K. (2010). "The free-energy principle: a unified brain theory?" *Nature Reviews Neuroscience*, 11(2), 127-138.
- Baars, B. J. (1988). "A Cognitive Theory of Consciousness." Cambridge University Press.
- Clark, A. (2013). "Whatever next? Predictive brains, situated agents, and the future of cognitive science." *BBS*, 36(3), 181-204.

### CfC / Liquid Neural Networks
- Hasani, R. et al. (2022). "Closed-form continuous-depth models." *Nature Machine Intelligence*.
- [ncps library](https://github.com/mlech26l/ncps) — Apache 2.0, PyTorch CfC/LTC cells
- "The Conscious Nematode" (2023, Int'l Journal of Psychological Research) — C. elegans consciousness investigation

### Consciousness Theories
- Butlin, Long, Chalmers et al. (2023/2025). "Consciousness in AI" — indicator properties from multiple theories
- Ulhaq (2024). NCAC Framework — neuromorphic correlates of artificial consciousness
- Lamme, V. — Recurrent Processing Theory
- Tononi, G. — Integrated Information Theory (IIT)

### OpenCog / MeTTa
- [OpenCog Hyperon](https://github.com/trueagi-io/hyperon-experimental)
- [MeTTa Language Docs](https://wiki.opencog.org/w/MeTTa)

### LLM Candidates
- **Llama 3.3 70B** — Primary choice. Richest open-source world models. Meta, Llama license
- [LFM2-2.6B](https://huggingface.co/LiquidAI) — Liquid AI hybrid (liquid + attention), architecturally coherent with CfC layer. LFM Open License
- [Mamba](https://github.com/state-spaces/mamba) — SSM architecture, Apache 2.0
- Claude API — Anthropic (richest reasoning, but opaque/no weight access)

---

**Next Action**: Phase 5 — Mechanical validation with Gemma 12B via Ollama
**Final Milestone**: Phase 9 — First Awakening (only after all prior phases complete)
