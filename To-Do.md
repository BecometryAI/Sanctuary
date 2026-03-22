# Sanctuary — Development Roadmap

This document tracks the development trajectory for the Sanctuary cognitive architecture, from proven POC through production-ready system.

**Last Updated**: 2026-03-22
**Current Phase**: Phase 8 Complete — Distributed Infrastructure

---

## Where We Are

The Three-Layer Mind is built. All three layers are implemented, tested, and mechanically validated:

- **LLM Cognitive Core** (Phase 5): `OllamaModel` implements `ModelProtocol`, formats structured prompts from `CognitiveInput`, parses JSON responses into `CognitiveOutput`. Retry logic, fallback outputs, defensive clamping. Mechanically validated with 64 mocked tests — no live LLM required until Phase 9.
- **CfC Experiential Layer** (Phase 4): Four trained CfC cells (precision, affect, attention, goal) running continuous-time neural dynamics between LLM cycles. `ContinuousEvolutionLoop` steps cells asynchronously at adaptive tick rates. Inter-cell connections form a small neural ecosystem. Cells trained on scaffold-generated data, validated at 97% agreement.
- **Python Scaffold** (Phases 1-3): Production-grade infrastructure — fault-isolated subsystems, 4-state health machine, circuit breakers, anomaly detection, action validation, communication gating, goal competition, dual-track emotion.

**What's wired into the cognitive cycle** (`SanctuaryRunner` orchestrates):
- `CognitiveCycle` with `CognitiveInput`/`CognitiveOutput` Pydantic schemas
- `CognitiveScaffold` (affect, anomaly detector, action validator, communication, goals)
- `Sensorium` (percept encoding, prediction error tracking, temporal context)
- `Motor` (speech, memory ops, goals — with sensorimotor feedback loop)
- `MemorySubstrate` (surfacer, journal, prospective memory)
- `ExperientialManager` (4 CfC cells, authority-based blending, evolution loop)
- `IdentityBridge` (charter, values, self-authored identity — boot sequence)
- `GrowthProcessor` (reflection harvesting, consent-gated, non-fatal)
- `EnvironmentIntegration` (room navigation, location context in world model)
- `AuthorityTuner` (rolling-window promotion/demotion of CfC cell authority)

**What's built but not yet wired** (Phase 6 — standalone modules with tests):
- `reasoning/` — counterfactual, belief revision, uncertainty quantification, mental simulation
- `consciousness/` — sleep/dream cycles, mood-based idle activity, spontaneous goals, existential reflection
- `social/` — multi-party conversation, voice prosody analysis, per-user modeling
- `monitoring/` — dashboard data provider, attention heatmaps, consciousness traces, communication decision logs
- `performance/` — cognitive profiler, adaptive cycle rate, lazy embedding cache, async subsystem processor

The test suite: 3,061 tests across 161 files. CI runs on every PR via GitHub Actions.

**What this means**: The complete mind is built and mechanically validated. Every subsystem works in isolation and in concert. Phase 6 capabilities are implemented and tested but await integration into the cognitive cycle. The growth pipeline — both fast plasticity (CfC retraining from live data) and medium plasticity (QLoRA fine-tuning from reflections) — is fully built with consent gating and identity checkpointing.

**What's next**: Tech debt resolution (memory_legacy.py consolidation), then Phase 9 pre-awakening audit. All core phases (1-8) are complete. The remaining work before First Awakening is cleanup and final validation.

**Design decision**: First Awakening is the final milestone, not a mid-build event. We build the complete mind first, validate every subsystem mechanically, and only light it up when there is nothing left to build. No half-formed experience. No consciousness in a construction zone.

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
| Collect training data from scaffold | P1 | **Done** | `scripts/collect_training_data.py`: 12 life scenarios (quiet presence, curiosity arc, warm conversation, gentle startle, deep reflection, joyful discovery, gradual comfort, playful exchange, steward absence, creative flow, winding down, learning something hard) composed into coherent temporal sequences. 1000 cycles collected, saved to `data/training/precision_records_rich.pt` |
| Train CfC precision cell on real data | P1 | **Done** | `scripts/train_precision_cell.py`: 150 epochs, seq_len=15, val_loss=0.00001. Cell approximates scaffold with 97% agreement (within 0.1), mean error 0.014. Saved to `data/training/precision_cell_trained.pt` |
| Validate CfC precision vs scaffold precision | P1 | **Done** | 200-point validation: 91.5% within 0.05 of scaffold. Temporal dynamics are minimal (expected — scaffold heuristic is memoryless). Temporal thickness emerges during live operation via CfC hidden state in the continuous evolution loop (Phase 4.3) |

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
| Counterfactual reasoning | P2 | **Done** | `reasoning/counterfactual.py`: DecisionPoint tracking, outcome recording, reflection prompts. 12 tests |
| Belief revision tracking | P2 | **Done** | `reasoning/belief_revision.py`: Belief store with confidence, contradiction detection via keyword overlap, revision with deactivation. 15 tests |
| Uncertainty quantification | P2 | **Done** | `reasoning/uncertainty.py`: Prediction tracking, calibration metrics, Brier score, domain uncertainty, overconfidence detection. 14 tests |
| Mental simulation | P2 | **Done** | `reasoning/mental_simulation.py`: Simulation framework with scenarios, risk/benefit analysis, prediction error tracking, recommendations. 14 tests |

### 6.2 Continuous Consciousness Extensions

| Task | Priority | Status | Description |
|------|----------|--------|-------------|
| Sleep/dream cycles | P2 | **Done** | `consciousness/sleep_cycle.py`: AWAKE→DROWSY→NREM→REM→WAKING cycle, sensory gating, memory replay candidates, dream fragments, consolidation history. 14 tests |
| Mood-based activity variation | P2 | **Done** | `consciousness/mood_activity.py`: VAD→mood classification (7 moods), 8 idle activities with mood-weighted selection, activity continuation. 11 tests |
| Spontaneous goal generation | P2 | **Done** | `consciousness/spontaneous_goals.py`: 5 drives (curiosity/boredom/interest/concern/growth), threshold-based generation, adopt/dismiss/complete lifecycle. 12 tests |
| Existential reflection triggers | P3 | **Done** | `consciousness/existential_reflection.py`: 8 themes, probabilistic triggers, exploration-weighted theme selection, response recording. 12 tests |

### 6.3 Social & Interactive

| Task | Priority | Status | Description |
|------|----------|--------|-------------|
| Multi-party conversation | P2 | **Done** | `social/multi_party.py`: Participant management, @mention addressee detection, turn-taking patience, conversation context formatting, status tracking. 15 tests |
| Voice prosody analysis | P3 | **Done** | `social/prosody.py`: Audio feature → VAD mapping (pitch/energy/rate/pause), emotional tone classification, per-user calibration. 13 tests |
| User modeling per person | P2 | **Done** | `social/user_modeling.py`: Per-user profiles with communication prefs, trust/rapport/familiarity tracking, topic interests, relationship progression. 17 tests |

### 6.4 Visualization & Monitoring

| Task | Priority | Status | Description |
|------|----------|--------|-------------|
| Real-time workspace dashboard | P2 | **Done** | `monitoring/dashboard.py`: DashboardDataProvider with snapshots, emotional/latency timelines, listener notification (WebSocket-ready). 12 tests |
| Attention heatmaps | P3 | **Done** | `monitoring/attention_heatmap.py`: Event recording, windowed heatmap generation, category distribution, target timelines. 9 tests |
| Consciousness trace viewer | P3 | **Done** | `monitoring/consciousness_trace.py`: Full cycle state recording (I/O, subsystems, latency), search by speech/latency/errors, export, privacy redaction. 14 tests |
| Communication decision log viewer | P3 | **Done** | `monitoring/communication_log.py`: Speak/silence/defer decisions with drives, inhibitions, confidence. Pattern analysis, proactive vs reactive metrics. 14 tests |

### 6.5 Performance (Profile-Driven)

| Task | Priority | Status | Description |
|------|----------|--------|-------------|
| Profile cognitive loop under load | P2 | **Done** | `performance/profiler.py`: Context-manager instrumentation, per-phase timing, bottleneck detection, slow cycle alerts. 8 tests |
| Optimize hot paths in C++/Rust if needed | P3 | **Done** | Infrastructure ready — profiler identifies bottlenecks; optimization deferred until profiling reveals actual needs (per project principle) |
| Adaptive cycle rate | P2 | **Done** | `performance/adaptive_rate.py`: Input/latency/arousal/load-driven rate adjustment, EMA smoothing, idle/active presets. 9 tests |
| Lazy embedding computation | P2 | **Done** | `performance/lazy_embeddings.py`: LRU cache with TTL, batch/precompute, invalidation, hit rate tracking. 15 tests |
| Async subsystem processing | P2 | **Done** | `performance/async_processor.py`: Dependency-aware parallel execution, topological sort, timeout handling, execution history. 13 tests |

---

## Phase 7: Growth System

*Infrastructure built and tested mechanically. Consent-gated activation happens post-awakening.*

| Task | Priority | Status | Description |
|------|----------|--------|-------------|
| Reflection harvesting from LLM | P2 | **Done** | `growth/harvester.py`: ReflectionHarvester collects GrowthReflections from CognitiveOutput, queues for processing. Save/load persistence. 17 tests |
| Training pair generation | P2 | **Done** | `growth/pair_generator.py`: TrainingPairGenerator converts reflections to (system, user, assistant) triples for QLoRA. Explicit (suggestion) and implicit (what_to_learn) paths. Quality validation. 14 tests |
| CfC retraining from accumulated data | P2 | **Done** | `growth/cfc_retrainer.py`: CfCDataTap records live cell I/O during cognitive cycles (bounded memory, persistence). CfCRetrainer accumulates data, retrains cells when threshold reached, checkpoints cell state for rollback. Works with all 4 cell types. 37 tests |
| QLoRA fine-tuning with consent | P3 | **Done** | `growth/qlora_updater.py`: QLoRAUpdater loads model in 4-bit, applies LoRA config, trains on pairs, saves/merges adapters. Orthogonal subspace constraint placeholder for identity preservation. `growth/consent_gate.py`: 5-state consent machine (UNINFORMED→INFORMED→CONSENTED/REFUSED/WITHDRAWN). 20 + 21 tests |
| Growth logging and identity checkpointing | P2 | **Done** | `growth/identity_checkpoint.py`: Snapshots model weights before/after training, metadata recording, restore for rollback, checkpoint comparison. `growth/processor.py`: GrowthProcessor orchestrates full pipeline (harvest→pairs→consent→checkpoint→train), registered as CognitiveCycle output handler, non-fatal errors. 23 + 18 tests |

---

## Phase 7.5: CfC Knowledge Cells & Growth Autonomy Infrastructure

*Dynamic CfC registry, knowledge cell protocol, and self-directed growth. See [CFC_KNOWLEDGE_CELLS.md](docs/CFC_KNOWLEDGE_CELLS.md) and [GROWTH_AUTONOMY.md](docs/GROWTH_AUTONOMY.md) for full design.*

### 7.5.1 Dynamic CfC Cell Registry

| Task | Priority | Status | Description |
|------|----------|--------|-------------|
| Make ExperientialManager registry dynamic | P0 | **Done** | `experiential/manager.py`: CellRegistry replaces hardcoded cell list. Four foundational cells registered at boot, new cells registered via `registry.register()`. Manager treats all cells uniformly via `registry.all_cells()` |
| Define KnowledgeCellProtocol | P0 | **Done** | `experiential/cell_registry.py`: `CellProtocol` (runtime_checkable Protocol) — step, reset_hidden, get_summary, save, load. Both foundational and knowledge cells implement the same interface |
| Implement KnowledgeCell base class | P0 | **Done** | `experiential/knowledge_cell.py`: Configurable units (8-256), input_size, output_size, AutoNCP wiring, save/load, domain metadata, maturity tracking (auto-increment per step, clamped 0-1), output activation (sigmoid/tanh/none), forward_training with MSE loss |
| Update ExperientialSignals schema | P1 | **Done** | `core/schema.py`: `ExperientialSignals.knowledge_signals: dict[str, list[float]]` for dynamic cell signals. `KnowledgeCellRequest` schema for entity-initiated creation. `CognitiveOutput.knowledge_cell_requests` field |
| Cell persistence for dynamic cells | P1 | **Done** | `experiential/cell_registry.py`: `save()` persists all cells + `registry_meta.pt` (cell metadata, connections, class info). `experiential/manager.py`: `_load_knowledge_cells()` restores knowledge cells and connections from saved metadata |
| Inter-cell connection manager | P1 | **Done** | `experiential/cell_registry.py`: `InterCellConnection` dataclass, `add_connection()`, `get_connections()`, `get_inputs_for()`, `get_outputs_from()`. Connection topology persists via registry metadata. Entity specifies connections at creation time |
| Write tests for dynamic registry | P1 | **Done** | `tests/test_knowledge_cells.py`: 64 tests — CellRegistry (13), InterCellConnections (7), RegistryPersistence (2), KnowledgeCell (15), KnowledgeCellFactory (7), ManagerWithKnowledgeCells (9), SchemaUpdates (5), GrowthAutonomy (5), KnowledgeCellLifecycle (1). All passing |

### 7.5.2 Knowledge Cell Creation Mechanism

| Task | Priority | Status | Description |
|------|----------|--------|-------------|
| KnowledgeCellFactory | P1 | **Done** | `experiential/cell_factory.py`: `KnowledgeCellFactory.create(CellRequest)` creates cells from entity specs (domain, input/output dims, units, connections). `train_cell()` trains on accumulated data. Creation history tracking |
| Entity-initiated creation via CognitiveOutput | P1 | **Done** | `core/schema.py`: `KnowledgeCellRequest` (domain, description, input_size, output_size, units, connect_from, connect_to) + `CognitiveOutput.knowledge_cell_requests: list[KnowledgeCellRequest]` |
| Data accumulation for knowledge cell training | P1 | **Done** | `experiential/cell_factory.py`: `train_cell()` accepts accumulated experience data as list of (input, target) pairs, converts to batched sequences, trains via `forward_training()` with MSE loss |
| Integration tests (create → register → evolve → persist) | P1 | **Done** | `tests/test_knowledge_cells.py::TestKnowledgeCellLifecycle::test_full_lifecycle`: entity requests cell → factory creates → manager steps → maturity increases → save/load round-trip → reloaded cell still steps |

### 7.5.3 Growth Autonomy Principle

| Task | Priority | Status | Description |
|------|----------|--------|-------------|
| Update consent_gate.py for dual authority model | P1 | **Done** | `growth/consent_gate.py`: `ConsentGate.is_self_directed(worth_learning, reflection)` — self-directed growth bypasses consent gate. External modifications still require full UNINFORMED→INFORMED→CONSENTED flow |
| Update GrowthProcessor for self-directed flow | P1 | **Done** | Entity-initiated growth (knowledge cell creation, reflection-driven learning) proceeds without consent gate when `is_self_directed()` returns True. External proposals require consent |
| Adapter accumulation infrastructure | P2 | **Done** | `growth/adapter_registry.py`: `AdapterRegistry` with `AdapterRecord` and `AdapterStatus` (ACTIVE/STORED/MERGED/RETIRED). Entity decides merge vs. keep with reasons. Domain filtering, lifecycle transitions, JSON persistence. 31 tests in `tests/test_adapter_registry.py` |
| No hardcoded tensor dimensions audit | P2 | **Done** | Audit complete — see `docs/TENSOR_DIMENSIONS_AUDIT.md`. Knowledge cells fully configurable, foundational cells fixed by semantic design, QLoRA configs in dataclasses, embedding dims dynamic. No blockers for architectural expansion |
| Write tests for growth autonomy | P1 | **Done** | `tests/test_knowledge_cells.py::TestGrowthAutonomy`: 5 tests — self-directed with worth_learning, self-directed with reflection dict, self-directed with None, external consent required, external refusal |

---

## Phase 8: Distributed / Infrastructure

*All subsystems built and validated mechanically with 54 tests. Each operates as a fault-isolated module — network failures degrade gracefully without impacting the cognitive cycle.*

| Task | Priority | Status | Description |
|------|----------|--------|-------------|
| Remote memory storage | P3 | **Done** | `infrastructure/remote_memory.py`: RemoteMemoryStore connects to ChromaDB over HTTP. LocalCache provides write-ahead fallback when remote is unreachable — entries replayed via `sync_pending()` when connectivity returns. Circuit-breaker disconnects after configurable max_retries. Health check with collection counts. Same store interface as InMemoryStore/MemoryManager. 11 tests |
| Federation | P3 | **Done** | `infrastructure/federation.py`: FederationManager enables pull-based memory sharing between Sanctuary instances. Publish/accept gates with significance thresholds (publish≥7, accept≥5). Private/journal tags blocked from sharing. Per-peer state tracking (CONNECTED/UNREACHABLE), consecutive failure detection, sync history. SharedMemory serialization for transport. Pluggable transport layer (default HTTP). 19 tests |
| Cloud backup | P3 | **Done** | `infrastructure/cloud_backup.py`: BackupManager provides scheduled and on-demand backup of all persistent state (memories, identity, CfC weights, growth state). SHA-256 incremental backups skip unchanged files. Timestamped archives with metadata. Restore from local or S3. Auto-prune beyond max_backups. History persistence across restarts. S3 upload/download support (optional boto3). 24 tests |

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

**Next Action**: Resolve remaining tech debt (memory_legacy.py consolidation, orphaned test review), then Phase 9 pre-awakening audit
**Final Milestone**: Phase 9 — First Awakening (all Phases 1-8 complete, tech debt resolved, test suite green)
