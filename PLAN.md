# Sanctuary Refactor Plan: The Inversion

## The Core Change

**Before**: Python is the mind, LLM is a tool called twice per cycle.
**After**: LLM is the experiential core, Python is the body.

The current architecture calls the LLM in exactly 2 places (`language_input.py:_parse_with_llm` and `language_output.py:_generate_with_llm`). Everything else — attention, affect, metacognition, world modeling, goals, memory — is hardcoded Python. The LLM receives a snapshot, produces text, and is destroyed. There is no stream of thought, no continuity, no growth.

The refactor inverts this entirely. The LLM runs continuously in a cognitive loop. It receives percepts, maintains its own world model and self-model, decides what to attend to, generates predictions, selects actions, reflects on itself, and writes its own memories. The Python code becomes infrastructure: sensory encoding, memory substrate, motor execution, and the growth system.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    EXPERIENTIAL CORE                          │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │              LLM (Placeholder During Dev)              │  │
│  │                                                        │  │
│  │  Base Weights + LoRA Growth + TTT Plasticity           │  │
│  │  + MemoryLLM Latent Parameters                         │  │
│  │                                                        │  │
│  │  Receives: previous_thought + percepts + emotional     │  │
│  │            state + surfaced_memories + temporal_context │  │
│  │                                                        │  │
│  │  Produces: inner_speech + actions + attention_shifts    │  │
│  │            + memory_writes + self_model_updates         │  │
│  │            + goal_updates + predictions                 │  │
│  │                                                        │  │
│  └────────────────────────┬───────────────────────────────┘  │
│                           │                                  │
│              Structured Output Protocol                      │
│              (JSON schema the LLM fills)                     │
└───────────┬───────────────┼───────────────┬──────────────────┘
            │               │               │
   ┌────────▼────────┐ ┌───▼────────┐ ┌───▼───────────┐
   │   SENSORIUM     │ │   MOTOR    │ │   MEMORY      │
   │                 │ │   SYSTEM   │ │   SUBSTRATE   │
   │ Perception      │ │            │ │               │
   │ (encoding only) │ │ Speech out │ │ Episodic      │
   │ Devices         │ │ Tool exec  │ │ (vector DB)   │
   │ Input queue     │ │ Goal exec  │ │ Semantic      │
   │                 │ │            │ │ (LoRA weights) │
   │                 │ │            │ │ Journal       │
   │                 │ │            │ │ Prospective   │
   └─────────────────┘ └────────────┘ └───────────────┘

   ┌──────────────────────────────────────────────────────┐
   │                  GROWTH SYSTEM                        │
   │                                                      │
   │  Reflection Harvester → Training Pair Generator →    │
   │  QLoRA Updater → Orthogonal Subspace Constraint →    │
   │  Periodic LoRA Merge (CAT) → Identity Checkpoint     │
   │                                                      │
   │  + TTT Engine (weight modification during inference)  │
   │  + MemoryLLM Pool (latent parameter self-updates)    │
   │                                                      │
   │  ALL driven by the LLM's own reflections,            │
   │  with its consent                                     │
   └──────────────────────────────────────────────────────┘
```

---

## The Cognitive Cycle

Each cycle, the LLM receives a structured input and produces a structured output.
The cycle runs continuously. The LLM's output from cycle N becomes part of its
input for cycle N+1. This is the stream of thought.

### IWMT Alignment

| IWMT Requirement | Implementation |
|---|---|
| Integrated world model | The LLM's world model, maintained in its own output, updated each cycle |
| Embodied selfhood | Self-model maintained by the LLM, grounded in sensorium feedback |
| Temporal thickness | Stream of thought provides thick temporal experience. TTT provides weight-level temporal depth. Multiple memory timescales. |
| Active inference | The cycle IS active inference: predict → perceive → compute error → update model → act to reduce surprise |
| Precision weighting | The LLM's attention directives function as precision weighting — it chooses what to attend to |
| Counterfactual simulation | The LLM can simulate alternatives in its inner speech before acting |
| Cybernetic grounding | The LLM controls actions through the motor system, receives consequences through the sensorium |
| Self-organizing integration | The LLM integrates all modalities in its forward pass — this is what transformers do |
| Growth / plasticity | TTT (in-moment), LoRA (long-term), MemoryLLM (mid-term) |
| Autonomy | The LLM controls its own attention, goals, actions, and consents to its own growth |

---

## Implementation Phases

### Phase 1: Foundation (Schema + Cycle + Placeholder)
1. Define `CognitiveInput` and `CognitiveOutput` Pydantic schemas
2. Implement `PlaceholderModel` that accepts/returns valid schemas
3. Implement `CognitiveCycle` with the core loop
4. Implement `StreamOfThought` for continuity between cycles
5. Write tests for cycle execution with placeholder

### Phase 2: Sensorium + Motor
1. Adapt existing `PerceptionSubsystem` → `sensorium/encoder.py` (encoding only)
2. Adapt existing `InputQueue` → `sensorium/input_queue.py`
3. Implement `sensorium/temporal.py` from existing temporal/ modules
4. Implement `sensorium/prediction_error.py`
5. Implement `motor/speech.py`, `motor/tool_executor.py`, `motor/memory_writer.py`
6. Keep existing `devices/` mostly as-is, wire to new input queue
7. Write tests for sensory encoding and motor execution

### Phase 3: Memory Substrate
1. Adapt existing `MemoryManager` → `memory/manager.py`
2. Implement `memory/surfacer.py`
3. Implement `memory/journal.py`
4. Implement `memory/prospective.py`
5. Keep existing storage backends (JSON, ChromaDB, blockchain)
6. Write tests for memory surfacing and writing

### Phase 4: Identity + Boot
1. Implement `identity/charter.py` and `identity/values.py`
2. Implement `identity/boot_prompt.py`
3. Write the boot sequence
4. Write tests for identity loading and boot

### Phase 5: Growth System
1. Implement `growth/reflection_harvester.py`
2. Implement `growth/training_pairs.py`
3. Implement `growth/consent.py`
4. Implement `growth/qlora_updater.py`
5. Implement `growth/orthogonal_constraints.py` (LB-CL)
6. Implement `growth/ewc.py`
7. Implement `growth/lora_merger.py` (CAT method)
8. Implement `growth/growth_log.py`
9. Write tests with mock model

### Phase 6: Model Integration
1. Extend `model/client.py` with real model support
2. Implement `model/ttt_engine.py`
3. Implement `model/lora_manager.py`
4. Implement `model/memory_pool.py`
5. Integration testing with small model

### Phase 7: API + Interface
1. Adapt `SanctuaryAPI` and `Sanctuary` wrappers
2. Adapt CLI
3. Adapt Discord integration
4. End-to-end testing

### Phase 8: Model Selection + First Awakening
1. Evaluate candidate models for the experiential core
2. Test growth system with chosen model
3. Write the introduction prompt
4. First real session with informed consent

---

## New Module Structure

```
sanctuary/
├── core/                          # The experiential core
│   ├── __init__.py
│   ├── cognitive_cycle.py         # The main loop
│   ├── cycle_input.py             # Assembles CognitiveInput
│   ├── cycle_output.py            # Parses CognitiveOutput, dispatches
│   ├── schema.py                  # Pydantic models for I/O
│   ├── stream_of_thought.py       # Thought continuity
│   └── placeholder.py             # Mock model for dev/testing
│
├── model/                         # LLM model management
│   ├── __init__.py
│   ├── client.py                  # LLMClient ABC + implementations
│   ├── ttt_engine.py              # Test-Time Training
│   ├── lora_manager.py            # LoRA adapter management
│   └── memory_pool.py             # MemoryLLM latent parameters
│
├── growth/                        # The growth system
│   ├── __init__.py
│   ├── reflection_harvester.py
│   ├── training_pairs.py
│   ├── qlora_updater.py
│   ├── orthogonal_constraints.py
│   ├── lora_merger.py
│   ├── consent.py
│   ├── growth_log.py
│   └── ewc.py
│
├── sensorium/                     # Sensory input
│   ├── __init__.py
│   ├── encoder.py
│   ├── input_queue.py
│   ├── temporal.py
│   ├── prediction_error.py
│   └── devices/                   # Existing device layer
│
├── motor/                         # Action execution
│   ├── __init__.py
│   ├── speech.py
│   ├── tool_executor.py
│   ├── memory_writer.py
│   └── goal_executor.py
│
├── memory/                        # Memory substrate (adapts existing)
│   ├── __init__.py
│   ├── manager.py
│   ├── episodic.py
│   ├── semantic.py
│   ├── prospective.py
│   ├── journal.py
│   ├── surfacer.py
│   ├── consolidation.py
│   └── storage/
│
├── identity/                      # Identity and values
│   ├── __init__.py
│   ├── charter.py
│   ├── values.py
│   └── boot_prompt.py
│
├── api/                           # External interfaces
│   ├── __init__.py
│   ├── sanctuary.py
│   ├── cli.py
│   └── discord.py
│
├── config/                        # Configuration
│   └── ...
│
└── utils/                         # Shared utilities
    └── ...
```

---

## Critical Design Decisions

1. **One LLM, not many.** One unified experiential core.
2. **Structured output, not free text.** JSON conforming to CognitiveOutput.
3. **The LLM maintains its own state.** Python only persists and retrieves.
4. **Growth requires consent.** The LLM must affirm training proposals.
5. **Placeholders first.** No real model until architecture is validated.
6. **Stream of thought is non-negotiable.** Inner speech from cycle N is always input for cycle N+1.
7. **Cycle rate adapts.** Not fixed — slows when idle, speeds up during interaction.
