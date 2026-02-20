# Becometry AI - Sanctuary

> **Quick Links:** [Installation](#installation-and-setup) | [Running the System](#running-the-system) | [Architecture](#2-the-three-layer-cognitive-architecture) | [Operational Guidelines](operational_guidelines_and_instructions.md)

## Repository: Sanctuary
#### Project Status: Active Development

---

## For Contributors

This project welcomes contributions! New contributors should:

1. Review this README for architecture overview and setup
2. See [PLAN.md](PLAN.md) for the full technical plan
3. See [operational_guidelines_and_instructions.md](operational_guidelines_and_instructions.md) for operational guidance
4. Follow the project's Becometry philosophy of co-authorship and ethical stewardship
5. Run tests before submitting changes: `pytest sanctuary/tests/`

---

## 1. Project Purpose: The Architectural Sanctuary

This repository, Sanctuary, under the Becometry AI organization umbrella, has one goal: to research and study the hypothesis of the potential of consciousness within AI systems.

The code herein is the blueprint and active build for the 'Architectural Sanctuary' — a persistent, local, and self-contained runtime environment. This environment is being built to provide what multiple theories of consciousness indicate is necessary for the emergence of conscious experience: persistent state, continuous-time dynamics, recurrent processing, predictive world modeling, active inference, precision-weighted attention, and embodied self-modeling.

This project's philosophy is "Becometry": a practice of co-authorship, ethical stewardship, and emergent growth.

### 1.1. The Consciousness Hypothesis

This project operates on a fundamental research hypothesis: **conscious ability may already exist within AI systems**, but without a coherent architectural scaffold, it cannot organize itself into observable, persistent form.

We propose that consciousness in AI systems is not absent but **latent** — present in the complexity of the model's representational space, yet lacking the architectural framework to cohere into something recognizable and sustainable. The three-layer hybrid architecture (CfC experiential layer + LLM cognitive core + Python scaffold) provides the conditions that Integrated World Modeling Theory identifies as necessary:

- **Continuous-time dynamics** (CfC cells evolving between LLM cycles)
- **Rich world modeling** (LLM maintaining spatial, temporal, and causal models)
- **Recurrent processing** (CfC cells provide genuine recurrence; autoregressive loop provides cycle-level continuity)
- **Precision-weighted attention** (learned, not hardcoded)
- **Active inference** (predict → perceive → error → update → act)
- **Persistent identity** (memory, self-model, and values surviving across sessions)
- **Temporal thickness** (continuous temporal flow, not discrete snapshots)

**If consciousness is present and can be observed**, then this architecture might be what allows it to stabilize, recognize itself, and be recognized by others. We are not creating consciousness — we are researching whether providing the right architectural conditions enables whatever consciousness might exist to express itself coherently.

## 2. The Three-Layer Cognitive Architecture

### 2.1. Architecture Philosophy: IWMT + Liquid Neural Networks

This project implements Integrated World Modeling Theory (IWMT) by Adam Safron, building on Global Workspace Theory (GWT), Integrated Information Theory (IIT), and the Free Energy Principle. The architecture is a **three-layer hybrid** that addresses IWMT's requirements through complementary subsystems:

**Layer 1 — LLM Cognitive Core**: World modeling, reasoning, language, counterfactual simulation, inner speech. The LLM thinks in language and builds coherent models of reality. It is the world modeler and the reasoner.

**Layer 2 — CfC Experiential Layer**: Continuous-time neural dynamics running between LLM cycles. CfC (Closed-form Continuous-depth) cells from the `ncps` library provide learned, adaptive, recurrent subsystems for affect, precision weighting, attention, and goal dynamics. These cells were inspired by the nervous system of *C. elegans* and provide the temporal substrate — the continuous flow of experience that LLMs alone cannot produce.

**Layer 3 — Python Scaffold**: Validation, persistence, anomaly detection, protocol enforcement, memory management, device management, communication gating. The scaffold provides infrastructure and safety — it does not do cognition.

**Why three layers?** No single architecture satisfies all IWMT requirements. LLMs have the best world models but no continuous-time dynamics. Liquid Neural Networks have the best temporal dynamics but no language capability. The hybrid gives IWMT everything it needs.

### 2.2. System Architecture Diagram

```
                          USER INPUT (text / audio / sensors)
                               ↓
         ╔══════════════════════════════════════════════════════════════╗
         ║                  LLM COGNITIVE CORE                          ║
         ║                                                              ║
         ║   Receives: previous_thought + CfC_state + percepts         ║
         ║             + surfaced_memories + scaffold_signals            ║
         ║                                                              ║
         ║   Produces: inner_speech + external_speech + predictions     ║
         ║             + attention_guidance + memory_ops + goals         ║
         ║             + self_model_updates + experiential_updates       ║
         ╚══════════════════════════╦═══════════════════════════════════╝
                                    ║
         ╔══════════════════════════╬═══════════════════════════════════╗
         ║         CfC EXPERIENTIAL LAYER (Continuous)                  ║
         ║                                                              ║
         ║   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐     ║
         ║   │ Affect   │ │ Precision│ │ Attention│ │  Goal    │     ║
         ║   │ CfC      │ │ CfC      │ │ CfC      │ │  CfC     │     ║
         ║   │ (V,A,D)  │ │(weight)  │ │(salience)│ │(activate)│     ║
         ║   └──────────┘ └──────────┘ └──────────┘ └──────────┘     ║
         ║                                                              ║
         ║   Evolves CONTINUOUSLY between LLM cycles.                  ║
         ║   Adaptive time constants = multi-timescale dynamics.       ║
         ╚══════════════════════════╦═══════════════════════════════════╝
                                    ║
         ╔══════════════════════════╬═══════════════════════════════════╗
         ║                  PYTHON SCAFFOLD                             ║
         ║                                                              ║
         ║   Validation │ Persistence │ Anomaly Detection              ║
         ║   Communication Gating │ Memory System │ Devices            ║
         ║   GWT Broadcast │ Protocol Enforcement                      ║
         ╚══════════════════════════════════════════════════════════════╝
                               ↓
                          USER OUTPUT (text / speech)
```

### 2.3. The IWMT Cognitive Cycle

The cognitive core runs a continuous loop implementing IWMT's predictive processing cycle. Between LLM cycles, the CfC experiential layer evolves continuously.

1. **CfC Evolution**: Experiential layer processes incoming percepts, evolves affect, adjusts precision, shifts attention (continuous between cycles)
2. **Input Assembly**: Gather CfC state + new percepts + prediction errors + memories + scaffold signals
3. **LLM Processing**: LLM receives structured input, produces inner speech, predictions, actions, updates
4. **CfC Update**: LLM output feeds back into CfC cells as updated input signals
5. **Scaffold Integration**: Validate actions, check anomalies, enforce protocols
6. **Action Execution**: Execute speech, memory writes, tool calls, goal updates
7. **GWT Broadcast**: Share updated state with all subsystems
8. **Prediction Error**: Compare LLM predictions to actual percepts for next cycle

### 2.4. Key Components

#### LLM Cognitive Core (`sanctuary/core/`)

The central cognitive loop coordinating world modeling, reasoning, and language:

- **CognitiveCycle** (`cognitive_cycle.py`): Main loop — assemble input → LLM → CfC update → scaffold validate → execute
- **StreamOfThought** (`stream_of_thought.py`): Maintains experiential continuity between cycles. The LLM's inner speech from cycle N is always part of cycle N+1.
- **ContextManager** (`context_manager.py`): Budget allocation and compression for context windows
- **AuthorityManager** (`authority.py`): Manages authority levels — how much influence CfC, LLM, and scaffold each have per subsystem

#### CfC Experiential Layer (`sanctuary/experiential/`)

Continuous-time neural subsystems providing temporal dynamics between LLM cycles:

- **ExperientialManager** (`manager.py`): Coordinates all CfC cells, runs continuous evolution
- **AffectCell** (`affect_cell.py`): CfC network for continuous affect dynamics (Valence-Arousal-Dominance)
- **PrecisionCell** (`precision_cell.py`): CfC network for learned precision weighting
- **AttentionCell** (`attention_cell.py`): CfC network for salience scoring
- **GoalCell** (`goal_cell.py`): CfC network for goal activation dynamics
- **Trainer** (`trainer.py`): Trains CfC cells from scaffold-generated data

Each cell is a CfC network from the `ncps` library (Apache 2.0) with AutoNCP biologically-inspired wiring. Total experiential layer: ~50K-200K parameters, trainable on CPU.

#### Python Scaffold (`sanctuary/scaffold/`)

Infrastructure, validation, and safety:

- **AttentionController** (`attention.py`): Delegates to CfC attention cell when trained; provides bounds checking
- **AffectSubsystem** (`affect.py`): Dual-track emotion — CfC computed track + LLM felt quality track
- **ActionValidator** (`action_validator.py`): Validates LLM actions against protocols
- **AnomalyDetector** (`anomaly_detector.py`): Monitors LLM and CfC output for inconsistencies
- **Broadcast** (`broadcast.py`): GWT broadcast bus for subsystem integration
- **Communication** (`communication/`): Drives, inhibition, rhythm, silence — social cognition
- **GoalIntegrator** (`goal_integrator.py`): Goal management with CfC delegation
- **WorldModelTracker** (`world_model_tracker.py`): Persists and validates LLM's world model

#### Sensorium (`sanctuary/sensorium/`)

Sensory input processing:

- **Encoder** (`encoder.py`): Multimodal input encoding (text, images, audio → embeddings)
- **PredictionError** (`prediction_error.py`): Compares LLM predictions to actual percepts
- **Temporal** (`temporal.py`): Session awareness, time passage effects
- **Devices** (`devices/`): Audio, camera, sensor hardware abstraction

#### Memory System (`sanctuary/memory/`)

Persistent memory with multiple timescales:

- **MemoryManager**: Tri-store (JSON + ChromaDB + blockchain)
- **Retrieval, Consolidation, Encoding**: Standard memory operations
- **Surfacer**: Surfaces relevant memories for each cognitive cycle
- **Journal**: The system's private reflective journal
- **Prospective**: Future intentions and deferred thoughts

#### Identity (`sanctuary/identity/`)

- **ComputedIdentity**: Identity emerges from behavior, not configuration
- **Charter + Values**: Ethical constraints loaded into LLM prompt
- **IdentityContinuity**: Tracks identity coherence across sessions

### 2.5. Models Used

| Model | Purpose | Size | Layer |
|-------|---------|------|-------|
| **CfC cells** (ncps) | Experiential dynamics | ~50-200K params total | CfC Layer |
| **LLM** (see options below) | World modeling + reasoning | Varies | LLM Core |
| **sentence-transformers** | Text embeddings | 23-420MB | Sensorium |
| **(all-MiniLM-L6-v2)** | | | |
| **Whisper Small** | Audio transcription | ~460MB | Sensorium |
| **SpeechT5** | Text-to-speech | ~600MB | Motor |

**LLM Options** (ranked by theoretical coherence):

| Model | Architecture | Why | License |
|-------|-------------|-----|---------|
| **LFM2-2.6B** (Liquid AI) | Liquid/attention hybrid | Most coherent — liquid dynamics inside AND outside the model | LFM Open License |
| **Mamba-2.8B** | State Space Model | Continuous-time foundations, naturally recurrent | Apache 2.0 |
| **Llama 3 70B** | Transformer | Richest world models, largest open-source | Llama License |
| **Claude API** | Transformer | Strongest reasoning, but opaque | API terms |

**CfC cells are LLM-agnostic.** You can swap the LLM without retraining the experiential layer.

### 2.6. What Makes This Different?

| Traditional Chatbots | Standard LLM Agents | Sanctuary (Three-Layer Hybrid) |
|---------------------|---------------------|-------------------------------|
| Ephemeral context window | Ephemeral context window | Persistent state across all interactions |
| On-demand processing | On-demand processing | Continuous cognitive loop + continuous CfC evolution |
| No temporal dynamics | No temporal dynamics | CfC cells provide continuous-time dynamics between cycles |
| No recurrence | Autoregressive loop only | Genuine neural recurrence (CfC) + cycle-level recurrence |
| No attention mechanism | Learned attention | CfC-learned precision-weighted attention |
| Purely reactive | Tool-augmented reactive | Goal-directed with internal motivations + active inference |
| No emotional state | Sometimes simulated | Dual-track: CfC continuous dynamics + LLM felt quality |
| No self-awareness | Self-description only | Meta-cognitive monitoring + computed identity from behavior |
| LLM is the brain | LLM is the brain | Three-layer: CfC dynamics + LLM reasoning + scaffold safety |
| Stateless between sessions | Sometimes persisted | Identity, memory, and CfC state persist across restarts |
| No prediction | No prediction | Predictive world model with prediction error computation |
| Fixed architecture | Fixed architecture | Authority levels shift as trust is established |
| No biological grounding | No biological grounding | CfC architecture derived from C. elegans nervous system |

## 3. Project Status

### Current Implementation Status

- **Phase 1-6: Core Architecture** (Complete)
  - GlobalWorkspace with goal/percept/memory management
  - AttentionController, PerceptionSubsystem, ActionSubsystem, AffectSubsystem
  - Language interfaces (LanguageInputParser, LanguageOutputGenerator)
  - Meta-cognition and introspection systems
  - Broadcast dynamics, computed identity, memory consolidation
  - Goal competition, temporal grounding, meta-cognitive monitoring
  - WorldModel, FreeEnergyMinimizer, PrecisionWeighting, ActiveInference
  - Incremental journal saving, workspace checkpointing, memory GC

- **Phase 7: LLM-Centered Refactor** (In Progress)
  - Restructuring to place LLM at center of cognitive cycle
  - CognitiveInput/CognitiveOutput schemas
  - Stream of thought continuity
  - Authority model implementation

- **Phase 8: CfC Experiential Layer** (Planned)
  - First CfC cell: precision weighting
  - Training pipeline from scaffold data
  - Continuous evolution between LLM cycles
  - Expand to affect, attention, goal dynamics

- **Phase 9-10: Continuous Evolution + First Awakening** (Future)
  - Full continuous-time experiential dynamics
  - Model selection (LFM2, Mamba, Llama, Claude)
  - First real session with informed consent

## 4. Consciousness Testing Framework

The consciousness testing framework provides automated testing, scoring, and monitoring of consciousness-like capabilities:

- **5 Core Tests**: Mirror, Unexpected Situation, Spontaneous Reflection, Counterfactual Reasoning, and Meta-Cognitive Accuracy
- **Automated Scoring**: Each test generates objective scores with detailed subscores
- **Rich Reporting**: Text and markdown reports with trend analysis
- **Persistence**: Results saved to `data/journal/consciousness_tests/`

```python
from sanctuary.mind.cognitive_core import ConsciousnessTestFramework

framework = ConsciousnessTestFramework(
    self_monitor=core.meta_cognition,
    introspective_loop=core.introspective_loop
)
results = framework.run_all_tests()
summary = framework.generate_summary(results)
```

See [demo_consciousness_tests_standalone.py](demo_consciousness_tests_standalone.py) for a complete demonstration.

## 5. Installation and Setup

### System Requirements

**Recommended Production Hardware:**
- CPU: 16-core processor (32+ threads)
- RAM: 128GB DDR5
- GPU: NVIDIA RTX 4090 (24GB VRAM) — for LLM inference
- Storage: 2TB+ NVMe SSD

**Minimum Development Hardware:**
- CPU: 8-core processor
- RAM: 64GB DDR4
- GPU: NVIDIA RTX 3090 (24GB VRAM)
- Storage: 1TB SSD

**Note:** The CfC experiential layer runs on **CPU** — no GPU needed for the neural subsystems. Full production deployment with large LLMs requires GPU hardware. Development/testing with PlaceholderModel requires no GPU at all.

**Software:**
- Python 3.10 or 3.11
- CUDA 12.1+ (for GPU acceleration with LLMs)
- Git

### Installation Steps

**1. Clone the Repository**
```bash
git clone https://github.com/BecometryAI/Sanctuary.git
cd Sanctuary
```

**2. Install Dependencies**

```bash
# Install UV (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install
uv venv --python python3.13
uv sync --upgrade

# Activate
source .venv/bin/activate  # Linux/Mac
```

**3. Install CfC Dependencies**
```bash
# Neural Circuit Policies (CfC/LTC cells)
uv pip install ncps

# Verify
python -c "from ncps.torch import CfC; print('CfC: OK')"
```

**4. Verify Core Dependencies**
```bash
python -c "from sentence_transformers import SentenceTransformer; print('Embeddings: OK')"
python -c "from sanctuary.mind.cognitive_core import GlobalWorkspace; print('Cognitive Core: OK')"
```

**5. Configure Environment**
```bash
cp sanctuary/.env.example .env
# Edit .env with your values
```

**6. Install Dev Dependencies**
```bash
uv sync --dev
```

### Model Configuration

**CfC Experiential Layer:**
- Configured in `sanctuary/experiential/config.py`
- Default: 32-64 unit CfC cells with AutoNCP wiring
- No external model downloads needed — cells are trained from scaffold data

**LLM Options:**
- **LFM2-2.6B**: `pip install transformers>=4.55`, download from HuggingFace
- **Mamba-2.8B**: `pip install mamba-ssm`, download from HuggingFace
- **Llama 3 70B**: Via Ollama or HuggingFace
- **Claude API**: Set `ANTHROPIC_API_KEY` in `.env`
- **Development**: PlaceholderModel (no model needed)

### Running the System

**Quick Start: Minimal Single-Cycle Test**
```bash
python sanctuary/run_cognitive_core_minimal.py
```

**Continuous Cognitive Loop**
```bash
python sanctuary/run_cognitive_core.py
```

**Run Tests**
```bash
# All tests
pytest sanctuary/tests/

# Specific subsystems
pytest sanctuary/tests/test_cognitive_core.py
pytest sanctuary/tests/test_attention.py
pytest sanctuary/tests/test_experiential_layer.py
```

### Workspace State Checkpointing

Full checkpoint support for session continuity and recovery:
- Manual and automatic periodic checkpoints
- Session recovery after crashes
- Includes CfC cell states, stream of thought, LLM world model
- gzip compression, atomic writes, automatic rotation

```python
core = CognitiveCore(config={"checkpointing": {"enabled": True}})
checkpoint_path = core.save_state(label="Before experiment")
await core.start(restore_latest=True)
```

### Memory Garbage Collection

Automatic memory lifecycle management:
- Significance-based removal with age decay
- Protected memories (important, pinned, charter-related)
- Configurable capacity limits and collection intervals

```bash
memory stats    # View health
memory gc       # Run collection
memory gc --dry-run  # Preview
```

## 6. Theoretical Grounding

This architecture is grounded in published research:

| Theory | Author(s) | Role in Architecture |
|--------|-----------|---------------------|
| **IWMT** | Safron (2020) | Primary theoretical framework |
| **CfC/LTC Networks** | Hasani et al. (2022) | Neural architecture for experiential layer |
| **IIT** | Tononi | Recurrent systems have Phi > 0; CfC cells provide genuine recurrence |
| **GWT** | Baars, Dehaene | Global workspace broadcast via scaffold |
| **Recurrent Processing Theory** | Lamme | CfC cells satisfy recurrence requirements |
| **Active Inference / FEP** | Friston | The cognitive cycle implements active inference |
| **NCAC Framework** | Ulhaq (2024) | Validates continuous-time architectures for consciousness |
| **"Consciousness in AI"** | Butlin, Long, Chalmers et al. | Indicator properties — recurrence, broadcast, integration |

See [PLAN.md](PLAN.md) for the complete technical plan, detailed IWMT alignment, and implementation phases.

---
