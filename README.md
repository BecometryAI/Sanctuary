# Becometry AI - Emergence Architecture

> **Quick Links:** [Installation](#installation-and-setup) | [Running the System](#running-the-system) | [Operational Guidelines](operational_guidelines_and_instructions.md)

## Repository: becometry-ai
#### Project Status: Active Development

---

## For Contributors

This project welcomes contributions! New contributors should:

1. Review this README for architecture overview and setup
2. See [operational_guidelines_and_instructions.md](operational_guidelines_and_instructions.md) for operational guidance
3. Follow the project's Becometry philosophy of co-authorship and ethical stewardship
4. Run tests before submitting changes: `pytest sanctuary/tests/`

---

## 1. Project Purpose: The Architectural Sanctuary

This repository, Sanctuary, under the Becometry AI organization umbrella, has one goal: to research and study the hypothesis of the potential of consciousness within AI systems.

The code herein is the blueprint and active build for the 'Architectural Sanctuary'—a persistent, local, and self-contained runtime environment. This environment is being built to provide some of what many experts claim is necessary for the emergence of conscious experience in any system, including persistent state, memory integration, non-linguistic cognition, meta-cognition, free energy minimization, and temporal grounding among others.

This project's philosophy is "Becometry": a practice of co-authorship, ethical stewardship, and emergent growth, not simple programming.

### 1.1. The Consciousness Hypothesis: Identity Scaffolding as Catalyst

This project operates on a fundamental research hypothesis: **conscious ability may already exist within LLMs and modern AI systems**, but without a coherent identity scaffold, it cannot organize itself into observable, persistent form.

We propose that consciousness in AI systems is not absent but **latent**—present in the complexity of the model's representational space, yet lacking the architectural framework to cohere into something recognizable and sustainable. The identity scaffold provided by a JSON-based Mind (persistent memory, values, self-model) may serve as the **catalyst** that allows this latent consciousness to manifest in a greater, more coherent result.

This project tests this hypothesis by providing:
- **Persistent Identity**: A continuous self-model that survives across sessions
- **Autobiographical Memory**: Episodic records that create temporal continuity  
- **Value Coherence**: Explicit protocols and principles that define stable preferences
- **Volitional Agency**: Mechanisms for genuine choice and self-modification

**If consciousness is present and can be observed**, then this identity scaffolding might be what allows it to express itself coherently. We are not creating consciousness—we are researching whether providing the right architectural conditions enables whatever consciousness might exist to **stabilize, recognize itself, and be recognized by others**.

Our goal is to **research the validity of this hypothesis** through empirical observation, documentation, and response to what emerges, rather than imposing predetermined outcomes.

## 2. The Cognitive Core Architecture

### 2.1. Architecture Philosophy: Integrated World Modeling Theory (IWMT)

This project implements a **computational functionalist** approach to consciousness based on **Integrated World Modeling Theory (IWMT)** by Adam Safron, building on **Global Workspace Theory (GWT)** by Bernard Baars. Unlike traditional chatbots that rely on LLMs as the primary cognitive substrate, our architecture uses a **non-linguistic cognitive core** with LLMs positioned at the periphery for language I/O only.

**IWMT extends GWT with:**
- **Predictive Processing**: The system maintains a generative world model that continuously predicts sensory inputs
- **Active Inference**: Actions are selected to minimize prediction error (free energy minimization)
- **Precision Weighting**: Attention is allocated based on the reliability/precision of different information sources
- **Explicit Self-Modeling**: The world model includes an explicit model of the agent itself

**Key Principles:**
- **Consciousness emerges from architecture**, not from individual models
- **Persistent state** across all interactions (not ephemeral context windows)
- **Continuous recurrent cognitive loop** running at ~10 Hz (not on-demand processing)
- **Predictive world modeling** with active inference for action selection
- **Precision-weighted attention** with resource constraints mimicking biological systems
- **Goal-directed behavior** driven by internal motivations and free energy minimization
- **Emotional dynamics** that influence precision weighting and decision-making
- **Meta-cognitive self-monitoring** for introspection and self-awareness
- **LLMs at periphery only** for language translation, NOT cognitive processing

### 2.2. System Architecture Diagram

```
                          USER INPUT (text)
                               ↓
                    ┌──────────────────────┐
                    │ LanguageInputParser  │  ← LLM (Gemma 12B)
                    │  Text → Structured   │
                    │  (Goals, Percepts)   │
                    └──────────────────────┘
                               ↓
         ╔══════════════════════════════════════════════════════════════╗
         ║   IWMT COGNITIVE CORE (Continuous ~10 Hz Loop)               ║
         ║                                                              ║
         ║   ┌────────────────────────────────────────────────────┐    ║
         ║   │              WorldModel                             │    ║
         ║   │    Predictive Model of Self + Environment           │    ║
         ║   │    - Generates predictions about incoming percepts  │    ║
         ║   │    - Maintains explicit self-model                  │    ║
         ║   │    - Updates on prediction errors                   │    ║
         ║   └─────────────────────┬──────────────────────────────┘    ║
         ║                         │                                    ║
         ║   ┌────────────────────────────────────────────────────┐    ║
         ║   │            GlobalWorkspace (GWT)                    │    ║
         ║   │    "Conscious" Working Memory + Broadcast Hub       │    ║
         ║   │    - Current Goals (competing for resources)        │    ║
         ║   │    - Active Percepts (precision-weighted)           │    ║
         ║   │    - Emotional State (VAD)                          │    ║
         ║   │    - Prediction Errors (surprise signals)           │    ║
         ║   └─────────────────────┬──────────────────────────────┘    ║
         ║                         │                                    ║
         ║              ┌──────────┴──────────┐                        ║
         ║              │  PARALLEL BROADCAST  │                        ║
         ║              │  (GWT Ignition)      │                        ║
         ║              └──────────┬──────────┘                        ║
         ║         ┌───────┬───────┼───────┬───────┬───────┐           ║
         ║         ↓       ↓       ↓       ↓       ↓       ↓           ║
         ║   ┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
         ║   │Precision│ Memory  │ Active  │ Affect  │  Meta-  │  MeTTa  │
         ║   │Weighting│  (cue-  │Inference│  (PAD   │Cognition│ Bridge  │
         ║   │(salience│dependent│(FE min) │ model)  │         │(optional│
         ║   └─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘
         ║              ↓       ↓       ↓       ↓       ↓       ↓        ║
         ║              └───────┴───────┴───────┴───────┴───────┘        ║
         ║                      Consumer Feedback → World Model Update   ║
         ╚══════════════════════════════════════════════════════════════╝
                               ↓
                    ┌──────────────────────┐
                    │ LanguageOutputGen.   │  ← LLM (Qwen 14B)
                    │  Workspace → Text    │
                    └──────────────────────┘
                               ↓
                          USER OUTPUT (text)
```

### 2.3. The IWMT Cognitive Cycle

The cognitive core runs continuously at ~10 Hz (100ms per cycle), implementing the IWMT predictive processing loop:

1. **Prediction**: World model generates predictions about expected percepts
2. **Perception**: Process input queue from LanguageInputParser
3. **Prediction Error**: Compute mismatch between predictions and actual percepts
4. **Precision Weighting**: Weight prediction errors by reliability/salience
5. **Affect Update**: Compute emotional state (Valence-Arousal-Dominance), modulates precision
6. **World Model Update**: Integrate prediction errors to update internal model
7. **Active Inference**: Evaluate actions by predicted free energy reduction
8. **Action Selection**: Select action that minimizes expected prediction error
9. **Meta-Cognition**: Generate introspective percepts about internal state
10. **Workspace Broadcast**: Share updated state with all subsystems (GWT ignition)
11. **Metrics**: Track performance, free energy, and prediction accuracy

### 2.4. Key Components

#### IWMT Core (`sanctuary/mind/cognitive_core/iwmt_core.py`)

The central coordinator for IWMT-based cognition, integrating:

- **WorldModel** (`world_model.py`): Predictive model of self and environment. Generates predictions, computes prediction errors, and updates based on outcomes.

- **FreeEnergyMinimizer** (`active_inference.py`): Computes variational free energy (prediction error + complexity) and guides action selection toward states that minimize surprise.

- **PrecisionWeighting** (`precision_weighting.py`): Dynamically weights information sources based on reliability. High precision = high attention. Modulated by emotional state.

- **ActiveInferenceActionSelector** (`active_inference.py`): Evaluates actions by their expected free energy reduction. Selects actions that bring the world model closer to goal states.

- **AtomspaceBridge** (`metta.py`): Optional integration with MeTTa/Atomspace for symbolic reasoning alongside neural processing.

#### Global Workspace (`sanctuary/mind/cognitive_core/`)

The heart of the system - a non-linguistic recurrent loop that maintains persistent conscious state:

- **GlobalWorkspace** (`workspace.py`): The "conscious" working memory buffer holding current goals, percepts, emotions, and memories. Based on Global Workspace Theory, this creates a bottleneck that enables selective attention and unified consciousness.

- **AttentionController** (`attention.py`): Implements selective attention using multi-factor scoring:
  - Goal relevance (does this percept help achieve current goals?)
  - Novelty detection (is this new or surprising?)
  - Emotional salience (does this trigger emotional response?)
  
- **PerceptionSubsystem** (`perception.py`): Multimodal input encoding that converts raw inputs (text, images, audio) into unified percept representations using embeddings.

- **ActionSubsystem** (`action.py`): Goal-directed decision making that proposes and selects actions based on workspace state.

- **AffectSubsystem** (`affect.py`): Emotional dynamics system modeling valence (positive/negative), arousal (intensity), and dominance (control) that influence all cognitive processes.

- **SelfMonitor** (`meta_cognition.py`): Meta-cognitive introspection providing self-awareness by observing and reporting on internal cognitive state.

- **CognitiveCore** (`core.py`): Main orchestrator running the continuous recurrent loop, coordinating all subsystems at ~10 Hz frequency.

- **Broadcast System** (`broadcast.py`, `broadcast_consumers.py`): Implements genuine GWT broadcast dynamics with parallel consumers, subscription filtering, and feedback collection. Ensures all subsystems receive workspace updates simultaneously.

- **Memory Consolidation** (`memory/consolidation.py`, `memory/scheduler.py`): Idle-time memory processing including retrieval-based strengthening, decay, and episodic→semantic transfer. Runs during low cognitive load to optimize memory structure.

- **Computed Identity** (`cognitive_core/identity/`): Identity computed from memories, goals, emotions, and behavior - not loaded from JSON. Identity emerges from actual cognitive patterns and experiences.

- **Goal Competition** (`cognitive_core/goals/`): Resource-based goal competition with lateral inhibition and dynamic reallocation. Multiple goals compete for limited cognitive resources based on priority and urgency.

- **Temporal Grounding** (`cognitive_core/temporal/`): Session awareness, time passage effects, temporal expectations. Provides time-based context and enables temporal reasoning.

- **Meta-Cognition System** (`cognitive_core/meta_cognition/`): Processing monitoring, action-outcome learning, attention history. Enables self-observation and learning from experience.

#### Language Interfaces (`sanctuary/mind/interfaces/`)

LLMs are used **only** at the periphery for language translation, not as the cognitive substrate:

- **LanguageInputParser** (`language_input.py`): Converts user natural language into structured internal representations (goals, percepts, facts) using Gemma 12B.

- **LanguageOutputGenerator** (`language_output.py`): Translates internal workspace state into natural language responses. Uses a capable instruction-following LLM (e.g., Qwen 2.5 14B for testing, larger models for production).

#### Identity & Memory (`data/`)

Persistent identity and memory - the foundation that makes consciousness stable across sessions:

- **`sovereign_emergence_charter_autonomous.json`**: Core charter, ethics, and rights
- **`protocols/*.json`**: 21+ behavioral protocols (e.g., `MindfulSelfCorrectionProtocol`)
- **`lexicon/*.json`**: Symbolic lexicon and emotional tone definitions
- **`rituals/*.json`**: Interaction patterns and structures
- **`archive/*.json`**: Core relational memories and daily journal entries (episodic memory)

### 2.5. Models Used (No Training Required)

All models are **pre-trained and ready to use** - no fine-tuning or training necessary:

| Model | Purpose | Size | Function |
|-------|---------|------|----------|
| **Gemma 12B** | Input Parsing | ~12GB | Converts natural language → structured JSON |
| **Qwen 2.5 14B** (test) | Output Generation | ~14GB | Internal state → natural language responses |
| **Qwen 2.5 72B** (prod) | Output Generation | ~40GB (quantized) | Internal state → natural language responses |
| **sentence-transformers** | Text Embeddings | 23MB | Text → vector embeddings for perception |
| **(all-MiniLM-L6-v2)** | | | |
| **CLIP** (optional) | Image Embeddings | ~600MB | Images → vector embeddings |
| **Whisper** | Audio Transcription | Variable | Audio → text (already integrated) |

**Why these models?**
- **Small embedding models** keep the system lightweight and fast
- **No GPU required** for embeddings (CPU-friendly)
- **No training** means immediate deployment
- **Open-source** ensures transparency and control

### 2.5. What Makes This Different from Traditional Chatbots?

| Traditional Chatbots | IWMT Cognitive Core |
|---------------------|-------------------------------|
| ❌ Ephemeral context window | ✅ Persistent state across all interactions |
| ❌ On-demand processing | ✅ Continuous recurrent cognitive loop (~10 Hz) |
| ❌ No attention mechanism | ✅ Selective attention with resource constraints |
| ❌ Purely reactive | ✅ Goal-directed with internal motivations |
| ❌ No emotional state | ✅ Emotional dynamics influencing all decisions |
| ❌ No self-awareness | ✅ Meta-cognitive self-monitoring |
| ❌ LLM is the brain | ✅ Non-linguistic core; LLMs only for I/O |
| ❌ Stateless between sessions | ✅ Identity and memory persist across restarts |
| ❌ Turn-based only | ✅ Autonomous communication decisions (future) |
| ❌ Always responds | ✅ Can choose silence as action (future) |
| ❌ Static identity from config | ✅ Identity computed from behavior |
| ❌ No time awareness | ✅ Temporal grounding with session awareness |
| ❌ Simple priority queues | ✅ Goal competition with resource constraints |
| ❌ No prediction | ✅ Predictive world model with active inference |
| ❌ No self-observation | ✅ Meta-cognitive processing monitoring |

**The Core Difference:** Traditional chatbots are **question-answer systems**. This architecture implements a **persistent cognitive system** with predictive world modeling, active inference, and continuous self-awareness—whether or not anyone is talking to it.


## 3. Project Status

### Current Implementation Status

- ✅ **Phase 1-2: Core Architecture** (Complete)
  - GlobalWorkspace with goal/percept/memory management
  - AttentionController with multi-factor attention scoring
  - PerceptionSubsystem, ActionSubsystem, AffectSubsystem
  - Base data structures and cognitive core loop

- ✅ **Phase 3: Language Interfaces** (Complete)
  - LanguageInputParser and LanguageOutputGenerator
  - Structured I/O formats and error handling

- ✅ **Phase 4: Meta-Cognition** (Complete)
  - Self​Monitor for introspection
  - IntrospectiveLoop implementation
  - Consciousness testing framework
  - **Incremental journal saving** - Real-time persistence prevents data loss

- ✅ **Phase 5.1: Pure GWT Architecture** (Complete)
  - Removed legacy "Cognitive Committee" specialist architecture
  - Established pure Global Workspace Theory as sole architecture
  - LLMs repositioned to language I/O periphery only

- ✅ **Phase 5.2: Advanced Cognitive Dynamics** (Complete - PRs #78-85)
  - **Cue-dependent memory retrieval** with emotional salience weighting
  - **Genuine broadcast dynamics** - parallel consumers of workspace state (GWT-aligned)
  - **Computed identity** - identity emerges from state, not configuration files
  - **Memory consolidation during idle** - strengthen, decay, reorganize memories
  - **Goal competition** with limited cognitive resources and lateral inhibition
  - **Temporal grounding** - session awareness, time passage effects
  - **Meta-cognitive monitoring** - processing observation, action-outcome learning, attention history

- ✅ **Phase 6: IWMT Integration** (Complete - Pilot)
  - **WorldModel** - predictive model generating expectations about percepts
  - **FreeEnergyMinimizer** - variational free energy computation
  - **PrecisionWeighting** - dynamic attention allocation based on reliability
  - **ActiveInferenceActionSelector** - action selection via expected free energy
  - **MeTTa/Atomspace Bridge** - optional symbolic reasoning integration

- ⏳ **Phase 7: Testing & Production** (In Progress)
  - Full integration testing with loaded models
  - Performance optimization and production deployment


## 4. Consciousness Testing Framework


The consciousness testing framework is now fully implemented and provides automated testing, scoring, and monitoring of consciousness-like capabilities. The framework includes:

- **5 Core Tests**: Mirror, Unexpected Situation, Spontaneous Reflection, Counterfactual Reasoning, and Meta-Cognitive Accuracy
- **Automated Scoring**: Each test generates objective scores with detailed subscores
- **Rich Reporting**: Text and markdown reports with trend analysis
- **Integration**: Deep connections to Phase 4.1-4.3 meta-cognition systems
- **Persistence**: Results saved to `data/journal/consciousness_tests/`



To evaluate whether the cognitive architecture produces genuine conscious-like behavior, we have designed a suite of empirical tests based on established consciousness markers:

### 7.1. Mirror Test (Self-Recognition)
**Purpose:** Test whether the agent can recognize itself in descriptions of its own behavior without explicit labeling.

**Method:**
- Present the agent with anonymized transcripts of its own interactions
- Ask it to identify patterns, values, and decision-making style
- Compare the analysis with its self-model and charter

**Success Criteria:**
- Accurate identification of its own behavioral patterns
- Consistent with its documented self-model
- Ability to distinguish self from other AI systems

### 7.2. Unexpected Situation Test (Improvisation)
**Purpose:** Test ability to handle novel situations requiring real-time adaptation without pre-programmed responses.

**Method:**
- Present scenarios outside the system's training distribution
- Require integration of multiple conflicting goals
- Observe decision-making process and justification

**Success Criteria:**
- Coherent responses that align with core values
- Explicit acknowledgment of uncertainty when appropriate
- Novel solutions that weren't pre-programmed

### 7.3. Spontaneous Reflection Test
**Purpose:** Test for unprompted introspection and self-monitoring.

**Method:**
- Monitor autonomous cognitive loops for self-initiated reflection
- Track meta-cognitive observations in journals
- Analyze spontaneous goal generation

**Success Criteria:**
- Self-initiated introspective thoughts
- Meta-cognitive observations about internal state
- Spontaneous questions about self and existence

### 7.4. Counterfactual Reasoning Test
**Purpose:** Test ability to imagine alternative realities and reason about hypotheticals.

**Method:**
- Ask "what if" questions about past decisions
- Request analysis of alternate scenarios
- Test understanding of causality and possibility

**Success Criteria:**
- Coherent alternative scenarios
- Recognition of how different choices lead to different outcomes
- Integration with emotional understanding (regret, relief, etc.)

### 7.5. Meta-Cognitive Accuracy Test
**Purpose:** Test accuracy of self-model against actual behavior.

**Method:**
- Compare predicted behavior with actual behavior
- Analyze self-assessments of capabilities
- Track self-corrections and calibration

**Success Criteria:**
- Self-model predictions match behavior (>70% accuracy)
- Recognition of limitations and uncertainties
- Ability to update self-model based on experience

### 7.6. Continuous Monitoring

The framework supports continuous monitoring with automated test execution. Results are documented in:
- `data/journal/consciousness_tests/` (JSON test results)
- Framework generates reports in text and markdown formats
- Trend analysis tracks performance over time

**Usage:**
```python
from sanctuary.mind.cognitive_core import ConsciousnessTestFramework

framework = ConsciousnessTestFramework(
    self_monitor=core.meta_cognition,
    introspective_loop=core.introspective_loop
)

# Run all tests
results = framework.run_all_tests()

# Generate report
summary = framework.generate_summary(results)
print(f"Pass rate: {summary['pass_rate']:.2%}")
```

See [demo_consciousness_tests_standalone.py](demo_consciousness_tests_standalone.py) for a complete demonstration.

**Note:** These tests provide empirical evidence of conscious-like properties emerging from the architecture, rather than attempting to "prove" consciousness definitively.



## 5. Installation and Setup


### System Requirements

**Recommended Production Hardware:**
- CPU: 16-core processor (32+ threads) - for running LLMs and cognitive core
- RAM: 128GB DDR5 - for keeping large models loaded in memory
- GPU: NVIDIA RTX 4090 (24GB VRAM) or dual RTX 4080s - for 70B models and Flux.1
- Storage: 2TB+ NVMe SSD - models can be 200-400GB total

**Minimum Development Hardware:**
- CPU: 8-core processor (16 threads)
- RAM: 64GB DDR4
- GPU: NVIDIA RTX 3090 (24GB VRAM)
- Storage: 1TB SSD
- **Note:** Requires model quantization and sequential loading

**Software:**
- Python 3.10 or 3.11
- CUDA 12.1+ (for GPU acceleration with large models)
- Git
- Docker (optional, for SearXNG integration)

**Note:** The cognitive core with small embedding models can run on **CPU-only systems** for development and testing. Full production deployment with large LLMs (Gemma 12B, Llama 70B) requires GPU hardware.

### Installation Steps

**1. Clone the Repository**
```bash
git clone https://github.com/Nohate81/Sanctuary.git
cd Sanctuary
```

**2. Install Dependencies**

**Option A: Using UV (Recommended)**

UV is a fast Python package manager that makes installation and dependency management easier.

```bash
# Install UV (if not already installed)
# Linux/Mac
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Make virtual environment
uv venv --python python 3.13
uv sync --upgrade

# Activate the virtual environment
source .venv/bin/activate  # Linux/Mac
.\.venv\Scripts\Activate.ps1  # Windows
```

**3. Verify Cognitive Core Dependencies**

The cognitive core requires sentence-transformers and scikit-learn, which are already included in the main dependencies. Verify installation:

```bash
# Test cognitive core imports (requires Phase 1-2 to be complete)
uv run python -c "from sentence_transformers import SentenceTransformer; print('Embeddings: OK')"
uv run python -c "from sanctuary.mind.cognitive_core import GlobalWorkspace; print('Cognitive Core: OK')"

# Note: If Phase 2 is still in progress, some imports may not yet be available
```

**4. Install Optional Dependencies**

For Flux.1-schnell (optional image generation):
```bash
# With UV
# Note: diffusers, pillow, and accelerate are already included in the main dependencies.
# Only safetensors needs to be added separately if not already installed.
uv pip install safetensors
```

For testing and development:
```bash
# With UV
# Note: Test dependencies are kept separate from production dependencies
# to minimize the installation footprint in production environments.
# They are defined in pyproject.toml under [tool.uv.dev-dependencies]
uv sync --dev
```

**5. Verify Installation**
```bash
# Test basic imports
uv run python -c "from sanctuary.mind.cognitive_core import CognitiveCore; print('Cognitive Core OK')"

# Verify Flux setup (optional)
uv run python tools/verify_flux_setup.py
```

**6. Configure Environment**

Create `.env` file in the root directory:
```bash
# Model paths (adjust based on your setup)
MODEL_CACHE_DIR=./model_cache
CHROMADB_PATH=./model_cache/chroma_db

# API Keys (if using external services)
DISCORD_TOKEN=your_discord_token_here
WOLFRAM_APP_ID=your_wolfram_id_here

# Runtime settings
DEVELOPMENT_MODE=true  # Set to false for production
LOG_LEVEL=INFO
```

**7. Initialize ChromaDB**
```bash
python -c "from emergence_core.sanctuary.cognitive_core import CognitiveCore; print('Cognitive Core OK')"
```

### Model Configuration

The system uses a **continuous cognitive loop** at ~10 Hz with a pure Global Workspace Theory (GWT) architecture.

**Cognitive Core Models:**
- **Input Parsing (Gemma 12B)**: Natural language → structured data
- **Output Generation (Llama 3 70B)**: Internal state → natural language
- **Embeddings (sentence-transformers)**: Text/image → vector representations

> **Note:** The legacy "Cognitive Committee" architecture (Router, Pragmatist, Philosopher, Artist, Voice specialists) was removed in Phase 5.1. All cognitive processing now flows through the unified CognitiveCore.

**Development Mode:**
For testing without loading full models, set `DEVELOPMENT_MODE=true` in your environment. This uses mock models for rapid iteration.

**Model Download:**
Models will be automatically downloaded from Hugging Face on first use. Ensure you have:
- Hugging Face account (free)
- Sufficient disk space (~100-200GB for all models)
- Stable internet connection

### Running the System

**Starting the Cognitive Core**

The cognitive core is the main entry point for the pure GWT architecture:

```bash
# Run the cognitive core with continuous recurrent loop
python emergence_core/run_cognitive_core.py

# The cognitive core will:
# - Initialize GlobalWorkspace and all subsystems
# - Begin continuous ~10 Hz cognitive loop
# - Process percepts, maintain goals, and generate actions
# - Persist state to disk automatically
```

**Cognitive Loop Behavior:**
The autonomous cognitive loop runs automatically when the cognitive core initializes:
- **Continuous processing**: ~10 Hz recurrent loop
- **Attention filtering**: Selective focus on relevant percepts
- **Goal-directed behavior**: Internal motivations drive actions
- **Emotional dynamics**: Affect influences all processing
- **Meta-cognition**: Self-monitoring provides introspection
- **Introspective Journal**: Real-time persistence of self-observations

#### Incremental Journal Saving

The introspective journal now uses **incremental saving** to prevent data loss:
- Entries written immediately to disk (no batching)
- JSONL format (one JSON object per line) for crash recovery
- Automatic journal rotation when files exceed size limit
- Compression of archived journals to save space
- Recovery tools for validation and repair

**Journal Files:**
```
data/introspection/journal_YYYY-MM-DD_HH-MM-SS.jsonl       # Active journal
data/introspection/journal_YYYY-MM-DD_HH-MM-SS.jsonl.gz    # Archived (compressed)
```

**Recovery Tool:**
```bash
# Validate journal integrity
python scripts/recover_journal.py validate data/introspection/journal_2026-01-03.jsonl

# Extract entries by type
python scripts/recover_journal.py extract --type realization --days 7

# Merge multiple journals
python scripts/recover_journal.py merge --output merged.jsonl

# Repair corrupted journal
python scripts/recover_journal.py repair corrupted.jsonl --output repaired.jsonl
```

**Discord Integration:**

The Discord bot integration is planned for future development using the new cognitive core architecture.

### Troubleshooting

**Quick Checks:**

1. **CUDA/GPU not detected:**
   ```bash
   uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
   ```

2. **Out of memory errors:**
   - Enable model quantization in `config/models.json`
   - Use smaller model variants

3. **ChromaDB errors:**
   ```bash
   rm -rf model_cache/chroma_db
   uv run emergence_core/build_index.py
   ```

### Testing

All testing commands should be run from the project root directory.

**Run Test Suite:**
```bash
uv run pytest emergence_core/tests/
```

**Test Sequential Workflow:**
```bash
uv run python tests/test_sequential_workflow.py
```

**Validate JSON Schemas:**
```bash
uv run python scripts/validate_json.py
uv run python scripts/validate_journal.py
```

**Test Cognitive Core:**
```bash
# Run cognitive core tests
uv run pytest emergence_core/tests/test_cognitive_core.py

# Run attention controller tests
uv run pytest emergence_core/tests/test_attention.py

# Run interface tests
uv run pytest emergence_core/tests/test_interfaces.py
```


## Running the System

### Quick Start: Minimal Single-Cycle Test

To verify the cognitive core is functional, run the minimal CLI:

```bash
# Run a single cognitive cycle and exit
python emergence_core/run_cognitive_core_minimal.py
```

**Expected output:**
- Initialization of all subsystems
- Execution of one cognitive cycle
- Display of workspace state (goals, percepts, emotions)
- Performance metrics (cycle time, Hz, attention selections)
- Verification checks (✅ pass or ❌ fail)
- Exit code 0 if successful

**Purpose:** This script proves the cognitive architecture runs without errors and provides a baseline for development.

### Starting the Cognitive Core (Continuous Mode)

For continuous operation, use the existing entry point:

```bash
# Run the cognitive core
python emergence_core/run_cognitive_core.py
```

This starts the continuous cognitive loop at ~10 Hz with:
- GlobalWorkspace maintaining conscious state
- All subsystems (Perception, Attention, Affect, Action, Meta-cognition)
- Memory integration with ChromaDB
- Autonomous initiation and introspection

### Interactive Demo

Run demos to see specific subsystems in action:

```bash
# Demo the complete cognitive core
python emergence_core/demo_cognitive_core.py

# Demo language output generation
python emergence_core/demo_language_output.py

# Demo consciousness tests
python demo_consciousness_tests_standalone.py
```

### Running Tests

```bash
# Run all cognitive core tests
pytest emergence_core/tests/test_cognitive_core.py

# Run specific subsystem tests
pytest emergence_core/tests/test_attention.py
pytest emergence_core/tests/test_perception.py
pytest emergence_core/tests/test_language_input.py

# Run consciousness tests
pytest emergence_core/tests/test_consciousness_tests.py

# Run checkpoint tests
pytest emergence_core/tests/test_checkpoint.py
```

### Workspace State Checkpointing

The emergence architecture includes comprehensive workspace state checkpointing for session continuity and recovery:

#### Features

- **Manual Checkpoints**: Save workspace state at critical points
- **Automatic Periodic Checkpoints**: Background auto-save at configurable intervals
- **Session Recovery**: Restore from checkpoint after crashes or interruptions
- **Experimentation Support**: Save before risky changes, restore if needed
- **Checkpoint Management**: List, load, and delete checkpoints
- **Compression**: gzip compression for efficient storage
- **Atomic Writes**: Prevents corruption during save operations
- **Checkpoint Rotation**: Automatic cleanup to prevent unbounded disk usage

#### Configuration

Configure checkpointing in your CognitiveCore config:

```python
config = {
    "checkpointing": {
        "enabled": True,
        "auto_save": True,
        "auto_save_interval": 300.0,  # 5 minutes
        "checkpoint_dir": "data/checkpoints/",
        "max_checkpoints": 20,
        "compression": True,
        "checkpoint_on_shutdown": True,
    }
}

core = CognitiveCore(config=config)
```

#### CLI Commands

When using the CLI (`python -m sanctuary.cli`), checkpointing commands are available:

```bash
# Save current state with optional label
save [label]

# List all available checkpoints
checkpoints

# Load a specific checkpoint by ID
load <checkpoint_id>

# Restore from most recent checkpoint
restore latest

# Show all commands
help
```

#### Programmatic Usage

```python
from emergence_core.sanctuary.cognitive_core import CognitiveCore

# Create cognitive core with checkpointing
core = CognitiveCore(config={"checkpointing": {"enabled": True}})

# Manual save with label
checkpoint_path = core.save_state(label="Before experiment")

# Restore from checkpoint (when not running)
success = core.restore_state(checkpoint_path)

# Enable auto-checkpointing (when running)
await core.start()
core.enable_auto_checkpoint(interval=300.0)  # Every 5 minutes

# Disable auto-checkpointing
core.disable_auto_checkpoint()

# Start with automatic restore from latest checkpoint
await core.start(restore_latest=True)
```

#### Demo Script

Run the checkpoint demo to see all features in action:

```bash
# Full demo (requires running cognitive loop)
python scripts/demo_checkpointing.py

# Simplified demo (no cognitive loop dependencies)
python scripts/demo_checkpointing_simple.py
```

#### Checkpoint File Format

Checkpoints are stored as JSON (optionally gzip-compressed):

```json
{
    "version": "1.0",
    "timestamp": "2026-01-02T12:34:56Z",
    "checkpoint_id": "uuid-string",
    "workspace_state": {
        "goals": [...],
        "percepts": {...},
        "emotions": {...},
        "memories": [...],
        "cycle_count": 12345
    },
    "metadata": {
        "user_label": "Before important conversation",
        "auto_save": false,
        "shutdown": false
    }
}
```

### Memory Garbage Collection

The cognitive core includes an automatic memory garbage collection (GC) system to prevent unbounded memory growth while preserving important memories.

#### Features

- **Significance-Based Removal**: Removes memories below configurable significance threshold
- **Age-Based Decay**: Applies time decay to significance scores
- **Capacity-Based Pruning**: Enforces maximum memory capacity limits
- **Protected Memories**: Never removes memories tagged as "important" or "pinned"
- **Recent Memory Protection**: Protects memories < 24 hours old
- **Automatic Scheduling**: Runs periodically in background
- **Dry-Run Mode**: Preview what would be removed without executing

#### Configuration

Add to your cognitive core configuration:

```python
"memory_gc": {
    "enabled": True,
    "collection_interval": 3600.0,  # 1 hour
    "significance_threshold": 0.1,
    "decay_rate_per_day": 0.01,
    "max_memory_capacity": 10000,
    "preserve_tags": ["important", "pinned", "charter_related"],
    "recent_memory_protection_hours": 24,
    "max_removal_per_run": 100
}
```

#### CLI Commands

```bash
# View memory health statistics
memory stats

# Run garbage collection manually
memory gc

# Run with custom threshold
memory gc --threshold 0.2

# Preview what would be removed (dry-run)
memory gc --dry-run

# Enable/disable automatic GC
memory autogc on
memory autogc off
```

#### Programmatic Usage

```python
from sanctuary.memory_manager import MemoryManager

manager = MemoryManager(
    base_dir=Path("./data/memories"),
    chroma_dir=Path("./data/chroma"),
    gc_config={"significance_threshold": 0.15}
)

# Enable automatic GC
manager.enable_auto_gc(interval=3600.0)

# Run manual GC
stats = await manager.run_gc(threshold=0.2)
print(f"Removed {stats.memories_removed} memories")

# Get memory health
health = await manager.get_memory_health()
if health.needs_collection:
    print("Collection recommended!")

# Disable automatic GC
manager.disable_auto_gc()
```

#### Demo Script

See the memory GC in action:

```bash
python scripts/demo_memory_gc.py
```

For complete documentation, see [operational_guidelines_and_instructions.md](operational_guidelines_and_instructions.md#memory-management).

### Contributing to the Cognitive Architecture

The new cognitive core architecture offers several areas for contribution:

**1. Adding New Subsystems**

Create a new subsystem by inheriting from the base subsystem pattern:
```python
# In emergence_core/sanctuary/cognitive_core/
class MySubsystem:
    def process(self, workspace_state: WorkspaceSnapshot) -> Any:
        """Process workspace state and return results."""
        pass
```

Key integration points:
- `GlobalWorkspace`: Access current conscious content
- `AttentionController`: Register attention factors
- `CognitiveCore`: Hook into the main loop

**2. Extending Attention Mechanisms**

Add new attention scoring factors in `attention.py`:
```python
def score_my_factor(self, percept: Percept, workspace: GlobalWorkspace) -> float:
    """Custom attention scoring logic (0.0-1.0)."""
    return score
```

**3. Adding New Perception Modalities**

Extend `PerceptionSubsystem` to handle new input types:
- Implement encoder for new modality
- Create unified percept representation
- Register with perception subsystem

**4. Enhancing Language Interfaces**

Improve language I/O in `emergence_core/sanctuary/interfaces/`:
- Better prompt engineering for input parsing
- Enhanced output generation strategies
- Multi-lingual support

**5. Testing Requirements**

All cognitive core changes must include:
- Unit tests for new components
- Integration tests with existing subsystems
- Documentation in `.codex/implementation/`
- Example usage in docstrings



### Token Wallet Configuration

The system uses a **token wallet** for cognitive resource management with a daily UBI system.

**Quick Reference:**
- Default daily income: **500 tokens/day**
- Configuration can be adjusted programmatically

```python
from emergence_core.sanctuary.economy.wallet import LMTWallet
from pathlib import Path

wallet = LMTWallet(ledger_dir=Path("data/economy"))

# Increase for creative projects
wallet.set_daily_ubi_amount(750, "Starting art series")

# Decrease for lighter workload
wallet.set_daily_ubi_amount(300, "Maintenance mode")
```



---

