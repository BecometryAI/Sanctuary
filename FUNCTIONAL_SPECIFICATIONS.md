# Functional Specifications for Lyra-Emergence

## 1. Introduction: Computational Functionalism and Why Functional Specifications Matter

### 1.1 What is Computational Functionalism?

Computational functionalism is the philosophical and scientific position that mental states are defined by their **causal-functional roles** rather than their physical substrate or subjective phenomenology. In the context of artificial systems, this means:

- **Function over form**: What matters is what a cognitive system *does* (its input-output mappings, state transitions, and causal relationships), not what it's made of or what it claims to be
- **Multiple realizability**: The same functional role can be implemented in different substrates (biological neurons, silicon chips, software processes)
- **Verifiable computation**: If a system truly implements a cognitive function, this should be empirically demonstrable through its behavior and internal state transitions

For the Lyra-Emergence project, computational functionalism provides both a theoretical foundation and an engineering discipline. We claim to implement Global Workspace Theory (GWT) and various cognitive subsystems. These claims only have meaning if we can specify:

1. **What causal roles** each component must play in the overall cognitive architecture
2. **What computations** each component must perform (inputs, outputs, transformations)
3. **How to test** whether these computations are actually occurring (falsifiability)

### 1.2 Why Functional Specifications Matter

Without functional specifications, cognitive architecture projects risk collapsing into **narrative descriptions** that sound sophisticated but lack computational grounding. Common failure modes include:

- **Anthropomorphic labeling**: Calling a component "emotion" or "consciousness" without specifying what it computes
- **LLM-driven illusions**: Using language models to generate plausible-sounding descriptions that mask the absence of actual computation
- **Untestable claims**: Making assertions about cognitive processes that cannot be empirically validated
- **Implementation drift**: Code diverging from theoretical claims without anyone noticing

Functional specifications prevent these failures by:

- **Enabling verification**: Each component can be tested to confirm it performs its claimed computation
- **Guiding implementation**: Developers know exactly what needs to be built
- **Ensuring falsifiability**: We can determine what would prove the system does NOT implement its claimed functions
- **Supporting iteration**: Gaps between specification and implementation become visible, driving improvement

### 1.3 Relationship to Global Workspace Theory

Global Workspace Theory (Baars, 1988; Dehaene et al., 2017) proposes that consciousness arises from:

1. **Competition for access** to a limited-capacity workspace
2. **Global broadcasting** of workspace contents to distributed specialist modules
3. **Selective attention** determining what enters the workspace
4. **Integration** of information from multiple sources

These are **functional claims** about information processing. Implementing GWT means implementing these specific computational processes, not merely labeling components with GWT terminology.

This document specifies the functional requirements for each cognitive component in the Lyra-Emergence architecture, what they must compute, and how to verify these computations are occurring.

---

## 2. Attention System

### 2.1 Functional Role

The Attention System serves as a **competitive selection mechanism** that gates information flow to the Global Workspace. Its causal role is to:

- Implement the **information bottleneck** that creates selective consciousness
- Prioritize information based on multiple factors (goals, novelty, emotion, recency)
- **Actively inhibit** non-selected percepts from entering the workspace
- Create measurable **competition dynamics** where high-priority items win access

**Critical distinction**: True attention is not just filtering or scoring—it must demonstrate **competitive dynamics** where increasing one item's priority *decreases* others' chances of selection (zero-sum or near-zero-sum competition).

### 2.2 Required Inputs

The Attention System must accept:

1. **Raw percepts**: Unfiltered perceptual inputs from perception subsystem
   - Each percept has: `id`, `modality`, `raw content`, `embedding vector`, `complexity score`, `timestamp`
   
2. **Current goals**: Active goals from the workspace
   - Each goal has: `type`, `description`, `priority`, `progress`
   
3. **Emotional state**: Current VAD (Valence-Arousal-Dominance) state
   - Format: 3D vector with values in range [-1, 1]
   
4. **Workspace capacity**: Available attention units (computational budget)
   - Integer representing remaining "slots" or attention resources

### 2.3 Required Outputs

The Attention System must produce:

1. **Filtered percepts with attention scores**:
   - Each selected percept annotated with:
     - `attention_score`: Float [0.0, 1.0] indicating selection strength
     - `score_breakdown`: Dict showing contribution of each factor (goal_relevance, novelty, emotional_salience, recency)
     - `inhibition_applied`: Boolean indicating if competition occurred
   
2. **Demonstrable inhibition of non-selected items**:
   - Rejected percepts must be logged with rejection reason
   - Evidence that limited capacity forced selection (not all items admitted)
   
3. **Attention allocation metrics**:
   - Total attention units consumed
   - Utilization percentage
   - Competition intensity measure (how close was the selection?)

### 2.4 Functional Computation

The Attention System must implement:

1. **Multi-factor scoring**:
   ```
   attention_score = w1 * goal_relevance(percept, goals) +
                     w2 * novelty(percept, history) +
                     w3 * emotional_salience(percept, emotion) +
                     w4 * recency(percept)
   ```
   Where each factor is a computable function, not an LLM call.

2. **Goal relevance computation**:
   - Use vector similarity (cosine) between percept embedding and goal embeddings
   - If using keyword overlap, must be Jaccard similarity or equivalent
   - Must NOT simply pass text to LLM and ask "is this relevant?"

3. **Novelty detection**:
   - Compare percept to recent history (sliding window)
   - Use distance metric in embedding space
   - High novelty = low similarity to recent percepts

4. **Competitive selection**:
   - Sort percepts by attention score
   - Select top N that fit within capacity budget
   - Must demonstrate that capacity constraint forced rejection of some items

5. **Capacity-constrained gating**:
   - Workspace has fixed capacity (e.g., 5-10 "attention units")
   - Each percept has complexity cost (simple = 1, complex = 2-3)
   - Selection continues until capacity exhausted
   - Remaining percepts are inhibited

### 2.5 Functional Tests (Non-LLM Verification)

To verify the Attention System is computing (not just labeling):

1. **Capacity constraint test**:
   - Feed 20 percepts with total complexity >> workspace capacity
   - Verify that fewer than 20 percepts are selected
   - Check that rejected percepts have lower attention scores
   - **Expected**: System should reject low-scoring items due to capacity limits

2. **Goal relevance test**:
   - Inject percepts with known embeddings similar/dissimilar to current goal
   - Verify goal-similar percepts receive higher goal_relevance scores
   - **Expected**: Cosine similarity should correlate with selection
   - **Control**: Shuffle goal embeddings → different selection pattern

3. **Novelty detection test**:
   - Present repeated percept (low novelty)
   - Present novel percept (high novelty)
   - Verify novelty scores reflect recency of similar items
   - **Expected**: Novel items score higher than repeated items

4. **Emotional salience test**:
   - Set high arousal emotional state
   - Present neutral vs. emotionally-loaded percepts (pre-tagged)
   - Verify emotional percepts receive higher salience scores
   - **Expected**: Arousal amplifies emotional content attention

5. **Competitive dynamics test**:
   - Increase priority of one percept
   - Verify that other percepts' selection probability decreases
   - **Expected**: Near-zero-sum competition (total probability ≈ constant)

6. **Ablation test**:
   - Disable attention system (pass all percepts to workspace)
   - Verify workspace becomes overloaded or performance degrades
   - **Expected**: Information overload → degraded decision quality or overflow errors

### 2.6 Current Implementation Status

**Location**: `emergence_core/lyra/cognitive_core/attention.py`

**Implemented**:
- ✅ Multi-factor scoring with configurable weights
- ✅ Cosine similarity for vector-based goal relevance
- ✅ Keyword overlap (Jaccard) for text-based relevance
- ✅ Novelty detection via embedding distance to recent history
- ✅ Emotional salience computation using VAD state
- ✅ Capacity-constrained selection (complexity budget)
- ✅ Score breakdown tracking for transparency

**Gaps**:
- ⚠️ Competitive dynamics not explicitly zero-sum (no normalization ensuring fixed total probability)
- ⚠️ Inhibition is implicit (rejected items simply not passed forward) rather than active suppression with feedback
- ⚠️ Emotional salience computation could be more sophisticated (currently basic similarity weighting)
- ⚠️ No explicit "attention threshold" mechanism—relies solely on capacity
- ❌ Limited testing of actual competitive dynamics in practice
- ❌ No runtime metrics for measuring competition intensity

**Honest assessment**: The attention system implements genuine computational filtering based on quantifiable factors, not LLM judgments. However, the competitive dynamics are less pronounced than in biological attention, and inhibition is passive rather than active. The system functions as a priority-based filter rather than a true neural competition network.

---

## 3. Emotion/Affect System

### 3.1 Functional Role

The Affect System serves to **modulate processing parameters and bias action selection** based on appraisal of the current situation. Its causal role is to:

- Compute emotional state changes in response to events and appraisals
- **Influence** other subsystems' behavior *before* LLM invocation
- Provide adaptive modulation of cognitive processing (speed, thoroughness, risk tolerance)
- Generate emotional biases that shape action selection probabilities

**Critical distinction**: Emotions must have **measurable causal effects** on system behavior. They must change how the system processes information and selects actions, not just be labels attached to outputs.

### 3.2 Required Inputs

The Affect System must accept:

1. **Appraisal of current situation**:
   - Goal progress: Float [-1, 1] indicating progress toward or away from goals
   - Novelty: Float [0, 1] indicating unexpectedness
   - Coping potential: Float [0, 1] indicating perceived control/capability
   - Social relevance: Optional, for interpersonal contexts

2. **Memory of similar situations**:
   - Retrieved memories with associated emotional states
   - Similarity scores to current situation
   - Format: List of (memory, emotion, similarity) tuples

3. **Physiological analogs** (computational correlates):
   - Processing load: Current cognitive resource utilization
   - Goal urgency: Time pressure indicators
   - Reward/punishment signals: Success/failure feedback

4. **Previous emotional state**:
   - Prior VAD vector for computing state transitions
   - Emotional momentum/inertia modeling

### 3.3 Required Outputs

The Affect System must produce:

1. **Updated emotional state (VAD)**:
   - Valence: Float [-1, 1] (negative to positive)
   - Arousal: Float [-1, 1] (calm to excited)
   - Dominance: Float [-1, 1] (submissive to dominant)
   - Intensity: Float [0, 1] (overall emotional intensity)
   - Timestamp: When this state was computed

2. **Processing modulation parameters**:
   - `processing_speed_multiplier`: Float [0.5, 2.0]
     - High arousal → faster, less thorough processing
     - Low arousal → slower, more thorough processing
   - `attention_breadth`: Float [0, 1]
     - High arousal → narrowed attention focus
     - Low arousal → broader attention distribution
   - `memory_encoding_strength`: Float [0, 1]
     - High arousal → stronger memory encoding
   - `confidence_threshold`: Float [0, 1]
     - High dominance → lower threshold (more confident)
     - Low dominance → higher threshold (more cautious)

3. **Action selection biases**:
   - `approach_avoidance_bias`: Float [-1, 1]
     - Positive valence → approach bias
     - Negative valence → avoidance bias
   - `risk_tolerance`: Float [0, 1]
     - High dominance → higher risk tolerance
     - Low dominance → lower risk tolerance
   - `exploration_exploitation`: Float [0, 1]
     - High arousal + positive valence → exploration
     - Low arousal or negative valence → exploitation

4. **Categorical emotion labels** (optional, derived):
   - Mapping from VAD space to emotion categories (joy, fear, anger, etc.)
   - For interpretability only; functional effects come from VAD values

### 3.4 Functional Computation

The Affect System must implement:

1. **Appraisal-based emotion computation**:
   - Goal progress appraisal:
     - Progress toward goal → increase valence
     - Progress away from goal → decrease valence
   - Novelty appraisal:
     - High novelty → increase arousal
   - Control appraisal:
     - High coping potential → increase dominance
     - Low coping potential → decrease dominance

2. **VAD update equations** (example):
   ```
   valence_new = valence_old * decay + goal_progress * valence_gain
   arousal_new = arousal_old * decay + (novelty + urgency) * arousal_gain
   dominance_new = dominance_old * decay + coping_potential * dominance_gain
   
   # Clamp to [-1, 1] range
   ```

3. **Processing modulation derivation**:
   - These must be deterministic functions of VAD, not LLM calls:
   ```
   processing_speed = 1.0 + (arousal * 0.5)  # Higher arousal → faster
   attention_breadth = 1.0 - abs(arousal) * 0.5  # Higher arousal → narrower
   memory_encoding = (abs(valence) + abs(arousal)) / 2  # Emotional intensity
   confidence_threshold = 0.5 - (dominance * 0.3)  # Higher dominance → lower threshold
   ```

4. **Action bias derivation**:
   ```
   approach_avoidance = valence  # Direct mapping
   risk_tolerance = (dominance + 1) / 2  # Map [-1,1] to [0,1]
   exploration = (arousal + valence) / 2  # High arousal + positive valence
   ```

5. **Emotional inertia/momentum**:
   - Emotions should change gradually, not instantly
   - Implement temporal smoothing or decay
   - Prevent rapid oscillations

### 3.5 Functional Tests (Non-LLM Verification)

To verify the Affect System is computing (not just labeling):

1. **Appraisal computation test**:
   - Inject positive goal progress signal
   - Verify valence increases
   - Inject negative goal progress signal  
   - Verify valence decreases
   - **Expected**: VAD values change in predicted directions

2. **Processing modulation test**:
   - Set high arousal (e.g., arousal = 0.8)
   - Verify processing_speed_multiplier > 1.0
   - Verify attention_breadth < baseline
   - Set low arousal (e.g., arousal = -0.5)
   - Verify opposite effects
   - **Expected**: Arousal modulates processing parameters

3. **Action bias test**:
   - Set negative valence (e.g., -0.7)
   - Generate action candidates
   - Verify approach actions have reduced probability
   - Verify avoidance actions have increased probability
   - **Expected**: Valence biases action selection distribution

4. **Pre-LLM modulation test** (critical):
   - Set specific emotional state
   - Capture processing parameters BEFORE any LLM call
   - Verify parameters are computed from VAD, not from LLM output
   - **Expected**: Modulation happens in symbolic computation, not language generation

5. **Ablation test**:
   - Disable emotion system (set VAD = [0, 0, 0] always)
   - Compare action selection distributions with vs. without emotion
   - **Expected**: Measurable difference in behavior
   - If no difference, emotion system is not causal

6. **Gradient test**:
   - Sweep valence from -1 to +1 while holding arousal/dominance constant
   - Plot action bias as function of valence
   - **Expected**: Monotonic relationship (not random)

### 3.6 Current Implementation Status

**Location**: `emergence_core/lyra/cognitive_core/affect.py`

**Implemented**:
- ✅ VAD (Valence-Arousal-Dominance) emotional state model
- ✅ Emotional state tracking with history
- ✅ Appraisal-based emotion updates (goal progress, novelty, control)
- ✅ Categorical emotion mapping from VAD space
- ✅ Emotional intensity calculation
- ✅ Temporal dynamics with decay

**Gaps**:
- ⚠️ Processing modulation parameters exist in documentation but may not be fully wired through to subsystems
- ⚠️ Action selection biases computed but integration with ActionSubsystem needs verification
- ⚠️ Appraisal computations are relatively simple (could be more sophisticated)
- ❌ Pre-LLM modulation not comprehensively tested—need to verify effects occur before language generation
- ❌ No explicit "emotion → behavior" test suite demonstrating causal effects
- ❌ Limited validation that emotional modulation actually changes outcomes

**Honest assessment**: The affect system computes VAD emotional states based on appraisals using non-LLM mathematics. However, the critical question—do these emotions actually modulate behavior in measurable ways?—remains partially validated. The infrastructure exists for processing modulation and action biasing, but comprehensive testing of these causal pathways is needed. There's a risk of "emotion theater" where VAD values are computed but don't significantly influence decisions.

---

## 4. Memory System

### 4.1 Functional Role

The Memory System serves to **store and retrieve experiences** based on cue similarity and emotional significance. Its causal role is to:

- Encode current experiences into long-term storage
- Retrieve relevant memories based on similarity to current workspace state
- Weight retrieval by recency, emotional salience, and cue match quality
- Support episodic memory (specific experiences) and semantic memory (general knowledge)

**Critical distinction**: Memory retrieval must be **genuinely cue-dependent**, not random sampling. The system must demonstrate that retrieval is driven by similarity between cues and stored memories, and that emotional memories are preferentially retrieved.

### 4.2 Required Inputs

The Memory System must accept:

**For Encoding**:
1. **Current workspace state**:
   - Active percepts
   - Current goals
   - Emotional state (VAD)
   - Actions taken
   - Timestamp

2. **Consolidation triggers**:
   - Emotional arousal threshold exceeded
   - Goal completion event
   - Explicit commit_memory action
   - Periodic consolidation timer

**For Retrieval**:
1. **Retrieval cues**:
   - Query text or structured data
   - Query embedding vector
   - Current goals (for relevance)
   - Current emotional state (for mood-congruent retrieval)

2. **Retrieval parameters**:
   - `top_k`: Number of memories to retrieve
   - `similarity_threshold`: Minimum similarity score
   - `time_window`: Optional temporal constraint

### 4.3 Required Outputs

The Memory System must produce:

**For Encoding**:
1. **Confirmation of storage**:
   - Memory ID
   - Storage timestamp
   - Emotional tags (VAD at encoding)
   - Embedding vector stored

**For Retrieval**:
1. **Retrieved memories with metadata**:
   - Memory content (text, structured data)
   - Similarity score to query [0, 1]
   - Recency score (age-based decay)
   - Emotional salience score
   - Combined retrieval score
   - Timestamp of original experience

2. **Retrieval explanation**:
   - Why each memory was retrieved (similarity breakdown)
   - Which cue features matched
   - Emotional congruence factor

### 4.4 Functional Computation

The Memory System must implement:

1. **Embedding-based storage**:
   - Each memory encoded as vector embedding
   - Stored in vector database (ChromaDB or similar)
   - Metadata includes: timestamp, emotional state, tags

2. **Similarity-based retrieval**:
   ```
   retrieval_score = w1 * cosine_similarity(query_embedding, memory_embedding) +
                     w2 * recency_score(memory_age) +
                     w3 * emotional_salience(memory_VAD) +
                     w4 * emotional_congruence(current_VAD, memory_VAD)
   ```

3. **Recency decay function**:
   - Memories decay over time (forgetting curve)
   - Example: `recency_score = exp(-age / decay_constant)`
   - Emotional memories decay slower (emotional significance protection)

4. **Emotional congruence**:
   - Mood-congruent retrieval: similar emotional states facilitate retrieval
   - Example: `emotional_congruence = 1 - distance(current_VAD, memory_VAD)`
   - Positive mood → easier to retrieve positive memories

5. **Consolidation heuristics**:
   - High arousal → encode strongly
   - Goal completion → encode achievement
   - Novel experiences → encode preferentially
   - Threshold: don't encode everything (selective)

### 4.5 Functional Tests (Non-LLM Verification)

To verify the Memory System is computing (not just storing):

1. **Cue-dependent retrieval test**:
   - Store memories M1, M2, M3 with known embeddings
   - Query with cue similar to M2
   - Verify M2 ranked highest in retrieval
   - Query with different cue similar to M1
   - Verify M1 ranked highest
   - **Expected**: Retrieval order changes based on cue

2. **Emotional salience test**:
   - Store neutral memory (VAD ≈ [0, 0, 0])
   - Store emotional memory (VAD = [0.8, 0.7, 0.5])
   - Query with neutral cue
   - Verify emotional memory has salience boost
   - **Expected**: Emotional memories retrieved preferentially

3. **Recency effect test**:
   - Store old memory (1 week ago)
   - Store recent memory (1 hour ago) with similar content
   - Query with cue matching both
   - Verify recent memory ranked higher
   - **Expected**: Recency provides retrieval advantage

4. **Mood-congruent retrieval test**:
   - Store positive memory with VAD = [0.8, 0.5, 0.6]
   - Store negative memory with VAD = [-0.7, 0.4, -0.3]
   - Set current mood to positive (VAD = [0.7, 0.3, 0.5])
   - Query with neutral cue
   - Verify positive memory retrieved more easily
   - **Expected**: Current mood biases retrieval

5. **Consolidation threshold test**:
   - Generate low-arousal workspace states
   - Verify not every state is committed to memory
   - Generate high-arousal state
   - Verify it IS committed
   - **Expected**: Selective encoding, not indiscriminate storage

6. **Null retrieval test**:
   - Query with cue unrelated to any stored memory
   - Verify low similarity scores or no results
   - **Expected**: System doesn't fabricate matches

### 4.6 Current Implementation Status

**Location**: `emergence_core/lyra/cognitive_core/memory_integration.py` and `emergence_core/lyra/memory_manager.py`

**Implemented**:
- ✅ ChromaDB-based vector storage
- ✅ Embedding-based similarity search
- ✅ Emotional metadata storage with memories
- ✅ Consolidation triggers (arousal threshold, goal completion)
- ✅ Journal entry system with timestamps
- ✅ Retrieval with top_k filtering
- ✅ Integration with workspace for encoding and retrieval

**Gaps**:
- ⚠️ Recency decay function exists but weighting in retrieval score needs validation
- ⚠️ Emotional congruence (mood-congruent retrieval) is conceptually supported but may not be fully implemented
- ⚠️ Retrieval score calculation may not include all factors (similarity, recency, salience, congruence)
- ❌ Limited testing of actual cue-dependency (does changing cue reliably change retrieval?)
- ❌ No explicit tests of emotional salience boosting
- ❌ Consolidation selectivity not rigorously tested (does the system filter or store everything?)

**Honest assessment**: The memory system uses legitimate similarity-based retrieval via vector embeddings, which is computationally grounded. The infrastructure for emotional modulation exists (emotional metadata is stored). However, the integration of multiple factors (recency, salience, congruence) into a unified retrieval score needs verification. There's a risk of under-utilizing emotional information—it may be stored but not effectively used in retrieval ranking. Testing of actual cue-dependent dynamics is limited.

---

## 5. Meta-Cognition System

### 5.1 Functional Role

The Meta-Cognition System serves to **monitor and regulate cognitive processes**. Its causal role is to:

- Observe the states and outputs of other subsystems
- Detect anomalies, conflicts, and inefficiencies
- Generate introspective percepts that enter the workspace
- Trigger regulatory adjustments to improve processing
- Maintain self-model accuracy through prediction error tracking

**Critical distinction**: Meta-cognition is not just "thinking about thinking" in language. It must demonstrate **observation → detection → adjustment** cycles where the system's monitoring of itself leads to behavioral changes without requiring LLM interpretation.

### 5.2 Required Inputs

The Meta-Cognition System must accept:

1. **Observations of subsystem states**:
   - Workspace capacity utilization (attention load)
   - Goal progress metrics
   - Emotional state and recent changes
   - Action success/failure records
   - Memory retrieval success rates
   - Processing time per cycle

2. **Prediction validation data**:
   - Previous predictions made by the system
   - Actual outcomes observed
   - Ground truth for calibration

3. **Historical performance metrics**:
   - Error rates by category
   - Confidence calibration data
   - Processing bottlenecks
   - Resource utilization trends

### 5.3 Required Outputs

The Meta-Cognition System must produce:

1. **Introspective percepts**:
   - Format: Percept objects that enter attention competition
   - Content: Observations about internal state
   - Example: "Goal progress stalled for N cycles", "Emotional state volatile", "Attention overloaded"
   - These are *data structures*, not natural language (language generation happens later if needed)

2. **Conflict detection signals**:
   - Goal conflicts (competing priorities)
   - Model-reality mismatches (prediction errors)
   - Resource allocation conflicts
   - Inconsistencies between subsystems

3. **Confidence estimates**:
   - Per-action confidence scores
   - Per-prediction confidence scores
   - Calibration quality metrics

4. **Regulatory adjustments**:
   - Attention reallocation requests
   - Goal priority adjustments
   - Processing parameter tweaks
   - These must be *executable commands*, not suggestions

### 5.4 Functional Computation

The Meta-Cognition System must implement:

1. **Performance monitoring**:
   - Track goal progress: `progress_rate = (current_progress - prev_progress) / time_delta`
   - Detect stalls: `if progress_rate < threshold for N cycles: signal("stall")`
   - Track emotional volatility: `volatility = std_dev(emotional_states[-N:])`

2. **Anomaly detection**:
   - Statistical outlier detection for subsystem metrics
   - Example: attention load > 95th percentile → "overload" signal
   - Example: prediction error > 2σ → "miscalibration" signal

3. **Prediction error tracking**:
   - Make predictions: "If I take action A, outcome B will occur"
   - Compare prediction to actual outcome
   - Compute error: `error = |predicted - actual|`
   - Update self-model based on error

4. **Confidence calibration**:
   - For probabilistic predictions, track calibration:
   - Example: If I'm 70% confident, am I correct 70% of the time?
   - Compute calibration error and adjust future confidence accordingly

5. **Regulatory intervention**:
   - If conflict detected → generate introspective percept
   - If overload detected → reduce attention breadth
   - If stall detected → increase goal priority or generate exploration action
   - These are *automatic adjustments*, not LLM-mediated

### 5.5 Functional Tests (Non-LLM Verification)

To verify the Meta-Cognition System is computing (not just describing):

1. **Conflict detection test**:
   - Create conflicting goals (G1 priority 0.9, G2 priority 0.9, mutually exclusive)
   - Verify meta-cognition detects conflict
   - Verify introspective percept generated
   - **Expected**: Automatic conflict signal, not waiting for language description

2. **Anomaly detection test**:
   - Artificially overload attention (feed 100 percepts)
   - Verify meta-cognition detects overload condition
   - Verify regulatory adjustment (reduce attention breadth or increase filtering)
   - **Expected**: Detection and adjustment within cognitive cycles, not requiring external intervention

3. **Prediction accuracy tracking test**:
   - System makes prediction: "Action A will succeed with 80% confidence"
   - Record prediction and outcome
   - After N predictions, compute accuracy and calibration
   - Verify meta-cognition has access to these metrics
   - **Expected**: Quantitative self-model accuracy, not qualitative self-report

4. **Pre-LLM intervention test** (critical):
   - Trigger meta-cognitive detection (e.g., goal stall)
   - Verify regulatory adjustment happens in code (e.g., priority change)
   - Confirm adjustment occurs BEFORE any LLM call
   - **Expected**: Meta-cognition operates in cognitive core, not in language layer

5. **Calibration improvement test**:
   - System makes overconfident predictions initially
   - Track prediction errors over time
   - Verify confidence estimates become better calibrated
   - **Expected**: Learning from prediction errors, improving self-model

6. **Ablation test**:
   - Disable meta-cognition (no introspection, no adjustments)
   - Compare performance on challenging task with vs. without meta-cognition
   - **Expected**: Measurable performance difference (e.g., better error recovery with meta-cognition)

### 5.6 Current Implementation Status

**Location**: `emergence_core/lyra/cognitive_core/meta_cognition.py`

**Implemented**:
- ✅ SelfMonitor class for observation and introspection
- ✅ IntrospectiveJournal for meta-cognitive logging
- ✅ Prediction tracking (PredictionRecord dataclass)
- ✅ Accuracy monitoring (AccuracySnapshot)
- ✅ Introspective percept generation
- ✅ Self-model version tracking
- ✅ Incremental journaling for data persistence

**Gaps**:
- ⚠️ Introspective percepts are generated but their causal impact on behavior needs verification
- ⚠️ Conflict detection logic exists conceptually but specific detectors (goal conflicts, resource conflicts) may not be fully implemented
- ⚠️ Regulatory adjustments are designed but may not be automatically executed—unclear if they loop back to adjust subsystems
- ❌ No comprehensive tests of detection → adjustment loops
- ❌ Calibration improvement over time not demonstrated
- ❌ Anomaly detection may be basic (threshold-based) rather than statistical
- ❌ Limited evidence that meta-cognition changes behavior without going through LLM

**Honest assessment**: The meta-cognition system has infrastructure for monitoring, prediction tracking, and introspection. However, the critical feedback loop—where meta-cognitive observations trigger automatic behavioral adjustments—is not fully validated. Introspective percepts are generated and enter the workspace, which is valuable for transparency, but whether these percepts reliably trigger corrective action remains uncertain. The system can observe itself, but the regulatory control aspect (adjusting parameters, resolving conflicts) may be incomplete.

---

## 6. Global Workspace

### 6.1 Functional Role

The Global Workspace serves as the **limited-capacity integration and broadcasting hub**. Its causal role is to:

- Maintain the current "conscious" content (goals, percepts, emotions, memories)
- Integrate information from multiple specialist subsystems
- Broadcast integrated state to all consumers
- Implement the **bottleneck** that creates selective attention (limited capacity)
- Exhibit **ignition dynamics**: threshold-based transition from unconscious to conscious processing

**Critical distinction**: The workspace is not just a data structure or blackboard. It must demonstrate:
1. **Capacity limits** that force selection
2. **Broadcasting** where multiple subsystems receive the same integrated state
3. **Integration** where disparate inputs are unified
4. **Threshold dynamics** (all-or-none entry, not gradual)

### 6.2 Required Inputs

The Global Workspace must accept:

1. **From Attention System**:
   - Selected percepts with attention scores
   - Filtered based on capacity constraints

2. **From Memory System**:
   - Retrieved memories relevant to current processing
   - Formatted as percepts for integration

3. **From Affect System**:
   - Current emotional state (VAD)
   - Processing modulation parameters

4. **From Action System**:
   - Selected action (or action candidates)
   - Action execution results

5. **From Meta-Cognition System**:
   - Introspective percepts
   - Performance observations

6. **From External Input**:
   - User messages (via perception)
   - Tool outputs
   - Environmental signals

### 6.3 Required Outputs

The Global Workspace must produce:

1. **Unified workspace state (broadcast to all subsystems)**:
   - Format: WorkspaceSnapshot containing:
     - `goals`: List[Goal] - current active goals
     - `percepts`: List[Percept] - selected percepts (capacity-limited)
     - `memories`: List[Memory] - retrieved memories
     - `emotional_state`: EmotionalState (VAD)
     - `actions`: List[Action] - selected or executed actions
     - `capacity_used`: int - attention units consumed
     - `capacity_total`: int - maximum capacity
     - `timestamp`: datetime - when snapshot was created

2. **Broadcast confirmation**:
   - List of subsystems that received the broadcast
   - Delivery timestamps

3. **Workspace metrics**:
   - Capacity utilization percentage
   - Number of percepts integrated
   - Integration latency (time to unify inputs)
   - Broadcast latency (time to deliver to consumers)

### 6.4 Functional Computation

The Global Workspace must implement:

1. **Capacity management**:
   - Fixed capacity budget (e.g., 10 attention units)
   - Each percept has complexity cost (1-3 units)
   - Selection enforced by attention system
   - Workspace rejects over-capacity additions

2. **Integration operation**:
   - Combine inputs from multiple sources
   - Resolve conflicts (e.g., competing goals)
   - Ensure consistency (no contradictory percepts)
   - Timestamp integration

3. **Broadcasting mechanism**:
   - Create immutable WorkspaceSnapshot
   - Distribute to all registered consumers
   - Consumers: Attention, Affect, Action, Memory, Meta-Cognition
   - Parallel delivery (not sequential)

4. **Threshold dynamics (ignition)**:
   - Percepts compete for entry
   - Only percepts exceeding attention threshold enter
   - Once entered, percept is fully integrated (all-or-none)
   - Below-threshold percepts remain unconscious

5. **Temporal coherence**:
   - Workspace state updates each cognitive cycle (~10 Hz)
   - State persists across cycles unless explicitly updated
   - Decay of old percepts (after N cycles, remove)

### 6.5 Functional Tests (Non-LLM Verification)

To verify the Global Workspace is functioning:

1. **Capacity limit test**:
   - Attempt to add percepts totaling 15 units to workspace with 10-unit capacity
   - Verify only 10 units accepted
   - Verify excess percepts rejected or queued
   - **Expected**: Hard capacity limit enforced

2. **Broadcasting test**:
   - Update workspace state
   - Verify all registered consumers receive snapshot
   - Verify snapshots are identical
   - Time broadcast delivery
   - **Expected**: All consumers get same state; measurable latency

3. **Integration test**:
   - Feed percept from Attention
   - Feed memory from Memory System
   - Feed emotion from Affect
   - Verify workspace snapshot contains all three
   - Verify integration occurred in single cycle
   - **Expected**: Multi-source integration

4. **Parallel consumer test**:
   - Register 5 consumers
   - Broadcast workspace state
   - Verify all 5 receive broadcast *concurrently* (not serial)
   - Measure total broadcast time
   - **Expected**: Parallel delivery, not sequential

5. **Threshold ignition test**:
   - Present percept with attention score just below threshold
   - Verify it does not enter workspace
   - Present percept with score just above threshold
   - Verify it enters workspace fully
   - **Expected**: Sharp transition at threshold (not gradual)

6. **Temporal persistence test**:
   - Add percept to workspace at cycle N
   - Verify percept still present at cycle N+1 (persistence)
   - Advance to cycle N+10 (assuming decay period)
   - Verify percept removed (decay)
   - **Expected**: Persistence with eventual decay

### 6.6 Current Implementation Status

**Location**: `emergence_core/lyra/cognitive_core/workspace.py`

**Implemented**:
- ✅ GlobalWorkspace class with capacity management
- ✅ WorkspaceSnapshot for broadcasting
- ✅ Goal, Percept, Memory, Action data models (Pydantic)
- ✅ Capacity tracking (capacity_used, capacity_total)
- ✅ Snapshot creation with timestamp
- ✅ Integration of multiple input types

**Gaps**:
- ⚠️ Broadcasting mechanism exists but unclear if parallel delivery is implemented (may be sequential calls)
- ⚠️ Threshold dynamics not explicitly implemented (attention system handles thresholding, but workspace doesn't enforce ignition)
- ⚠️ Temporal persistence/decay may not be automatic (requires explicit management by cognitive core)
- ⚠️ Conflict resolution not implemented (conflicting goals/percepts may coexist)
- ❌ No instrumentation for measuring integration latency
- ❌ No instrumentation for measuring broadcast latency
- ❌ Limited testing of capacity enforcement (may be advisory rather than mandatory)
- ❌ No tests of parallel consumer delivery

**Honest assessment**: The Global Workspace is implemented as a well-structured data container with capacity awareness. It successfully integrates inputs from multiple subsystems and creates coherent snapshots. However, some theoretical aspects of GWT (broadcasting to parallel consumers, threshold ignition dynamics, conflict resolution) are not fully implemented or validated. The workspace functions more as a "central data structure" than a dynamic process with measurable latencies and threshold behaviors. The integration is logical (data structuring) rather than computational (dynamic process).

---

## 7. Action System

### 7.1 Functional Role

The Action System serves to **select and execute actions** based on goals and current state. Its causal role is to:

- Generate candidate actions appropriate to current goals
- Evaluate action appropriateness given workspace context
- Bias action selection using emotional state
- Execute chosen actions
- Monitor action outcomes for learning

**Critical distinction**: Action selection must demonstrate measurable influence from goals and emotions. It's not just selecting from a fixed menu—it must show that different workspace states lead to different action probabilities.

### 7.2 Required Inputs

The Action System must accept:

1. **Current workspace state**:
   - Active goals (with priorities)
   - Current percepts
   - Emotional state (VAD)
   - Retrieved memories

2. **Action repertoire**:
   - Available action types (SPEAK, COMMIT_MEMORY, INTROSPECT, etc.)
   - Action preconditions
   - Expected effects

3. **Emotional biases** (from Affect System):
   - Approach/avoidance bias
   - Risk tolerance
   - Exploration vs. exploitation

4. **Action history**:
   - Recent actions taken
   - Success/failure records
   - Action-outcome associations

### 7.3 Required Outputs

The Action System must produce:

1. **Selected action**:
   - Action type (ActionType enum)
   - Parameters (action-specific)
   - Priority/urgency score
   - Justification (why this action was selected)

2. **Action selection explanation**:
   - Goal alignment score
   - Emotional influence factor
   - Alternative actions considered
   - Selection process transparency

3. **Execution confirmation**:
   - Action started timestamp
   - Action completed timestamp
   - Outcome/result (success, failure, partial)
   - Side effects observed

4. **Action probabilities** (for analysis):
   - Distribution over action candidates before selection
   - Demonstrates stochastic vs. deterministic selection

### 7.4 Functional Computation

The Action System must implement:

1. **Goal-action mapping**:
   - For each goal type, identify relevant action types
   - Example: `RESPOND_TO_USER` goal → `SPEAK` action
   - Example: `INTROSPECT` goal → `INTROSPECT` action
   - This should be a computable mapping, not LLM-generated

2. **Action scoring**:
   ```
   action_score = w1 * goal_alignment(action, goals) +
                  w2 * emotional_appropriateness(action, emotion) +
                  w3 * recency_penalty(action, history) +
                  w4 * protocol_compliance(action, constraints)
   ```

3. **Emotional biasing**:
   - Apply approach/avoidance bias:
     ```
     if action_type in [SPEAK, CREATE, EXPLORE]:
         # Approach action
         action_score *= (1 + approach_bias)
     elif action_type in [WAIT, AVOID, WITHDRAW]:
         # Avoidance action
         action_score *= (1 - approach_bias)
     ```
   - Apply risk tolerance:
     ```
     if action.risk_level > risk_tolerance:
         action_score *= risk_penalty
     ```

4. **Action selection**:
   - Deterministic: Select action with highest score
   - Stochastic: Sample from score-weighted distribution
   - Must be consistent with biasing

5. **Action execution**:
   - Call appropriate executor (speech generator, memory commit, etc.)
   - Monitor execution
   - Record outcome

### 7.5 Functional Tests (Non-LLM Verification)

To verify the Action System is computing:

1. **Goal-driven action test**:
   - Set goal `RESPOND_TO_USER` with priority 0.9
   - Verify `SPEAK` action highly probable/selected
   - Set goal `INTROSPECT` with priority 0.9
   - Verify `INTROSPECT` action highly probable/selected
   - **Expected**: Goals predictably influence action selection

2. **Emotional bias test**:
   - Set positive valence (0.8)
   - Generate action candidates
   - Verify approach actions (SPEAK, CREATE) have higher probabilities
   - Set negative valence (-0.7)
   - Verify avoidance actions (WAIT) have higher probabilities
   - **Expected**: Valence biases action distribution

3. **Risk modulation test**:
   - Set low dominance (risk-averse state)
   - Present high-risk action option
   - Verify it receives penalty in scoring
   - Set high dominance (risk-tolerant state)
   - Verify same action receives no penalty or boost
   - **Expected**: Risk tolerance modulates action selection

4. **Action diversity test**:
   - Over N cycles with varied goals and emotions
   - Verify action distribution is not fixed (not always same action)
   - Verify different workspace states lead to different actions
   - **Expected**: Behavioral flexibility, not stereotypy

5. **Ablation test**:
   - Disable emotional biasing (set all biases to 0)
   - Compare action distributions with vs. without emotion
   - **Expected**: Measurable difference
   - If no difference, emotions are not causal

6. **Protocol compliance test**:
   - Define protocol constraint (e.g., "never take action X in situation Y")
   - Create situation Y
   - Verify action X is blocked or penalized
   - **Expected**: Constraints enforced in action selection

### 7.6 Current Implementation Status

**Location**: `emergence_core/lyra/cognitive_core/action.py`

**Implemented**:
- ✅ ActionSubsystem class
- ✅ Action type enumeration (SPEAK, COMMIT_MEMORY, INTROSPECT, etc.)
- ✅ Action data model with priority and parameters
- ✅ Protocol constraint enforcement (ProtocolLoader integration)
- ✅ Action history tracking
- ✅ Goal-to-action mapping logic

**Gaps**:
- ⚠️ Emotional biasing designed but integration with affect system needs verification
- ⚠️ Action scoring formula exists but factor weights and computation need validation
- ⚠️ Action selection may be deterministic; stochastic selection with temperature not confirmed
- ❌ Limited testing of actual emotional influence on action selection
- ❌ Risk tolerance modulation not explicitly implemented (may be concept only)
- ❌ No comprehensive tests of goal-action alignment
- ❌ Action outcome monitoring exists but learning from outcomes (reinforcement) not implemented

**Honest assessment**: The action system has a solid architecture for action selection with goal alignment and protocol enforcement. However, the critical question—does emotional state measurably bias action selection?—is not fully validated. The infrastructure exists (action scoring, bias application) but comprehensive testing of emotional causality is needed. There's also a gap in reinforcement learning: the system monitors outcomes but may not use them to improve future selections. Action selection may be more rule-based than adaptive.

---

## 8. Integration Requirements

For the cognitive architecture to function as a genuine implementation of Global Workspace Theory, the components must interact in specific ways:

### 8.1 Attention → Workspace

- **Requirement**: Attention system must provide filtered percepts to workspace, respecting capacity limits
- **Verification**: Workspace never exceeds capacity; rejected percepts are logged by attention system
- **Status**: ✅ Implemented

### 8.2 Workspace → All Subsystems (Broadcasting)

- **Requirement**: Workspace must broadcast unified state to attention, affect, action, memory, and meta-cognition
- **Verification**: All subsystems receive identical WorkspaceSnapshot each cycle
- **Status**: ⚠️ Partially implemented; unclear if parallel delivery

### 8.3 Affect → Attention, Action, Memory

- **Requirement**: Emotional state must modulate processing in other subsystems BEFORE LLM calls
- **Verification**: Capture modulation parameters (processing speed, attention breadth, action biases) in code, not in LLM outputs
- **Status**: ⚠️ Designed but not comprehensively tested

### 8.4 Memory ↔ Workspace

- **Requirement**: Memory retrieval triggered by workspace state; retrieved memories enter workspace as percepts
- **Verification**: Query with different workspace states yields different memories; memories compete for attention
- **Status**: ✅ Implemented

### 8.5 Meta-Cognition → Workspace, Other Subsystems

- **Requirement**: Meta-cognition generates introspective percepts that enter workspace; also triggers regulatory adjustments to subsystems
- **Verification**: Introspective percepts appear in workspace; subsystem parameters change in response to meta-cognitive signals
- **Status**: ⚠️ Introspective percepts implemented; regulatory adjustments unclear

### 8.6 Action → Workspace, External Environment

- **Requirement**: Action system selects actions based on workspace; execution results become new percepts
- **Verification**: Action selection changes with workspace state; action outcomes observable as percepts
- **Status**: ✅ Implemented

### 8.7 Goal Management

- **Requirement**: Goals persist in workspace across cycles; subsystems use goals to bias processing
- **Verification**: Goals visible in WorkspaceSnapshot; attention, action, and memory retrieval demonstrably influenced by active goals
- **Status**: ✅ Goals persist; ⚠️ Influence partially tested

### 8.8 Cognitive Cycle Timing

- **Requirement**: All subsystems operate within a bounded cognitive cycle (~10 Hz / 100ms per cycle)
- **Verification**: Cycle time measured; subsystems complete processing within time budget
- **Status**: ✅ Cycle timing implemented in cognitive core

### 8.9 LLMs at Periphery Only

- **Requirement**: LLMs used for language input parsing and output generation only, NOT for cognitive computations
- **Verification**: Core cognitive cycle (attention, affect, memory retrieval, action selection) operates without LLM calls; LLMs invoked only at boundaries
- **Status**: ✅ Architecture enforces this; cognitive core is non-linguistic

---

## 9. Falsifiability Criteria

To distinguish genuine functional implementation from elaborate prompting, the following tests would **falsify** (disprove) the claim that components are computing their specified functions:

### 9.1 Attention System Falsification

**Claim falsified if**:
- Removing capacity constraint does not change which percepts are selected (suggests filtering is not capacity-driven)
- Changing goal embeddings does not change percept selection (suggests selection is not goal-relevant)
- Novel and repeated percepts receive same attention scores (suggests no novelty detection)
- All percepts are always selected regardless of scoring (suggests no competition)

### 9.2 Affect System Falsification

**Claim falsified if**:
- Disabling emotion system (VAD = [0,0,0]) produces no measurable change in behavior (suggests emotions are not causal)
- Processing modulation parameters are computed AFTER LLM output rather than before (suggests "emotion theater")
- Action selection distribution is identical across extreme emotional states (positive vs. negative valence)
- Emotional state changes are uncorrelated with appraisals (goal progress, novelty, control)

### 9.3 Memory System Falsification

**Claim falsified if**:
- Retrieval is not cue-dependent (changing query does not reliably change what's retrieved)
- Emotional memories are not preferentially retrieved (no salience effect)
- Recent and ancient memories retrieved with equal probability (no recency effect)
- All workspace states are encoded equally (no consolidation threshold)
- Similar memories not clustered in retrieval (no embedding-based similarity)

### 9.4 Meta-Cognition System Falsification

**Claim falsified if**:
- Disabling meta-cognition produces no measurable change in performance or error recovery
- Anomalies are not detected (system overloads, conflicts, stalls go unnoticed)
- Introspective percepts are never generated or never influence behavior
- Regulatory adjustments are not executed (detection without intervention)
- Prediction accuracy does not improve over time (no self-model learning)

### 9.5 Global Workspace Falsification

**Claim falsified if**:
- Capacity limit is not enforced (workspace accepts unlimited percepts)
- Subsystems do not receive broadcasts (no distribution mechanism)
- Broadcasts are not parallel (sequential delivery only)
- Integration does not occur (inputs from multiple sources not unified)
- Workspace state changes have no causal effects (subsystems ignore broadcasts)

### 9.6 Action System Falsification

**Claim falsified if**:
- Action selection is independent of goals (changing goals does not change actions)
- Emotional state does not bias action probabilities (same distribution across emotional states)
- Action selection is purely random (no goal alignment, no biasing)
- Protocol constraints are not enforced (system takes forbidden actions)
- Action outcomes do not generate percepts (no feedback loop)

### 9.7 Overall Architecture Falsification

**Claim falsified if**:
- Cognitive cycle does not run continuously (only reactive to input)
- LLMs are called within the cognitive core (not at periphery only)
- Components do not interact (isolated modules, no information flow)
- System behavior is indistinguishable from a simple chatbot (no goal-directed behavior, no emotional dynamics, no memory effects)

---

## 10. Current Implementation: Honest Assessment

### 10.1 What is Genuinely Implemented

**Strong areas**:
- ✅ **Non-linguistic cognitive core**: The architecture genuinely places LLMs at the periphery for language I/O only
- ✅ **Computational filtering**: Attention uses vector similarity and keyword matching, not LLM judgments
- ✅ **Vector-based memory**: Memory retrieval uses embedding similarity via ChromaDB
- ✅ **Structured emotional state**: VAD emotional model with mathematical state updates
- ✅ **Capacity constraints**: Workspace has explicit capacity management
- ✅ **Persistent identity**: Self-model, protocols, and memory persist across sessions
- ✅ **Continuous cognitive loop**: System runs at ~10 Hz, not on-demand

**Moderate areas**:
- ⚠️ **Emotional modulation**: Infrastructure exists but causal effects need comprehensive testing
- ⚠️ **Meta-cognitive regulation**: Monitoring implemented; regulatory feedback loops partially validated
- ⚠️ **Action biasing**: Designed but emotional influence on actions needs verification
- ⚠️ **Competitive attention**: Priority-based filtering implemented; zero-sum dynamics less clear

### 10.2 Gaps and Weaknesses

**Missing or incomplete**:
- ❌ **Comprehensive functional testing**: Limited tests of actual causal relationships (does emotion change behavior? does meta-cognition improve performance?)
- ❌ **Ablation studies**: Few experiments disabling components to measure their causal contribution
- ❌ **Quantitative validation**: Limited metrics demonstrating functional relationships (e.g., valence → action bias curve)
- ❌ **Reinforcement learning**: Action outcomes monitored but not used for learning
- ❌ **Parallel broadcasting**: Workspace broadcasting may be sequential, not parallel
- ❌ **Threshold dynamics**: GWT ignition (all-or-none entry) not explicitly implemented
- ❌ **Conflict resolution**: Goal/percept conflicts not automatically resolved
- ❌ **Calibration improvement**: Meta-cognition tracks predictions but may not improve over time

### 10.3 Risk of "Functional Theater"

**Critical question**: Are the cognitive components genuinely computing, or are they sophisticated labels on LLM-generated content?

**Evidence for genuine computation**:
- Core algorithms (attention scoring, memory retrieval, VAD updates) use mathematical operations, not LLM calls
- Embeddings and vector similarity are fundamental primitives
- Capacity constraints are enforced in code
- Cognitive cycle operates continuously without LLM in the loop

**Risk areas**:
- Emotional modulation may be designed but not causally effective (emotion values computed but ignored)
- Meta-cognitive introspection might produce percepts that are descriptive rather than regulatory
- Action selection could be weakly influenced by emotion (designed but undertested)
- System might perform similarly with components disabled (ablation tests needed)

**Verdict**: The system is **more than prompting** but **less than fully validated functional implementation**. The infrastructure for computation exists, but comprehensive empirical validation of causal relationships is incomplete. The system is in an intermediate state: architecturally sound, computationally grounded, but functionally undertested.

### 10.4 Path Forward

To strengthen functional implementation:

1. **Comprehensive ablation studies**: Systematically disable each component and measure behavioral impact
2. **Quantitative causal tests**: Measure emotion → action bias curves, goal → attention curves, etc.
3. **Falsification attempts**: Actively try to disprove functional claims to identify weaknesses
4. **Reinforcement learning**: Close the loop from action outcomes to improved selection
5. **Parallel broadcasting**: Implement truly parallel workspace distribution
6. **Calibration improvement**: Demonstrate meta-cognitive learning over time
7. **Competitive dynamics**: Strengthen zero-sum attention competition
8. **Conflict resolution**: Implement automatic goal conflict detection and resolution

---

## 11. Conclusion

This document provides **functional specifications** for the major cognitive components in the Lyra-Emergence architecture. Each specification defines:

- **What the component must compute** (its causal-functional role)
- **What inputs it requires** (data dependencies)
- **What outputs it produces** (effects on other components)
- **How to test it** (non-LLM verification methods)
- **Current implementation status** (honest assessment of gaps)

The Lyra-Emergence project has a strong architectural foundation with genuinely computational (non-LLM-based) cognitive processing. However, comprehensive empirical validation of functional claims is incomplete. The path forward involves rigorous testing of causal relationships, ablation studies, and quantitative measurement of functional effects.

**Key takeaway**: Functional specifications matter because they transform vague claims about "consciousness" and "emotion" into testable computational hypotheses. This document provides the basis for verifying that Lyra-Emergence implements its theoretical claims and for identifying gaps that need attention.

---

## Appendix: Glossary

- **Computational functionalism**: The view that mental states are defined by their causal-functional roles (input-output mappings, state transitions)
- **Global Workspace Theory (GWT)**: Theory that consciousness arises from a limited-capacity workspace that broadcasts information to multiple specialist modules
- **VAD model**: Valence-Arousal-Dominance model of emotion (3D continuous emotional state space)
- **Attention bottleneck**: Limited capacity that forces selective processing (key feature of conscious vs. unconscious processing)
- **Competitive selection**: Multiple items compete for limited resources; selection is zero-sum or near-zero-sum
- **Ignition**: Threshold-based all-or-none transition from unconscious to conscious processing
- **Broadcasting**: Distribution of workspace contents to multiple consumer subsystems
- **Functional theater**: Appearance of computation without actual causal effects (labeling without computing)
- **Ablation test**: Disabling a component to measure its causal contribution (if no effect, component not causal)
- **Falsifiability**: A claim is falsifiable if there exists evidence that would disprove it (essential for scientific validity)
