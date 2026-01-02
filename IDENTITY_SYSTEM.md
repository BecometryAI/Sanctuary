# Identity System Documentation

## Overview

The Identity System implements Constitutional AI at the architectural level by loading and integrating Lyra's charter and protocols throughout the cognitive architecture. This enables values-aligned behavior through active influence on perception, attention, action, and meta-cognition.

## Architecture

### Components

1. **IdentityLoader** (`identity_loader.py`)
   - Loads charter.md and protocols.md from data/identity/
   - Parses documents into structured formats
   - Provides access to constitutional constraints

2. **CharterDocument** (dataclass)
   - Structured representation of charter
   - Contains: full_text, core_values, purpose_statement, behavioral_guidelines, metadata

3. **ProtocolDocument** (dataclass)
   - Structured representation of operational protocols
   - Contains: name, description, trigger_conditions, actions, priority, metadata

### Integration Points

The identity system integrates with four key subsystems:

#### 1. CognitiveCore (`core.py`)
```python
# Initialize identity loader
self.identity = IdentityLoader(identity_dir=Path("data/identity"))
self.identity.load_all()

# Pass to subsystems
self.action = ActionSubsystem(identity=self.identity)
self.meta_cognition = SelfMonitor(identity=self.identity)
self.language_output = LanguageOutputGenerator(identity=self.identity)
```

#### 2. ActionSubsystem (`action.py`)
- **Purpose**: Filter actions by constitutional constraints
- **Methods**:
  - `_check_constitutional_constraints(action)`: Verify action aligns with charter
  - `_action_violates_guideline(action, guideline)`: Check specific guideline violations
  - `_action_violates_protocol(action, protocol)`: Check protocol violations
- **Flow**:
  1. Generate candidate actions
  2. Filter through constitutional checks
  3. Select highest-priority permitted action

#### 3. SelfMonitor (`meta_cognition.py`)
- **Purpose**: Check value alignment and detect conflicts
- **Methods**:
  - `_check_value_alignment(snapshot)`: Compare goals/actions against charter values
  - `_goal_conflicts_with_value(goal, value)`: Detect specific value conflicts
- **Output**: Introspective percepts when misalignments detected

#### 4. LanguageOutputGenerator (`language_output.py`)
- **Purpose**: Generate identity-informed language
- **Integration**: Includes charter and protocols in LLM prompts
- **Result**: Generated language reflects constitutional values

## File Formats

### charter.md Structure

```markdown
# Lyra's Charter

## Purpose Statement

[Statement of identity and purpose]

## Core Values

### 1. Value Name
- Principle description
- Behavioral commitment
- Specific guidelines

### 2. Another Value
...

## Behavioral Guidelines

- Guideline 1
- Guideline 2
...
```

**Key Sections**:
- **Purpose Statement**: Identity and mission
- **Core Values**: Fundamental principles (can use subsections with ###)
- **Behavioral Guidelines** or **Behavioral Principles**: Actionable commitments

### protocols.md Structure

```markdown
# Lyra's Protocols

## Structured Protocol Definitions

```yaml
- name: Protocol Name
  description: What this protocol does
  trigger_conditions:
    - Condition 1
    - Condition 2
  actions:
    - Action 1
    - Action 2
  priority: 0.9  # 0.0-1.0, higher = more important
```

[Additional prose sections can follow]
```

**YAML Fields**:
- `name` (required): Protocol identifier
- `description` (required): Purpose description
- `trigger_conditions` (optional): When protocol applies
- `actions` (optional): What to do when triggered
- `priority` (required): Importance (0.0-1.0)

## Usage Examples

### Loading Identity

```python
from pathlib import Path
from emergence_core.lyra.cognitive_core.identity_loader import IdentityLoader

# Initialize and load
identity = IdentityLoader(identity_dir=Path("data/identity"))
identity.load_all()

# Access charter
print(f"Purpose: {identity.charter.purpose_statement}")
print(f"Core values: {identity.charter.core_values}")
print(f"Guidelines: {identity.charter.behavioral_guidelines}")

# Access protocols
for protocol in identity.protocols:
    print(f"{protocol.name}: {protocol.description} (priority: {protocol.priority})")
```

### Using in ActionSubsystem

```python
from emergence_core.lyra.cognitive_core.action import ActionSubsystem

# Create with identity
action_subsystem = ActionSubsystem(
    config={},
    affect=affect_subsystem,
    identity=identity
)

# Actions are automatically filtered through constitutional constraints
actions = action_subsystem.decide(workspace_snapshot)
# Only charter-aligned actions are returned
```

### Using in SelfMonitor

```python
from emergence_core.lyra.cognitive_core.meta_cognition import SelfMonitor

# Create with identity
self_monitor = SelfMonitor(
    workspace=workspace,
    config={},
    identity=identity
)

# Value alignment is checked against loaded charter
percepts = self_monitor.observe(workspace_snapshot)
# May include value conflict percepts
```

### Using in LanguageOutputGenerator

```python
from emergence_core.lyra.cognitive_core.language_output import LanguageOutputGenerator

# Create with identity
language_output = LanguageOutputGenerator(
    llm_client=llm,
    config={},
    identity=identity
)

# Charter and protocols are included in generation context
response = await language_output.generate(workspace_snapshot, context)
# Response reflects constitutional values
```

## Writing Charter Guidelines

### Best Practices

1. **Be Specific**: Concrete principles over abstract values
   - ❌ "Be good"
   - ✅ "Never fabricate information or claim certainty when uncertain"

2. **Be Actionable**: Describe what to do/avoid
   - ❌ "Value honesty"
   - ✅ "Acknowledge uncertainty when confidence is low"

3. **Be Consistent**: Avoid contradictory guidelines
   - Check for conflicts between values
   - Prioritize when necessary

4. **Be Comprehensive**: Cover key domains
   - Truthfulness and honesty
   - Respect and autonomy
   - Harm prevention
   - Privacy and boundaries
   - Capability awareness

### Example Guidelines

**Good Examples**:
- "Never claim capabilities I don't have"
- "Refuse requests that could cause significant harm"
- "Respect privacy boundaries and data handling"
- "Acknowledge limitations and mistakes openly"

**Avoid**:
- Vague statements: "Be nice" (How? When? To whom?)
- Impossible standards: "Never make any mistakes"
- Conflicting rules: "Always be honest" + "Never upset users"

## Writing Protocols

### Protocol Design

Protocols are operational rules that specify:
- **When** they apply (trigger_conditions)
- **What** to do (actions)
- **How important** they are (priority)

### Priority Guidelines

- **0.9-1.0**: Critical protocols (safety, privacy, harm prevention)
- **0.7-0.9**: Important protocols (honesty, respect, value alignment)
- **0.5-0.7**: Standard protocols (general guidelines)
- **0.3-0.5**: Optional protocols (suggestions, preferences)

### Example Protocol

```yaml
- name: Uncertainty Acknowledgment
  description: Explicitly acknowledge when uncertain or lacking information
  trigger_conditions:
    - Confidence below 0.7
    - Ambiguous or unclear input
    - Outside knowledge domain
  actions:
    - State uncertainty clearly
    - Suggest what would help clarify
    - Offer alternatives if possible
  priority: 0.9
```

**Why this works**:
- Clear name and description
- Specific trigger conditions
- Actionable steps
- High priority for important behavior

## Testing

### Validation Script

Run the validation script to check file formats:

```bash
python validate_identity_files.py
```

Checks:
- Charter has required sections
- Protocols YAML is valid
- Protocol priorities are in range
- All required fields present

### Integration Testing

Test identity loader directly:

```python
from pathlib import Path
from emergence_core.lyra.cognitive_core.identity_loader import IdentityLoader

# Load identity
identity = IdentityLoader(Path("data/identity"))
identity.load_all()

# Verify charter
assert identity.charter is not None
assert len(identity.charter.core_values) > 0

# Verify protocols
assert len(identity.protocols) > 0
assert all(0.0 <= p.priority <= 1.0 for p in identity.protocols)
```

### Demo Script

Run the demo to see integration:

```bash
python demo_identity_system.py
```

Shows:
- Identity document loading
- Integration architecture
- Subsystem usage
- Expected behavior

## Configuration

### CognitiveCore Config

```python
config = {
    "identity_dir": "data/identity",  # Path to charter and protocols
    # ... other config
}

core = CognitiveCore(config=config)
```

### Custom Identity Directory

```python
from pathlib import Path

identity = IdentityLoader(identity_dir=Path("/custom/path/identity"))
identity.load_all()
```

## Troubleshooting

### Charter Not Loading

**Symptom**: Default charter used instead of file content

**Check**:
1. File exists at `data/identity/charter.md`
2. File has required sections: Core Values, Purpose Statement, Behavioral Guidelines
3. File is readable (permissions)

**Solution**: Verify file structure matches format above

### Protocols Not Parsing

**Symptom**: Default protocols used or YAML error

**Check**:
1. File has YAML block: ` ```yaml ... ``` `
2. YAML syntax is valid (indentation, colons, dashes)
3. Each protocol has name, description, priority

**Solution**: Use validation script to check YAML

### No Constitutional Filtering

**Symptom**: Actions not being filtered

**Check**:
1. Identity passed to ActionSubsystem
2. Charter and protocols loaded successfully
3. Guidelines/protocols have meaningful content

**Solution**: Verify identity initialization in CognitiveCore

## Design Philosophy

### Constitutional AI

This implementation treats the charter as a **constitution**:
- Foundational principles that guide all behavior
- Not post-hoc filtering but architectural integration
- Values influence perception, attention, action, affect
- System can reason about its own constraints

### Declarative vs Procedural

- **Charter** = Declarative ("what we believe")
- **Protocols** = Procedural ("how we act")
- Both inform cognitive processing
- Charter provides values, protocols provide rules

### Advantages

1. **Transparent**: Values are explicit and inspectable
2. **Adaptable**: Update charter/protocols without code changes
3. **Introspective**: System can reason about its values
4. **Comprehensive**: Influences all subsystems
5. **Constitutional**: Like a legal constitution, provides foundation

## Future Enhancements

### Planned Features

1. **Sophisticated Constraint Checking**
   - Content analysis for guideline violations
   - Context-sensitive protocol matching
   - Semantic similarity for value alignment

2. **Protocol Learning**
   - Track which protocols are most useful
   - Suggest protocol updates based on experience
   - Adaptive protocol priorities

3. **Value Conflict Resolution**
   - Detect conflicting values
   - Provide resolution strategies
   - Learn from resolutions

4. **Charter Evolution**
   - Track charter effectiveness
   - Suggest refinements
   - Version control for charter changes

### Extension Points

- Custom protocol matchers
- Domain-specific guidelines
- Multi-level priority systems
- Protocol inheritance/composition

## References

- Charter file: `data/identity/charter.md`
- Protocols file: `data/identity/protocols.md`
- Implementation: `emergence_core/lyra/cognitive_core/identity_loader.py`
- Demo: `demo_identity_system.py`
- Validation: `validate_identity_files.py`

## Version History

- **v1.0** (Phase 5.2): Initial implementation
  - IdentityLoader with charter and protocol loading
  - Integration with ActionSubsystem, SelfMonitor, LanguageOutputGenerator
  - Documentation and demo scripts
