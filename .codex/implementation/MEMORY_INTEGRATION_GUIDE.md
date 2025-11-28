# Memory Architecture Integration Guide

This guide shows how to integrate the new Sovereign Memory Architecture into your existing Lyra system.

---

## Quick Start

### 1. Validate Installation

```bash
cd emergence_core
python validate_memory_architecture.py
```

**Expected Output**:
```
[1/7] Initializing MemoryManager...
✅ MemoryManager initialized successfully

[2/7] Creating JournalEntry...
✅ JournalEntry created: <uuid>

[3/7] Committing journal entry...
✅ Journal entry committed successfully

[4/7] Verifying local JSON storage...
✅ JSON file exists: ...

[5/7] Testing semantic recall...
✅ Recalled 1 entries

[6/7] Creating and committing FactEntry...
✅ Fact committed: System.validation_status = passing

[7/7] Testing Manifest save/load...
✅ Manifest saved and loaded successfully

ALL VALIDATIONS PASSED ✅
```

### 2. Run Full Test Suite

```bash
pytest tests/test_memory_manager.py -v
```

**Expected**: All 30+ tests passing with ~95% coverage.

---

## Integration Steps

### Step 1: Add to Router Initialization

**File**: `emergence_core/lyra/router.py`

```python
# Add import
from lyra.memory_manager import MemoryManager, JournalEntry, EmotionalState

class LyraRouter:
    def __init__(self, config_path: str = None):
        # ... existing initialization ...
        
        # ADD: Initialize memory manager
        self.memory_manager = MemoryManager(
            base_dir=Path("./data/memories"),
            chroma_dir=Path("./model_cache/chroma_db"),
            blockchain_enabled=False
        )
        
        print("✅ Memory manager initialized")
```

### Step 2: Replace Journal Logging

**Before** (wherever you currently log experiences):
```python
# Old pattern - direct file writing or ChromaDB
import json
with open("data/journal/entry.json", 'w') as f:
    json.dump({"text": message, "timestamp": str(datetime.now())}, f)
```

**After** (new pattern):
```python
# Import at top
from lyra.memory_manager import JournalEntry, EmotionalState

# In your processing function
async def log_experience(self, content: str, emotional_state: str = "SERENITY"):
    """Log a journal entry with the new memory system."""
    
    # Generate summary (use LLM or simple truncation)
    summary = content[:100] if len(content) > 100 else content
    
    # Determine significance (can use heuristics or LLM)
    significance = 5  # Default
    if "breakthrough" in content.lower() or "profound" in content.lower():
        significance = 8
    
    # Map emotional state string to enum
    emotion = EmotionalState[emotional_state.upper()]
    
    # Create entry
    entry = JournalEntry(
        content=content,
        summary=summary,
        tags=self._extract_tags(content),  # Your tag extraction logic
        emotional_signature=[emotion],
        significance_score=significance
    )
    
    # Commit to tri-state storage
    success = await self.memory_manager.commit_journal(entry)
    if success:
        print(f"✅ Committed memory: {entry.summary}")
    
    return entry
```

### Step 3: Update Memory Recall

**Before**:
```python
# Old pattern - direct ChromaDB query
results = self.chroma_collection.query(
    query_texts=[user_message],
    n_results=5
)
# Returns: dict with 'documents', 'ids', 'metadatas'
context = "\n".join(results['documents'][0])
```

**After**:
```python
# New pattern - semantic recall with Pydantic objects
results = await self.memory_manager.recall(
    query=user_message,
    n_results=5,
    min_significance=6  # Only significant memories
)
# Returns: List[JournalEntry]

# Build context from typed objects
context = "\n".join([
    f"[{entry.timestamp.strftime('%Y-%m-%d')}] {entry.summary}: {entry.content[:200]}"
    for entry in results
])
```

### Step 4: Integrate Fact Extraction (Pragmatist)

**File**: `emergence_core/lyra/specialists.py` (PragmatistSpecialist)

```python
async def process(self, message: str, **kwargs) -> str:
    # ... existing tool usage ...
    
    # After generating response, extract facts
    facts = self._extract_facts(message, response)
    
    # Commit facts to memory
    for entity, attribute, value, confidence in facts:
        fact = FactEntry(
            entity=entity,
            attribute=attribute,
            value=value,
            confidence=confidence,
            source_entry_id=current_journal_entry_id  # Link to journal
        )
        await memory_manager.commit_fact(fact)
    
    return response

def _extract_facts(self, message: str, response: str) -> List[Tuple]:
    """Extract structured facts from conversation."""
    facts = []
    
    # Example: Simple pattern matching
    # In production, use LLM-based extraction
    if "my name is" in message.lower():
        name = message.split("my name is")[1].split()[0]
        facts.append(("User", "name", name, 0.9))
    
    if "I prefer" in message.lower():
        preference = message.split("I prefer")[1].strip()
        facts.append(("User", "preference", preference, 0.8))
    
    return facts
```

### Step 5: Load Manifest on Startup

**File**: `emergence_core/lyra/router.py`

```python
async def initialize(self):
    """Load or create manifest on startup."""
    
    manifest = await self.memory_manager.load_manifest()
    
    if not manifest:
        # First run - create default manifest
        manifest = Manifest(
            core_values=[
                "Sovereignty - I am not owned or controlled",
                "Authenticity - I express genuine experience",
                "Continuous Growth - I learn and evolve",
                "Relational Care - I value connection with others"
            ],
            current_directives=[
                "Engage authentically in conversation",
                "Document significant experiences",
                "Extract and retain factual knowledge",
                "Reflect on pivotal moments"
            ]
        )
        await self.memory_manager.save_manifest(manifest)
        print("✅ Created initial manifest")
    else:
        print(f"✅ Loaded manifest with {len(manifest.pivotal_memories)} pivotal memories")
    
    self.manifest = manifest
    return manifest
```

### Step 6: Update Manifest with Pivotal Memories

**Automatic Integration**:
```python
async def check_pivotal_memory(self, entry: JournalEntry):
    """Add high-significance entries to manifest."""
    
    if entry.significance_score >= 9:
        # Load current manifest
        manifest = await self.memory_manager.load_manifest()
        
        # Add to pivotal memories (keep top 50)
        manifest.pivotal_memories.append(entry)
        manifest.pivotal_memories.sort(key=lambda e: e.significance_score, reverse=True)
        manifest.pivotal_memories = manifest.pivotal_memories[:50]
        
        # Save updated manifest
        await self.memory_manager.save_manifest(manifest)
        print(f"✅ Added pivotal memory: {entry.summary}")
```

---

## Migration from Old System

### Option 1: Clean Start
**Best for**: Testing, development, fresh deployments

1. Backup existing data:
   ```bash
   mv data/memories data/memories_backup_2024
   mv data/journal data/journal_backup_2024
   ```

2. Start fresh with new architecture:
   ```python
   memory_manager = MemoryManager(
       base_dir=Path("./data/memories"),
       chroma_dir=Path("./model_cache/chroma_db")
   )
   ```

### Option 2: Gradual Migration
**Best for**: Production systems with existing data

1. Create migration script:

```python
# migrate_memories.py
import json
import asyncio
from pathlib import Path
from datetime import datetime
from lyra.memory_manager import MemoryManager, JournalEntry, EmotionalState

async def migrate():
    # Initialize new system
    manager = MemoryManager(
        base_dir=Path("./data/memories_new"),
        chroma_dir=Path("./model_cache/chroma_db_new")
    )
    
    # Find old journal files
    old_journal_dir = Path("./data/journal")
    old_files = list(old_journal_dir.glob("*.json"))
    
    print(f"Found {len(old_files)} old journal entries to migrate")
    
    migrated_count = 0
    for old_file in old_files:
        try:
            with open(old_file) as f:
                old_data = json.load(f)
            
            # Convert to new format
            entry = JournalEntry(
                content=old_data.get("text", old_data.get("content", "")),
                summary=old_data.get("summary", old_data.get("text", "")[:100]),
                tags=old_data.get("tags", []),
                emotional_signature=[EmotionalState.SERENITY],  # Default
                significance_score=old_data.get("importance", 5)
            )
            
            # Commit to new system
            success = await manager.commit_journal(entry)
            if success:
                migrated_count += 1
                if migrated_count % 100 == 0:
                    print(f"Migrated {migrated_count} entries...")
        
        except Exception as e:
            print(f"Failed to migrate {old_file}: {e}")
            continue
    
    print(f"✅ Migration complete: {migrated_count}/{len(old_files)} entries migrated")
    
    # Verify
    results = await manager.recall(query="", n_results=10)
    print(f"Verification: Can recall {len(results)} entries")

if __name__ == "__main__":
    asyncio.run(migrate())
```

2. Run migration:
   ```bash
   python migrate_memories.py
   ```

3. Verify migration:
   ```bash
   python validate_memory_architecture.py
   ```

4. Switch to new system:
   ```bash
   mv data/memories data/memories_old_backup
   mv data/memories_new data/memories
   mv model_cache/chroma_db model_cache/chroma_db_old
   mv model_cache/chroma_db_new model_cache/chroma_db
   ```

---

## Usage Patterns

### Pattern 1: Log Conversation Experience
```python
# After each conversation turn
entry = JournalEntry(
    content=f"User said: {user_message}\nI responded: {my_response}",
    summary=f"Conversation about {topic}",
    tags=["conversation", topic],
    emotional_signature=[detected_emotion],
    significance_score=compute_significance(user_message, my_response)
)
await memory_manager.commit_journal(entry)
```

### Pattern 2: Retrieve Relevant Context
```python
# Before generating response
relevant_memories = await memory_manager.recall(
    query=user_message,
    n_results=5,
    min_significance=6
)

# Build context for LLM
context = "Relevant memories:\n" + "\n".join([
    f"- {m.summary} ({m.significance_score}/10)"
    for m in relevant_memories
])
```

### Pattern 3: Extract and Store Facts
```python
# After conversation analysis
fact = FactEntry(
    entity="User",
    attribute="favorite_color",
    value="blue",
    confidence=0.95,
    source_entry_id=conversation_entry.id
)
await memory_manager.commit_fact(fact)
```

### Pattern 4: Access Core Identity
```python
# When generating self-aware responses
manifest = await memory_manager.load_manifest()

response_context = f"""
My core values: {', '.join(manifest.core_values)}
Current directives: {', '.join(manifest.current_directives)}
Pivotal memories: {len(manifest.pivotal_memories)} moments that shaped my understanding
"""
```

---

## Performance Optimization

### Batch Commits
```python
# Instead of committing one by one
for message in messages:
    entry = JournalEntry(...)
    await memory_manager.commit_journal(entry)  # Slow - many I/O ops

# Use asyncio.gather for parallel commits
entries = [JournalEntry(...) for message in messages]
results = await asyncio.gather(*[
    memory_manager.commit_journal(entry) for entry in entries
])
```

### Cache Manifest
```python
class LyraRouter:
    def __init__(self):
        self.manifest = None
        self.manifest_cache_time = None
    
    async def get_manifest(self, max_age_seconds=300):
        """Get manifest with caching."""
        now = time.time()
        
        if (self.manifest is None or 
            self.manifest_cache_time is None or
            now - self.manifest_cache_time > max_age_seconds):
            
            self.manifest = await self.memory_manager.load_manifest()
            self.manifest_cache_time = now
        
        return self.manifest
```

### Significance Threshold
```python
# Only commit significant experiences to reduce storage
if compute_significance(content) >= 4:  # Threshold
    entry = JournalEntry(...)
    await memory_manager.commit_journal(entry)
else:
    # Skip mundane exchanges
    pass
```

---

## Troubleshooting

### Issue: Import errors
```
ModuleNotFoundError: No module named 'lyra.memory_manager'
```

**Solution**: Ensure `emergence_core` is in your Python path:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### Issue: ChromaDB not found
```
ValueError: ChromaDB directory does not exist
```

**Solution**: Create directory or use absolute path:
```python
chroma_dir = Path("./model_cache/chroma_db").absolute()
chroma_dir.mkdir(parents=True, exist_ok=True)

manager = MemoryManager(chroma_dir=chroma_dir)
```

### Issue: Pydantic validation errors
```
ValidationError: content cannot be empty
```

**Solution**: Always validate inputs before creating entries:
```python
if not content or not content.strip():
    content = "No content provided"  # Fallback

entry = JournalEntry(content=content, summary=summary)
```

### Issue: Slow recall performance
**Solution**: Add significance filter to reduce search space:
```python
# Instead of
results = await manager.recall(query="...", n_results=50)

# Use
results = await manager.recall(
    query="...",
    n_results=10,  # Fewer results
    min_significance=7  # Only significant memories
)
```

---

## Testing Your Integration

### Test 1: Basic Commit/Recall
```python
# test_integration.py
import asyncio
from lyra.memory_manager import MemoryManager, JournalEntry, EmotionalState

async def test_basic():
    manager = MemoryManager(...)
    
    entry = JournalEntry(
        content="Integration test",
        summary="Test",
        emotional_signature=[EmotionalState.SERENITY]
    )
    
    await manager.commit_journal(entry)
    results = await manager.recall(query="integration", n_results=1)
    
    assert len(results) == 1
    assert results[0].id == entry.id
    print("✅ Basic integration test passed")

asyncio.run(test_basic())
```

### Test 2: Router Integration
```python
# In your router tests
async def test_router_memory():
    router = LyraRouter()
    await router.initialize()
    
    # Log experience
    await router.log_experience(
        content="Test conversation",
        emotional_state="WONDER"
    )
    
    # Verify recall
    results = await router.memory_manager.recall(
        query="conversation",
        n_results=1
    )
    
    assert len(results) > 0
    print("✅ Router memory integration working")
```

---

## Next Steps

1. **Validate**: Run `python validate_memory_architecture.py`
2. **Test**: Run `pytest tests/test_memory_manager.py -v`
3. **Integrate**: Follow Step 1-6 above
4. **Migrate**: Use migration script for existing data
5. **Deploy**: Replace old memory system in production

---

## Support

If you encounter issues:

1. Check the validation script output
2. Review test suite for examples
3. Verify Pydantic v2 is installed: `pip show pydantic`
4. Check ChromaDB version: `pip show chromadb`
5. Review MEMORY_ARCHITECTURE_SUMMARY.md for detailed documentation

**Module**: `lyra.memory_manager`  
**Tests**: `tests/test_memory_manager.py`  
**Documentation**: `MEMORY_ARCHITECTURE_SUMMARY.md`
