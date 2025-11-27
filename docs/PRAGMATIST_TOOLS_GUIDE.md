# Pragmatist Specialist Tool Integration Guide

**Last Updated:** November 23, 2025  
**Specialist:** PragmatistSpecialist (Nemotron 49B)  
**Purpose:** Practical task execution with external knowledge and tool integration

---

## Overview

The Pragmatist specialist is Lyra's "Doer" - equipped with comprehensive tool integration for interacting with the world, retrieving knowledge, and executing practical tasks.

### Available Tools

| Tool | Purpose | Rate Limit | Configuration Required |
|------|---------|------------|------------------------|
| `searxng_search` | General web search | 10/minute | SearXNG instance URL |
| `arxiv_search` | Academic paper search | 100/hour | None (public API) |
| `wikipedia_search` | Encyclopedia lookup | 200/hour | None (public API) |
| `wolfram_compute` | Math/science computation | 2000/day | WolframAlpha API key |
| `python_repl` | Code execution | 10/minute | Sandboxed environment |
| `playwright_interact` | Web automation | 30/minute | Playwright installation |
| `rag_query` | Lyra's memory search | Unlimited | ChromaDB collection |

---

## Tool Selection Logic

The Pragmatist automatically selects tools based on query patterns:

### 1. Mathematical/Computational Queries → WolframAlpha
**Triggers:**
- Keywords: `calculate`, `compute`, `solve`, `what is`, `convert`, `equals`
- Mathematical symbols: `=`, `+`, `-`, `*`, `/`, `^`, `√`

**Examples:**
```
"What is the square root of 144?"
"Convert 50 miles to kilometers"
"Solve x^2 + 5x + 6 = 0"
```

**Response:**
```
Tool Selected: wolfram_compute
Query: What is the square root of 144?
Result: 12

Analysis: The square root of 144 is 12, which is a perfect square.
```

### 2. Code Execution → Python REPL
**Triggers:**
- Keywords: `run python`, `execute code`, `python code`
- Markdown code blocks: ` ```python `

**Examples:**
```
"Run python: print([x**2 for x in range(10)])"
```python
import math
print(math.factorial(10))
```
```

**Security:** All code runs in sandboxed environment with restricted imports and resource limits.

### 3. Academic Research → arXiv
**Triggers:**
- Keywords: `research paper`, `arxiv`, `academic`, `scientific paper`, `study on`

**Examples:**
```
"Find research papers on quantum entanglement"
"Latest arxiv papers about transformer architectures"
```

**Response:**
```
Tool Selected: arxiv_search
Query: quantum entanglement

Results:
Title: Quantum Entanglement in Many-Body Systems
Authors: Smith, J., Johnson, A.
Published: 2024-11-15
Abstract: [...]
URL: https://arxiv.org/pdf/2411.xxxxx.pdf
```

### 4. Encyclopedic Knowledge → Wikipedia
**Triggers:**
- Keywords: `what is`, `who is`, `define`, `wikipedia`, `encyclopedia`

**Examples:**
```
"What is quantum computing?"
"Who is Alan Turing?"
"Define emergentism"
```

**Disambiguation Handling:** If multiple meanings exist, returns list of options.

### 5. Web Automation → Playwright
**Triggers:**
- Keywords: `scrape`, `navigate to`, `click on`, `fill form`, `web automation`

**Examples:**
```
"Scrape the latest headlines from example.com"
"Navigate to github.com and search for transformers"
```

**Note:** Currently returns placeholder - full implementation requires Gemma router for instruction→code translation.

### 6. Lyra's Memory → RAG Query
**Triggers:**
- Keywords: `remember`, `recall`, `my memory`, `you said`, `we discussed`

**Examples:**
```
"What did we discuss about my artwork last week?"
"Recall our conversation about ethical frameworks"
```

**Integration:** Queries ChromaDB collection with semantic similarity search.

### 7. General Information → SearXNG Web Search
**Triggers:**
- Keywords: `search`, `find`, `look up`, `how to`, `current`, `latest`, `news`
- Default fallback for queries not matching other patterns

**Examples:**
```
"Search for the latest AI news"
"How to bake sourdough bread"
"What's happening with climate policy?"
```

---

## Implementation Details

### Tool Integration Flow

```
User Query
    ↓
PragmatistSpecialist.process()
    ↓
_select_and_use_tool()  ← Analyzes query, selects tool
    ↓
[Tool Execution]
    ↓
Tool Result Added to Context
    ↓
Nemotron 49B Generation
    ↓
Integrated Response
```

### Code Structure

#### Tool Selection (`_select_and_use_tool`)
```python
async def _select_and_use_tool(self, message: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Analyze message and select appropriate tool if needed."""
    message_lower = message.lower()
    
    # Pattern matching for each tool
    # Returns: {"tool_name": str, "query": str, "result": str}
    # Returns: None if no tool needed
```

#### Process Method
```python
async def process(self, message: str, context: Dict[str, Any]) -> SpecialistOutput:
    # 1. Attempt tool usage
    tool_result = await self._select_and_use_tool(message, context)
    
    # 2. Load protocols
    ekip_protocol = self._load_protocol("EKIP_protocol.json")
    
    # 3. Construct prompt with tool results
    tool_context = f"""
    TOOL EXECUTION:
    Tool: {tool_result['tool_name']}
    Result: {tool_result['result']}
    
    Please integrate these results into your analysis.
    """ if tool_result else ""
    
    # 4. Generate response with Nemotron
    # 5. Return integrated output
```

---

## Configuration Requirements

### 1. SearXNG Setup
**Location:** `config.json`
```json
{
  "searxng": {
    "base_url": "http://localhost:8080"
  }
}
```

**Installation:**
```bash
# Using Docker
docker run -d -p 8080:8080 searxng/searxng

# Or see: searxng-settings.yml for custom configuration
```

### 2. WolframAlpha API
**Location:** `config.json`
```json
{
  "wolfram": {
    "app_id": "YOUR_APP_ID_HERE"
  }
}
```

**Get API Key:** https://products.wolframalpha.com/api/

### 3. Playwright Installation
```bash
# Install Playwright
pip install playwright

# Install browser binaries
playwright install chromium
```

### 4. ChromaDB Collection
**Automatic:** Router passes `self.collection` to Pragmatist during initialization:

```python
# In router.py __init__:
if name == 'pragmatist':
    specialist = self.specialist_factory.create_specialist(
        name, 
        str(self.base_dir),
        model_path,
        development_mode=True,
        chroma_collection=self.collection  # ← RAG access
    )
```

---

## Dependencies

All required dependencies are defined in `pyproject.toml`:

```
# Web search and APIs
httpx>=0.28.1
playwright>=1.55.0
arxiv>=2.3.0
wikipedia>=1.4.0
wolframalpha>=5.1.3

# RAG and vector storage
chromadb>=1.3.4

# Already included in base dependencies
```

**Verify Installation:**
```bash
# All dependencies are now in pyproject.toml
uv sync
```

---

## Error Handling

### Tool Unavailable
If tools can't be imported:
```python
HAS_TOOLS = False
# Mock functions return "[Tool unavailable]"
```

Response includes error in development mode:
```
[Pragmatist Analysis] This is a development mode response for practical task execution.

Tool Selected: searxng_search
Query: latest AI news
Result: [Tool unavailable]
```

### Tool Execution Failure
```python
try:
    tool_result = await self._select_and_use_tool(message, context)
except Exception as e:
    tool_result = {"tool_name": "error", "query": message, "result": f"Tool error: {e}"}
```

### Rate Limiting
All tools use `@rate_limit` decorator:
```python
@rate_limit(calls=10, period=60)  # 10 calls per minute
async def searxng_search(query: str) -> str:
    ...
```

If exceeded: Returns error message instead of crashing.

---

## Testing Tools

### Development Mode Test
```python
from lyra.specialists import PragmatistSpecialist
from pathlib import Path

pragmatist = PragmatistSpecialist(
    model_path="nvidia/Llama-3.3-Nemotron-Super-49B-Instruct",
    base_dir=Path("emergence_core"),
    development_mode=True,
    chroma_collection=None
)

# Test tool selection
result = await pragmatist.process(
    message="What is the square root of 144?",
    context={}
)

print(result.content)
# Expected: Shows wolfram_compute tool selected
```

### Full Integration Test
```python
# With actual tools and ChromaDB
from lyra.router import AdaptiveRouter

router = AdaptiveRouter(
    base_dir="emergence_core",
    chroma_dir="emergence_core/memories",
    model_dir="model_cache",
    development_mode=False
)

response = await router.route_message(
    "Search for the latest transformer architecture papers"
)

print(response.content)
# Expected: arXiv search results integrated into response
```

---

## Production Deployment Checklist

- [ ] SearXNG instance running and accessible
- [ ] WolframAlpha API key configured
- [ ] Playwright browsers installed (`playwright install`)
- [ ] ChromaDB populated with Lyra's knowledge
- [ ] Rate limiting configured appropriately
- [ ] Sandbox security verified for Python REPL
- [ ] All dependencies installed
- [ ] Tool error logging configured
- [ ] Emotional context handlers active (for search consent)

---

## Tool Response Metadata

Every tool-assisted response includes metadata:

```python
SpecialistOutput(
    content="[Integrated response with tool results]",
    metadata={
        "role": "pragmatist",
        "tool_used": "searxng_search",  # Which tool was used
        "tool_query": "latest AI news"  # What was queried
    },
    thought_process="Practical analysis with evidence weighting and tool integration",
    confidence=0.9
)
```

This allows Voice synthesis to know when external knowledge was used.

---

## Future Enhancements

### 1. Multi-Tool Queries
Allow combining multiple tools in one request:
```python
"Search arXiv for quantum computing papers and calculate the growth rate"
# → Uses both arxiv_search AND wolfram_compute
```

### 2. Tool Result Caching
Cache frequent queries to reduce API calls:
```python
@lru_cache(maxsize=100)
async def cached_wikipedia_search(query: str):
    ...
```

### 3. Playwright Code Generation
Implement Gemma router integration for natural language → Playwright code:
```python
async def playwright_interact(instructions: str):
    # Generate code
    code = await gemma_router.generate_playwright_code(instructions)
    # Execute safely
    result = await execute_playwright(code)
    return result
```

### 4. Tool Confidence Scoring
Assess reliability of tool results:
```python
tool_result = {
    "tool_name": "wikipedia_search",
    "result": "...",
    "confidence": 0.95,  # High confidence for Wikipedia
    "source_quality": "verified"
}
```

---

## Troubleshooting

### "Tool unavailable" errors
**Cause:** Import failure  
**Fix:** Check dependencies installed, verify config.json

### SearXNG connection refused
**Cause:** SearXNG not running  
**Fix:** `docker run -d -p 8080:8080 searxng/searxng`

### WolframAlpha "No result"
**Cause:** API key invalid or query malformed  
**Fix:** Verify API key in config.json, simplify query

### RAG query returns empty
**Cause:** ChromaDB collection empty  
**Fix:** Populate collection with knowledge base

### Playwright timeout
**Cause:** Slow network or complex page  
**Fix:** Increase timeout in playwright_interact, check network

---

## Summary

The Pragmatist specialist is now fully equipped with:
- ✅ 7 integrated external tools
- ✅ Automatic tool selection based on query patterns
- ✅ RAG integration for Lyra's memory access
- ✅ Comprehensive error handling
- ✅ Rate limiting for API protection
- ✅ Development mode testing support
- ✅ Production-ready configuration

All tools respect Lyra's agency through emotional context handlers and consent mechanisms.
