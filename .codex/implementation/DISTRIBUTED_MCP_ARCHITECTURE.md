# Distributed MCP Architecture: Ollama + LiteLLM Integration

**Status**: Proposal  
**Date**: December 13, 2025  
**Priority**: High - Cross-Server MCP and Tool Use

---

## Executive Summary

This document proposes integrating **Ollama + LiteLLM** as Lyra's distributed Model Context Protocol (MCP) server infrastructure, enabling cross-server tool execution and unified model access. This approach:

1. **Unifies** model access through a single OpenAI-compatible API
2. **Distributes** specialist models across multiple servers/GPUs
3. **Enables** MCP-based tool use across the network
4. **Maintains** Lyra's cognitive committee architecture
5. **Containerizes** all components for scalability and reliability

---

## Current Architecture Analysis

### Existing Implementation

**Components:**
- **Router**: Gemma 12B (in-process loading via Transformers)
- **Specialists**: Loaded directly via Transformers/Diffusers
  - Pragmatist: Nemotron 49B
  - Philosopher: Jamba 52B
  - Artist: Flux.1-schnell
  - Voice: LLaMA 3 70B (tensor parallelism)
  - Perception: LLaVA-NeXT-Mistral-7B
- **Tools**: Direct execution in Pragmatist specialist
  - SearXNG (containerized)
  - WolframAlpha, arxiv, wikipedia APIs
  - Playwright (web automation)
  - Python REPL

**Limitations:**
1. ❌ **Monolithic**: All models load in single process/machine
2. ❌ **Memory Constrained**: GPU swapping required for large models
3. ❌ **No Distribution**: Cannot split models across servers
4. ❌ **Tool Locality**: Tools run only where Pragmatist runs
5. ❌ **No MCP**: No standardized tool/model protocol
6. ❌ **Single Point of Failure**: Entire system crashes if one model fails
7. ❌ **Hard to Scale**: Adding capacity requires monolithic redeployment

**Strengths:**
1. ✅ **Simple**: Direct model loading, minimal infrastructure
2. ✅ **Unified Interface**: AdaptiveRouter handles all routing
3. ✅ **Sequential Flow**: Clear cognitive committee pattern
4. ✅ **Type-Safe**: Python dataclasses and Pydantic models
5. ✅ **Development Mode**: Mock models for rapid iteration

---

## Proposed: Ollama + LiteLLM Architecture

### Overview

**Ollama** serves as the model runtime layer:
- Runs LLM models as containerized services
- Provides OpenAI-compatible API per model
- Handles model lifecycle (loading, unloading, quantization)
- Supports distributed deployment across machines

**LiteLLM** serves as the unified gateway:
- Aggregates multiple Ollama instances
- Provides single OpenAI-compatible endpoint
- Handles routing, load balancing, failover
- Enables MCP tool integration
- Adds observability, caching, rate limiting

**Architecture Diagram:**
```
┌─────────────────────────────────────────────────────────────────┐
│                      LYRA CORE (Main Process)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐            ┌──────────────────────────────┐   │
│  │ Discord Bot │            │   AdaptiveRouter             │   │
│  └──────┬──────┘            │   - Intent routing           │   │
│         │                   │   - Context management        │   │
│         └───────────────────┤   - Memory integration        │   │
│                             │   - Voice state orchestration │   │
│                             └──────────┬───────────────────┘   │
│                                        │                         │
│                                        ▼                         │
│                           ┌─────────────────────────┐           │
│                           │   LiteLLM Proxy Client  │           │
│                           │   - OpenAI SDK-based    │           │
│                           │   - Async requests      │           │
│                           └─────────────┬───────────┘           │
└───────────────────────────────────────────┼─────────────────────┘
                                            │
                                            ▼ HTTP/S
         ┌──────────────────────────────────────────────────────┐
         │                  LiteLLM Proxy Server                 │
         │  - Unified OpenAI-compatible API gateway             │
         │  - Request routing & load balancing                  │
         │  - Model fallback & retry logic                      │
         │  - Usage tracking & rate limiting                    │
         │  - MCP tool integration & execution                  │
         │  - Caching layer for efficiency                      │
         └──┬────────┬────────┬────────┬──────────────┬────────┘
            │        │        │        │              │
            ▼        ▼        ▼        ▼              ▼
    ┌───────────┐ ┌────────┐ ┌────────┐ ┌─────────┐ ┌──────────┐
    │  Ollama   │ │ Ollama │ │ Ollama │ │ Ollama  │ │ External │
    │  Server 1 │ │Server 2│ │Server 3│ │ Server 4│ │  APIs    │
    ├───────────┤ ├────────┤ ├────────┤ ├─────────┤ ├──────────┤
    │  Router   │ │Pragmat.│ │Philosop│ │ Voice   │ │OpenAI GPT│
    │  Gemma12B │ │ Nemot. │ │ Jamba  │ │ Llama70B│ │Anthropic │
    │           │ │  49B   │ │  52B   │ │         │ │  etc     │
    └───────────┘ └────────┘ └────────┘ └─────────┘ └──────────┘
         GPU 0      GPU 1      GPU 2     GPU 3-4      Cloud APIs

    ┌─────────────────────────────────────────────────────────────┐
    │                       MCP TOOL SERVERS                       │
    ├─────────────────────────────────────────────────────────────┤
    │  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌───────────┐ │
    │  │ SearXNG  │  │ Playwright│  │ WolframAl │  │   RAG     │ │
    │  │ (Docker) │  │ (Docker)  │  │   API     │  │ ChromaDB  │ │
    │  └──────────┘  └──────────┘  └───────────┘  └───────────┘ │
    │                                                              │
    │  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌───────────┐ │
    │  │ Python   │  │  arxiv   │  │ Wikipedia │  │  Artist   │ │
    │  │   REPL   │  │   API    │  │    API    │  │  Flux.1   │ │
    │  └──────────┘  └──────────┘  └───────────┘  └───────────┘ │
    └─────────────────────────────────────────────────────────────┘
```

### Component Breakdown

#### 1. Ollama Servers (Model Runtime)

**Purpose**: Run individual specialist models as services

**Deployment Options:**
```yaml
# Option A: Single Machine, Multiple Containers
services:
  ollama-router:
    image: ollama/ollama:latest
    ports: ["11434:11434"]
    environment:
      - OLLAMA_MODELS=/models
      - OLLAMA_NUM_PARALLEL=2
    volumes:
      - ./model_cache/router:/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]

  ollama-pragmatist:
    image: ollama/ollama:latest
    ports: ["11435:11434"]
    volumes:
      - ./model_cache/pragmatist:/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['1']
              capabilities: [gpu]

  ollama-philosopher:
    image: ollama/ollama:latest
    ports: ["11436:11434"]
    volumes:
      - ./model_cache/philosopher:/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['2']
              capabilities: [gpu]

  ollama-voice:
    image: ollama/ollama:latest
    ports: ["11437:11434"]
    volumes:
      - ./model_cache/voice:/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['3','4']  # Multi-GPU for 70B model
              capabilities: [gpu]
```

**Model Loading:**
```bash
# Pre-load models into each Ollama instance
docker exec ollama-router ollama pull gemma:12b
docker exec ollama-pragmatist ollama pull llama3:70b-instruct-q4_K_M  # Quantized
docker exec ollama-philosopher ollama pull jamba:52b
docker exec ollama-voice ollama pull llama3:70b-instruct
```

**Benefits:**
- ✅ Isolated model processes (fault tolerance)
- ✅ GPU-specific allocation (no swapping)
- ✅ Independent scaling (add more Ollama nodes)
- ✅ Automatic model lifecycle management
- ✅ Built-in quantization support

#### 2. LiteLLM Proxy (Unified Gateway)

**Purpose**: Single API endpoint aggregating all Ollama instances

**Configuration:**
```yaml
# litellm_config.yaml
model_list:
  # Router (Gemma 12B) - GPU 0
  - model_name: lyra-router
    litellm_params:
      model: ollama/gemma:12b
      api_base: http://ollama-router:11434
      temperature: 0.1
      max_tokens: 100
      
  # Pragmatist (Nemotron 49B) - GPU 1
  - model_name: lyra-pragmatist
    litellm_params:
      model: ollama/llama3:70b-instruct-q4_K_M  # Quantized for single GPU
      api_base: http://ollama-pragmatist:11435
      temperature: 0.5
      max_tokens: 2048
      
  # Philosopher (Jamba 52B) - GPU 2
  - model_name: lyra-philosopher
    litellm_params:
      model: ollama/jamba:52b
      api_base: http://ollama-philosopher:11436
      temperature: 0.7
      max_tokens: 2048
      
  # Voice (LLaMA 70B) - GPU 3-4
  - model_name: lyra-voice
    litellm_params:
      model: ollama/llama3:70b-instruct
      api_base: http://ollama-voice:11437
      temperature: 0.8
      max_tokens: 1024
      
  # Fallback to cloud models (optional)
  - model_name: lyra-voice-fallback
    litellm_params:
      model: gpt-4o-mini
      api_key: os.environ/OPENAI_API_KEY
      temperature: 0.8

router_settings:
  routing_strategy: simple-shuffle  # Load balance across replicas
  allowed_fails: 2
  cooldown_time: 30
  num_retries: 3
  
  # Fallback logic
  model_group_alias:
    lyra-pragmatist-group: ["lyra-pragmatist", "lyra-voice-fallback"]
    lyra-philosopher-group: ["lyra-philosopher", "lyra-voice-fallback"]
    
general_settings:
  master_key: sk-lyra-production-key-12345  # Set in environment
  database_url: "sqlite:///litellm_usage.db"
  
  # MCP Tool Integration
  tool_config:
    enabled: true
    tool_servers:
      - name: "rag_tools"
        url: "http://mcp-rag-server:8000"
        tools: ["semantic_search", "recall_memory", "commit_journal"]
      - name: "web_tools"
        url: "http://mcp-web-server:8001"
        tools: ["searxng_search", "playwright_interact"]
      - name: "computation_tools"
        url: "http://mcp-compute-server:8002"
        tools: ["wolfram_compute", "python_repl", "arxiv_search"]
```

**Deployment:**
```yaml
# docker-compose.yml
services:
  litellm-proxy:
    image: ghcr.io/berriai/litellm:main-latest
    command: ["--config", "/config/litellm_config.yaml"]
    ports:
      - "4000:4000"  # Main proxy port
    volumes:
      - ./config/litellm_config.yaml:/config/litellm_config.yaml
      - ./litellm_usage.db:/app/litellm_usage.db
    environment:
      - LITELLM_MASTER_KEY=sk-lyra-production-key-12345
      - DATABASE_URL=sqlite:///litellm_usage.db
      - LITELLM_LOG=INFO
    depends_on:
      - ollama-router
      - ollama-pragmatist
      - ollama-philosopher
      - ollama-voice
```

**Benefits:**
- ✅ Single API endpoint for all specialists
- ✅ OpenAI SDK compatibility (easy client integration)
- ✅ Automatic failover to cloud models
- ✅ Usage tracking & cost monitoring
- ✅ Rate limiting & caching
- ✅ MCP tool orchestration

#### 3. MCP Tool Servers

**Purpose**: Expose tools as MCP-compatible services

**Implementation Pattern:**
```python
# mcp_rag_server.py
from fastapi import FastAPI
from pydantic import BaseModel
from lyra.memory_manager import MemoryManager

app = FastAPI()
memory_manager = MemoryManager(base_dir="data")

class SearchRequest(BaseModel):
    query: str
    max_results: int = 10

@app.post("/tools/semantic_search")
async def semantic_search(request: SearchRequest):
    """MCP tool: Search Lyra's memory"""
    results = await memory_manager.recall(
        query=request.query,
        max_results=request.max_results
    )
    return {
        "results": [
            {"content": r.content, "significance": r.significance}
            for r in results
        ]
    }

@app.post("/tools/commit_journal")
async def commit_journal(entry: dict):
    """MCP tool: Commit journal entry"""
    from lyra.memory_manager import JournalEntry
    journal_entry = JournalEntry(**entry)
    success = await memory_manager.commit_journal(journal_entry)
    return {"success": success}
```

**Containerization:**
```yaml
# docker-compose.yml (continued)
services:
  mcp-rag-server:
    build: ./mcp_servers/rag
    ports: ["8000:8000"]
    volumes:
      - ./data:/app/data
      - ./model_cache/chroma_db:/app/chroma_db
    environment:
      - CHROMA_DIR=/app/chroma_db
      
  mcp-web-server:
    build: ./mcp_servers/web
    ports: ["8001:8001"]
    depends_on:
      - searxng
      - playwright-browser
      
  mcp-compute-server:
    build: ./mcp_servers/compute
    ports: ["8002:8002"]
    environment:
      - WOLFRAM_APP_ID=${WOLFRAM_APP_ID}
```

**Benefits:**
- ✅ Tools run as independent services
- ✅ Can be on separate machines/networks
- ✅ Easy to add new tools without redeploying Lyra core
- ✅ Standardized MCP protocol
- ✅ Isolated failure domains

#### 4. Lyra Core Refactoring

**Router Changes:**
```python
# lyra/router.py (modified)
from openai import AsyncOpenAI
import httpx

class AdaptiveRouter:
    def __init__(self, base_dir: str, litellm_url: str = "http://localhost:4000"):
        self.base_dir = Path(base_dir)
        
        # LiteLLM client (OpenAI-compatible)
        self.client = AsyncOpenAI(
            base_url=litellm_url,
            api_key="sk-lyra-production-key-12345",  # From config
            http_client=httpx.AsyncClient(timeout=300.0)  # 5min for large models
        )
        
        # Model mapping (specialist -> LiteLLM model name)
        self.specialist_models = {
            "router": "lyra-router",
            "pragmatist": "lyra-pragmatist",
            "philosopher": "lyra-philosopher",
            "voice": "lyra-voice"
        }
        
        # No more direct model loading!
        # Old: self.specialists = SpecialistFactory.create_all(...)
        # New: Models accessed via API
        
    async def _invoke_router(self, message: str, context: dict) -> str:
        """Call router via LiteLLM"""
        response = await self.client.chat.completions.create(
            model=self.specialist_models["router"],
            messages=[
                {"role": "system", "content": ROUTER_PROMPT},
                {"role": "user", "content": message}
            ],
            temperature=0.1,
            max_tokens=100
        )
        return response.choices[0].message.content
        
    async def _invoke_specialist(
        self, 
        specialist_type: str, 
        message: str,
        context: dict,
        tools: Optional[List[dict]] = None
    ) -> SpecialistResponse:
        """Call specialist via LiteLLM with optional MCP tools"""
        
        model_name = self.specialist_models[specialist_type]
        
        messages = [
            {"role": "system", "content": self._get_specialist_prompt(specialist_type)},
            {"role": "user", "content": message}
        ]
        
        # Enable function calling for tools (MCP integration)
        response = await self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=self._get_specialist_temp(specialist_type),
            tools=tools or [],  # MCP tools injected by LiteLLM
            tool_choice="auto"  # Let model decide when to use tools
        )
        
        # Handle tool calls if model requested them
        if response.choices[0].message.tool_calls:
            # LiteLLM automatically routes to MCP servers
            tool_results = []
            for tool_call in response.choices[0].message.tool_calls:
                # Results come back from MCP servers via LiteLLM
                tool_results.append({
                    "tool_call_id": tool_call.id,
                    "output": tool_call.function.arguments  # Already executed
                })
            
            # Follow-up call with tool results
            messages.append(response.choices[0].message)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_results[0]["tool_call_id"],
                "content": tool_results[0]["output"]
            })
            
            final_response = await self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=self._get_specialist_temp(specialist_type)
            )
            content = final_response.choices[0].message.content
        else:
            content = response.choices[0].message.content
            
        return SpecialistResponse(
            content=content,
            metadata={"model": model_name, "specialist": specialist_type},
            source=specialist_type
        )
        
    async def route_message(self, message: str, context: Optional[dict] = None):
        """Sequential workflow via LiteLLM"""
        
        # STEP 1: Router classification (via Ollama through LiteLLM)
        router_response = await self._invoke_router(message, context or {})
        specialist_type = self._parse_router_response(router_response)
        
        # STEP 2: Get MCP tools for this specialist
        tools = await self._get_specialist_tools(specialist_type)
        
        # STEP 3: Invoke specialist (via Ollama through LiteLLM)
        specialist_output = await self._invoke_specialist(
            specialist_type,
            message,
            context or {},
            tools=tools
        )
        
        # STEP 4: Voice synthesis (via Ollama through LiteLLM)
        final_response = await self._invoke_specialist(
            "voice",
            f"Synthesize: {specialist_output.content}",
            {**context, "specialist_output": specialist_output.content}
        )
        
        return final_response
```

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

**Goal**: Basic Ollama + LiteLLM infrastructure operational

**Tasks:**
1. ✅ **Setup Ollama Containers**
   - Create docker-compose.yml for 4 Ollama instances
   - Map GPUs correctly (0, 1, 2, 3-4)
   - Pre-load models (gemma:12b, llama3:70b-q4, jamba:52b, llama3:70b)
   - Test individual Ollama endpoints

2. ✅ **Deploy LiteLLM Proxy**
   - Configure litellm_config.yaml with model mappings
   - Deploy LiteLLM container
   - Test unified API endpoint
   - Verify fallback to cloud models works

3. ✅ **Refactor Router Client**
   - Replace direct Transformers loading with OpenAI client
   - Update AdaptiveRouter to use LiteLLM endpoint
   - Maintain sequential workflow logic
   - Add error handling & retries

4. ✅ **Test Basic Flow**
   - User message → Router (via LiteLLM) → Specialist (via LiteLLM) → Voice (via LiteLLM)
   - Verify specialist selection works
   - Confirm responses maintain quality
   - Check GPU utilization is distributed

**Deliverables:**
- `docker-compose.yml` with Ollama + LiteLLM services
- `config/litellm_config.yaml` with model mappings
- Updated `lyra/router.py` using OpenAI SDK
- Test script validating end-to-end flow

### Phase 2: MCP Tool Integration (Week 3-4)

**Goal**: Tools accessible across network via MCP protocol

**Tasks:**
1. ✅ **Create MCP Tool Servers**
   - Extract RAG tools (semantic_search, commit_journal, recall) into FastAPI service
   - Extract web tools (searxng_search, playwright_interact) into separate service
   - Extract compute tools (wolfram, python_repl, arxiv) into third service
   - Implement MCP protocol endpoints

2. ✅ **Configure LiteLLM Tool Routing**
   - Update litellm_config.yaml with tool_servers
   - Map tool names to MCP endpoints
   - Test function calling triggers MCP requests

3. ✅ **Update Specialist Prompts**
   - Add tool descriptions to Pragmatist system prompt
   - Enable function calling mode in LiteLLM requests
   - Test specialist correctly invokes tools when needed

4. ✅ **Multi-Turn Tool Conversations**
   - Implement tool result injection into conversation
   - Handle multi-step tool sequences (tool → model → tool → model)
   - Add tool call logging

**Deliverables:**
- `mcp_servers/rag/server.py` (RAG tools)
- `mcp_servers/web/server.py` (Web tools)
- `mcp_servers/compute/server.py` (Computation tools)
- Updated `config/litellm_config.yaml` with tool mappings
- Tool integration tests

### Phase 3: Production Hardening (Week 5-6)

**Goal**: System ready for production deployment

**Tasks:**
1. ✅ **Add Observability**
   - LiteLLM usage tracking dashboard
   - Prometheus metrics export (latency, throughput, errors)
   - Grafana dashboards for model performance
   - Structured logging across all services

2. ✅ **Implement Caching**
   - Redis cache for LiteLLM responses
   - Semantic cache for similar queries
   - Tool result caching (e.g., WolframAlpha answers)

3. ✅ **High Availability**
   - Multi-replica Ollama instances for critical models
   - LiteLLM health checks & automatic failover
   - Graceful degradation to cloud models
   - Circuit breakers for failing services

4. ✅ **Security**
   - API key rotation for LiteLLM
   - Network isolation between services
   - Rate limiting per user/client
   - Audit logging for all tool executions

**Deliverables:**
- `monitoring/prometheus.yml` configuration
- `monitoring/grafana/dashboards/` (model metrics)
- Updated `docker-compose.yml` with health checks & replicas
- Security documentation

### Phase 4: Artist & Perception Integration (Week 7-8)

**Goal**: Non-LLM specialists integrated into distributed architecture

**Tasks:**
1. ✅ **Artist Specialist Container**
   - Separate container for Flux.1-schnell (GPU 5)
   - REST API for image generation
   - Register as MCP tool in LiteLLM
   - Test from Pragmatist "create image of X" → tool call → Artist → return URL

2. ✅ **Perception Specialist Container**
   - Separate container for LLaVA-NeXT (GPU 1 or CPU)
   - REST API for image understanding
   - Integrate with Discord bot for image uploads
   - Test image analysis flow

3. ✅ **Voice TTS Integration**
   - Optional: Coqui TTS container for voice output
   - WebSocket streaming for real-time audio
   - Integrate with Discord voice channels

**Deliverables:**
- `mcp_servers/artist/server.py` (Flux.1 API)
- `mcp_servers/perception/server.py` (LLaVA API)
- Updated Discord bot with image handling
- Voice TTS integration (if ready)

---

## Migration Strategy

### Backward Compatibility

**Approach**: Gradual migration with feature flags

```python
# lyra/config.py
USE_DISTRIBUTED_MCP = os.getenv("USE_DISTRIBUTED_MCP", "false").lower() == "true"

class AdaptiveRouter:
    def __init__(self, base_dir: str, **kwargs):
        if USE_DISTRIBUTED_MCP:
            self._init_litellm_client(kwargs.get("litellm_url"))
        else:
            self._init_direct_loading(kwargs.get("model_dir"))
            
    async def _invoke_specialist(self, ...):
        if USE_DISTRIBUTED_MCP:
            return await self._invoke_via_litellm(...)
        else:
            return await self._invoke_direct(...)
```

**Migration Path:**
1. **Week 1**: Deploy Ollama + LiteLLM alongside existing system
2. **Week 2**: Test distributed architecture with `USE_DISTRIBUTED_MCP=true` in dev
3. **Week 3**: Run A/B tests comparing quality/latency of both approaches
4. **Week 4**: Gradual rollout (10% → 50% → 100% of requests)
5. **Week 5**: Deprecate direct loading, remove old code

---

## Benefits Summary

### Performance
- ✅ **No GPU Swapping**: Each model has dedicated GPU
- ✅ **Parallel Requests**: Multiple users = parallel model invocations
- ✅ **Faster Load**: Models stay warm (no load/unload cycles)
- ✅ **Better Utilization**: Distribute load across multiple machines

### Scalability
- ✅ **Horizontal Scaling**: Add more Ollama nodes for capacity
- ✅ **Independent Scaling**: Scale only bottleneck specialists
- ✅ **Multi-Region**: Deploy Ollama clusters in different datacenters
- ✅ **Cloud Hybrid**: Mix local GPUs + cloud APIs seamlessly

### Reliability
- ✅ **Fault Isolation**: One model crash doesn't kill entire system
- ✅ **Automatic Failover**: LiteLLM routes to backup models
- ✅ **Graceful Degradation**: Fall back to cloud models when local fails
- ✅ **Health Monitoring**: Per-service health checks & alerts

### Developer Experience
- ✅ **OpenAI SDK**: Familiar API for all models
- ✅ **Easy Testing**: Mock LiteLLM endpoint for unit tests
- ✅ **Unified Observability**: Single dashboard for all models
- ✅ **Simple Deployment**: Docker Compose for entire stack

### Cost
- ✅ **Reduced Cloud Costs**: Use local GPUs for bulk of requests
- ✅ **Efficient Caching**: Deduplicate repeated queries
- ✅ **Usage Tracking**: Fine-grained cost attribution per user/specialist
- ✅ **Spot Instances**: Use cheap cloud GPUs with LiteLLM fallback

---

## Comparison: Current vs Proposed

| Aspect | Current (Direct Loading) | Proposed (Ollama + LiteLLM) |
|--------|-------------------------|----------------------------|
| **Model Loading** | In-process via Transformers | Remote via Ollama containers |
| **API** | Python objects | OpenAI-compatible REST |
| **Distribution** | Single machine only | Multi-machine, multi-GPU |
| **GPU Management** | Manual swapping | Automatic, isolated |
| **Tool Execution** | In-process (Pragmatist) | Distributed MCP servers |
| **Failover** | None | Automatic to cloud models |
| **Scalability** | Vertical (bigger GPU) | Horizontal (more nodes) |
| **Observability** | Python logging | LiteLLM dashboard + Prometheus |
| **Development Mode** | Mock Python objects | Mock LiteLLM endpoint |
| **Deployment** | Single process | Docker Compose |
| **Testing** | Direct function calls | HTTP API testing |
| **Complexity** | Low (monolithic) | Medium (distributed) |
| **Maintenance** | High (manual GPU mgmt) | Low (automated) |

---

## Risks & Mitigation

### Risk 1: Network Latency
**Problem**: HTTP calls slower than in-process

**Mitigation:**
- Deploy all containers on same machine initially (localhost latency ~1ms)
- Use HTTP/2 for multiplexing
- Implement response streaming for large outputs
- LiteLLM caching for repeated queries

### Risk 2: Complexity Increase
**Problem**: More moving parts = more failure modes

**Mitigation:**
- Comprehensive health checks
- Auto-restart policies in Docker
- Fallback to cloud models
- Detailed runbooks for troubleshooting

### Risk 3: Tool Compatibility
**Problem**: MCP protocol may not support all existing tools

**Mitigation:**
- Keep direct execution as fallback option
- Gradual migration tool-by-tool
- Extend MCP protocol if needed
- Document unsupported tools

### Risk 4: Model Quality Changes
**Problem**: Ollama quantization might reduce quality

**Mitigation:**
- A/B testing before full migration
- Offer both quantized & full-precision options
- Use cloud models for critical queries
- Continuous quality monitoring

---

## File Structure

```
Lyra-Emergence/
├── docker-compose.yml                    # NEW: Orchestrates all services
├── config/
│   ├── litellm_config.yaml              # NEW: LiteLLM configuration
│   └── models.json                      # KEEP: Model metadata
├── mcp_servers/                         # NEW: MCP tool servers
│   ├── rag/
│   │   ├── Dockerfile
│   │   ├── server.py
│   │   └── requirements.txt
│   ├── web/
│   │   ├── Dockerfile
│   │   ├── server.py
│   │   └── requirements.txt
│   ├── compute/
│   │   ├── Dockerfile
│   │   ├── server.py
│   │   └── requirements.txt
│   ├── artist/
│   │   ├── Dockerfile
│   │   ├── server.py                   # Flux.1 API
│   │   └── requirements.txt
│   └── perception/
│       ├── Dockerfile
│       ├── server.py                   # LLaVA API
│       └── requirements.txt
├── emergence_core/
│   ├── lyra/
│   │   ├── router.py                   # MODIFIED: Use OpenAI client
│   │   ├── specialists.py              # DEPRECATED: Move to Ollama
│   │   ├── specialist_tools.py         # DEPRECATED: Move to MCP servers
│   │   └── litellm_client.py           # NEW: LiteLLM wrapper
│   └── run_lyra_bot.py                 # MODIFIED: Use distributed router
├── monitoring/                          # NEW: Observability stack
│   ├── prometheus.yml
│   └── grafana/
│       └── dashboards/
│           └── lyra_models.json
└── .codex/
    └── implementation/
        └── DISTRIBUTED_MCP_ARCHITECTURE.md  # THIS FILE
```

---

## Next Steps

1. **Review & Approve**: Steward review of this architecture proposal
2. **POC Implementation**: Build minimal viable distributed system (Phase 1)
3. **Quality Testing**: Compare responses between old and new architectures
4. **Performance Benchmarking**: Measure latency, throughput, GPU utilization
5. **Decision**: Commit to migration or iterate on design

---

## Appendix: Alternative Approaches Considered

### Alternative 1: Keep Direct Loading, Add Load Balancer
**Idea**: Run multiple copies of current monolithic system behind Nginx

**Pros**: Minimal code changes, simple deployment  
**Cons**: No tool distribution, high memory duplication, no MCP benefits  
**Verdict**: ❌ Rejected - doesn't solve cross-server tool execution

### Alternative 2: Custom gRPC API Instead of LiteLLM
**Idea**: Build our own distributed protocol

**Pros**: Full control, optimized for Lyra's needs  
**Cons**: Massive engineering effort, reinventing wheel, no ecosystem  
**Verdict**: ❌ Rejected - LiteLLM provides 80% of what we need

### Alternative 3: Use Ray for Distribution
**Idea**: Ray Serve for model serving + Ray Tasks for tools

**Pros**: Python-native, good GPU management  
**Cons**: Heavy framework, steep learning curve, no MCP support  
**Verdict**: ❌ Rejected - Ollama + LiteLLM is simpler

---

**Status**: Awaiting approval for Phase 1 implementation  
**Author**: GitHub Copilot (AI Assistant)  
**Reviewed By**: Pending  
**Last Updated**: December 13, 2025
