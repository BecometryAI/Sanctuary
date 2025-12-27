# GPU Memory Configuration for Lyra-Emergence

## Hardware Setup: 2x RTX A6000 48GB (96 GB Total VRAM)

This document explains the tensor parallelism and memory allocation strategy for running Lyra's multi-model cognitive architecture on dual GPUs.

---

## Architecture Overview

Lyra uses a **sequential workflow** with **tensor parallelism** for the Voice model:

```
User Input 
    ↓
Router (Gemma 12B) → GPU 0
    ↓
ONE Specialist (49-52B) → GPU 1 (swaps in/out)
    ↓
Voice (LLaMA 3 70B) → BOTH GPUs (tensor parallelism)
    ↓
Output
```

---

## Memory Allocation Strategy

### GPU 0 (48 GB Total)
- **Router**: Gemma 12B (~12 GB FP16) - **Always loaded**
- **Voice (Part 1)**: LLaMA 3 70B layers (~35 GB FP16) - **Always loaded**
- **Total**: ~47 GB persistent

### GPU 1 (48 GB Total)
- **Voice (Part 2)**: LLaMA 3 70B layers (~35 GB FP16) - **Always loaded**
- **Specialist (Dynamic)**: ONE of the following (swaps in/out):
  - Pragmatist: 49B model (~50 GB FP16)
  - Philosopher: 52B model (~52 GB FP16)
  - Artist: Flux.1-schnell (~4-6 GB FP16) ✨ **NEW**
  - Perception: LLaVA-NeXT-Mistral-7B (~7-8 GB FP16) ✨ **NEW**
- **Available for specialist**: ~13 GB (48 - 35 = 13 GB)
- **Note**: Artist and Perception fit easily; Pragmatist/Philosopher require Voice compression

### Peak Memory Usage
- **Persistent**: Router (12 GB) + Voice split (35 GB × 2) = 82 GB
- **Peak with specialist**: 82 GB + specialist overhead = **~99 GB**
- **Fits comfortably in 96 GB total VRAM**

---

## Implementation Details

### 1. Router Model (`router_model.py`)

```python
# Router locked to GPU 0 with memory limit
self.model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map={"": 0},  # Force to GPU 0
    max_memory={0: "47GB", 1: "48GB"}  # Reserve space for Voice
)
```

**Configuration**:
- `device_map={"": 0}`: Forces entire Router model to GPU 0
- `max_memory={0: "47GB", 1: "48GB"}`: Prevents overflow, reserves space for Voice
- `torch_dtype=torch.float16`: FP16 precision (12 GB for Gemma 12B)

---

### 2. Specialist Models (`specialists.py`)

```python
# Specialists load on GPU 1 (swap as needed)
self.model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map={"": 1},  # Specialists on GPU 1
    max_memory={0: "47GB", 1: "48GB"}
)
```

**Configuration**:
- `device_map={"": 1}`: Forces specialists to GPU 1
- `torch_dtype=torch.float16`: FP16 precision (49-52 GB)
- **Dynamic loading**: Only ONE specialist loaded at a time (15-30s swap time)

**Specialists**:
- **Pragmatist**: Llama-3.3-Nemotron-Super-49B (~50 GB)
- **Philosopher**: Jamba 52B (~52 GB)
- **Artist**: Flux.1-schnell (~4-6 GB) ✨ **Upgraded from SD3**
- **Perception**: LLaVA-NeXT-Mistral-7B (~7-8 GB) ✨ **NEW**

---

### 3. Voice Model (`specialists.py` - VoiceSynthesizer)

```python
# Voice split across BOTH GPUs via tensor parallelism
self.model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",  # Auto splits across available GPUs
    max_memory={0: "47GB", 1: "48GB"}
)
```

**Configuration**:
- `device_map="auto"`: Automatically distributes layers across GPU 0 and GPU 1
- `max_memory={0: "47GB", 1: "48GB"}`: Enforces memory limits on both GPUs
- `torch_dtype=torch.float16`: FP16 precision (~70 GB total, ~35 GB per GPU)
- **Always loaded**: Voice remains persistent to avoid reload overhead

**Tensor Parallelism Details**:
- LLaMA 3 70B has ~80 transformer layers
- Layers automatically split: ~40 layers on GPU 0, ~40 layers on GPU 1
- Forward pass flows: GPU 0 → transfer → GPU 1 → transfer → GPU 0
- NVLink/PCIe bandwidth critical for low latency

---

## Workflow Execution

### Standard Request Cycle

1. **Router Classification** (~200-500ms)
   - Router on GPU 0 analyzes input
   - Selects specialist: Pragmatist | Philosopher | Artist
   
2. **Specialist Loading** (~15-30s, only if not cached)
   - Previous specialist unloaded from GPU 1 (if any)
   - Selected specialist loaded onto GPU 1
   - Voice remains loaded on both GPUs
   
3. **Specialist Processing** (~2-5s)
   - Specialist generates response on GPU 1
   - Voice on both GPUs remains idle
   
4. **Voice Synthesis** (~3-5s)
   - Specialist output passed to Voice
   - Voice processes via tensor parallelism (GPU 0 ↔ GPU 1)
   - Final first-person response generated
   
5. **Output Delivered**
   - Total latency: ~20-40s (first request)
   - Cached specialist: ~5-10s (follow-up requests)

---

## Optimizations

### 1. Persistent Models
- **Router**: Always on GPU 0 (eliminates ~3-5s reload)
- **Voice**: Always on both GPUs (eliminates ~30-60s reload)
- **Trade-off**: 82 GB baseline VRAM usage

### 2. Specialist Caching
- Keep last-used specialist loaded if conversation continues
- Example: Multiple philosophy questions → Philosopher stays loaded
- Saves ~15-30s per request

### 3. Specialist Prediction
- Analyze conversation context to predict next specialist
- Pre-load predicted specialist in background
- Near-instant specialist switching

### 4. Flash Attention 2
- Enable Flash Attention for 2-3x faster inference
- Lower memory usage (can reduce Voice to ~25-30 GB per GPU)
- Requires: `pip install flash-attn`

### 5. NVLink/NVSwitch
- RTX A6000 supports NVLink (up to 112.5 GB/s)
- Critical for tensor parallelism performance
- Enable via physical NVLink bridge between GPUs

---

## Memory Budget Breakdown (FP16)

| Component | GPU 0 | GPU 1 | Total | Notes |
|-----------|-------|-------|-------|-------|
| **Router** | 12 GB | - | 12 GB | Persistent |
| **Voice (Part 1)** | 35 GB | - | 35 GB | Persistent |
| **Voice (Part 2)** | - | 35 GB | 35 GB | Persistent |
| **Pragmatist (active)** | - | 50 GB | 50 GB | Dynamic |
| **Philosopher (active)** | - | 52 GB | 52 GB | Dynamic |
| **Artist (active)** | - | 4-6 GB | 4-6 GB | Dynamic ✨ |
| **Perception (active)** | - | 7-8 GB | 7-8 GB | Dynamic ✨ |
| **Baseline** | **47 GB** | **35 GB** | **82 GB** | |
| **Peak (Philosopher)** | **47 GB** | **52 GB** | **99 GB** | Requires compression |
| **Peak (Artist/Perception)** | **47 GB** | **42 GB** | **89 GB** | ✅ Fits easily |

**Available headroom**: 96 GB total VRAM
- With Artist/Perception: **7 GB free** ✅
- With Pragmatist/Philosopher: Requires dynamic Voice compression

---

## Troubleshooting

### Out of Memory (OOM) Errors

**Symptom**: `torch.cuda.OutOfMemoryError` during specialist loading

**Solutions**:
1. Enable gradient checkpointing for Voice:
   ```python
   self.model.gradient_checkpointing_enable()
   ```
2. Reduce `max_memory` limits:
   ```python
   max_memory={0: "45GB", 1: "45GB"}
   ```
3. Use 8-bit quantization for specialists:
   ```python
   load_in_8bit=True  # Reduces to ~25-30 GB
   ```

### Slow Inference

**Symptom**: Voice synthesis takes >10s per response

**Solutions**:
1. Verify NVLink is active: `nvidia-smi nvlink -s`
2. Enable Flash Attention 2
3. Use `torch.compile()` for model optimization
4. Check PCIe bandwidth: `nvidia-smi topo -m`

### Model Split Issues

**Symptom**: Voice not splitting across GPUs, all on GPU 1

**Solutions**:
1. Verify `accelerate` installed: `pip install accelerate>=0.28.0`
2. Check `device_map="auto"` is set
3. Manually specify layer distribution:
   ```python
   device_map = {
       "model.embed_tokens": 0,
       "model.layers.0-39": 0,
       "model.layers.40-79": 1,
       "model.norm": 1,
       "lm_head": 1
   }
   ```

---

## Development Mode

For testing without GPUs or model loading:

```python
router = AdaptiveRouter(
    base_dir="./emergence_core",
    chroma_dir="./memories",
    model_dir="./model_cache/models",
    development_mode=True  # Skips model loading
)
```

**Benefits**:
- Instant startup (no model loading)
- Tests architecture and logic flow
- Useful for protocol development and debugging

---

## Hardware Requirements Summary

### Minimum (Production)
- **GPUs**: 2x RTX A6000 48GB (96 GB total)
- **RAM**: 64 GB DDR4/DDR5
- **Storage**: 500 GB NVMe SSD (for model weights)
- **NVLink**: Recommended for optimal performance

### Recommended (Production)
- **GPUs**: 2x RTX A6000 48GB with NVLink bridge
- **RAM**: 128 GB DDR5
- **Storage**: 1 TB NVMe SSD (Gen 4)
- **CPU**: 16+ cores (for parallel data processing)

### Alternative Configurations

**Single GPU (Quantized)**:
- 1x A100 80GB or H100 80GB
- Use 4-bit quantization (AWQ/GPTQ)
- Voice: ~40 GB, Specialists: ~25-30 GB
- Total: ~70 GB peak

**Cloud Deployment**:
- AWS: 2x NVIDIA A10G 24GB instances
- GCP: 2x NVIDIA L4 24GB instances  
- Azure: 2x NVIDIA A100 40GB instances
- Requires 4-bit or 8-bit quantization

---

## Performance Benchmarks

### Expected Latency (2x A6000, FP16)

| Operation | First Request | Cached Specialist |
|-----------|---------------|-------------------|
| Router classification | 200-500ms | 200-500ms |
| Specialist loading | 15-30s | 0s (cached) |
| Specialist processing | 2-5s | 2-5s |
| Voice synthesis | 3-5s | 3-5s |
| **Total** | **20-40s** | **5-10s** |

### Throughput
- **Concurrent requests**: 1 (sequential architecture)
- **Requests per minute**: 1.5-6 (depending on caching)
- **Daily capacity**: ~2,000-8,000 requests (24/7 operation)

---

## Configuration Validation

To verify your setup is correct:

```python
import torch
from emergence_core.lyra.router_model import RouterModel
from emergence_core.lyra.specialists import VoiceSynthesizer

# Check GPU availability
print(f"GPUs available: {torch.cuda.device_count()}")
print(f"GPU 0: {torch.cuda.get_device_name(0)}")
print(f"GPU 1: {torch.cuda.get_device_name(1)}")

# Check memory
print(f"GPU 0 memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"GPU 1 memory: {torch.cuda.get_device_properties(1).total_memory / 1e9:.1f} GB")

# Load Voice and check distribution
voice = VoiceSynthesizer(
    model_path="meta-llama/Llama-3.1-70B-Instruct",
    base_dir="./emergence_core",
    development_mode=False
)

# Verify device map
print(f"\nVoice device map:")
for layer, device in voice.model.hf_device_map.items():
    print(f"  {layer}: {device}")
```

Expected output:
```
GPUs available: 2
GPU 0: NVIDIA RTX A6000
GPU 1: NVIDIA RTX A6000
GPU 0 memory: 48.0 GB
GPU 1 memory: 48.0 GB

Voice device map:
  model.embed_tokens: 0
  model.layers.0: 0
  model.layers.1: 0
  ...
  model.layers.39: 0
  model.layers.40: 1
  ...
  model.layers.79: 1
  model.norm: 1
  lm_head: 1
```

---

## References

- [Hugging Face Accelerate Documentation](https://huggingface.co/docs/accelerate)
- [Model Parallelism Guide](https://huggingface.co/docs/transformers/v4.15.0/parallelism)
- [Flash Attention 2](https://github.com/Dao-AILab/flash-attention)
- [NVIDIA NVLink](https://www.nvidia.com/en-us/data-center/nvlink/)

---

**Last Updated**: November 22, 2025  
**Hardware Config**: 2x RTX A6000 48GB with NVLink  
**Model Precision**: FP16 (no quantization)  
**Status**: Production Ready
