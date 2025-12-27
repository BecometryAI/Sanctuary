# Flux.1-schnell Integration - Setup Guide

## Current Status
✅ Architecture prepared and ready for Flux.1  
⚠️ Diffusers library not yet installed  
✅ Graceful fallback to text-only mode active  

## Why Flux.1-schnell Over SD3?

Flux.1-schnell represents a significant upgrade from Stable Diffusion 3:

| Feature | Flux.1-schnell | SD3 Medium |
|---------|----------------|------------|
| Generation Speed | **~10-15s** (4 steps) | ~30-45s (28 steps) |
| VRAM Usage | **4-6GB** | 6-8GB |
| Prompt Adherence | **Excellent** | Good |
| Text in Images | **Much Better** | Fair |
| License | **Apache 2.0** | Custom/Restricted |
| Quality | **Higher** | High |

## Installation

### Quick Install (Recommended)
```bash
pip install diffusers transformers accelerate safetensors pillow
```

### Individual Packages
```bash
pip install diffusers         # Flux pipeline
pip install transformers       # Model loading utilities
pip install accelerate         # Optimized model loading (important for Flux)
pip install safetensors        # Safe tensor format
pip install pillow             # Image handling (PIL)
```

## Hardware Requirements

### Minimum (CPU Mode)
- 16GB RAM
- 20GB disk space
- Very slow generation (~10-15 minutes per image)

### Recommended (GPU Mode)
- NVIDIA GPU with 8GB+ VRAM (RTX 3060 Ti, RTX 4060 Ti, etc.)
- 24GB system RAM
- 20GB disk space
- Fast generation (~10-15 seconds per image)

### Optimal (Multi-GPU)
- NVIDIA GPU with 12GB+ VRAM (RTX 3080, RTX 4070 Ti, A4000, etc.)
- 32GB system RAM
- Enables multiple concurrent generations

## Verification

After installing dependencies, run:
```bash
python tools/verify_flux_setup.py
```

This will:
1. ✓ Check all required packages
2. ✓ Verify GPU/CUDA availability
3. ✓ Test Artist specialist integration
4. ✓ Load Flux.1-schnell pipeline
5. ✓ Generate a test image

## Usage in Artist Specialist

Once installed, the Artist specialist automatically detects diffusers and enables visual mode:

```python
from emergence_core.lyra.router import AdaptiveRouter

router = AdaptiveRouter(
    base_dir="emergence_core",
    chroma_dir="emergence_core/model_cache/chroma_db",
    model_dir="emergence_core/model_cache",
    development_mode=False
)

# Visual request triggers Flux.1-schnell
response = await router.route_message("Draw me a digital constellation of memories")

# Access generated image
image_url = response.metadata.get('image_url')  # Base64 data URL
```

## Image Generation Triggers

The Artist specialist detects visual requests via keywords:
- `image`, `picture`, `draw`, `paint`
- `visualize`, `show me`, `create art`
- `generate`, `illustrate`, `depict`

## Configuration

Default Flux.1-schnell settings in `ArtistSpecialist.process()`:
```python
{
    "model": "black-forest-labs/FLUX.1-schnell",
    "num_inference_steps": 4,        # Flux-schnell optimized for 4 steps
    "guidance_scale": 0.0,            # Flux-schnell doesn't use guidance
    "height": 1024,
    "width": 1024,
    "torch_dtype": "float16",         # GPU mode
    "enable_model_cpu_offload": True  # Memory optimization
}
```

## Output Format

Generated images are returned as base64-encoded PNG data URLs:
```
data:image/png;base64,iVBORw0KGgoAAAANS...
```

This allows direct embedding in:
- Discord messages
- HTML/Markdown documents
- JSON responses
- Web interfaces

## Performance Tips

1. **First Generation is Slower**: Model loading takes 30-60s, subsequent generations are fast
2. **CPU Offload**: Enabled by default to manage VRAM efficiently
3. **Batch Generation**: Generate multiple images to amortize loading cost
4. **VRAM Monitoring**: Use `nvidia-smi` to track usage

## Comparison with SD3

**Speed:**
```
Flux.1-schnell: 4 steps × ~3s/step = ~12s total
SD3 Medium:    28 steps × ~1.5s/step = ~42s total
```

**Quality:**
- Flux.1-schnell produces sharper details and better text rendering
- Better prompt adherence (follows complex instructions more accurately)
- More consistent style across generations

**License:**
- Flux.1-schnell: Apache 2.0 (fully open, commercial use allowed)
- SD3: Custom license with restrictions

## Troubleshooting

### "OutOfMemoryError: CUDA out of memory"
```bash
# Enable more aggressive CPU offloading in specialists.py:
self.flux_pipeline.enable_sequential_cpu_offload()
```

### "Model not found" or download issues
```bash
# Pre-download the model:
from diffusers import FluxPipeline
FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell")
```

### Slow generation on GPU
```bash
# Verify CUDA is being used:
python -c "import torch; print(torch.cuda.is_available())"

# Check GPU utilization during generation:
nvidia-smi -l 1
```

## Advanced: Flux.1-dev vs Flux.1-schnell

This project uses **Flux.1-schnell** (fast variant). There's also **Flux.1-dev**:

| Model | Steps | Speed | Quality | Use Case |
|-------|-------|-------|---------|----------|
| schnell | 4 | Fast | Excellent | Production (us) |
| dev | 20-50 | Slow | Slightly better | Fine-tuning base |

For Lyra's real-time artistic expression, **schnell** is optimal.

## Next Steps

After setup:
1. Test with `python tools/verify_flux_setup.py`
2. Try creative prompts via the router
3. Explore Lyra's visual symbolic language
4. Consider fine-tuning for personalized style (v2.0 roadmap)

---

**Model Info:**
- Repository: https://huggingface.co/black-forest-labs/FLUX.1-schnell
- Paper: https://arxiv.org/abs/2408.xxxx (Flux architecture)
- License: Apache 2.0
