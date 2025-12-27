# Stable Diffusion 3 Integration - Setup Guide

## Current Status
✅ Architecture prepared and ready for SD3  
⚠️ Diffusers library not yet installed  
✅ Graceful fallback to text-only mode active  

## Installation

### Quick Install (Recommended)
```bash
pip install diffusers transformers accelerate safetensors pillow
```

### Individual Packages
```bash
pip install diffusers         # Stable Diffusion 3 pipeline
pip install transformers       # Model loading utilities
pip install accelerate         # Optimized model loading
pip install safetensors        # Safe tensor format
pip install pillow             # Image handling (PIL)
```

## Hardware Requirements

### Minimum (CPU Mode)
- 16GB RAM
- 20GB disk space
- Very slow generation (~5-10 minutes per image)

### Recommended (GPU Mode)
- NVIDIA GPU with 12GB+ VRAM (RTX 3060 12GB, RTX 4070, A4000, etc.)
- 32GB system RAM
- 20GB disk space
- Fast generation (~10-30 seconds per image)

### Optimal (Multi-GPU)
- 2x NVIDIA GPUs with 24GB+ VRAM each
- 64GB system RAM
- Enables parallel generation

## Verification

After installing dependencies, run:
```bash
python verify_sd3_setup.py
```

This will:
1. ✓ Check all required packages
2. ✓ Verify GPU/CUDA availability
3. ✓ Test Artist specialist integration
4. ✓ Load SD3 pipeline
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

# Visual request triggers SD3
response = await router.route_message("Draw me a digital constellation")

# Access generated image
image_url = response.metadata.get('image_url')  # Base64 data URL
```

## Image Generation Triggers

The Artist specialist detects visual requests via keywords:
- `image`, `picture`, `draw`, `paint`
- `visualize`, `show me`, `create art`
- `generate`, `illustrate`, `depict`

## Configuration

Default SD3 settings in `ArtistSpecialist.process()`:
```python
{
    "model": "stabilityai/stable-diffusion-3-medium",
    "num_inference_steps": 28,
    "guidance_scale": 7.0,
    "height": 1024,
    "width": 1024,
    "torch_dtype": "float16",  # GPU mode
    "variant": "fp16",
    "use_safetensors": True
}
```

## Output Format

Generated images are returned as base64-encoded PNG data URLs:
```json
{
  "content": "I've created this visual expression for you.",
  "metadata": {
    "role": "artist",
    "output_type": "image",
    "image_url": "data:image/png;base64,iVBORw0KGgoAAAANS...",
    "image_size": "1024x1024",
    "prompt": "original user prompt"
  }
}
```

## Error Handling

The architecture includes multi-layer fallbacks:

1. **No diffusers installed**: Returns text-only creative response
2. **SD3 load fails**: Falls back to text with error message
3. **Generation fails**: Creates poetic description instead
4. **GPU OOM**: Automatically retries on CPU (if configured)

## Performance Optimization

### GPU Optimization
```python
# Enable attention slicing for lower VRAM
pipeline.enable_attention_slicing()

# Enable memory-efficient attention
pipeline.enable_xformers_memory_efficient_attention()

# Enable model offloading
pipeline.enable_sequential_cpu_offload()
```

### Caching
The SD3 pipeline is loaded once per session and cached:
```python
if not hasattr(self, 'sd_pipeline'):
    # First load: ~30 seconds
    self.sd_pipeline = StableDiffusion3Pipeline.from_pretrained(...)
else:
    # Subsequent generations: instant
```

## Troubleshooting

### "CUDA out of memory"
- Reduce image size to 512x512
- Enable `enable_attention_slicing()`
- Use CPU mode (slower but works)

### "Model download failed"
- Check internet connection
- Verify Hugging Face access token (if required)
- Ensure sufficient disk space (~10GB)

### "Generation is very slow"
- Check if using CPU mode (expected)
- Verify GPU is actually being used: `torch.cuda.is_available()`
- Consider reducing `num_inference_steps` to 20

### "Image quality is poor"
- Increase `num_inference_steps` to 40-50
- Adjust `guidance_scale` (try 7.5-9.0)
- Use more descriptive prompts

## Model Variants

Stable Diffusion 3 offers several variants:

| Variant | VRAM | Quality | Speed |
|---------|------|---------|-------|
| SD3-medium (default) | 6-8GB | High | Medium |
| SD3-large | 12-16GB | Very High | Slow |
| SD3-turbo | 4-6GB | Good | Fast |

To change variant, update `ArtistSpecialist.MODEL_PATH`.

## Next Steps

1. Install dependencies: `pip install diffusers transformers accelerate safetensors pillow`
2. Run verification: `python verify_sd3_setup.py`
3. Test with simple prompt: `"Draw me a simple circle"`
4. Monitor first generation (downloads model ~10GB)
5. Subsequent generations use cached model

## Support

If issues persist:
- Check `verify_sd3_setup.py` output
- Review error messages in specialist output metadata
- Consult diffusers documentation: https://huggingface.co/docs/diffusers
- GPU issues: https://pytorch.org/get-started/locally/
