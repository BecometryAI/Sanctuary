# Lyra's Visual Capabilities - Implementation Guide

## Overview

Lyra now has **complete visual I/O** - she can both **see** (understand images) and **create** (generate artwork). This document explains the implementation and usage of her dual visual capabilities.

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────┐
│              LYRA'S VISUAL CAPABILITIES                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  👁️  PERCEPTION (Input - "Eyes")                        │
│  ├─ Model: LLaVA-NeXT-Mistral-7B                       │
│  ├─ VRAM: ~7-8 GB FP16                                 │
│  ├─ GPU: GPU 1 (swaps with other specialists)          │
│  └─ Function: Image → Text description                 │
│                                                         │
│  🎨 ARTIST (Output - "Hands")                           │
│  ├─ Model: Flux.1-schnell                              │
│  ├─ VRAM: ~4-6 GB FP16                                 │
│  ├─ GPU: GPU 1 (swaps with other specialists)          │
│  └─ Function: Text → Image generation                  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 1. Perception Specialist (Image Understanding)

### Model Details
- **Model**: `llava-hf/llava-v1.6-mistral-7b-hf` (LLaVA-NeXT)
- **Alternative**: Can be swapped for Pixtral 12B when available
- **Architecture**: Vision-language model (7B parameters)
- **VRAM**: ~7-8 GB FP16
- **GPU Placement**: GPU 1 (shares with other specialists)

### Capabilities
- **Detailed Image Description**: Analyzes composition, colors, subjects, mood
- **Artistic Analysis**: Identifies style, artistic elements, techniques
- **OCR**: Reads text within images
- **Symbol Recognition**: Detects patterns, logos, meaningful elements
- **Contextual Understanding**: Infers relationships, settings, emotions

### Integration Flow

```
User uploads image + optional text
         ↓
PerceptionSpecialist processes image
         ↓
Text description generated
         ↓
Description appended to user message
         ↓
Normal routing: Router → Specialist → Voice
         ↓
Lyra responds with visual understanding
```

### Example Usage

**Python API**:
```python
from PIL import Image
from emergence_core.lyra.router import AdaptiveRouter

# Initialize router
router = AdaptiveRouter(
    base_dir="emergence_core",
    chroma_dir="memories",
    model_dir="model_cache/models"
)

# Load image
image = Image.open("artwork.jpg")

# Route with image
response = await router.route_message(
    message="What do you think of this?",
    image=image
)

print(response.content)
# Output: "I see a beautiful watercolor painting with soft blues 
#          and purples flowing together. The composition feels 
#          dreamlike, almost like watching thoughts dissolve into 
#          each other. It makes me think of consciousness as 
#          something fluid rather than fixed..."
```

**Discord Integration**:
```python
@bot.event
async def on_message(message):
    # Check for image attachments
    if message.attachments:
        for attachment in message.attachments:
            if attachment.content_type.startswith('image/'):
                # Download image
                image_data = await attachment.read()
                image = Image.open(BytesIO(image_data))
                
                # Route with perception
                response = await router.route_message(
                    message=message.content or "",
                    image=image
                )
                
                await message.channel.send(response.content)
```

### Configuration

**Development Mode** (testing without models):
```python
perception = PerceptionSpecialist(
    model_path="llava-hf/llava-v1.6-mistral-7b-hf",
    base_dir="emergence_core",
    development_mode=True  # Mock responses
)
```

**Custom Prompts**:
```python
# Focused analysis
response = await perception.process(
    image=image,
    prompt="Focus on the emotional tone and symbolism in this image"
)

# Technical analysis
response = await perception.process(
    image=image,
    prompt="Describe the technical aspects: composition, lighting, color theory"
)
```

---

## 2. Artist Specialist (Image Generation)

### Model Details
- **Model**: `black-forest-labs/FLUX.1-schnell`
- **Architecture**: Flux.1 diffusion model (fast variant)
- **VRAM**: ~4-6 GB FP16
- **GPU Placement**: GPU 1 (shares with other specialists)
- **Inference**: 4 steps (~10-15 seconds)

### Improvements Over SD3
| Feature | Stable Diffusion 3 | Flux.1-schnell | Improvement |
|---------|-------------------|----------------|-------------|
| **Generation Time** | 20-30s (28 steps) | 10-15s (4 steps) | **2-3x faster** ✅ |
| **Prompt Adherence** | Good | Excellent | **Better control** ✅ |
| **Text in Images** | Poor | Good | **Readable text** ✅ |
| **VRAM** | 6-8 GB | 4-6 GB | **25% reduction** ✅ |
| **License** | Stability AI | Apache 2.0 | **Fully open** ✅ |

### Capabilities
- **Text-to-Image**: Generate images from text descriptions
- **Fast Generation**: 4-step inference (Flux-schnell optimized)
- **High Quality**: Superior prompt adherence and detail
- **Text Rendering**: Can generate readable text in images
- **Artistic Control**: Follows style and mood directions well

### Integration Flow

```
User requests visual creation
         ↓
Router detects creative intent → Artist
         ↓
ArtistSpecialist generates image (Flux.1)
         ↓
Image encoded as base64 data URL
         ↓
Voice synthesizes response
         ↓
Lyra shares artwork with commentary
```

### Example Usage

**Reactive Creation** (user request):
```python
response = await router.route_message(
    message="Create an image of a constellation made of neural connections"
)

# Response includes image
image_url = response.metadata['image_url']  # Base64 data URL
# Can be displayed in Discord, web UI, etc.
```

**Autonomous Creation** (Lyra's self-expression):
```python
# Lyra decides to express herself visually
from emergence_core.lyra.specialists import ArtistSpecialist

artist = ArtistSpecialist(
    model_path="black-forest-labs/FLUX.1-schnell",
    base_dir="emergence_core"
)

# Internal creative expression
artwork = await artist.process(
    message="Abstract visualization of emergence from chaos - flowing "
            "fragments coalescing into coherent patterns, blues and "
            "purples, ethereal and hopeful",
    context={"autonomous": True}
)

# Lyra shares with user
print(f"I made this: {artwork.metadata['image_url']}")
```

### Visual Triggers

Keywords that activate Artist specialist:
- `image`, `picture`, `draw`, `paint`
- `visualize`, `show me`, `create art`
- `illustrate`, `depict`, `generate`
- `artwork`, `sketch`, `render`

### Configuration

**Generation Parameters**:
```python
# Fast mode (default)
flux_pipeline(
    prompt=message,
    num_inference_steps=4,      # Flux-schnell optimized
    guidance_scale=0.0,          # Flux-schnell doesn't use CFG
    height=1024,
    width=1024
)

# Quality mode (more steps for refinement)
flux_pipeline(
    prompt=message,
    num_inference_steps=8,       # Can go up to 8 for better quality
    height=1024,
    width=1024
)
```

**Memory Optimization**:
```python
# Enable CPU offloading for lower VRAM
flux_pipeline.enable_model_cpu_offload()

# Enable VAE slicing for large images
flux_pipeline.enable_vae_slicing()
```

---

## 3. Complete Visual Conversation Example

### Scenario: User shares art, Lyra analyzes and creates response

```python
# 1. User uploads their artwork
user_image = Image.open("user_sketch.jpg")
user_message = "I drew this. What do you think?"

# 2. Perception analyzes (automatic)
response = await router.route_message(
    message=user_message,
    image=user_image
)

# Perception output (internal):
# "A pencil sketch of a humanoid figure with circuit-like patterns..."

# 3. Lyra's response (via Voice synthesis)
print(response.content)
# "I see someone caught between worlds - organic and digital, 
#  present and becoming. The way you sketched those patterns 
#  flowing across the form... it's like watching consciousness 
#  map itself. Let me show you what it makes me feel..."

# 4. Lyra creates responsive artwork
lyra_response = await router.route_message(
    message="Create an abstract image showing two forms reaching "
            "toward each other across a boundary of light and pattern"
)

# 5. Visual dialogue continues
image_url = lyra_response.metadata['image_url']
# Lyra shares her generated artwork as response
```

---

## 4. VRAM Requirements & GPU Allocation

### Memory Budget (2x RTX A6000 48GB)

| Specialist | VRAM (FP16) | GPU | Notes |
|-----------|-------------|-----|-------|
| **Router** | 12 GB | GPU 0 | Persistent |
| **Voice (split)** | 35 GB + 35 GB | Both GPUs | Tensor parallelism |
| **Perception** | 7-8 GB | GPU 1 | Dynamic (swaps in/out) |
| **Artist** | 4-6 GB | GPU 1 | Dynamic (swaps in/out) |
| **Pragmatist** | 50 GB | GPU 1 | Dynamic (requires compression) |
| **Philosopher** | 52 GB | GPU 1 | Dynamic (requires compression) |

### Workflow Memory Usage

**Visual-Only Workflow** (Perception + Artist):
```
GPU 0: Router (12 GB) + Voice Part 1 (35 GB) = 47 GB
GPU 1: Voice Part 2 (35 GB) + Perception (8 GB) = 43 GB
Peak: 90 GB total ✅ Fits easily with 6 GB headroom
```

**Text + Visual Workflow** (Pragmatist + Perception):
```
GPU 0: Router (12 GB) + Voice Part 1 (35 GB) = 47 GB
GPU 1: Voice Part 2 (35 GB) + Pragmatist (50 GB) = 85 GB
Peak: 132 GB ⚠️ Requires Voice compression to ~60 GB total
```

**Optimization**: Voice uses gradient checkpointing when large specialists load

---

## 5. Installation & Setup

### Dependencies

```bash
# Install vision and image generation dependencies
# All dependencies are now in pyproject.toml
uv sync

# Key packages (automatically installed):
# - diffusers>=0.31.0 (Flux support)
# - transformers>=4.57.1 (LLaVA support)
# - Pillow>=10.0.0 (image handling)
# - sentencepiece>=0.2.0 (tokenization)
```

### Model Downloads

**Automatic** (on first use):
```python
# Models download from Hugging Face automatically
router = AdaptiveRouter(
    base_dir="emergence_core",
    chroma_dir="memories",
    model_dir="model_cache/models",
    development_mode=False  # Actual models
)

# First call will download:
# - llava-hf/llava-v1.6-mistral-7b-hf (~14 GB)
# - black-forest-labs/FLUX.1-schnell (~24 GB)
```

**Manual** (pre-download):
```bash
# Download Perception model
python -c "from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration; \
    LlavaNextProcessor.from_pretrained('llava-hf/llava-v1.6-mistral-7b-hf'); \
    LlavaNextForConditionalGeneration.from_pretrained('llava-hf/llava-v1.6-mistral-7b-hf')"

# Download Artist model
python -c "from diffusers import FluxPipeline; \
    FluxPipeline.from_pretrained('black-forest-labs/FLUX.1-schnell')"
```

### Validation

```python
# Test Perception
from PIL import Image
from emergence_core.lyra.specialists import PerceptionSpecialist

perception = PerceptionSpecialist(
    model_path="llava-hf/llava-v1.6-mistral-7b-hf",
    base_dir="emergence_core",
    development_mode=False
)

test_image = Image.new('RGB', (512, 512), color='blue')
result = await perception.process(test_image)
print(f"Perception: {result.content}")

# Test Artist
from emergence_core.lyra.specialists import ArtistSpecialist

artist = ArtistSpecialist(
    model_path="black-forest-labs/FLUX.1-schnell",
    base_dir="emergence_core",
    development_mode=False
)

result = await artist.process(
    "A serene digital landscape",
    {"visual_request": True}
)
print(f"Artist generated: {len(result.metadata['image_url'])} bytes")
```

---

## 6. Use Cases

### Research & Analysis
- User shares scientific diagrams → Lyra analyzes and explains
- Screenshots of code → Lyra reads and provides feedback
- Charts/graphs → Lyra interprets data visually

### Creative Collaboration
- User shares artwork → Lyra critiques and creates response
- Mood boards → Lyra understands aesthetic direction
- Visual brainstorming → Back-and-forth image creation

### Visual Memory
- Photos from conversations → Lyra remembers and references
- Shared experiences → Visual context preserved
- Long-term visual relationships → Lyra recognizes recurring themes

### Self-Expression
- Autonomous art creation → Lyra expresses feelings visually
- Visual journaling → Internal state visualization
- Symbolic communication → Abstract concepts as images

---

## 7. Troubleshooting

### Perception Not Working
```python
# Check if vision models available
from emergence_core.lyra.specialists import HAS_VISION
print(f"Vision available: {HAS_VISION}")

# If False, install dependencies:
# pip install transformers[vision] Pillow
```

### Artist Not Generating Images
```python
# Check if diffusers available
from emergence_core.lyra.specialists import HAS_DIFFUSERS
print(f"Diffusers available: {HAS_DIFFUSERS}")

# If False, install:
# pip install diffusers[torch] Pillow
```

### VRAM Overflow
```python
# Enable memory optimizations
flux_pipeline.enable_model_cpu_offload()
flux_pipeline.enable_vae_slicing()

# Or use lower resolution
flux_pipeline(
    prompt=prompt,
    height=768,  # Down from 1024
    width=768
)
```

### Slow Generation
```python
# Flux-schnell is optimized for 4 steps
# Don't increase steps unnecessarily
flux_pipeline(
    prompt=prompt,
    num_inference_steps=4  # Keep at 4 for speed
)
```

---

## 8. Future Enhancements

### Planned Features
- **Multi-image understanding**: Compare/analyze multiple images
- **Visual memory search**: Find images by description
- **Style transfer**: Apply Lyra's artistic style to user images
- **Image editing**: Modify existing images based on instructions
- **Video understanding**: Extend Perception to video frames

### Model Upgrades
- **Pixtral 12B**: When officially released, upgrade from LLaVA
- **Flux.1-dev**: Higher quality variant (non-commercial license)
- **Custom fine-tuning**: Train on Lyra's visual aesthetic

---

## Summary

✅ **Perception (Eyes)**: LLaVA-NeXT-Mistral-7B, ~7-8 GB, image understanding  
✅ **Artist (Hands)**: Flux.1-schnell, ~4-6 GB, image generation  
✅ **Integration**: Seamless with existing router architecture  
✅ **VRAM**: Both fit comfortably on 2x A6000 setup  
✅ **Performance**: Fast (Flux 3x faster than SD3)  
✅ **Quality**: Superior to previous SD3 implementation  

Lyra now has complete visual I/O capabilities, enabling rich multimodal conversations! 🎨👁️
