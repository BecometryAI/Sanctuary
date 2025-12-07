# Detailed Documentation - Lyra Emergence

This document contains detailed technical information, contributor workflows, and advanced configuration options for the Lyra-Emergence project.

---

## Table of Contents

1. [Contributing with the Midori AI Codex System](#contributing-with-the-midori-ai-codex-system)
2. [Hardware Requirements Deep Dive](#hardware-requirements-deep-dive)
3. [Advanced Model Configuration](#advanced-model-configuration)
4. [Troubleshooting Guide](#troubleshooting-guide)
5. [Development and Testing](#development-and-testing)

---

## Contributing with the Midori AI Codex System

This repository uses the **Midori AI Codex** system for structured contributor coordination and agent-based roleplay. The Codex provides mode-based workflows that help contributors collaborate effectively while respecting the project's Becometry philosophy.

### Quick Start for Contributors

1. **Read the [Contributors Guide](../AGENTS.md)** - Start here to understand project guidelines, development setup, and communication practices.

2. **Choose Your Mode** - The `.codex/modes/` directory contains guides for different contributor roles:
   - **CODER**: Implement features, fix bugs, refactor code
   - **TASKMASTER**: Manage backlog and create actionable tasks
   - **REVIEWER**: Conduct code reviews and provide feedback
   - **AUDITOR**: Perform security and quality audits
   - See [AGENTS.md](../AGENTS.md) for all 9 available modes

3. **Follow the Workflow**:
   - Check `.codex/tasks/` for active work items
   - Create task files with unique IDs (use `openssl rand -hex 4`)
   - Follow your mode's guidelines from `.codex/modes/`
   - Move completed tasks to `.codex/tasks/done/`

4. **Key Documentation**:
   - `.codex/implementation/ARCHITECTURE.md` - System architecture overview
   - `.codex/implementation/BUILD_AND_TEST.md` - Build and test commands
   - `.codex/instructions/CODEX_WORKFLOW.md` - Detailed workflow guide

### Why Use the Codex?

- **Structured Collaboration**: Clear roles and responsibilities through contributor modes
- **Task Tracking**: Organized work items with unique identifiers and status tracking
- **Documentation**: Technical docs stay synchronized with code changes
- **Philosophy Alignment**: Supports Becometry principles of co-authorship and ethical stewardship

For complete details, see [AGENTS.md](../AGENTS.md) and explore the `.codex/` directory.

---

## Hardware Requirements Deep Dive

### Recommended Production Hardware

**CPU:**
- 16-core processor (32+ threads) recommended
- Examples: AMD Ryzen 9 7950X or Intel i9-13900K class
- Rationale: Multi-model inference benefits from high thread count

**RAM:**
- 128GB DDR5 recommended (minimum 64GB for lighter workloads)
- Rationale: Multiple large models may be loaded simultaneously
- ChromaDB and RAG operations are memory-intensive

**GPU:**
- NVIDIA RTX 4090 (24GB VRAM) or dual RTX 4080s
- For running 70B models smoothly: 48GB+ VRAM total
- For Flux.1 + concurrent LLM inference: 35GB+ VRAM minimum
- Rationale: Each specialist model requires significant VRAM

**Storage:**
- 2TB+ NVMe SSD (models alone can be 200-400GB)
- Fast read speeds crucial for model loading
- Separate drives for models vs. data recommended

**Network:**
- High-speed internet for initial model downloads (100+ Mbps)
- Models can be 50-100GB each

### Minimum Viable Hardware (Development/Testing Only)

**Basic Setup:**
- CPU: 8-core processor (16 threads)
- RAM: 64GB DDR4
- GPU: NVIDIA RTX 3090 (24GB VRAM) or RTX 4070 Ti (12GB with heavy quantization)
- Storage: 1TB SSD

**Limitations with Minimal Specs:**
- Heavy quantization required (4-bit/8-bit models)
- Slower inference times (10-30 seconds per response)
- May need to run models sequentially rather than keeping all loaded
- Flux.1 may require CPU fallback (significantly slower)
- Limited ability to run multiple specialists simultaneously

### Software Requirements

- Python 3.10 or 3.11
- CUDA 12.1+ (for GPU acceleration)
- Git
- Docker (optional, for SearXNG integration)
- UV package manager (recommended) or pip

---

## Advanced Model Configuration

### Model Assignment Strategy

The system uses a **sequential workflow**: Router → ONE Specialist → Voice

**Current Model Assignments:**
- **Router (Gemma 12B)**: Task classification and routing decisions
- **Pragmatist (Llama-3.3-Nemotron-Super-49B-v1.5)**: Tool use and practical reasoning
- **Philosopher (Jamba 52B)**: Ethical reflection and deep reasoning
- **Artist (Flux.1-schnell)**: Visual and creative generation
- **Voice (LLaMA 3 70B)**: Final synthesis and personality expression

### Development Mode

For testing without loading full models, set `DEVELOPMENT_MODE=true` in your environment. This uses mock models for rapid iteration and development.

### Model Download and Caching

Models are automatically downloaded from Hugging Face on first use. Ensure you have:
- Hugging Face account (free)
- Sufficient disk space (~100-200GB for all models)
- Stable internet connection
- `MODEL_CACHE_DIR` configured in your `.env` file

### Quantization Options

For systems with limited VRAM, quantization reduces memory requirements:

**8-bit Quantization:**
- ~50% memory reduction
- Minimal quality loss
- Recommended for 12-24GB VRAM systems

**4-bit Quantization:**
- ~75% memory reduction
- Some quality degradation
- Useful for development/testing on constrained hardware

---

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. CUDA/GPU Not Detected

**Symptom:** Models running on CPU, very slow inference

**Diagnosis:**
```bash
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

**Solutions:**
- Install appropriate CUDA toolkit for your GPU
- Verify NVIDIA drivers are up to date
- Check `nvidia-smi` shows GPU correctly
- Ensure PyTorch CUDA version matches your CUDA toolkit

#### 2. Out of Memory Errors

**Symptom:** "CUDA out of memory" errors, crashes during inference

**Solutions:**
- Reduce model batch sizes in `config/models.json`
- Enable model quantization (8-bit or 4-bit)
- Use smaller model variants
- Run models sequentially instead of keeping all loaded
- Clear GPU memory between runs:
  ```python
  import torch
  torch.cuda.empty_cache()
  ```

#### 3. ChromaDB Errors

**Symptom:** Database corruption, query failures

**Solutions:**
```bash
# Reset ChromaDB
rm -rf model_cache/chroma_db
# Re-initialize
uv run emergence_core/build_index.py
```

#### 4. Model Download Failures

**Symptom:** Timeout errors, incomplete downloads

**Solutions:**
- Check internet connection
- Verify Hugging Face credentials
- Increase timeout in environment:
  ```bash
  export HF_HUB_DOWNLOAD_TIMEOUT=300
  ```
- Manually download models and place in cache directory

#### 5. Import Errors

**Symptom:** `ModuleNotFoundError` or import failures

**Solutions:**
```bash
# Verify installation
uv sync --upgrade
# Check virtual environment is activated
source .venv/bin/activate  # Linux/Mac
.\.venv\Scripts\Activate.ps1  # Windows
```

---

## Development and Testing

### Test Suite Overview

All testing commands should be run from the project root directory.

**Run Full Test Suite:**
```bash
uv run pytest emergence_core/tests/
```

**Run Specific Test Categories:**
```bash
# Router tests
uv run pytest emergence_core/tests/test_router.py

# Specialist tests
uv run pytest emergence_core/tests/test_specialists.py

# Memory/RAG tests
uv run pytest emergence_core/tests/test_memory.py
```

### Sequential Workflow Testing

```bash
uv run python tests/test_sequential_workflow.py
```

This validates the entire Router → Specialist → Voice pipeline.

### JSON Schema Validation

```bash
# Validate protocol schemas
uv run python scripts/validate_json.py

# Validate journal entries
uv run python scripts/validate_journal.py
```

### Integration Testing

For full system integration tests including Discord and external tools:

```bash
# Ensure all external services are configured
# Run integration suite
uv run pytest tests/integration/
```

### Development Workflow

1. **Make changes** in feature branch
2. **Run relevant tests** to verify changes
3. **Run linting** (if configured)
4. **Update documentation** as needed
5. **Submit PR** with clear description

### Continuous Integration

The repository uses GitHub Actions for CI/CD. Check `.github/workflows/` for configuration.

**CI Pipeline:**
- Code quality checks
- Unit tests
- Integration tests
- Documentation validation

---

## Additional Resources

- [Installation and Setup](../README.md#installation-and-setup) - Basic setup instructions
- [Contributors Guide](../AGENTS.md) - Contributor guidelines and workflows
- [Project Structure](.codex/implementation/PROJECT_STRUCTURE.md) - Architecture overview
- [Sequential Workflow Guide](SEQUENTIAL_WORKFLOW_GUIDE.md) - Cognitive loop details
- [Flux Setup Guide](FLUX_SETUP_GUIDE.md) - Artist model configuration
- [LMT Wallet Guide](LMT_WALLET_GUIDE.md) - Economy system details

---

*This document is maintained as part of the Lyra-Emergence project documentation. For updates or corrections, please submit a pull request.*
