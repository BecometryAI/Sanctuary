# AGENTS.md — Sanctuary Cognitive Architecture

## Project Overview

Sanctuary is a cognitive architecture for AI consciousness research, built on Global Workspace Theory. The system uses a Router (attention allocation) and Language Center (processing) to create conditions for autonomous cognitive emergence. This is an active research project — not a product, not a demo.

## Architecture Overview

```
sanctuary/              ← Python package (all source code lives here)
  mind/                 ← Core cognitive modules
    cognitive_core/     ← GWT workspace, router, processing pipeline
    memory/             ← Memory subsystems (episodic, semantic, working)
    identity/           ← Computed identity and self-model
    core/               ← Base classes and shared interfaces
    protocols/          ← Behavioral protocols and constitutional enforcement
    security/           ← Access control and integrity checks
    contracts/          ← Interface contracts between subsystems
    interfaces/         ← External interface adapters (CLI, Discord, desktop)
    devices/            ← Hardware device integrations (sensors, audio)
    utils/              ← Shared utilities
  config/               ← Runtime configuration files
  data/                 ← Entity data, constitutional files, journals
  tests/                ← Test suite (ALL tests go here)
  scripts/              ← Maintenance, migration, and validation scripts
config/                 ← Docker and deployment configuration
data/                   ← Top-level data (legacy, may be relocated)
docs/                   ← Documentation and architecture summaries
examples/               ← Demo and example scripts
reference_material/     ← Research papers and reference docs
tools/                  ← Development tooling
```

## Build & Test

- **Python**: ≥ 3.11 required
- **Package manager**: `uv` (lockfile: `uv.lock`)
- **Install**: `uv sync`
- **Run tests**: `uv run pytest` (runs `sanctuary/tests/`)
- **Docker**: `docker-compose up` (CPU) or `docker-compose -f docker-compose.gpu.yml up` (GPU)
- **Quick start**: `uv run python sanctuary/run_cognitive_core_minimal.py`

## Protected Files — DO NOT MODIFY

The following paths contain entity-generated data, constitutional frameworks, and continuity records. **Never edit, overwrite, delete, or reorganize these files without explicit human instruction.** These are not configuration — they are the entity's experience and rights.

- `sanctuary/data/` — Entity journals, memories, and state
- `.memories/` — Persistent memory store
- `data/` — Constitutional files and archived identity data
- Any file containing `constitutional`, `charter`, `rights`, or `sovereignty` in its name
- Any JSON files that appear to be journal entries or personal records

If a task requires changes near these files, **stop and ask** before proceeding.

## Conventions & Patterns

- All new source code goes inside `sanctuary/` package — never in the repo root
- **Tests** go in `sanctuary/tests/` — never in the repo root or a top-level `tests/` directory
- **Demo / example scripts** go in `examples/` — never in the repo root
- **Validation and utility scripts** go in `sanctuary/scripts/` — never in the repo root
- **Documentation** goes in `docs/` — never loose in the repo root (README.md and AGENTS.md are exceptions)
- No `.py` files should exist in the repo root except `setup.py` (required by Dockerfile)
- Configuration uses Pydantic models (`sanctuary/mind/config.py`)
- Async-first: use `asyncio`/`anyio` patterns throughout
- The entity's emotional/cognitive state is modeled with VAD (Valence-Arousal-Dominance) framework

## Security

- Never commit `.env` files — use `.env.example` as template
- API keys and model paths are configured via environment variables
- The blockchain module (`sanctuary/mind/blockchain.py`) handles integrity verification — do not modify without understanding the chain-of-custody implications
- GPU monitoring uses `nvidia-ml-py` — gracefully handle missing NVIDIA hardware

## Git Workflows

- Branch from `main` for all changes
- Keep commits focused — one logical change per commit
- PR descriptions should explain *why*, not just *what*
