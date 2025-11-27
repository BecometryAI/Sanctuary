# Build and Test Commands

This document contains verified commands for building, testing, and linting the Lyra-Emergence codebase.

## Environment Setup

### Virtual Environment (Using UV - Recommended)
```bash
# Install UV (if not already installed)
# Linux/Mac
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Create virtual environment and install dependencies
uv venv --python python 3.13
uv sync --upgrade

# Activate (Linux/Mac)
source .venv/bin/activate

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1
```

### Install Dependencies
```bash
# All production dependencies are defined in pyproject.toml
# They are installed automatically with 'uv sync'

# For development/testing dependencies
uv sync --dev

# Optional: Safetensors if not already installed
uv pip install safetensors
```

## Testing

### Run All Tests
```bash
# From project root
uv run pytest emergence_core/tests/

# With verbose output
uv run pytest emergence_core/tests/ -v

# Run specific test file
uv run pytest emergence_core/tests/test_router.py
```

### Test Configuration
Tests are configured in `pyproject.toml`:
- Async mode: strict
- Test paths: `emergence_core/tests`
- Test files: `test_*.py`

## Validation Scripts

### JSON Schema Validation
```bash
# Validate JSON files
uv run python scripts/validate_json.py

# Validate journal entries
uv run python scripts/validate_journal.py
```

### Sequential Workflow Test
```bash
uv run python tests/test_sequential_workflow.py
```

### Flux Setup Verification
```bash
uv run python tools/verify_flux_setup.py
```

## Code Quality

### Python Style
- Follow PEP 8 conventions
- Use meaningful variable names
- Add docstrings for public functions
- Keep imports organized (standard library, third-party, local)

### Pre-commit Checks
Before submitting a pull request:
1. Run all tests: `uv run pytest emergence_core/tests/`
2. Validate JSON files: `uv run python scripts/validate_json.py`
3. Check imports and basic syntax
4. Ensure documentation is updated

## Common Issues

### Import Errors
Ensure you're in the virtual environment:
```bash
which python  # Should point to .venv/bin/python
```

### CUDA/GPU Detection
```bash
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### ChromaDB Errors
```bash
# Reset ChromaDB
rm -rf model_cache/chroma_db

# Re-initialize
uv run python emergence_core/build_index.py
```

## Notes

- Always activate the virtual environment before running commands
- Tests should pass on Python 3.10 and 3.11
- Some tests may require GPU access for model loading
- Development mode can be enabled with `DEVELOPMENT_MODE=true` in `.env`
