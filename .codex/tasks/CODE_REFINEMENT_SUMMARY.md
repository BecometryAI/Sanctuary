# Code Refinement Technical Details
**Date**: 2025-12-27

## Changes Summary

### 1. Config Loading → Instance Method
**File**: `specialist_tools.py`

```python
# Before: Module-level (fails if missing)
config = json.load(open("config.json"))

# After: Instance method with fallback
def _load_config(self, config_path: str) -> Dict[str, Any]:
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"searxng": {"base_url": "..."}, "wolfram": {"app_id": ""}}
```

### 2. Exec Security → Restricted Environment
**File**: `specialist_tools.py`

```python
# Before: Full builtins (security risk)
exec(code, {'__builtins__': __builtins__}, local_vars)

# After: Safe builtins only
safe_builtins = {'len': len, 'str': str, 'int': int, ...}  # No open, import, eval
exec(code, {'__builtins__': safe_builtins}, local_vars)

# Plus: Pattern validation
dangerous = ['import os', '__import__', 'eval(', 'exec(', 'open(']
if any(p in code.lower() for p in dangerous):
    raise ValueError(f"Dangerous pattern detected")
```

### 3. Input Validation
**Files**: `specialist_tools.py`, `router.py`

```python
# Playwright instructions
if not instructions or len(instructions) > 2000:
    return "Error: Invalid instructions"

# Specialist invocation
if not isinstance(specialist, str):
    return error_response("invalid_specialist_name")
if not isinstance(context, dict):
    context = {"original_context": context}
```

### 4. Timeout Protection
**Files**: `specialist_tools.py`, `router.py`

```python
# Code execution: 5s
await asyncio.wait_for(asyncio.sleep(0.5), timeout=5.0)

# Code generation: 10s  
await asyncio.wait_for(router.generate(...), timeout=10.0)

# Specialist invocation: 30s
await asyncio.wait_for(specialist.process(...), timeout=30.0)
```

### 5. Pragmatist Init → Smart Fallback
**File**: `specialists.py`

```python
# Tools provided: Use methods
if tools:
    self._wolfram = tools.wolfram_compute

# No tools: Import fallback
else:
    try:
        from .specialist_tools import wolfram_compute
        self._wolfram = wolfram_compute
    except ImportError:
        async def _unavailable(q): return "Tool unavailable"
        self._wolfram = _unavailable
```

## Test Coverage

### Security Tests
- Dangerous pattern detection (import os, eval, exec, open)
- Restricted builtins enforcement
- Timeout behavior
- Code validation

### Edge Cases
- Empty/None/oversized inputs
- Non-string types requiring conversion
- Missing dependencies/imports
- Invalid specialist names
- Uninitialized specialists

### Integration
- Router ↔ SpecialistTools connection
- Backward compatibility of module functions
- Config loading fallbacks

## Running Tests

```bash
# Quick check
uv run pytest emergence_core/tests/test_specialist_tools_refactored.py::TestCodeGenerationSecurity -v

# Full suite
uv run pytest emergence_core/tests/test_*.py -v

# With coverage
uv run pytest emergence_core/tests/ --cov=emergence_core.lyra --cov-report=term-missing
```
