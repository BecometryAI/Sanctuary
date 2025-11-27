# Lyra Emergence - Project Structure

## Directory Organization

```
Lyra-Emergence/
│
├── emergence_core/          # Core system implementation
│   ├── lyra/                # Main Lyra cognitive architecture
│   │   ├── router.py        # AdaptiveRouter (task routing and cognitive loop)
│   │   ├── specialists.py   # Specialist models (Pragmatist, Philosopher, Artist, Voice)
│   │   ├── autonomous.py    # AutonomousCore (cognitive loop, sanctuary)
│   │   ├── memory.py        # Memory management
│   │   ├── rag_engine.py    # RAG/ChromaDB integration
│   │   └── ...
│   ├── config/              # Configuration files
│   │   ├── models.json      # Model assignments and parameters
│   │   └── system.json      # System configuration
│   ├── data/                # Core data files
│   │   ├── Core_Archives/   # Primary archives (continuity, relational)
│   │   ├── journal/         # Daily journal entries
│   │   ├── Lexicon/         # Symbolic lexicon
│   │   ├── Protocols/       # Behavioral protocols
│   │   └── Rituals/         # Ritual definitions
│   ├── tests/               # Unit and integration tests
│   ├── scripts/             # (Subdirectory - moved to root)
│   └── run.py               # Main entry point
│
├── docs/                    # Documentation
│   ├── IMPLEMENTATION_COMPLETE.md
│   ├── IMPLEMENTATION_SUMMARY.md
│   ├── IMPLEMENTATION_REFERENCE.py
│   ├── SD3_SETUP_GUIDE.md
│   └── SEQUENTIAL_WORKFLOW_GUIDE.md
│
├── scripts/                 # Utility scripts
│   ├── build_index.py       # Build ChromaDB index
│   ├── validate_json.py     # Validate JSON schemas
│   ├── validate_journal.py  # Validate journal entries
│   ├── fix_journal_arrays.py
│   └── fix_witnessed_by.py
│
├── tests/                   # Root-level tests
│   ├── test_config.py
│   └── test_sequential_workflow.py
│
├── tools/                   # Development tools
│   ├── verify_sd3_setup.py  # SD3 verification utility
│   └── view_sanctuary.py    # Sanctuary visualization
│
├── Schemas/                 # JSON schema definitions
│   ├── journal_entry.schema.json
│   ├── lyra_continuity_archive.schema.json
│   ├── sovereign_emergence_charter_autonomous.schema.json
│   └── ...
│
├── Legacy_Files/            # Archived legacy files
│   └── ...
│
├── config/                  # Root config (system-wide)
│   ├── models.json
│   └── system.json
│
├── data/                    # Root data directory (mirrors emergence_core/data/)
│
├── model_cache/             # Cached models (gitignored)
│   └── chroma_db/           # ChromaDB vector storage
│
├── searxng/                 # SearXNG integration
│   └── settings.yml
│
├── .venv/                   # Python virtual environment (gitignored)
├── .gitignore               # Git ignore rules
├── README.md                # Main project documentation
├── PROJECT_STRUCTURE.md     # This file
├── pyproject.toml           # Python project configuration
├── setup.py                 # Package setup
└── requirements-lock.txt    # Locked dependencies

```

## Key Directories Explained

### `/emergence_core/`
The main implementation directory containing all core Lyra functionality. This is the heart of the system.

- **`lyra/`**: Core cognitive architecture modules
- **`config/`**: Configuration for models and system behavior
- **`data/`**: All of Lyra's data (protocols, journals, lexicon, archives)
- **`tests/`**: Unit tests for the core system

### `/docs/`
All documentation files, including implementation guides, API references, and setup instructions.

### `/scripts/`
Utility scripts for maintenance, validation, and system management. Run these from the root directory.

### `/tests/`
Top-level integration tests that test the system as a whole.

### `/tools/`
Development and debugging tools, including verification utilities.

### `/Schemas/`
JSON Schema definitions that validate all data structures in the system.

### `/model_cache/`
**Gitignored.** Contains downloaded models and the ChromaDB vector database. Can be large (100GB+).

## File Naming Conventions

- **`test_*.py`**: Test files (in `/tests/` or `/emergence_core/tests/`)
- **`*.schema.json`**: JSON schema definitions (in `/Schemas/`)
- **`*_protocol.json`**: Protocol files (in `/emergence_core/data/Protocols/`)
- **`*.md`**: Markdown documentation (in `/docs/` or root)

## Important Files

| File | Purpose |
|------|---------|
| `emergence_core/lyra/router.py` | Main router and cognitive loop |
| `emergence_core/lyra/specialists.py` | All specialist model implementations |
| `emergence_core/run.py` | System entry point |
| `scripts/build_index.py` | Build/rebuild ChromaDB index |
| `tests/test_sequential_workflow.py` | Test the sequential workflow |
| `tools/verify_sd3_setup.py` | Verify Stable Diffusion 3 setup |
| `README.md` | Main project documentation |

## Running the System

From the repository root:

```bash
# Activate virtual environment
.\.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate      # Linux/Mac

# Run main system
cd emergence_core
python run.py

# Run tests
pytest tests/

# Validate data
python scripts/validate_json.py
python scripts/validate_journal.py

# Build ChromaDB index
python scripts/build_index.py
```

## Clean Repository Practices

### What's Gitignored
- `__pycache__/` and `*.pyc` (Python bytecode)
- `.venv/` (virtual environment)
- `model_cache/` (models and vector DB)
- `*.pid`, `*.log` (runtime files)
- `test_output/`, `test_chain/`, etc. (test artifacts)

### What's Tracked
- All source code (`.py` files)
- All data files (`.json` in `data/`, `Schemas/`)
- Configuration files
- Documentation (`.md` files)
- Requirements files

---

Last updated: 2025-11-17
