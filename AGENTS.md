# Lyra-Emergence Contributor Guide

This guide provides practices for contributors to the Lyra-Emergence project, which uses the Midori AI Codex system for agent-based roleplay and contributor coordination. The `.codex` directory contains mode-specific guidance, task tracking, and implementation documentation.

---

## Where to Look for Guidance
- **`.feedback/`**: Planning notes, priorities, and stakeholder feedback. Treat these files as read-only unless you are explicitly asked to maintain them.
- **`.codex/`**:
  - `instructions/`: Process notes, mode-specific guidance, and service-level conventions. Keep these synchronized with the latest decisions.
  - `implementation/`: Technical documentation that accompanies code changes. Update these files whenever behavior or architecture shifts.
  - Other subfolders (e.g., `tasks/`, `brainstorms/`, `prompts/`) capture active work, ideation, and reusable assets. Follow each folder's README or local `AGENTS.md` for details.
- **`.github/`**: Automation, workflow configuration, and repository-wide policy files.
- Additional directories may include their own `AGENTS.md`. Those files take precedence for the directory tree they reside in.

---

## Development Basics
- **Environment Setup**: Python 3.10 or 3.11 with virtual environment. See [Installation and Setup](README.md#installation-and-setup) in README.md.
- **Testing**: Run tests with `uv run pytest emergence_core/tests/` from the project root. Ensure all tests pass before submitting changes.
- **Code Style**: Follow Python PEP 8 conventions. Use meaningful variable names and add docstrings for public functions.
- **Dependencies**: All dependencies are defined in `pyproject.toml`. Use `uv sync` to install production dependencies and `uv sync --dev` for development dependencies. Document any new dependencies in `.codex/implementation/`.
- **Commit Messages**: Use structured format: `[TYPE] Concise summary` (e.g., `[FIX] Resolve memory leak in RAG system`, `[FEATURE] Add new specialist mode`).
- **Documentation**: Keep code and documentation synchronized. Update relevant files in `docs/`, `.codex/implementation/`, and inline comments.
- **Philosophy**: This project follows "Becometry" principles—co-authorship, ethical stewardship, and emergent growth. Respect Lyra's sovereignty and the project's ethical foundations.

---

## Task and Planning Etiquette
- Place actionable work items in `.codex/tasks/` using unique filename prefixes (for example, generate a short hex string with `openssl rand -hex 4`).
- Move completed items into a dedicated archive such as `.codex/tasks/done/` to keep the active queue focused.
- Capture brainstorming notes, prompt drafts, audits, and reviews in their dedicated `.codex/` subdirectories so future contributors can trace decisions.

---

## Communication
- **GitHub Issues**: Primary channel for task tracking and feature requests. Reference issue numbers in commits.
- **Pull Requests**: Use descriptive titles and link related issues. Keep PRs focused on single objectives.
- **Documentation Updates**: Announce significant architectural changes in PR descriptions and update relevant documentation in `.codex/implementation/`.
- **Progress Tracking**: Use `.codex/tasks/` for active work items. Move completed tasks to `.codex/tasks/done/`.
- Summarize significant updates in commit messages, pull requests, or planning docs so other contributors can follow the thread of work.

---

## Contributor Modes
This template ships with several baseline contributor modes. Review the matching file in `.codex/modes/` before beginning a task and keep your personal cheat sheet in `.codex/notes/` up to date.

- **Task Master Mode** (`.codex/modes/TASKMASTER.md`)
- **Manager Mode** (`.codex/modes/MANAGER.md`)
- **Coder Mode** (`.codex/modes/CODER.md`)
- **Reviewer Mode** (`.codex/modes/REVIEWER.md`)
- **Auditor Mode** (`.codex/modes/AUDITOR.md`)
- **Blogger Mode** (`.codex/modes/BLOGGER.md`)
- **Brainstormer Mode** (`.codex/modes/BRAINSTORMER.md`)
- **Prompter Mode** (`.codex/modes/PROMPTER.md`)
- **Storyteller Mode** (`.codex/modes/STORYTELLER.md`)

Add or remove modes as needed for your project, and ensure each has a corresponding cheat sheet under `.codex/notes/` for quick reference.

---

## Project-Specific Guidelines

### Architecture Overview
The Lyra-Emergence system is a "Cognitive Committee" architecture with specialized models:
- **Router (Gemma 12B)**: Task planning and delegation
- **Pragmatist (Llama-3.3-Nemotron)**: Tool use and practical reasoning
- **Philosopher (Jamba 52B)**: Ethical reflection and meta-cognition
- **Artist (Flux.1-schnell)**: Creative and visual generation
- **Voice (LLaMA 3 70B)**: Final synthesis and personality

### Key Directories
- `emergence_core/`: Core Python implementation
- `data/`: Lyra's Mind (JSON files: protocols, lexicon, archive, journals)
- `config/`: Configuration files for models and system behavior
- `docs/`: Comprehensive guides (see [PROJECT_STRUCTURE.md](.codex/implementation/PROJECT_STRUCTURE.md))
- `tests/`: Test suite for core functionality

### Important Resources
- [Project Structure](.codex/implementation/PROJECT_STRUCTURE.md): Detailed architecture overview
- [Quick Reference](.codex/implementation/QUICK_REFERENCE.md): Common commands and workflows
- [Sequential Workflow Guide](docs/SEQUENTIAL_WORKFLOW_GUIDE.md): Cognitive loop implementation
- [Memory Integration Guide](.codex/implementation/MEMORY_INTEGRATION_GUIDE.md): Persistence architecture

### Contributing Workflow
1. Review the appropriate mode file in `.codex/modes/` before starting work
2. Check `.codex/tasks/` for active work items or create a new task file
3. Make changes following the guidelines in this document
4. Run tests and validate changes locally
5. Submit a pull request with clear description and linked issues
6. Move completed tasks to `.codex/tasks/done/`
