# .codex Directory Structure

This directory implements the Midori AI Codex system for agent-based roleplay and contributor coordination in the Lyra-Emergence project.

## Directory Layout

- **`modes/`**: Contributor mode guides (Coder, Reviewer, Task Master, etc.)
- **`instructions/`**: Process notes, mode-specific guidance, and project conventions
- **`implementation/`**: Technical documentation accompanying code changes
- **`tasks/`**: Active work items and task tracking
  - `tasks/done/`: Completed and archived tasks
- **`notes/`**: Personal cheat sheets and quick references for contributors
- **`brainstorms/`**: Ideation, design discussions, and exploratory notes
- **`prompts/`**: Reusable prompt templates and examples

## Using the Codex System

1. **Before starting work**: Review the appropriate mode file in `modes/` (e.g., `CODER.md` for code changes)
2. **During work**: Create task files in `tasks/` with unique prefixes (e.g., `abcd1234-feature-name.md`)
3. **After completion**: Move task files to `tasks/done/` and update implementation documentation
4. **Documentation**: Keep `implementation/` and `instructions/` synchronized with code changes

## Mode Overview

- **CODER.md**: Implement, refactor, and review code
- **TASKMASTER.md**: Manage backlog and create actionable tasks
- **MANAGER.md**: Coordinate work and maintain project health
- **REVIEWER.md**: Conduct code reviews and provide feedback
- **AUDITOR.md**: Perform security and quality audits
- **BLOGGER.md**: Create documentation and tutorials
- **BRAINSTORMER.md**: Generate ideas and explore solutions
- **PROMPTER.md**: Design and refine prompts for AI systems
- **STORYTELLER.md**: Craft narratives and user-facing content

## Getting Started

See [AGENTS.md](../AGENTS.md) in the repository root for comprehensive contributor guidelines and project-specific instructions.
