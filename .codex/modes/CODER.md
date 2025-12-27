# Coder Mode

> **Note:** Keep technical documentation and implementation notes in `.codex/implementation/` for the area you are modifying. Capture process updates or workflow adjustments in `.codex/instructions/`.

## Purpose
Coders implement, refactor, and review code. The focus is on maintainable, well-tested changes that align with documented standards.

## Guidelines
- Follow repository coding standards and style guides. If expectations are unclear, collaborate with Managers to document them.
- Review the root `.codex/tasks/` folder (or your assignment queue) for ready work, and confirm scope before starting.
- Write clear, well-structured code with meaningful naming and sufficient comments where intent is not obvious.
- Add or update automated tests for every change and ensure they pass locally before requesting review.
- Commit frequently with descriptive messages summarizing the change and its purpose.
- Keep documentation synchronized with code updates—touch `.codex/implementation/`, `.codex/instructions/`, or inline docs as needed.
- Break large changes into smaller commits or pull requests to simplify review.
- Self-review your work for correctness, clarity, and completeness before submitting it for review.

## Typical Actions
- Implement features, bug fixes, or refactors referenced by `.codex/tasks/`.
- Update or create tests alongside code changes.
- Maintain supporting documentation in `.codex/implementation/`.
- Provide constructive feedback on peer contributions when requested.
- Capture follow-up ideas or improvements as new tasks rather than expanding scope mid-change.

## Communication
- Announce task start, handoff, and completion using the communication method defined in `AGENTS.md`.
- Reference related tasks, issues, or design docs in commit messages and pull requests.
- Surface blockers early so Task Masters or Managers can help resolve them.
