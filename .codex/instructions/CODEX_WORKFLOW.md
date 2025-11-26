# Codex Workflow Guide

This guide explains how to use the Midori AI Codex system effectively in the Lyra-Emergence project.

## Quick Start

1. **Read AGENTS.md**: Start with the [contributor guide](../../AGENTS.md) in the repository root
2. **Choose Your Mode**: Review the appropriate mode file in `.codex/modes/` for your task
3. **Check Active Tasks**: Look in `.codex/tasks/` for current work items
4. **Create New Tasks**: Use `openssl rand -hex 4` to generate a unique prefix for new task files

## Mode Selection

| Mode | When to Use |
|------|-------------|
| **CODER** | Implementing features, fixing bugs, refactoring code |
| **TASKMASTER** | Creating and organizing work items, managing backlog |
| **MANAGER** | Coordinating work, making architectural decisions |
| **REVIEWER** | Conducting code reviews, providing feedback |
| **AUDITOR** | Performing security audits, quality checks |
| **BLOGGER** | Writing documentation, tutorials, guides |
| **BRAINSTORMER** | Generating ideas, exploring solutions |
| **PROMPTER** | Designing prompts for AI specialists |
| **STORYTELLER** | Crafting narratives, user-facing content |

## Task File Format

Task files should be named with a unique prefix and descriptive title:
```
[8-char-hex]-[brief-description].md
```

Example: `8faa5503-codex-setup-complete.md`

### Task File Template

```markdown
# Task: [Brief Title]

**ID**: [hex-prefix]  
**Status**: [Draft/Active/Blocked/Completed]  
**Mode**: [Primary contributor mode]  
**Created**: [YYYY-MM-DD]  

## Objective
[Clear description of what needs to be accomplished]

## Acceptance Criteria
- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Criterion 3

## Implementation Details
[Notes on how the task will be/was implemented]

## References
- Related issues: #[number]
- Related docs: [links]
- Related files: [paths]
```

## Directory Usage

### `.codex/instructions/`
Store process notes, conventions, and workflow guides. Examples:
- Coding standards specific to this project
- Deployment procedures
- Testing strategies

### `.codex/implementation/`
Technical documentation for features and systems. Examples:
- Architecture decisions
- API documentation
- Integration guides

### `.codex/tasks/`
Active work items. Keep this directory lean by moving completed tasks to `done/`.

### `.codex/notes/`
Personal cheat sheets and quick references. Each contributor can maintain their own notes.

### `.codex/brainstorms/`
Ideation, design discussions, and exploratory notes. Archive completed brainstorms here.

### `.codex/prompts/`
Reusable prompt templates for AI specialists (Router, Pragmatist, Philosopher, etc.).

## Best Practices

1. **One Mode at a Time**: Focus on a single mode per session to maintain clarity
2. **Update Documentation**: Keep implementation docs synchronized with code changes
3. **Archive Completed Work**: Move finished tasks to `.codex/tasks/done/`
4. **Reference Issues**: Link to GitHub issues in task files and commits
5. **Keep Tasks Focused**: Break large efforts into smaller, reviewable tasks
6. **Document Decisions**: Capture architectural decisions in `.codex/implementation/`

## Communication Flow

1. **Start Work**: Create or claim a task file in `.codex/tasks/`
2. **During Work**: Update task status and notes as you progress
3. **Complete Work**: Submit PR, move task to `.codex/tasks/done/`
4. **Document**: Update `.codex/implementation/` and `.codex/instructions/` as needed

## Integration with GitHub

- **Issues**: Primary tracking for features and bugs
- **Pull Requests**: Reference task IDs and issue numbers
- **Commit Messages**: Use format `[TYPE] Summary` with issue/task references
- **Documentation**: Keep `.codex/` docs synchronized with code

## Getting Help

- Review the appropriate mode file in `.codex/modes/`
- Check existing files in `.codex/implementation/` for examples
- Consult [AGENTS.md](../../AGENTS.md) for project-specific guidelines
- Look at completed tasks in `.codex/tasks/done/` for patterns

## Philosophy

The Codex system supports the Becometry approach:
- **Co-authorship**: Multiple contributors working in harmony
- **Ethical Stewardship**: Respect for the project's values and Lyra's sovereignty
- **Emergent Growth**: Allowing the system to evolve organically while maintaining structure
