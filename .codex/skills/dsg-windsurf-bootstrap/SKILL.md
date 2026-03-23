---
name: dsg-windsurf-bootstrap
description: Bootstrap work in repositories that keep operating guidance under src/dsg-windsurf or .agent, including workflows, rules, skills, and memories that should shape how Codex works in the thread.
---

# DSG Windsurf Bootstrap

Use this skill at the start of work in repositories that keep agent guidance under `src/dsg-windsurf` or `.agent`.

## Goal

Load the minimum repo-specific context needed to work the way this repository expects, without bulk-reading everything.

## Bootstrap Procedure

1. Resolve the governance root:
   - Prefer `src/dsg-windsurf`
   - Fallback to `.agent`
   - If neither exists, say so and continue normally
2. Inventory the available files in:
   - `workflows/`
   - `rules/`
   - `skills/`
   - `memories/`
3. Tell the user which governance root was selected.
4. Load only the files needed for the current task:
   - Read the matching workflow when the user names one or the task clearly maps to one
   - Read relevant rules before code edits
   - Read skill files only when a workflow or task needs them
   - Read memories when they affect gates, approvals, or output behavior

## Workflow Mapping

- `brainstorming-workflow.md`: explore options and trade-offs without planning or coding
- `analyzer-workflow.md`: inspect the current design and compare conceptual paths
- `planning-workflow.md`: define and validate an implementation plan without coding
- `execution-workflow.md`: implement approved steps with context anchors and validation
- `testing-workflow.md`: plan and implement tests separately from production changes

If the user explicitly asks for a workflow, follow that workflow.

If the user asks for code directly, still inspect rules first and use the workflow that best matches the request's current phase.

## Rule Handling

Before editing code, read the relevant rule files from `rules/`.

In this repository, that typically means:
- `baby-step-ex.md`
- `architecture-principles.md`

Apply them as active constraints, not background reference material.

## Path Resolution Rule

Some workflow files may reference `.windsurf/...`.

Treat `.windsurf` as an alias for the resolved governance root in the current repository:
- use `src/dsg-windsurf/...` when that directory exists
- otherwise use `.agent/...`

## Context Discipline

- Do not load every skill body up front
- Do not summarize the entire governance system unless the user asks
- Keep startup output short: active root, relevant workflow, relevant rules, and any important gate behavior

## Current Repo Note

At the time this skill was created, this repository uses `.agent` and does not contain `src/dsg-windsurf`.
