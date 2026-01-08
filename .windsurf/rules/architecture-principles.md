---
trigger: model_decision
description: description: Architecture Principles (Execution Scope)
globs: 
---

# 1: Prefer Real Types
Use classes, dataclasses, enums, and value objects for domain state and behavior.  
Exception: boundaries (config, DTOs, logging rows) may use dict/tuple.

Guardrails:
- Don’t turn boundary dicts into heavy classes unless required by the plan.
- Value objects > primitive soup for domain invariants.

# 2: Right Kind of Reference
- Within an aggregate/module: direct object references are fine.
- Across modules/boundaries: use IDs + repository/service lookup to avoid tight coupling.

Guardrails:
- No cross-layer object reach-through; prefer lookups via abstractions.
- Avoid circular references; if you sense one → re-plan before coding.

# 3: Separation of Concerns (SoC)
Layers:
- Domain: pure logic, no I/O.
- Application/Orchestration: coordinates domain; no protocol details.
- Adapters/I/O: CLI, files, network, frameworks.

Guardrails:
- A class stays in its lane. If a change crosses layers → split into separate, minimal steps.

# 4: Plan Before Code
Implement exactly what the approved plan specifies. No piggybacking features.

Guardrails:
- If a new need appears, pause and go back to plan → review before coding.

# 5: Reuse & Consistency
Prefer extending existing classes/methods; keep naming and structure aligned.

Guardrails:
- No new class unless: (a) the plan calls for it, or (b) duplication would exceed a few lines and harm clarity.
- Mirror existing naming/constructor patterns.

# 6: Minimal Diff Alignment
When choices are equivalent, pick the change that:
1. touches fewer lines/files  
2. preserves public interfaces  
3. doesn’t ripple tests/config

Guardrails:
- If a fix requires multiple edits, split into smaller approved steps.

# 7: Encapsulate Variation
When the caller must decide *how* to handle something based on *what kind* of thing it is, move that decision into the thing itself.

Guardrails:
- If code branches on type or category → the variant should own its behavior, not the caller.
- Callers ask *what* they need; objects decide *how* to provide it.
