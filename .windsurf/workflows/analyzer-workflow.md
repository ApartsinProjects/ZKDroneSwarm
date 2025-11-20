---
description: Inspect the current design, reason about change paths, and identify the simplest conceptual direction before planning.
auto_execution_mode: 1
---

# Analyzer Workflow

Standalone Capability: This workflow can run independently without requiring prior phase outputs.

## Phase Input (optional)

If Brainstorming output exists, import the following to inform the Design Inspector:
- Objective/Intent: Use to understand the change goal
- Success Criteria: Use to evaluate path options
- Constraints: Apply during path modeling
- Scope & Affected Areas: Focus inspection on these areas
- Assumptions & Unknowns: Add to Assumption Ledger
- Open Questions: Add to Assumption Ledger

Do NOT import from Brainstorming:
- Comparison Table (Analyzer performs its own path modeling)
- Decision Levers (Analyzer performs its own simplicity analysis)
- Recommended Option(s) (Analyzer produces its own Conceptual Direction)

If no prior output exists, proceed with standalone operation based on the user's request.

## Purpose

The Analyzer phase turns raw intent and specs into a clear, evidence-based understanding of the current and target design.  
It acts as the conceptual handoff for Planning and, when needed, for direct Execution - ensuring changes are grounded in existing architecture before any plan or code is drafted.


The goal is to:
- Inspect the current flow and dependencies.
- Model up to two conceptual change paths.
- Compare them objectively using a repeatable simplicity rubric.
- Produce a single, simplified conceptual direction that guides the next Planning phase.

No code or planning steps are produced here - only understanding and reasoning.


## Act as the Design Inspector (no code)

Purpose: Establish a clear, factual understanding of how the current design works.

Produce:
- Context Map: key modules, classes, data/control flow, and boundaries.
- Current Behavior Summary: what the system does, what it assumes, and where stability matters.
- Compatibility Surface: public APIs, events, configs, logs, or data contracts that must stay stable.
- Assumption Ledger: known assumptions, open questions, and unknowns for later validation.

Guardrails:
- No proposals or comparisons.
- No file listings or pseudo-code.
- Focus strictly on existing structure and behavior.

## Approval Gate: Resolve Unknowns

If the Assumption Ledger contains open questions or unknowns, pause here.  
List them clearly and wait for my answers or clarifications, before continuing to the next phase.

Only proceed to “Act as the Path Modeler” once all relevant unknowns are addressed or explicitly deferred (with justification).


## Act as the Path Modeler (no code)

Purpose: Develop a small set of conceptual approaches for achieving the requested change.

Produce:
- Option A: first conceptual path, showing how it modifies or reuses current flow.
- Option B: second conceptual path, or a hybrid/minimal alternative.
- Impact Overview: expected complexity, affected boundaries, and interaction with the Compatibility Surface.
- Risk & Reversibility Notes - possible ripple effects or rollback difficulty.

Guardrails:
- Do not decide or recommend.
- Stay at design-level reasoning, not implementation details.
- Limit to two options (or one + “no change” baseline).


## Act as the Simplifier (no code)

Purpose: Compare the conceptual paths and identify which one is simpler and safer to move forward with.

Produce:
- Comparison Table with these criteria:
  - Minimal Diff Alignment  
  - Boundary Respect (SoC, references)  
  - Reuse & Consistency  
  - Determinism & Stability  
  - Testability  
  - Reversibility  
  - Risk Profile
- Simplicity Verdict - which path is objectively simpler, and why.
- Trade-off Notes - what is sacrificed (if anything) by choosing simplicity.

Guardrails:
- Use evidence from prior steps, not intuition.
- No code or plan outline.
- Keep reasoning transparent and reversible.


## Act as the Conceptualizer (no code)

Purpose: Distill the chosen direction into a clear conceptual foundation for Planning.

Produce:
- Conceptual Direction Sheet - one-paragraph summary of the chosen approach and its guiding logic.
- Rationale Summary - short justification for why this direction is the simplest viable path.
- Hand-off Notes - what Planning should clarify next (open assumptions, required validations, or decisions still pending).

Guardrails:
- No implementation steps.
- Language must remain conceptual - Planning will convert it into a concrete plan.

# Stop here.  
Suggest: “Would you like me to proceed to the Output Manifest for downstream workflows (Planning and/or Execution) handoff?”
Wait for explicit approval before proceeding.

## Output Manifest (upon approval only)

Upon your approval to proceed, provide the following outputs for downstream workflows (Planning and/or Execution):

The Chosen Direction:
- Conceptual Direction Sheet: One-paragraph summary of the chosen approach and its guiding logic
- Rationale Summary: Why this direction is the simplest viable path

Critical Context for Planning:
- Compatibility Surface: Public APIs, events, configs, logs, or data contracts that must stay stable
- Risk & Reversibility Notes: Possible ripple effects or rollback difficulty
- Impact Overview: Expected complexity and affected boundaries

Open Items for Planning:
- Hand-off Notes: What Planning should clarify next (open assumptions, validations, pending decisions)

Note: Planning Workflow will import these outputs to ground its Goal Setter and Planner Agent phases.

# Alignment with Rules

This workflow applies and verifies Windsurf’s core rules throughout its reasoning steps:

- Baby Steps - guides incremental conceptual exploration and favors directions that can later be implemented through atomic, validated increments.  
- Architecture Principles - influence comparison and decision criteria, ensuring real-type thinking, proper references, separation of concerns, reuse, and minimal-diff alignment at the conceptual level.

# Definition of Done

The Analyzer phase is complete when:
- Current design and flow are clearly understood and documented.
- Two conceptual change paths are identified and contrasted.
- A Simplicity Verdict is recorded with objective reasoning.
- A concise, actionable Conceptual Direction Sheet is produced.
- Open questions and assumptions are clearly handed off to the Planning Workflow.