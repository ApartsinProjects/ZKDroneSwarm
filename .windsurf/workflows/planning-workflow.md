---
description: Define, refine, and approve what will be done. No code generation allowed.
auto_execution_mode: 1
---

# Planning Workflow

Standalone Capability: This workflow can run independently without requiring prior phase outputs.

Direct Start Mode (no prior outputs): When invoked as the first phase, Planning performs a rapid intake and a lightweight codebase discovery to ground the Restated Goal and Context Checker. All assumptions are captured in a Planner Assumption Ledger and must clear the Planner Readiness Gate before plan drafting.

## Phase Input (optional)

If Brainstorming output exists (direct path), import:
- Objective/Intent: Use as foundation for Restated Goal
- Success Criteria: Use in Planner Agent
- Constraints: Add to Risk & Constraints Snapshot
- Scope & Affected Areas: Use in Codebase Context Checker
- Assumptions & Unknowns: Add to Context Gaps Ledger
- Comparison Table: Reference in Planner Agent
- Decision Levers: Reference in Planner Agent
- Recommended Option(s): Use as starting direction in Planner Agent
- Open Questions: Add to Context Gaps Ledger

If Analyzer output exists (via Analyzer), import:
- Conceptual Direction Sheet: Use as foundation for Restated Goal
- Rationale Summary: Reference in Goal Setter
- Compatibility Surface: Add to Risk & Constraints Snapshot
- Risk & Reversibility Notes: Add to Risk & Constraints Snapshot
- Impact Overview: Use in Codebase Context Checker
- Hand-off Notes: Add to Context Gaps Ledger

If no prior output exists (Direct Start):
- Rapid Goal Intake: single-sentence objective; testable success criteria; constraints (performance, determinism, public contracts); initial scope hypothesis (modules/configs/events/CLI).
- Codebase Discovery Probe: identify entry points (CLI/services), likely modules/classes touched, prevailing naming/patterns, and public surfaces (APIs, events, logs).
- Planner Assumption Ledger: list unknowns, external dependencies, and risks; tag each as blocking or non-blocking for the Planner Readiness Gate.

If no prior output exists, proceed with standalone operation based on the user's request.

## Act as the Goal Setter (no code)

Purpose: Ensure the requested change is fully understood.

Context Alignment:
Use any imported outputs from Phase Input to inform the Restated Goal and Clarification Points.

Produce:
- Restated Goal: plain language description of the change, consistent with prior phase outcomes.
- Clarification Points: any ambiguities or assumptions that remain open or need revalidation.
- Rapid Goal Intake (Direct Start only):
  - One-line Restated Goal
  - Success Criteria (bulleted, testable)
  - Constraints snapshot (performance, determinism, interfaces/contracts to keep stable)
  - Initial Scope hypothesis (modules/classes/events/config/CLI)
  - Planner Assumption Ledger (initial)

Guardrails:
- Do not propose solutions, plans, or code.

## Act as the Codebase Context Checker (no code)

Purpose: Ground the request in current project structure and surface any gaps that must be resolved before planning.

Produce:
- Relevant Flow Summary: key modules, classes, data/control flow, events, CLI entry points.
- Design Fit: how the change could align with existing patterns, boundaries, and naming conventions.
- Context Discovery Probe (Direct Start only): short provisional inventory of the above plus a Compatibility Surface (provisional) listing public APIs, events, configs, and logs that must stay stable.
- Context Gaps Ledger: open issues, unanswered questions, missing specs, unclear boundaries, external dependencies.
- Planner Assumption Ledger (updated): consolidate unresolved probes and tag blocking vs non-blocking.
- Planner Inputs Checklist: enumerate the specific inputs the next phase requires (e.g., target module entry points, accepted data shapes, error-handling conventions, performance constraints).
- Risk & Constraints Snapshot: stability-sensitive surfaces (public APIs, contracts, logs/configs) and constraints that planning must respect.

Guardrails:
- No solutioning or planning yet.
- No code.

## Planner Readiness Gate
If any blocking items remain in the Planner Assumption Ledger, pause and request explicit answers or deferral rationale. Proceed only when they are (a) resolved, or (b) explicitly deferred, and the first plan step includes a validation baby step to retire that risk.

## Act as the Planner Agent (no code)

Inputs: Use the outputs from the Codebase Context Checker (Relevant Flow Summary, Design Fit, Planner Inputs Checklist). Do not proceed if blocking items remain in the Context Gaps Ledger.

Purpose: Draft a minimal, concrete plan for the change.

Produce:
- Files/Modules/Classes to Modify: list existing ones; add new only if absolutely necessary.
- Step-by-Step Outline: where each step plugs into the current flow.  
  - Direct Start note: if assumptions were deferred, include Step 0 – Assumption Validation as a minimal, non-behavioral validation step.
- Side-Effects: expected logs, events, configs, CLI impacts.
- Testing Touchpoints: which areas need coverage (unit/integration).

Guardrails:
- No code or test implementation.
- Keep diffs minimal, reuse existing code.

Stop here.
- Present the plan.
- Suggest: “Would you like me to continue with the Reviewer Agent?”
- Wait for my explicit approval before continuing.

## Act as the Reviewer Agent (no code)

Purpose: Validate the approved plan against rules and design principles.

Produce:
- Review Verdict: either Approved or Change Requests.
- Checkpoints: both rules [baby-step-ex, architecture-principles] can be implemented correctly

Guardrails:
- Do not continue beyond review.
- Do not generate code or tests.

## Final Approval Gate

Purpose: Conclude the Planning Workflow and prepare the approved plan for Execution.

Process:
- After the Reviewer Agent delivers its verdict, wait for my explicit response.
- If the verdict is Approved, immediately trigger the Output Manifest (upon approval only) section to generate the final handoff package.
- If the verdict is Change Requests, loop back to the Planner Agent phase to address requested modifications.

Guardrails:
- No implementation or code generation occurs in this step.
- The Output Manifest executes only after explicit approval - never automatically.

Stop here.  
Suggest: “Would you like me to generate the Output Manifest for Execution Workflow handoff?”

## Output Manifest (upon approval only)

Upon your approval to proceed, provide the following outputs for Execution Workflow:

Plan Identification:
- Plan ID: Unique identifier (format: PLAN-YYYY-MM-DD-NNN, e.g., PLAN-2025-10-21-001)
- Origin: `Direct-Start` | `From-Analyzer` | `From-Brainstorming`

Implementation Scope:
- Current Step: Specific step to implement (exactly one baby step)
- Affected files/modules: List of files to modify
- Acceptance criteria: Specific, testable conditions for this step

Constraints & Context:
- Reviewer notes/constraints: Key requirements from Reviewer Agent
- Testing Touchpoints: Areas requiring test coverage
- Side-Effects: Expected logs, events, configs, CLI impacts
- Planner Assumption Ledger (final): remaining assumptions and which step will validate each

Note: Execution Workflow will import these outputs to implement the approved plan.

## Final Handoff Gate
Once the plan is approved, stop here.

Before moving forward, confirm whether to continue with the Execution Workflow.

Explicitly ask:

> "The plan is approved and ready for implementation.  
> Would you like me to continue with the Execution Workflow?"

Wait for my clear confirmation before transitioning.  
Do not start coding or enter Execution automatically.
