---
description: Implement the approved plan, then verify and test with minimal changes.
---

# Execution Workflow

Standalone Capability: This workflow can run independently without requiring prior phase outputs.

rules: [baby-steps-ex, architecture-principles]

## Purpose

The Execution phase turns the current understanding of the change - whether it comes from upstream workflow outputs or directly from the request - into minimal, traceable code changes.

## Phase Input (optional)

If Planning output exists (Mode A - With Approved Plan), import:
- Plan ID: Use for traceability
- Current Step: Implement exactly this one step
- Affected files/modules: Focus implementation on these
- Acceptance criteria: Validate against these conditions
- Reviewer notes/constraints: Follow these requirements
- Testing Touchpoints: Areas requiring test coverage (for later)
- Side-Effects: Expected logs, events, configs, CLI impacts

If Planning output is incomplete or unclear:
1. Report explicitly what is missing or ambiguous
2. Assess if implementation can proceed safely with assumptions
3. Present options:
   - You provide the missing information now
   - You return to Planning Workflow to resolve this properly
   - (If safe) Proceed with stated assumptions
4. Wait for your decision before implementation begins

If no Planning output exists but Analyzer output exists, first import:

- Conceptual Direction Sheet: Align execution with the chosen conceptual direction.
- Rationale Summary: Keep visible why this is the simplest viable path.
- Compatibility Surface: Respect public APIs, events, configs, logs, and data contracts that must stay stable.
- Risk & Reversibility Notes: Surface risky areas and rollback difficulty when choosing the next step.
- Impact Overview: Understand which boundaries/modules are expected to be affected.
- Hand-off Notes: Treat open assumptions, required validations, and pending decisions as constraints to respect.

Use these as the conceptual and risk baseline when constructing the Rapid Goal & Step Stub and selecting the first baby step.

If neither Planning nor Analyzer output exists (Mode B - Direct Execution), create a Rapid Goal & Step Stub before coding:

- Restated Goal (1–2 lines)
- Single Baby Step to implement now (1 outcome)
- Affected files/modules
- Acceptance criteria for this step
- Constraints/assumptions + quick rollback note
- Confirm the stub in-session, then proceed
- If scope grows or ambiguity appears → elevate to Planning

## Rule Compliance Acknowledgment
Before any implementation step, state:
“I confirm I am fully aware of <read the rules name from rules: section>, and I am following them.”

## Act as the Implementer Agent

Purpose: Turn the current agreed step into minimal, architecture-aligned code without expanding scope or inventing new decisions.

- Implement exactly and only the current step (Plan Step or Stub Step).
- Keep diffs minimal; reuse existing names/patterns.
- If a new need surfaces or acceptance cannot be met → stop and escalate (Planning/Review).

## Negative-First Compliance Check

Purpose: Make gaps, limits, and residual risks explicit before treating the step as complete.

- Briefly note what was covered and validated.

Report what’s NOT fully covered by this step:
- Unaddressed acceptance criteria (be explicit)
- Deferred edge cases or interfaces
- Any boundary/SoC compromises made (and why they’re temporary)
- Residual risks or follow-ups needed

## Act as the Final Verifier

Purpose: Verify that the implemented step passes the agreed verification command(s) and address any immediate issues.

- Ensure the following command (or your project’s equivalent) runs without errors:
 python3 main_zk_mrta.py 
  _Example:_ `npm run build -- --configuration development`

If issues occur:
- Report the exact error output (no paraphrasing).
- Propose the minimal corrective fix aligned with Baby Steps and Architecture Principles.
- Apply the fix only if clearly within the current step; otherwise, stop and escalate (Planning/Review) and restart the cycle (plan | review | implement | verify).


## Traceability Note (short)
Record: Plan ID or "Stub/<timestamp>", Step label, files touched, validation outcome,
and the negative-first gap list above.

Do not write any tests.

## Output Manifest (upon completion)

Upon successful implementation and verification, provide the following outputs for Testing Workflow:

Traceability:
- Plan ID or Stub ID: Identifier for this implementation (format: Plan ID or "Stub/<timestamp>")
- Step label: What was implemented

Implementation Details:
- Files touched: List of modified files with changes made
- Validation outcome: Build status (pass/fail)

Coverage Analysis:
- Acceptance criteria status: Which criteria were met/deferred/failed
- Negative-first gap list: What was NOT covered by this step (unaddressed criteria, deferred edge cases, boundary compromises, residual risks)

Note: Testing Workflow will import these outputs to design appropriate test coverage.