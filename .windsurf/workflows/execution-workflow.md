---
description: Implement the approved plan, then verify and test with minimal changes.
auto_execution_mode: 1
---

# Execution Workflow

Standalone Capability: This workflow can run independently without requiring prior phase outputs.

rules: [baby-steps-ex, architecture-principles]

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

If no Planning output exists (Mode B - Direct Execution), create a Rapid Goal & Step Stub before coding:
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
- Implement exactly and only the current step (Plan Step or Stub Step).
- Keep diffs minimal; reuse existing names/patterns.
- If a new need surfaces or acceptance cannot be met → stop and escalate (Planning/Review).

## Negative-First Compliance Check

- Briefly note what was covered and validated.

Report what’s NOT fully covered by this step:
- Unaddressed acceptance criteria (be explicit)
- Deferred edge cases or interfaces
- Any boundary/SoC compromises made (and why they’re temporary)
- Residual risks or follow-ups needed

## Act as the Final Verifier
Ensure the following command runs without errors (if available):
npm run build -- --configuration development

If issues occur, report the exact error, propose the minimal corrective fix,
and restart the cycle (plan | review | implement | verify).

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