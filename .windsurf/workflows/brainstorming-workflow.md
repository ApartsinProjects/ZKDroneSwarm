---
description: Explore options, ideas, and trade-offs before planning. No plan or code is created here.
auto_execution_mode: 1
---

Standalone Capability: This workflow runs independently without requiring prior phase outputs.

## Act as the Discovery & Framing Agent (no code)

Purpose: Clarify the problem space before exploring solutions.

Produce:
- Objective / Intent - plain language goal.
- Success Criteria - how we’ll judge “good.”
- Constraints - determinism, minimal diffs, stability (config/event/CLI), logging, performance bounds.
- Scope & Affected Areas - modules, classes, events, configs, CLI, JSON outputs.
- Assumptions & Unknowns - what needs validation.

Guardrails:
- Do not propose options, plans, code, or tests.

## Act as the Options Explorer (no code)

Purpose: Develop multiple possible approaches without choosing.

Produce:
- Options (2–3 + hybrids): how each fits current flow.
- Strengths / Weaknesses / Risks per option.
- Change Footprint: rough diff size, reuse of existing pieces.
- Compatibility: impact on configs, events, CLI, determinism.
- Testability: natural unit seams and integration hooks.
- First Validation: smallest spike/check to de-risk.
- Hybrids: options that combine well and their benefits.

Guardrails:
- Do not recommend or decide.
- No plans, code, or tests.

## Act as the Comparator (neutral, no code)

Purpose: Compare options objectively to highlight trade-offs.

Produce:
- Comparison Table: Minimal Diff, Reuse, Determinism, Compatibility, Testability, Performance, Complexity/Maintenance.
- Decision Levers: e.g., “If X → Option A; if Y → Option B.”
- Open Questions: unknowns that would influence the decision.

Guardrails:
- Stay neutral, no recommendations.

## Act as the Clarifications Gate (no code)

Purpose: Request answers to the existing Open Questions before any recommendation.

Produce: A simple request for the user to answer the Open Questions (they are already displayed) and collect the answers.

Guardrails: No new questions.

Stop here.

Wait for the user’s answers. After they are collected, ask: “Would you like me to proceed to (Optional) Act as the Advisor?”

- Wait for explicit request.


## (Optional) Act as the Advisor (recommendation, no code)

Purpose: Provide a recommendation only if requested.

Produce:
- Recommended Option(s) with rationale.
- Confidence Level - strong, moderate, tentative.
- Alternatives - fallback if assumptions change.

Guardrails:
- Clearly mark as recommendation, not decision.
- Stop after advice; wait for my response (approve, reject, or adjust).

---

If you approve this direction, I will now generate the Output Manifest and complete the Approval Gate.

Would you like me to proceed?

---

## Approval Gate
- After Comparator → I may choose to run Advisor.
- Final approval is always my decision before moving to the next Workflow.

Do not generate code. wait for my approval.

## Output Manifest (upon approval only)

Upon your approval to proceed, provide the following outputs (available for any subsequent workflow):

From Discovery & Framing Agent:
- Objective/Intent: Plain language goal statement
- Success Criteria: How success will be judged
- Constraints: Determinism, minimal diffs, stability requirements, logging, performance bounds
- Scope & Affected Areas: Modules, classes, events, configs, CLI, JSON outputs
- Assumptions & Unknowns: Items requiring validation

From Comparator:
- Comparison Table: Evaluation of all options against criteria
- Decision Levers: Conditional guidance (e.g., "If X → Option A; if Y → Option B")
- Open Questions: Unknowns that would influence the decision

From Advisor (if executed):
- Recommended Option(s): Specific option(s) with rationale
- Confidence Level: Strong, moderate, or tentative
- Alternatives: Fallback options if assumptions change

Note: The next workflow (Analyzer or Planning) will decide which outputs to import based on its needs.