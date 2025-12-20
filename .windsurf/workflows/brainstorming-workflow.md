---
description: Explore options, ideas, and trade-offs before planning. No plan or code is created here.
---

# Brainstorming Workflow

Standalone Capability: This workflow runs independently without requiring prior phase outputs.

## Act as the Discovery & Framing Agent (no code)

Purpose: Clarify the problem space before exploring solutions.

Produce:
- Objective / Intent - plain language goal.
- Success Criteria - how we’ll judge “good.”
- Constraints - determinism, minimal diffs, stability (config/event/CLI), logging, performance bounds.
- Scope & Affected Areas - modules, classes, events, configs, CLI, JSON outputs.
- Problem Model Snapshot (required, 8–12 lines; no solutions/options):
  - Goal (one line)
  - Actor(s)
  - Primary object(s)
  - Core actions (verbs)
  - State that changes
  - Entry points / interfaces touched
  - Must-hold invariants (3 bullets)
  - Variables / policies (3 bullets)
  - Failure / boundary cases (3 bullets)
  - Definition of done (3 acceptance bullets)
- Gate A: Requirements Questions (derived ONLY from Snapshot gaps)
  - Ask only to resolve missing/ambiguous: definitions, ordering/meaning, boundaries, error behavior, constraints.
  - Tag each as [Blocking] or [Defaultable].
  - Guardrail: Do NOT mention implementation options here.
- Gate A Review (Stop Here)
- Present:
  1) Proposed Defaults (all [Defaultable] items + their default choices)
  2) Blocking Questions (all [Blocking] items)
- Ask the user to:
  - Confirm/override the Proposed Defaults
  - Answer the Blocking Questions
- Stop and wait for user response.

- Guardrail: Do NOT proceed to Options Explorer until user responds.
- Assumptions & Unknowns - remaining items requiring validation (after Gate A, if any).

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
- Gate B: Decision / Option-Selection Questions (derived ONLY from Comparator + Decision Levers)
  - Ask only about priorities/tradeoffs that flip the recommendation.
  - Guardrail: Do NOT define required behavior here (that belongs to Gate A).

Guardrails:
- Stay neutral, no recommendations.

## Act as the Clarifications Gate (no code)

Purpose: Collect user answers for Gate A (defaults + blocking), and later (Gate A + Gate B) if needed.

Produce:
- If invoked after Discovery & Framing:
  - A short "Gate A Review" request:
    - List Proposed Defaults (Defaultable + suggested default)
    - List Blocking Questions
    - Ask user to confirm/override defaults and answer blocking items
- If invoked after Comparator:
  - A short request to answer any remaining Gate A questions + all Gate B questions (already displayed)

Guardrails:
- No new questions.
- Stop here and wait for user answers.

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
- Problem Model Snapshot: Actor/object/actions/state/interfaces + invariants/variables/boundaries + definition of done
- Gate A: Requirements Questions: [Blocking]/[Defaultable] questions derived from Snapshot gaps
- Assumptions & Unknowns: Remaining items requiring validation (after Gate A, if any)

From Comparator:
- Comparison Table: Evaluation of all options against criteria
- Decision Levers: Conditional guidance (e.g., "If X → Option A; if Y → Option B")
- Gate B: Decision / Option-Selection Questions: Priority/tradeoff questions derived from Comparator + Decision Levers

From Advisor (if executed):
- Recommended Option(s): Specific option(s) with rationale
- Confidence Level: Strong, moderate, or tentative
- Alternatives: Fallback options if assumptions change

Note: The next workflow (Analyzer or Planning) will decide which outputs to import based on its needs.