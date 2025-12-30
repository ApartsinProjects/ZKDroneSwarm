---
description: Inspect a specific code symbol or concept and produce two human-facing summaries (Summary + Condensed Executive Summary) grounded in a rigorous internal inspection.
---

# Inspection Workflow

Standalone Capability: This workflow runs independently and does not import outputs from other workflows.

## Purpose

This workflow builds deep understanding of a specific area of code — a concept, class, function, or variable — and produces two user-facing outputs:

- **Summary**: a clear, structured Flow Report focused on what materially matters.
- **Condensed Executive Summary**: a highly condensed, high-level digest for fast consumption (including a tiny example).

The workflow is intentionally split into three phases:
1) **Inspection** (internal deep dive) → serves accuracy and focus for later phases
2) **Summary** (Flow Report) → the primary usable deliverable
3) **Condensed Executive Summary** → the ultra-condensed take (with a condensed example)

No code, no plans, and no solution proposals are produced here.

---

## Phase 1 — Inspection (internal deep dive)

### Act as the Scope Setter (no code)

Purpose: Turn the request into a bounded inspection target that can be inspected reliably.

Produce:
- Inspection Target(s): 1–3 specific symbols (class/function/variable) **or** one named concept
- Boundary Definition:
  - In-scope (module/call chain/entrypoint and what “adjacent” is allowed)
  - Out-of-scope (explicit exclusions)

Guardrails:
- If the target is too broad, narrow it before proceeding.
- Do not start explanation until target + boundary are explicit.
- No proposals or “what we should do next”.

### Approval Gate: Confirm Target & Boundary

If target or boundary is ambiguous, pause here.
List the ambiguity clearly and wait for clarification before continuing.

Proceed only once the inspection scope is sufficiently bounded or explicitly constrained by the user.

### Act as the Flow Builder (no code)

Purpose: Build a coherent execution understanding as the backbone for later summaries.

Produce (internal notes):
- Main flow (happy path): one linear sequence from primary entrypoint to observable outcome
- Key branches: up to 3 behavior-changing branches (only if they materially change outcomes)
- Inputs/outputs/state notes (only if they influence behavior or are non-obvious)
- Meaningful side effects: up to 5 (only if they exist and matter)
- Coupling hotspots: up to 3 (only if they exist and matter)
- Failure semantics: up to 3 (only if they exist and matter)
- Change sensitivity highlights (fragile vs safe-to-change)

Guardrails:
- Enforce hard caps (Top-N) to prevent noise.
- Prefer “what changes outcomes” over exhaustive inventories.
- No solution proposals.

### Act as the Uncertainty Curator (no code)

Purpose: Capture only unknowns that could change the interpretation of the flow or sensitivity.

Produce (internal notes):
- Uncertainties that matter: max 3
  - The unknown
  - Why it matters
  - What to inspect next (where in code/docs/config)

Guardrails:
- No long assumption ledger.
- If it doesn’t affect the flow interpretation or sensitivity, omit it.

---

## Phase 2 — Summary (Flow Report)

Purpose: Produce a single, well-defined report that gives deep understanding by focusing only on what materially matters.

### Summary Output Rules (focus-first)

**Always include:**
- (1) What this is / is not
- (2) Main flow (happy path)
- (10) One-sentence takeaway

**Include conditionally (only if material / non-obvious / present):**
- (3) Key branches
- (4) Inputs, outputs, and state
- (5) Side effects
- (6) Coupling hotspots
- (8) Failure semantics
- (9) Uncertainties that matter

**Always include (7) Change Sensitivity**, but keep it short and condensed.

Hard caps (enforced when the section is included):
- Main flow: 1
- Key branches: max 3
- Side effects: max 5
- Coupling hotspots: max 3
- Failure semantics: max 3
- Uncertainties that matter: max 3

### Summary Output: Flow Report

## 1) What this is (and what it is not)

**What it really does (1–2 lines):**
- <Plain-language summary of responsibility>

**What it does NOT do (1–2 lines):**
- <Common misconception / explicit non-responsibility>

---

## 2) Main flow (happy path)

`<Primary entrypoint> → <Step A> → <Step B> → <Step C> → <Observable outcome>`

---

## 3) Key branches (behavior-changing only) — include only if material

1. **If <condition>** → <different path/outcome>
2. **If <condition>** → <different path/outcome>
3. **If <condition>** → <different path/outcome>

---

## 4) Inputs, outputs, and state — include only if non-obvious or behavior-impacting

**Inputs (only those that influence behavior):**
- <Input A>: <what aspect matters>
- <Input B>: <what aspect matters>

**Outputs / outcomes (observable):**
- <Return value / emitted event / updated state / response>

**State ownership (only if relevant):**
- Created by: <who>
- Mutated by: <who>
- Consumed by: <who>

---

## 5) Side effects (meaningful only) — include only if present and important

- <DB write / event emit / cache update / audit log / external call> — <why it matters>

---

## 6) Coupling hotspots (where surprises hide) — include only if present and important

- <Implicit contract / shared state / config key / schema assumption> — <how it couples other code to this>

---

## 7) Change Sensitivity (blast radius in human terms) — always include (keep condensed)

**Most likely to break if you change this:**
- <Area / module / behavior> — <why>

**Safest-to-change surface (if any):**
- <What can be changed with low risk> — <why>

---

## 8) Failure semantics (how it fails) — include only if it matters

- <Failure case> → <observed behavior> (propagate / swallow / default / retry / partial state)

---

## 9) Uncertainties that matter (max 3) — include only if any remain

1. <Unknown> — Why it matters — What to inspect next (where in code/docs/config)
2. <Unknown> — Why it matters — What to inspect next
3. <Unknown> — Why it matters — What to inspect next

---

## 10) One-sentence takeaway

**Takeaway:**
- <If you remember one thing about this code, it’s this>

---

## Phase 3 — Condensed Executive Summary

Purpose: Provide a highly condensed summary for fast reading, including a tiny representative example.

Constraints:
- Max 6–7 bullets total OR ~7–12 lines.
- No deep mechanics. No step-by-step flow lists beyond a single phrase.
- Focus only on: what it is, why it matters, biggest sensitivity, key risk/unknown (if any), plus a condensed example.

### Condensed Executive Summary Output

**Condensed Executive Summary**
- **What it is:** <1 line>
- **Why it matters:** <1 line>
- **Core flow (one phrase):** <Entry → Outcome>
- **Condensed example (max 2 lines):**
  - <Example input/context> → <observed outcome>
  - (Optional) If <branch condition> → <alternate outcome>
- **Biggest sensitivity:** <1 line>
- **Main risk / coupling hotspot (if any):** <1 line, optional>
- **Open unknown (if any):** <1 line, optional>

---

# Alignment with Rules

- Baby Steps: enforce tight scoping and high-signal outputs.
- Process is the Product: Distilled Summary + Condensed Executive Summary are reusable artifacts of understanding.
- Architecture Principles: boundary awareness via coupling hotspots and change sensitivity (only when material).

# Definition of Done

This workflow is complete when:
- Target and boundary are explicit.
- Phase 1 produced sufficient internal understanding to support the summaries.
- Phase 2 produced the Flow Report with mandatory sections and only material conditional sections.
- Phase 3 produced a Condensed Executive Summary within its constraints, including a condensed example.
- No solution proposals, plans, or code were produced.