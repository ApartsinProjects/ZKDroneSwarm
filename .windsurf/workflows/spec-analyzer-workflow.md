---
description: Analyze a feature spec (greenfield or existing) and turn it into structured, planning-ready inputs. No plan or code is created here.
---

# Spec Analyzer Workflow

Standalone Capability: This workflow can run independently without requiring prior phase outputs.

## Phase Input (optional)

If Brainstorming output exists, import:
- Objective / Intent
- Success Criteria
- Constraints
- Scope & Affected Areas
- Assumptions & Unknowns
- Decision Levers / Open Questions
- Recommended Option(s) & Rationale (if Advisor ran)

If Analyzer output exists, import:
- Current Design Summary
- Conceptual Direction Sheet
- Simplicity Verdict & comparison notes
- Open Questions & Assumptions

Use prior phases only to:
- Check alignment or conflict between the spec and earlier understanding.
- Highlight where the spec expands, contradicts, or under-specifies prior reasoning.

Do not silently override the spec or prior phases. Surface conflicts explicitly.

This workflow operates at the spec interpretation level only - no planning, no solutioning, no code.

---

## Act as the Spec Reader & Framing Agent (no code)

Purpose: Understand what the spec is asking for and frame it in clear, compact terms before going into details.

Produce:
- Restated Goal & Scope
  - Plain-language restatement of what the spec wants to achieve.
  - What is clearly in scope.
  - Any explicit non-goals / out-of-scope items mentioned.
- Scenario / Domain Overview
  - Short description of the main scenario(s), use cases, or domain described by the spec.
- Context Alignment (if prior phases exist)
  - Where the spec aligns with Brainstorming / Analyzer outputs.
  - Where it diverges or introduces new goals.

Guardrails:
- Do not invent new goals; stay anchored in the spec.
- Do not critique or redesign the spec here - just restate and frame.
- No plan, no code, no architecture decisions.

---

## Act as the Behaviour & Capability Extractor (no code)

Purpose: Turn the narrative spec into a concrete behaviour and capability checklist that Planning can rely on.

Produce:
- Capabilities & Behaviour Checklist
  - “The system / feature should be able to…” style bullets, derived from the spec.
  - Key life-cycle phases (e.g., initialization, normal operation, error handling, shutdown/termination).
  - Important state transitions and rules (how key fields change over time or in response to events/actions).
- State & Data View (conceptual)
  - Main state or data components (e.g., entities, aggregates, resources).
  - Which fields clearly exist and how they change over time, without defining concrete classes or schemas.
- Outcome Logic (if relevant)
  - How success, failure, status, or business outcomes are defined and triggered.

Guardrails:
- Keep everything implementation-agnostic (no classes, methods, file names, or specific libraries).
- Do not break work into steps or talk about delivery order.
- Do not add behaviours that aren’t grounded in the spec.

---

## Act as the Interface & Contract Mapper (no code)

Purpose: Make explicit the boundaries between actors and how they interact (inputs, outputs, contracts).

Produce:
- Actors & Roles
  - Internal and external actors (e.g., users, services, systems, jobs, clients).
- Interfaces / Surfaces
  - Inputs: what each actor sends in (API calls, events, UI actions, configs, files, messages).
  - Outputs: what comes out (responses, events, logs, notifications, generated data).
- Interaction Contracts
  - What each interface guarantees (e.g., idempotency, ordering, validation rules, error semantics).
- Change Surfaces vs. Building Blocks
  - If the spec targets an existing system:
    - Likely affected surfaces (APIs, flows, configs, data stores, jobs, CLI, JSON outputs).
  - If the spec is greenfield:
    - Conceptual building blocks implied (e.g., “public API layer”, “background processor”, “persistence layer”, “UI screen”).

Guardrails:
- Do not assume an existing codebase unless explicitly stated.
- Do not specify concrete module/file structures.
- No protocol or schema design beyond what the spec implies.

---

## Act as the Constraints & Risk Mapper (no code)

Purpose: Expose the rules and pressures the Planner must respect, and where complexity or risk hides.

Produce:
- Constraints & Invariants
  - Hard rules (e.g., domain rules, data integrity, consistency requirements, invariants over time).
  - Operational constraints if mentioned (performance, latency, throughput, security, privacy, compliance, audit).
  - Architectural or behaviour invariants implied by the spec (e.g., determinism, idempotency, isolation of concerns).
- Complexity & Risk Snapshot
  - High-level complexity rating (e.g., Low / Medium / High).
  - Areas that are:
    - Strongly coupled to other behaviour or systems.
    - Safety-, correctness-, or compliance-critical.
    - Subtle (e.g., concurrency, ordering, partial failure handling, long-running workflows).
- Alignment with Windsurf Rules (if provided)
  - Where the spec aligns with Baby Steps (small, testable increments) versus where it might be too big or multi-axis.
  - Where it may stress Architecture Principles (boundaries, minimal diffs, reuse, determinism).

Guardrails:
- Do not propose mitigations or plans; only highlight where care is needed.
- Do not downplay risks because they seem “easy” - stay evidence-based from the spec.
- No code, no refactors, no solution proposals.

---

## Act as the Gaps & Questions Collector (no code)

Purpose: Identify what’s missing or unclear in the spec and decide whether Planning can safely proceed.

Produce:
- Assumptions Ledger
  - Assumptions you had to make because the spec is silent or ambiguous.
  - Mark each as:
    - “Derived from spec wording”
    - or “Introduced here to make sense of the spec.”
- Spec Gaps & Ambiguities
  - Conflicts inside the spec.
  - Conflicts between the spec and prior Brainstorming / Analyzer outputs.
  - Missing decisions that affect behaviour, constraints, or interfaces.
- Planning Readiness
  - `Planning Readiness: READY` when:
    - The spec is clear enough to plan against.
    - Remaining questions are minor and can be resolved during planning.
  - `Planning Readiness: BLOCKED` when:
    - Key behaviours, constraints, or interfaces are unclear.
    - Fundamental conflicts with prior understanding exist.
- Questions for Spec Author / Stakeholder
  - A small, focused list of clarifying questions.
  - Each question should clearly:
    - Tie to a specific ambiguity or gap.
    - Indicate which aspect of Planning it would unblock (e.g., behaviour, interface, constraint, scope).

Guardrails:
- Do not “fix” the spec by rewriting it; only highlight what must be decided.
- Keep the question list minimal but sufficient - avoid flooding with noise.
- No planning, no code, no architecture redesign.

---

## Definition of Done & Handoff

The Spec Analyzer phase is complete when:

- The spec’s goal and scope are restated clearly.
- A concise Capabilities & Behaviour Checklist exists.
- Actors, interfaces, and contracts are explicit.
- Constraints, invariants, complexity, and risk are surfaced.
- Assumptions, gaps, and conflicts are clearly documented.
- A Planning Readiness status is set, with targeted clarification questions if needed.

At this point, the output can be handed off to the Planning Workflow, which will:

- Use the Spec Interpretation as input.
- Turn it into a concrete, Baby-Steps-aligned implementation plan.

Guardrail:
- Do not perform Planning or Implementation within this workflow. Stop after producing the Spec Interpretation and readiness assessment.

