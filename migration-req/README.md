# TabulaDrone Viewer Migration Requirements

This folder defines the high-level requirements for migrating the current Python-based TabulaDrone viewer to:

- `viewer-ng` for the Angular frontend
- `viewer-server` for the backend/API layer

This first pass is intentionally scoped to episodic mode only.

## Reading Order

1. `01-goals-non-goals.md`
2. `02-current-viewer-inventory.md`
3. `03-functional-requirements.md`
4. `04-visual-parity-requirements.md`
5. `05-data-contracts.md`
6. `06-api-requirements.md`
7. `07-frontend-architecture.md`
8. `08-backend-architecture.md`
9. `09-migration-phases.md`
10. `10-acceptance-criteria.md`
11. `11-risks-open-questions.md`

## Current Status

- Migration target confirmed feasible
- Phase 1 scope confirmed: episodic mode only
- Requirement set established at a high level
- Implementation planning not yet finalized

## Document Rules

- These documents define migration intent and constraints, not final implementation details.
- Where a technology choice is still open, the document should state the decision boundary rather than invent a final answer.
- If current behavior is unclear, prefer documenting the open question in `11-risks-open-questions.md`.
