# Risks and Open Questions

## Risks

## Contract Drift Risk

If the backend exposes raw artifact shapes directly, the Angular app may become tightly coupled to current log-file conventions and become hard to evolve.

## Scope Creep Risk

If continuous mode or live monitoring leaks into phase 1, the migration will become substantially larger and slower.

## Visual Over-Commit Risk

If the project treats Matplotlib pixel parity as mandatory, implementation complexity will rise without proportional analytical value.

## Analysis Parity Risk

Radar and Training Path behavior depend on episodic learning-state payloads and comparison logic. If these shapes are not normalized early, frontend complexity will grow quickly.

## Open Questions

- Which backend framework should power `tabula_server`?
- Which charting library should power Radar, Results, and Training Path visuals?
- Should the map be rendered with SVG, Canvas, or another browser-native approach?
- How much visual modernization is acceptable relative to the current Python viewer?
- Should episode detail and learning-state data be loaded through one endpoint or multiple endpoints?
- Should the backend precompute chart-ready derived values or should the frontend compute some of them?

## Current Recommendation

Keep these questions open during the requirement phase unless a decision is needed to unblock architecture or planning.
