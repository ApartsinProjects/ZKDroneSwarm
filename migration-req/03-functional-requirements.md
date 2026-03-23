# Functional Requirements

## Product Requirement

The web-based viewer must support episodic analysis workflows currently handled by the Python viewer.

## Run and Episode Browsing

- The system must list available runs.
- The system must list policies within a selected run.
- The system must list episodes within a selected policy.
- The system must expose enough metadata to help the user choose an episode.
- The system must support opening a specific episode directly.

## Episode Loading

- The frontend must be able to request an episodic episode payload by identifier.
- The frontend must be able to request the associated episodic learning-state payload when available.
- The system must handle missing optional learning-state data without failing the whole episode view.

## Map Visualization

- The viewer must render the world area using episode scenario and config data.
- The viewer must render drones, targets, world boundaries, and labels.
- The viewer must replay per-step target HP changes.
- The viewer must render engagement overlays for actions during playback.
- The viewer must support play, pause, and step-position awareness.

## Episode Navigation

- The viewer must support moving to previous and next episodes within the selected episode list.
- The viewer must keep UI state coherent when the selected episode changes.

## Info Analysis

- The viewer must display drone and target counts.
- The viewer must display weapon damage profile information.
- The viewer must display target class attribute information.

## Results Analysis

- The viewer must display outcome and termination reason.
- The viewer must display core summary metrics.
- The viewer must display per-drone rewards.
- The viewer must display the HP-history and active-target-history chart.

## Radar Analysis

- The viewer must display predicted reward radar data when episodic learning-state data includes agent match information.
- The viewer must support changing the selected agent.
- The viewer must degrade gracefully when radar data is unavailable.

## Training Path Analysis

- The viewer must display latent-space visualization for episodic learning-state data.
- The viewer must support changing the selected agent.
- The viewer must compare current episode state against episode 1 when both are available.
- The viewer must degrade gracefully when comparison data is unavailable.

## Error Handling

- The system must provide a user-visible error state when a requested run, policy, or episode cannot be found.
- The system must distinguish between missing optional analysis data and broken required episode data.

## Phase 1 Functional Exclusions

- No continuous-mode artifact browsing
- No live streaming or websocket updates
- No episode editing
- No simulation controls
