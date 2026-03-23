# Acceptance Criteria

## Purpose

This document defines how the migration will be judged complete for phase 1.

## Functional Acceptance

- A user can browse available runs, policies, and episodic episodes in the browser.
- A user can open an episodic episode and view its details without relying on local Python GUI execution.
- The web viewer supports Info, Results, Radar, and Training Path views when the required data exists.
- The map supports playback of episodic step data.
- Missing optional learning-state data does not prevent episode viewing.

## Data Acceptance

- The frontend uses stable API responses rather than direct filesystem assumptions.
- Episode and learning-state payloads are normalized consistently.
- Required image assets load correctly in the browser context.

## UX Acceptance

- Core episode workflows are understandable without terminal interaction.
- Empty, loading, and error states are explicit and readable.
- The viewer remains usable on standard desktop screen sizes.

## Analytical Acceptance

- A user can inspect the same episode-level facts currently available in the Python viewer.
- A user can replay target-state evolution through episode steps.
- A user can inspect predicted-reward and latent-space information for episodic CF episodes.

## Validation Samples

At minimum, acceptance should be checked against:

- one non-CF episodic policy episode
- one CF episodic policy episode with learning-state data
- one episode that lacks optional analysis data

## Explicit Phase 1 Exclusion

No acceptance criterion in this phase requires continuous-mode support.
