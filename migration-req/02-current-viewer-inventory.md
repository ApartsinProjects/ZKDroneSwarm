# Current Viewer Inventory

## Purpose

This document inventories the behavior of the current Python viewer so migration work can target parity deliberately instead of reinterpreting the product from scratch.

## Current Entry Point

The current viewer is launched from the command line and:

- discovers the latest run folder under `logs/run_*`
- finds episode files grouped by policy
- loads a selected episode JSON file
- loads related learning-state files when present
- opens a Matplotlib-based split-panel UI

## Current User Workflow

1. Open the latest or a specific episodic artifact
2. View the world map and episode metadata
3. Navigate between discovered episodes
4. Replay episode steps on the map
5. Switch between analysis tabs
6. Inspect collaborative-filtering outputs when learning-state data exists

## Current Functional Areas

## Episode Discovery

- Find the latest `run_*` folder
- Enumerate policy folders within that run
- Enumerate episode JSON files within each policy
- Sort episodes by step count in descending order inside each policy

## Map Area

- Render world bounds and background image
- Render drones and targets using image assets when present
- Display target HP labels
- Update target state during playback
- Show engagement overlays during step playback
- Provide previous and next episode navigation
- Provide play and pause controls for step replay

## Info Tab

- Show drone count and target count
- Show total episode count when present
- Show weapon damage profile table
- Show target class attribute table

## Results Tab

- Show outcome and termination reason
- Show total steps, targets destroyed, and total ammo used
- Show per-drone rewards
- Show HP and active-target history chart
- Provide overflow handling for long reward lists

## Radar Tab

- Show top-K predicted rewards for a selected agent
- Support switching selected agent
- Use learning-state data from episodic CF artifacts
- Color target labels by target class

## Training Path Tab

- Show latent-space positions for agent and targets
- Support switching selected agent
- Compare episode 1 state with current episode state
- Show target labels and highlighted best-target overlays

## Current Limitations and Quirks

- The viewer is tightly coupled to Matplotlib layout primitives
- The viewer reads files directly from the filesystem rather than through an API
- Learning-state loading is path-convention based
- Continuous-mode artifacts follow a different structure and are not in scope for phase 1
- Some panel behavior assumes specific episodic learning-state shapes

## Migration Implication

The current viewer is behaviorally rich but structurally favorable for migration because data loading and rendering are already separated into distinct modules.
