# Migration Phases

## Purpose

This document defines a practical migration order that supports incremental delivery and validation.

## Phase 1

Establish requirements and architecture baseline.

Outputs:

- migration requirement documents
- agreed scope and acceptance criteria

## Phase 2

Build backend run and episode discovery.

Outputs:

- run list API
- policy list API
- episode list API

## Phase 3

Build episode detail loading for episodic mode.

Outputs:

- episode detail API
- normalized replay-step payload
- learning-state loading API

## Phase 4

Build Angular shell and navigation.

Outputs:

- app layout
- run and policy selection
- episode selection flow

## Phase 5

Implement Info and Results parity.

Outputs:

- Info tab
- Results tab
- summary metrics and charts

## Phase 6

Implement map playback parity.

Outputs:

- world map rendering
- playback controls
- engagement overlays

## Phase 7

Implement CF analysis parity.

Outputs:

- Radar tab
- Training Path tab
- episodic episode-1 comparison support

## Phase 8

Perform parity pass and stabilization.

Outputs:

- UX cleanup
- bug fixes
- acceptance verification

## Sequencing Rule

Backend contract stabilization must happen before deep frontend feature implementation so the Angular app is not built on unstable assumptions.
