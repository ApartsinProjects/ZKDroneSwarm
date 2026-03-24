# Frontend Architecture

## Purpose

This document defines the high-level architectural expectations for `viewer-ng`.

## Architectural Role

`viewer-ng` is the presentation and interaction layer for viewer workflows.

It should:

- request normalized data from `tabula_server`
- manage UI navigation and selection state
- render maps, charts, and analysis panels
- remain isolated from raw log-file structure

It should not:

- read raw filesystem paths
- duplicate backend normalization logic
- become the source of truth for episodic data contracts

## High-Level UI Areas

- run and policy selection
- episode list and episode selection
- episode detail page
- map playback area
- analysis tabs
- empty and error states

## Suggested Frontend Boundaries

- routing and page composition
- API client layer
- viewer state layer
- map rendering module
- chart rendering module
- analysis-tab components

## State Responsibilities

The frontend should manage:

- currently selected run
- currently selected policy
- currently selected episode
- current playback step
- currently selected agent in Radar and Training Path
- tab selection and page-level loading state

## Frontend Design Constraints

- Prefer declarative rendering over imperative widget-style layout management.
- Keep playback logic separate from rendering primitives.
- Make empty and missing-data states explicit in the UI.
- Avoid embedding backend path assumptions in components.

## Open Choice Areas

The following may remain undecided in this phase:

- exact Angular state management style
- exact charting library
- exact map rendering approach

The chosen approach must support the parity requirements and data contracts defined in this folder.
