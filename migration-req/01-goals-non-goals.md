# Goals and Non-Goals

## Goal

Migrate the current Python-based TabulaDrone episode viewer into a web-based system with:

- an Angular frontend in `viewer-ng`
- a backend/API service in `tabula_server`

The migrated system must preserve the current viewer's core functionality and analytical value for episodic runs.

## Primary Goals

- Support episodic-mode browsing and visualization from generated log artifacts
- Preserve the current viewer's functional behavior for map playback and analysis panels
- Establish a clean backend contract so the frontend does not depend on raw filesystem layout
- Make the new viewer easier to extend and maintain than the current Matplotlib-based UI

## Secondary Goals

- Improve discoverability and navigation across runs, policies, and episodes
- Make the viewer usable in a browser without requiring a local Python GUI session
- Create a foundation for future enhancements without requiring another full rewrite

## Non-Goals For Phase 1

- Continuous-mode visualization
- Live monitoring of a running simulation
- Editing scenario configuration from the viewer
- Triggering or controlling simulation runs from the viewer
- Authentication, authorization, or multi-user collaboration
- Pixel-perfect recreation of Matplotlib layout behavior

## Scope Boundary

Phase 1 is complete when the web viewer can replace the current Python viewer for episodic analysis workflows that depend on:

- episode map playback
- run/policy/episode browsing
- Info tab
- Results tab
- Radar tab
- Training Path tab

## Success Definition

The migration is successful if a user who currently uses `python -m viewer show` can perform the same episodic analysis tasks in the web viewer with no meaningful loss of information.
