# Visual Parity Requirements

## Purpose

This document defines what visual parity means for phase 1 so the migration preserves analytical usefulness without forcing a pixel-for-pixel clone of Matplotlib.

## Parity Standard

Phase 1 requires functional visual parity, not exact rendering parity.

Functional visual parity means:

- the same information is visible
- the same analysis tasks are possible
- the same interactions are supported
- differences in layout or style do not reduce interpretation quality

## Elements That Must Remain Recognizable

- split between map-focused visualization and analysis content
- drones and targets displayed in consistent world coordinates
- target HP visibility during playback
- engagement overlay visibility during playback
- per-tab separation of Info, Results, Radar, and Training Path concerns
- color distinction between target classes
- clear selection state for the active agent in Radar and Training Path views

## Elements That May Be Modernized

- overall page layout
- tab appearance
- typography
- spacing and alignment
- panel chrome and navigation styling
- popup behavior for overflow reward details

## Required Visual Behaviors

- The map must preserve world-coordinate relationships.
- Playback must visually communicate step progression.
- Destroyed targets must remain visually distinguishable from active targets.
- Charts must remain readable at typical browser window sizes.
- Missing optional data must show clear empty-state messaging.

## Responsive Expectations

- Desktop layout is the primary target.
- The interface must still load and remain usable on smaller screens.
- If panels cannot fit side-by-side on narrow viewports, the layout may reflow as long as information remains accessible.

## Explicit Non-Requirement

The browser version does not need to reproduce Matplotlib widget placement or exact figure geometry.
