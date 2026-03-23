# Backend Architecture

## Purpose

This document defines the high-level architectural expectations for `viewer-server`.

## Architectural Role

`viewer-server` is the adapter layer between raw TabulaDrone artifacts and the Angular frontend.

It should:

- discover available runs, policies, and episodes
- read episodic episode and learning-state artifacts from disk
- normalize raw payloads into stable frontend DTOs
- expose API endpoints for the viewer

It should not:

- embed simulation logic
- re-run policies or environments
- push raw filesystem details into the frontend contract

## Input Sources

Phase 1 backend inputs are:

- episodic episode JSON files
- episodic learning-state JSON files
- image assets used by the current viewer

## Backend Responsibilities

- filesystem discovery
- resource identification
- payload loading
- normalization and derivation
- error translation
- static asset exposure

## Normalization Rules

- The backend should reconcile current filename and directory conventions into stable identifiers.
- The backend should prepare frontend-friendly shapes for tables, charts, and playback.
- The backend should explicitly mark optional analysis data as unavailable rather than forcing the frontend to infer absence.

## Error Handling Expectations

- distinguish resource-not-found from malformed-data cases
- isolate one broken artifact without collapsing the entire run browser
- surface enough diagnostic detail for development without making the API noisy for users

## Phase 1 Constraint

The backend may be implemented with a simple architecture as long as it preserves contract stability and keeps episodic mode reliable.
