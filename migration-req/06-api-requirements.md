# API Requirements

## Purpose

This document defines the minimum backend API responsibilities for `viewer-server` in phase 1.

## General Requirements

- The API must support episodic mode only in phase 1.
- The API must treat `logs/` artifacts as backend implementation detail.
- The API must return normalized JSON payloads for frontend consumption.
- The API must provide clear error responses for missing resources and invalid requests.

## Required Capabilities

## Run Discovery

The API must provide a way to:

- list available runs
- identify policies within a run
- return enough metadata for frontend navigation

## Episode Discovery

The API must provide a way to:

- list episodes for a given run and policy
- identify which episodes have summary and learning-state data
- expose a stable frontend identifier for each episode

## Episode Detail Loading

The API must provide a way to:

- load one normalized episodic episode detail payload
- load replay steps needed by the map and results views
- load summary and derived chart data

## Learning-State Loading

The API must provide a way to:

- load the episodic learning-state payload for the selected episode
- load the episode 1 learning-state payload when comparison is needed
- report absence of learning-state data cleanly

## Asset Serving

The API or static asset layer must provide access to image assets required by the viewer.

## Non-Functional API Expectations

- Episode load time should be acceptable for typical episodic payload sizes.
- The API should avoid forcing the frontend to make excessive chained requests for a single episode page.
- The API should keep response structure stable across frontend iterations.

## Phase 1 API Exclusions

- No live update endpoints
- No simulation-control endpoints
- No mutation endpoints
- No continuous-mode endpoints
