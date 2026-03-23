# Data Contracts

## Purpose

This document defines the frontend-facing data contracts for phase 1. The backend may read raw files from `logs/`, but the frontend should consume normalized API responses.

## Design Rule

The Angular frontend must not depend directly on raw file naming conventions or folder traversal logic.

## Core Contract Types

## Run Summary

Represents one available run.

Required fields:

- `runId`
- `displayName`
- `createdAt` when derivable
- `policies`

## Policy Summary

Represents one policy within a run.

Required fields:

- `policyId`
- `policyType`
- `episodeCount`
- `availablePanels`

## Episode Summary

Represents one episode entry in a list view.

Required fields:

- `episodeId`
- `episodeNum`
- `label`
- `stepCount`
- `hasSummary`
- `hasLearningState`

Optional fields:

- `isBest`
- `isFirst`
- `isMid`

## Episode Detail

Represents the normalized payload required to render the main episode page.

Required fields:

- `episode`
- `scenario`
- `config`
- `derived`

Where:

- `episode` contains metadata such as version, episode number, seed, and timestamp
- `scenario` contains drones, targets, world size, and policy metadata needed for rendering
- `config` contains class and weapon mappings needed for tables and visual scaling
- `derived` contains summary-friendly and chart-friendly data prepared for the frontend

## Replay Step

Represents one playback step.

Required fields:

- `stepNum`
- `actions`
- `rewards`
- `targetHps`
- `targetActive`

Optional fields:

- `targetAttributes`
- `ammoUsed`
- `overkill`

## Learning State Detail

Represents episodic learning-state analysis for CF policies.

Required fields when present:

- `episodeNum`
- `policyType`
- `numAgents`
- `numTargets`
- `preEpisode`
- `postEpisode`
- `entities`

Normalization rules:

- `preEpisode` may be absent only if the backend explicitly marks it unavailable
- `postEpisode` may be absent only if the backend explicitly marks it unavailable
- the frontend must not infer episodic comparison state from filename conventions

## Agent Analysis Payload

Each agent analysis object should normalize:

- `agentIndex`
- `agentVector2d`
- `targetVectors2d`
- `match`
- `metadata`

Where `match` may include:

- `predictedRewards`
- `rankedTargets`
- `bestTarget`

## Asset Contract

The frontend needs stable URLs or API-backed paths for:

- background image
- drone image
- active target image
- burning target image if retained
- destroyed target image

## Contract Stability Rule

If raw log formats evolve later, `viewer-server` should absorb that change so `viewer-ng` can keep a stable contract.
