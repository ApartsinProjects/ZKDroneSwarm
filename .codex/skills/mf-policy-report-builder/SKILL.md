---
name: mf-policy-report-builder
description: Generate an MF final report or deterministic policy_report.json for a completed ZK-MRTA matrix_factorization_cf run, defaulting to the latest logs/run_* directory and comparing the MF best episode against same-run baselines.
---

# MF Final Report Builder

Use this skill when the user asks to generate an MF final report or wants a deterministic post-execution report for a completed ZK-MRTA run.

## Goal

Generate a canonical `policy_report.json` from existing run artifacts without rerunning the simulation or modifying source logs.

## When to Use It

- The user says something like `generate an mf final report`
- The run artifacts already exist under `logs/run_*`
- The report should focus on `matrix_factorization_cf`
- The user wants MF learning progression plus comparison against available same-run baselines
- The output needs to be machine-readable and stable for dashboards, plots, or paper figures

## Fixed Requirements

- `matrix_factorization_cf` is required as the primary policy
- If no run is provided, use the most recent `logs/run_*`
- Use `matrix_factorization_cf/episodes_summary.json` as the authoritative source for `best_episode_path`
- Compare baselines against the MF best episode, not the MF final episode
- Treat MF final episode as learning-state context only
- Prefer episode `metrics` already stored in artifacts over recomputing them
- Record warnings and errors for missing or inconsistent artifacts
- Do not describe MF as centralized or parameter-sharing

## Expected Inputs

- Run directory such as `logs/run_20260411_162952`, or nothing to use the latest run
- `environment.json`
- `matrix_factorization_cf/episodes_summary.json`
- MF episode files under `matrix_factorization_cf/episodes/`
- MF learning-state files under `matrix_factorization_cf/learning_state/` when present
- Optional same-run baseline folders such as `random` or `max_damage_oracle`

## Canonical Output

- `logs/run_<id>/policy_report.json`

## Workflow

1. Resolve the target run directory.
2. Validate that `matrix_factorization_cf` artifacts exist.
3. Load `environment.json`, MF episode artifacts, and `episodes_summary.json`.
4. Use `best_episode_path` as the main MF comparison anchor.
5. Load MF learning-state checkpoints when available.
6. Load same-run baselines when available.
7. Emit a stable `policy_report.json`.

## Implementation Note

Use the bundled builder script in this skill directory to generate the report.
