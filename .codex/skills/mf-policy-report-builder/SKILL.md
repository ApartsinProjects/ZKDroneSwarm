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

### Report Structure (v0.2.0)

The generated report includes:

- **`metric_definitions`**: Canonical definitions for all metrics with formulas, sources, direction, and category
  - Covers all raw metrics: `steps`, `targets_neutralized`, `total_ammo_used`, `total_collisions`, `total_overkill`, `total_net_damage`, `total_gross_damage`, `total_latent_mismatch`
  - Covers derived metrics: `shots_per_target`, `avg_latent_match_quality`, `total_reward`, `mean_reward_per_agent`, `success`
  - Each definition includes: `description`, `formula`, `source`, `direction` (higher/lower_is_better), `category`

- **`episode_curves`**: Per-episode performance with trend analysis
  - All episode metrics from `episode_*.json` files
  - `trend_summary`: Direction, monotonicity, plateau detection for key metrics

- **`policy_summary`**: Best episode + aggregate training metrics
  - Includes: `avg_total_collisions`, `avg_total_overkill`, `avg_total_reward`, `avg_mean_reward_per_agent`

- **`learning_summary`**: Learning progression and convergence
  - `convergence_assessment`: `best_is_final`, `best_episode`, `convergence_status` (potentially_undertrained / early_peak / converged)
  - Epsilon progression and checkpoints

- **`comparison_vs_baseline`**: MF best episode vs baselines
  - Each metric tagged with `category` (efficiency / precision / coordination / task_completion / reward)
  - `category_directions`: Per-category performance breakdown (e.g., efficiency: improved, precision: worsened)
  - `overall_label`: Mixed / improved / worsened

- **`key_findings`**: Auto-surfaced actionable insights
  - Cross-baseline regressions (metrics worsened vs all baselines)
  - Active degradation (non-plateaued negative trends)
  - Convergence warnings
  - Target crowding (>40% agents targeting same best target)

- **`report_focus`**: Baseline roles explicitly labeled (`control` / `ceiling`)

## Workflow

1. Resolve the target run directory.
2. Validate that `matrix_factorization_cf` artifacts exist.
3. Load `environment.json`, MF episode artifacts, and `episodes_summary.json`.
4. Use `best_episode_path` as the main MF comparison anchor.
5. Load MF learning-state checkpoints when available.
6. Load same-run baselines when available.
7. Build metric definitions, trend summaries, convergence assessments, and key findings.
8. Emit a stable `policy_report.json`.

## Implementation Note

Use the bundled builder script in this skill directory to generate the report.
