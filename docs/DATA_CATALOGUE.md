# Data Catalogue

Registry of COMPLETE experimental data (not just summaries) so every run can be
re-analyzed without re-running. Connected to PROJECT_LOG.md by cycle number.

## Convention
- Each experiment saves FULL per-seed / per-config results (and per-round
  trajectories where relevant) to `results/pilots/<name>_<UTCstamp>.json` via
  `experiments/_results_io.py:save_results()`.
- The JSON includes `meta` (params, config grid, seeds, metric defs, date, git
  commit) and `raw` (per-seed metric VALUES, not just means).
- After each experiment: add a row below AND log the cycle in PROJECT_LOG.md.
- Never rely on stdout/scrollback alone for headline results.

## Catalogue
| cycle | experiment | data file | metric(s) | seeds | date |
|---|---|---|---|---|---|
| 1-17 | pre-catalogue pilots | (none; stdout only) | summary tables in PROJECT_LOG.md | 3-5 | 2026-05-22 |
| 18 | C8 unseen-pair generalization (FAIR, d_hat=8) | results/pilots/c8_generalize_20260522_104739.json | per-seed overall + unseen-pair skill, Tabular/RewardCF/BothCF | 5 | 2026-05-22 |
| 18 | C8 (superseded oracle-rank d=5) | results/pilots/c8_generalize_20260522_104053.json | same metrics, used TRUE rank (unfair) -> superseded by fair run | 5 | 2026-05-22 |

NOTE: cycles 1-17 predate structured saving; their SUMMARY tables are in
PROJECT_LOG.md and full stdout is in the (ephemeral) task-output files. Complete
structured saving starts at cycle 18. If a pre-catalogue result becomes a paper
headline, RE-RUN it with save_results() to capture complete data.
