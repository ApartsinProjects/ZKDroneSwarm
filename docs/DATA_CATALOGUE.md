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
| 19 | C12 dynamic target onboarding | results/pilots/c12_onboard_20260522_110552.json | per-seed CF & Tabular skill on NEW targets vs #probes (Theta(d) vs Theta(m)) | 5 | 2026-05-22 |
| 20 | C11 unseen-pair under masking (FIXED stateUniq) | results/pilots/c11_masking_20260522_111142.json | per-seed overall/unseen skill + corrected stateUniq vs rho | 5 | 2026-05-22 |
| 20 | C11 (superseded; buggy stateUniq) | results/pilots/c11_masking_20260522_110703.json | overall/unseen skill VALID; stateUniq buggy -> superseded | 5 | 2026-05-22 |
| 21 | C13 unseen-pair vs true rank | results/pilots/c13_rank_unseen_20260522_111736.json | per-seed Tabular/CF unseen + CF overall vs true d (D3 support) | 5 | 2026-05-22 |
| 23 | C14 method bake-off vs competitors (masked regime) | results/pilots/c14_compare_20260522_131827.json | per-seed overall+unseen skill + stateUniq for 8 methods x 3 rho (Random, UCBIndep, UCBHomo, Tabular, MFSGD, ESTR, RewardCF, BothCF); fair d_hat=8 | 5 | 2026-05-22 |
| 24 | C14b EXTENDED bake-off (+PTF +BPMF) | results/pilots/c14_compare_20260522_132640.json | per-seed overall+unseen skill + stateUniq for 10 methods x 3 rho (adds PTF, BPMF); fair d_hat=8; supersedes cycle-23 subset | 5 | 2026-05-22 |
| 25 | C15 masking-robustness crossover (fine rho) | results/pilots/c15_crossover_20260522_133526.json | per-seed overall+unseen skill + stateUniq, 7 methods x 8 rho {1.0..0.1}; fair d_hat=8; -> docs/figures/F5_crossover.png | 8 | 2026-05-22 |
| 26 | C16 anytime cumulative-reward AUC | results/pilots/c16_anytime_20260522_134648.json | per-seed cumulative-normalized skill TRAJECTORY (per round) for 10 methods x rho{1.0,0.25}; operational metric -> docs/figures/F6_anytime.png | 8 | 2026-05-22 |
| 27 | E9 crossover + HybridCF | results/pilots/c15_crossover_20260522_145626.json | unseen+overall skill, 8 methods incl HybridCF x 8 rho; -> F5_crossover.png (supersedes cycle-25) | 8 | 2026-05-22 |
| 27 | E9 anytime + HybridCF/ChoiceCF | results/pilots/c16_anytime_20260522_145857.json | anytime trajectories, 12 methods incl HybridCF/ChoiceCF x rho{1.0,0.25}; -> F6_anytime.png (supersedes cycle-26) | 8 | 2026-05-22 |

NOTE: cycles 1-17 predate structured saving; their SUMMARY tables are in
PROJECT_LOG.md and full stdout is in the (ephemeral) task-output files. Complete
structured saving starts at cycle 18. If a pre-catalogue result becomes a paper
headline, RE-RUN it with save_results() to capture complete data.
