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
| 28 | E3 two-channel grid (rho x sigma_obs) | results/pilots/e3_channels_20260522_153527.json | overall/unseen/anytime skill, 5 methods x 3 rho x 5 sigma_obs; both channels masked consistently; -> F7_channels.png | 8 | 2026-05-22 |
| 29 | E12 persistent vs iid masking (Theorem 4) | results/pilots/e12_iid_masking_20260522_154506.json | rawA: unseen/anytime/uniq vs rho for 6 methods x 2 modes; rawB: uniq vs T (RewardCF); -> F8_iid_vs_persistent.png | 8 | 2026-05-22 |
| 30 | E13 choice-only ablation + strict-ZK | results/pilots/e13_choice_20260522_155023.json | unseen+anytime vs rho for Tabular/ChoiceCF/ChoiceZK/RewardCF/BothCF; choice-channel value + strict-ZK robustness | 8 | 2026-05-22 |
| 31 | E2/E4/E6 scaling sweeps | results/pilots/e246_scaling_20260522_160250.json | unseen+anytime vs true d / horizon T / targets n / guessed d_hat, 5 methods; -> F9_scaling.png | 8 | 2026-05-22 |
| 32 | E7 newcomer cold-start | results/pilots/e7_newcomer_20260522_160904.json | newcomer skill on unseen vs #own probes; CF foldin (shrunk to pop prior) vs Tabular vs popularity; -> F10_newcomer.png | 10 | 2026-05-22 |
| 33 | E10 fusion (precision-gated + stacked) | results/pilots/e10_precgate_20260522_162536.json | overall skill vs sigma_obs for RewardCF/ChoiceCF/BothCF/BothCFPrec (+StackCF smoke); fusion-dominance attempts | 8 | 2026-05-22 |
| 34 | Converged config dominance vs PTF | results/pilots/conv_confirm_20260522_172716.json | unseen+anytime for RewardCF/HybridCF/HybridCFconv/PTF x 3 rho; paired HybridCFconv-PTF bootstrap CIs (ties rho=1 unseen, wins else) | 10 | 2026-05-22 |
| 35 | E8 active exploration confirmation | results/pilots/e8_active_20260522_182950.json | unseen+anytime for RewardCF/HybridCFconv/ActiveCFconv/PTF x 3 rho; paired ActiveCFconv-RewardCF bootstrap CIs (active exploration dominates eps-greedy) | 12 | 2026-05-22 |
| 36 | E14 generality (m, K, within) | results/pilots/e14_robust_20260522_184041.json | unseen+anytime vs drones m / clusters K / latent spread within; confirms conclusions general (incl K=m no-clustering) | 8 | 2026-05-22 |
| 37 | E15 broader baselines (SoftImpute/kNN-CF/BiasModel) | results/pilots/e15_morebase_20260522_191327.json | unseen+anytime vs rho for 3 new fair baselines + PTF + ours; paired CIs; -> F12_morebaselines.png | 8 | 2026-05-22 |
| 38 | Real tabula_drone simulator validation | results/pilots/tabula_bench_real.json | per-seed skill + per-episode learning curves for random/oracle/ucb_indep/mf/weighted_als in the real PettingZoo env; -> F13_realsim.png | 3 | 2026-05-22 |
| 39 | Assumption stress (approx low-rank + nonlinear) | results/pilots/stress_assump_20260522_200547.json | unseen+anytime+effective-rank vs nonlin/approx for 5 methods; graceful degradation; -> F14_stress.png | 8 | 2026-05-22 |
| 40 | P1-4 20-seed headline bootstrap CIs | results/pilots/headline20_20260522_210110.json | per-seed unseen+anytime for UCBIndep/PTF/RewardCF/HybridCFconv/ActiveCFconv x rho{1.0,0.25}; bootstrap 95% CI; -> docs/HEADLINE_TABLE.md | 20 | 2026-05-22 |

NOTE: cycles 1-17 predate structured saving; their SUMMARY tables are in
PROJECT_LOG.md and full stdout is in the (ephemeral) task-output files. Complete
structured saving starts at cycle 18. If a pre-catalogue result becomes a paper
headline, RE-RUN it with save_results() to capture complete data.
