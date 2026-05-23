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
| 41 | P1-6 consolidated method ablation | results/pilots/ablation12_20260522_214411.json | unseen+anytime for ActiveCFconv/HybridCFconv/RewardCFconv/RewardCFconv_noprec/PTF/ESTR x rho{1.0,0.25} + d_hat sweep {2,5,8,12,20}; FINDING: precision-off best unseen (0.584); -> docs/ABLATION_TABLE.md | 12 | 2026-05-22 |
| 41b | Precision on/off vs sigma_obs crossover | results/pilots/precision_sweep_20260523_034915.json | unseen+anytime, precision ON vs OFF (uniform) x sigma_obs{0.1..2.0} at rho=1.0; uniform wins unseen at ALL noise, precision edges anytime only at sigma>=1.0; -> docs/PRECISION_SWEEP.md | 12 | 2026-05-23 |
| 41c | Confidence-mechanism bake-off (incl EM/Bayesian) | results/pilots/confidence8_20260523_072729.json | unseen+anytime for uniform/full/relcap4/EM/EMshrink x (rho,sigma){4 conds}; FINDING: EM (variational Bayesian factorization w/ predictive-interval UCB) DOMINATES uniform, EMshrink best-unseen Pareto; -> docs/CONFIDENCE.md | 6 | 2026-05-23 |
| 42 | P1-8 contention (capacity-1 matching) | results/pilots/contention8_20260523_080621.json | earned-reward (Hungarian-normalized) + contention-free unseen + collision rate for ActiveCFconv/RewardCFconv/PTF/UCBIndep/Random x pool{240,60,30,15}; FINDINGS: categorical unseen survives (RewardCF 0.29-0.38 vs UCB ~0); operational gap narrows under severe contention; ActiveCF count-bonus BACKFIRES (synchronizes probes) under shared pool; -> docs/CONTENTION.md | 8 | 2026-05-23 |
| 43 | ChoiceEM (choice-informativeness joint EM) + rescue | results/pilots/choiceem8_20260523_103500.json | unseen+anytime for RewardCF/ChoiceCF/ChoiceEM(g0=.5)/ChoiceEM-rescue x sigma_obs{0.6,1.0,2.0}; HONEST NEGATIVE on skill: naive EM deadlocks (unseen 0.012, anytime 0.163); rescue (g0=0.1, warm_em=0.3) FIXES deadlock (anytime 0.163->0.217 ties ChoiceCF 0.219), confirming root cause, but learned gate gives no edge over fixed ramp (unseen 0.031<0.093). POSITIVE NICHE: at sigma_obs=2.0 noise-immune ChoiceCF beats RewardCF on BOTH unseen (0.093 vs 0.042) and anytime (0.219 vs 0.179), non-overlapping CIs; -> docs/CHOICEEM.md | 8 | 2026-05-23 |
| 44 | Contention WIN: +ContentionCF (fixed-offset) | results/pilots/contention8_20260523_090113.json | adds ContentionCF (RewardCF + fixed private per-target offset) to the pool sweep; WINS at severe contention (pool=15: 0.105 [0.096,0.114] vs ~0.05 argmax-CF/PTF, non-overlapping, ~2x) and pool=30; regime-dependent. De-confliction needs PRIVATE FIXED randomness (softmax/shared-signal backfire). -> docs/CONTENTION.md | 8 | 2026-05-23 |
| 45 | ARD rank self-determination (C7) | results/pilots/ard8_20260523_095245.json | RewardCF/EMCF/ARD-EMCF x d_hat{8,20}; ARD-EMCF recovers stable eff rank ~3.2 independent of d_hat, no overfit, anytime improves (0.46 vs 0.42); prunes to identified ~3 dims (< true d=5). -> docs/ARD.md | 8 | 2026-05-23 |
| 46 | c=n candidate-set independence | results/pilots/candset8_*.json | RewardCF/Tabular unseen vs training candidate size c_train{20,60,120,240=n}, fixed eval offer 20; CF stays 0.34-0.44 at ALL c incl c=n, Tabular ~0; categorical result is candidate-size-independent. -> docs/CANDSET.md | 8 | 2026-05-23 |
| 47 | H1 info-directed exploration (sample eff) | results/pilots/explore8_20260523_100338.json | anytime trajectory for eps-greedy/count-bonus/posterior-UCB(b=1)/collective-UCB(b=1)/collective(b=0.3) at rho=0.25; big-beta UCB OVER-explores; collective b=0.3 best FINAL anytime (0.356 vs count 0.333); count fastest early. -> docs/EXPLORE.md | 8 | 2026-05-23 |
| 48 | H2 adaptive ContentionAdaCF (self-tuning offset) | results/pilots/contention8_20260523_113229.json | adds ContentionAdaCF (offset scaled by own loss-rate, hard ZK scarcity gate offer<=4m) to the pool sweep. EXTENDS the contention win from pool=15 alone to pool<=60: beats fixed ContentionCF at pool=30 (0.153 vs 0.134) and pool=60 (0.205 vs 0.178), ties at pool=15 (0.100 vs 0.105); all no-offset methods <=0.06 at pool=15. HONEST LIMIT: at pool=240 (no contention) offset policies trail plain CF (0.25 vs 0.44) since they drop eps-exploration; fix = eps-greedy fallback when gate off (queued). -> docs/CONTENTION.md | 8 | 2026-05-23 |
| 49 | H9 ChoiceEM held-out gamma + heterogeneous-teammate SANITY | results/pilots/choicehetero8_20260523_115216.json | good learners + ORACLE or RANDOM special teammates (frac 0/33/50%), choice-only. SANITY PASSES for predictive (held-out) gamma: gamma(oracle) 0.49 >> gamma(random) 0.10 (~5x); in-sample WRONGLY inflates gamma(random) to 0.70 (Prop 9). ChoiceEM-pred > in-sample ChoiceEM on unseen+anytime everywhere; more robust than ramp at 50% random (0.033 vs 0.015). BOOTSTRAPPING: good-drone unseen leaps 0.089->0.55 with 50% oracle teammates (knowledge propagates via choices, ZK). HONEST: with mostly-good teammates the full-trust ramp matches/beats held-out conservatism (oracle 0.55 vs 0.49); reward channel beats all choice methods (oracle 0.62). -> docs/CHOICEHETERO.md | 8 | 2026-05-23 |

NOTE: cycles 1-17 predate structured saving; their SUMMARY tables are in
PROJECT_LOG.md and full stdout is in the (ephemeral) task-output files. Complete
structured saving starts at cycle 18. If a pre-catalogue result becomes a paper
headline, RE-RUN it with save_results() to capture complete data.
