# Project Log: Confidence-Aware Decentralized Collaborative Filtering (ZK-MRTA)

**Goal**: find a CLEAR, relatively SIMPLE framing/method for decentralized
multi-robot task allocation (via collaborative filtering) that beats all
baselines and is groundbreaking; accept a more complex extension only if no
simple option works. Updated after every cycle. Commit per cycle (revertible).

## Experimental harness (synthetic pilots)
- Low-rank world: m drones, n targets, rank d; reward R = P Uᵀ. Structures:
  one-hot modes, cluster_gauss (random cluster centers), gauss (continuous),
  optional reward sharpening (raises effective rank).
- Sample-starved, CHANGING candidate subsets: each step every drone is offered a
  random subset (size `cand`) and picks one; T pulls/drone < n (camping
  impossible -> generalisation required). Repeated assignment.
- Two-stage noise: own reward observed with sigma_own (clean); teammates'
  rewards observed with sigma_obs (noisy); CHOICES observed cleanly.
- Reward-sharing prob p (1 = rewards broadcast, 0 = decision-only).
- Metric: skill = (greedy_final_model - random)/(oracle - random) in [0,1];
  also online reward/step. greedy = exploit final model over fresh subsets.
- Policies: Tabular (own-row memorisation), RewardCF/MF (pool rewards),
  ChoiceCF (pool choices, BPR/implicit), DualConf (competence-weighted choices),
  DualEM (mixture EM), RewardCFRobust (per-teammate residual trust).
- Files: experiments/pilot_identifiability.py, pilot_choice_only.py,
  pilot_refit.py, pilot_structure.py, pilot_rank.py, pilot_starvation.py,
  pilot_bootstrap.py, pilot_em.py, pilot_confirm.py, pilot_noise.py,
  pilot_trust.py. Python: C:/Python314/python.exe.

## Experiments and results (chronological)

| # | experiment | key result |
|---|---|---|
| 1 | identifiability gate (offline, choices) | U IS recoverable from choices; at low data choices BEAT rewards (greedy 0.448 vs 0.356); choice recovery saturates ~0.45 (≈44% of oracle gap) |
| 2 | online single-pass RLS | conjugate core fixed RECOVERY (Urec 0.06->0.25) but lost to tabular (single-pass underfits bilinear) |
| 3 | warm-started batch refit (ALS) | **GATE PASSED**: RewardMF@p=1 greedy 0.687 vs Tabular 0.485 (+42%), Urec 0.639. Reward-observable CF win is SOLID |
| 4 | decision-only choices online | naive folding + separated + entangled all fail (bootstrap deadlock; corruption) |
| 5 | 2D phase diagram (starvation x sharing) | CF/Tab non-monotonic, peak +51% at ~21% coverage; at p=0 CF/Tab~0.9 flat (reward-sharing NECESSARY) |
| 6 | latent structure sweep | CF wins for ALL geometries (gap 0.10-0.34); more clusters help; reward NONLINEARITY (raises rank) collapses advantage |
| 7 | true-rank sweep | low-rank LOAD-BEARING: gap ~0 at full rank (d=30) AND at d=1 (no personalisation); sweet spot d~3-5 |
| 8 | competence-weighted bootstrap (DualConf) | first p=0 "win" +8% at n=120 (3 seeds) -- later shown to be noise |
| 9 | principled EM (mixture) | EM works at p=1 but FAILS at p=0 (model-agreement responsibility reintroduces deadlock). Lesson: infer competence from BEHAVIOUR not model-agreement |
| 10 | 5-seed confirmation | reward-observable ROCK-SOLID (CF/Tab 1.44-1.89). decision-only "win" EVAPORATES to parity (was a 3-seed fluke) |
| 11 | apples-to-apples channel (clean choices vs noisy rewards) | choice channel FLAT 0.423 vs noise; reward channel 0.728->0.368 (drops BELOW solo at sigma_obs=2). decisions NOISE-IMMUNE; crossover at high noise |
| 12 | per-teammate trust w/ faulty teammates | CHOICE-trust WORKS: Ch_comp robust, gap grows w/ faulty% (+0.159->+0.180). REWARD-trust = BUG (override bypassed). |
| 13 | P1 collaboration-harm threshold | naive pooling drops BELOW solo (reward@50%, choice always); competence-weighted CHOICE pooling robust to 50% faulty, stays ABOVE solo (0.435->0.419). reward-trust fix works (gain grows w/ faulty%) but only parity at 50% |
| 14 | MF approximation audit (RUNNING) | convergence/regularization check: are CF results understated by under-converged ALS? |
| 15 | RANSAC/consensus robust factorization (planned) | seed-grow consensus anchored on self, absolute residual threshold -> push robustness past 50% faulty; + trust-weighted hybrid |

### Headline result tables
**Cycle 5 (phase diagram, CF/Tab greedy ratio, 5 seeds):**
n=30:1.16, n=60:1.28, n=120:1.38, n=240:1.51(peak), n=480:1.33 at p=1; ~0.9 flat at p=0.

**Cycle 7 (true-rank, CF-Tab skill gap):**
d=1:0.04, d=2:0.29, d=3:0.33, d=5:0.32, d=8:0.09, d=15:0.14, d=30:0.01.

**Cycle 11 (noise, skill):** Tabular 0.395 flat; RewardCF 0.728/0.591/0.491/0.451/0.368
(sigma_obs 0.1-2.0); ChoiceCF_comp 0.423 flat. Crossover ~sigma_obs 1.5-2.

**Cycle 12 (faulty%, skill):** Ch_naive 0.276/0.251/0.225; Ch_comp 0.435/0.424/0.405
(faulty 0/20/40%). C-gain +0.159/+0.173/+0.180.

**Cycle 13 = P1 (collaboration-harm threshold, 5 seeds, SOLO=0.405-0.414):**
| faulty% | RewardCF | RewRobust | Ch_naive | Ch_comp |
|---|---|---|---|---|
| 0  | 0.636 | 0.653 | 0.276< | 0.435 |
| 20 | 0.527 | 0.545 | 0.251< | 0.424 |
| 35 | 0.451 | 0.490 | 0.230< | 0.423 |
| 50 | 0.335< | 0.404< | 0.265< | 0.419 |
(< = below solo). SANITY: solo flat (ignores teammates) OK; Ch_naive low even at
0% faulty because cold-start exploratory choices corrupt unweighted pooling
(consistent with bootstrap finding); RewRobust now > RewardCF (bug fixed),
gain grows with faulty% (+0.017->+0.069). FINDINGS: (1) naive collaboration is
HARMFUL (below solo); (2) competence-weighted CHOICE pooling robust to 50%
faulty, stays ABOVE solo, best method at 50%; (3) reward-trust limited at 50%
(in-range garbage hard to detect) -> decisions make unreliability more
DETECTABLE than outcomes. TO IMPROVE: RANSAC/consensus to push past 50%;
trust-weighted HYBRID (rewards when reliable, choices when not) to dominate
across the range; clearer-outlier faulty rewards to make reward-trust test fair.

## Key findings (consolidated)
1. **Reward-observable CF beats tabular** (+44-89%, robust) IFF: reward is
   low-rank-but-personalised (1<d<<min(m,n)), regime sample-starved with changing
   availability, AND rewards shared. Non-monotonic sweet spots in starvation and
   rank. Explains all prior negatives (Sets M/N were sample-rich).
2. **Decision-only choices ≈ tabular parity** (capped by choice-recovery ceiling
   ≈ tabular). Not a standalone win.
3. **Competence must be inferred from BEHAVIOUR** (consistency), not
   model-agreement (which deadlocks).
4. **Decisions are NOISE-IMMUNE**: under high observation noise, the clean choice
   channel beats the noisy reward channel; naive reward-pooling can fall below
   solo.
5. **Per-teammate trust (choice side) works**: competence weighting is robust to
   faulty teammates; advantage grows with faulty fraction.

## Current standing & next (the groundbreaking bet)
THESIS: naive collaboration is fragile (worse than solo past a threshold of
unreliable peers); inferring per-agent, per-decision CONFIDENCE makes
collaboration SAFE and robust. See docs/EXPERIMENT_PROGRAM.md (P1-P6).
- P1 (running): collaboration-harm threshold. P2: latent-confidence EM/VB. P3:
  adversarial + theory. P4-P6: noise channel, fusion, cold-start.
