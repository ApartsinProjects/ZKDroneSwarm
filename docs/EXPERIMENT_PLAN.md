# Experimentation Plan: from point estimates to a publication-grade evidence base

*Prepared 2026-05-22. Companion to PROJECT_LOG.md (history), BACKLOG.md (ideas),
DATA_CATALOGUE.md (saved data), THEORY.md (proofs), PAPER_DRAFT.md (write-up).*

## 1. Why more experiments

What we have so far is mostly POINT ESTIMATES at one default operating point
(m=30, n=240, d=5, K=10, d_hat=8, T=50, cand=20, sigma_own=0.10, sigma_obs=0.30),
with 5 to 8 seeds, sweeping one knob at a time (mostly the masking rho). That is
enough to establish the claims qualitatively. It is NOT yet enough for a strong
submission, which needs:

1. Tight confidence intervals (10 to 20 seeds) and significance tests on every
   headline number.
2. Curves, not points: each claim shown as a function of the relevant knob, with
   error bands.
3. A two-dimensional observability grid (masking rho x reward-noise sigma): we
   have swept rho with sigma fixed; the two channels have never been crossed.
4. Ablations that attribute the win to specific mechanisms (precision weighting,
   online updating, the choice channel, warm-start).
5. Robustness to the guessed rank d_hat (the fairness knob) and to the world
   parameters we have held fixed (clusters K, cluster tightness, n/T starvation).
6. Reviving a few parked method ideas (active exploration, precision-gated
   fusion, a probe-then-online hybrid, cold-start newcomers) that can both
   strengthen the method and add categorical results.

This plan is organized as: parameter taxonomy (Section 2), experiment suites
(Section 3), ablations (Section 4), statistical protocol (Section 5), and a
prioritized schedule (Section 6). Revived directions are flagged [REVIVE].

## 2. Parameter taxonomy (what to vary and why)

### 2a. World / latent-structure parameters
| param | meaning | current | sweep | why it matters | priority |
|---|---|---|---|---|---|
| d | TRUE latent rank | 5 | 1,2,3,5,8,12,20 | core: gap scales with low-rankness (C13). d=1 has no personalization; d->min(m,n) kills CF | P0 |
| K1,K2 | drone/target clusters (block model) | 10,10 | 2,5,10,20,40,n | effective rank = min(d,K1,K2); clustering lowers sample complexity (theory) | P1 |
| within | within-cluster spread (latent noise) | 0.15 | 0.0,0.05,0.15,0.3,0.6 | how clean the structure is; soft vs hard clusters | P1 |
| n | number of targets | 240 | 60,120,240,480,960 | sample starvation n/T; the unseen-pair and anytime edges grow with n | P0 |
| T | horizon (rounds) | 50 | 25,50,100,200,400 | n>>T regime; anytime advantage; convergence of all methods | P0 |
| m | number of drones | 30 | 10,30,60,120 | population size for recovering U from the broadcast | P2 |
| cand | offer size per round | 20 | 5,10,20,40,n | decision difficulty; small cand = sparser coverage | P2 |
| signed | reward in [-1,1] vs [0,1] | signed | signed, nonneg | reward-model robustness (sign/baseline artifacts) | P2 |
| link | reward link | bilinear | bilinear, mild nonlinear | nonlinearity raises effective rank: stress-test the assumption | P2 |

### 2b. Observability parameters (the two channels)
| param | meaning | current | sweep | channel | priority |
|---|---|---|---|---|---|
| rho | broadcast masking (fraction of teammates' events seen, persistent per drone) | swept | 1.0 to 0.05 | ACTION (you see the event or not) | P0 (done; add seeds) |
| sigma_obs | additive noise on observed teammate rewards | 0.30 | 0,0.1,0.3,0.6,1.0,2.0 | REWARD (value is noisy) | P0 (gap) |
| sigma_own | noise on a drone's own outcome | 0.10 | 0.05,0.1,0.2,0.4 | REWARD (self) | P2 |
| p_share | fraction of teammates who broadcast at all | 1.0 | 0.25,0.5,1.0 | overlaps with rho; lower priority | P2 |
| rho x sigma_obs | the two channels CROSSED | not done | full 2D grid | both | P0 |

### 2c. Algorithm / fairness parameters
| param | meaning | current | sweep | why | priority |
|---|---|---|---|---|---|
| d_hat | GUESSED rank given to all structured learners | 8 (true d=5) | 2,3,5,8,12,20 | robustness to rank misguess; the fairness knob; practicality | P0 |
| refit_every | online ALS cadence | 3 | 1,3,5,10 | online vs batch (isolates the anytime mechanism); cost | P1 |
| als_sweeps | ALS inner iterations | 8 | 3,8,15 | convergence audit (under-convergence understates us) | P1 |
| explore policy | eps-greedy / UCB / Thompson / info-gain | eps-decay | the four | estimation/decision separation; enables active exploration | P1 [REVIVE C10] |
| precision wt | weight obs by 1/sigma^2 vs uniform | on | on, off | is precision weighting the source of masking-robustness? | P1 (ablation) |
| competitor hp | PTF probe_frac, ESTR explore_frac, BPMF prior_var, MFSGD lr | defaults | tuned | fairness to baselines (avoid strawman) | P1 |

## 3. Experiment suites

Each suite states the QUESTION, the DESIGN (sweep), SEEDS, METRICS, EXPECTED
outcome, and rough COMPUTE. Metrics: skill (final-policy), unseen-pair skill,
anytime cumulative-reward skill (AUC and at K=T/4,T/2,T), state-uniqueness,
factor-recovery (subspace angle of U_hat vs U), and (where relevant) onboarding
skill vs probes and confidence calibration.

### E1. Headline confirmation with tight CIs [P0]
- Question: are the headline results (categorical unseen, masking-robustness,
  anytime) significant with proper error bars?
- Design: re-run C11 (unseen vs rho), C14b (10-method bake-off at rho in
  {1.0,0.5,0.25}), C15 (fine rho), C16 (anytime) at 20 seeds; add bootstrap 95%
  CIs and paired significance (ours vs each baseline per seed).
- Metrics: all. Expected: CIs separate ours from no-structure on unseen at every
  rho, and from all methods on anytime at every rho<1. Compute: ~few thousand
  cells, ~1 hr at 4 workers.

### E2. Rank scaling, both metrics [P0] (extends C13)
- Question: how do the unseen and anytime edges scale with true rank d?
- Design: d in {1,2,3,5,8,12,20}, fixed d_hat=8, all methods, 15 seeds.
- Metrics: unseen + anytime + factor-recovery. Expected: CF unseen decreases with
  d (more to complete); at d=1 CF ties tabular (no personalization); both vanish
  as d->min(m,n). Anytime edge largest at small d. Compute: ~30 min.

### E3. Two-channel observability grid (rho x sigma_obs) [P0] (fills the gap)
- Question: how do the reward channel (RewardCF), the action/choice channel
  (ChoiceCF), and the fusion (BothCF) trade off as masking AND reward-noise vary?
- Design: rho in {1.0,0.5,0.25,0.1} x sigma_obs in {0,0.3,0.6,1.0,2.0}; methods
  {Tabular, RewardCF, ChoiceCF, BothCF, best competitor PTF}; 15 seeds.
- Metrics: unseen + anytime. Expected (from the earlier apples-to-apples result):
  RewardCF best at low noise; as sigma_obs rises the clean CHOICE channel
  (ChoiceCF) overtakes; BothCF dominates the grid; masking hurts both channels;
  PTF degrades on both axes. This is the figure that shows we handle BOTH
  observability types. Compute: ~1 hr.

### E4. Sample starvation n/T [P0]
- Question: does the anytime advantage scale with starvation (n>>T)?
- Design: vary T in {25,50,100,200,400} at n=240, and n in {60,240,960} at T=50;
  all methods, 15 seeds.
- Metrics: anytime (primary) + final unseen. Expected: as n/T grows, UCBIndep
  anytime -> 0 (cannot pull each arm); CF anytime edge grows; as T grows large
  (n/T small) the gap shrinks and probe-then-fit catches up. Pins the regime where
  we win. Compute: ~45 min.

### E5. Cluster structure and onboarding [P1] (extends C12)
- Question: does block structure (clusters) lower onboarding/cold-start sample
  complexity, as the theory predicts?
- Design: K1=K2 in {2,5,10,20,40} x within in {0.05,0.15,0.3}; onboarding probes
  sweep; 15 seeds.
- Metrics: onboarding skill vs #probes (find the knee); compare to d and to m.
  Expected: fewer probes needed as clustering sharpens (type transfer); knee near
  d_hat for diffuse, below d_hat for tight clusters. Compute: ~45 min.

### E6. Guessed-rank robustness d_hat [P0]
- Question: how sensitive are we (and competitors) to misguessing the rank?
- Design: d_hat in {2,3,5,8,12,20} at true d=5; all structured methods; 15 seeds.
- Metrics: unseen + anytime. Expected: graceful over a wide band (d_hat >= d);
  mild degradation when d_hat < d; ours at least as robust as competitors. Builds
  the practicality case. Compute: ~30 min.

### E7. Newcomer cold-start (M2) [P1] [REVIVE]
- Question: can a late-joining drone with ZERO own history act well from the
  broadcast alone?
- Design: train the swarm for T0 rounds, then inject a fresh drone; it must select
  using only U recovered from the broadcast (fold-in its own p from a few self
  probes, or act on the type prior with zero probes). Compare tabular (random) vs
  CF (warm from swarm latent) vs hierarchical-prior CF (D7). 15 seeds.
- Metrics: newcomer reward vs #self-probes; collaborative gain M4. Expected:
  CATEGORICAL (tabular newcomer ~random; CF ~swarm-level immediately). Adds a
  second categorical result alongside unseen pairs. Compute: ~30 min.

### E8. Active (uncertainty-reducing) exploration (C10) [P1] [REVIVE, candidate]
- Question: does exploring the highest-posterior-variance target beat eps-greedy
  on sample efficiency, and does the broadcast make it COLLECTIVELY beneficial?
- Design: estimator with a factor posterior (BPMF or weighted-ALS with the normal-
  equation covariance); decision policy in {eps-greedy, Thompson, info-gain/
  D-optimal in latent space}. Optionally diversify probes across drones via the
  broadcast (avoid re-probing). 15 seeds, rho in {1.0,0.25}.
- Metrics: anytime + rounds-to-X%-oracle + factor-recovery speed. Expected:
  info-gain exploration reaches high skill in fewer rounds than eps-greedy;
  broadcast makes one drone's probe lift all estimates (collective). Strengthens
  the anytime and onboarding stories where we already win. Compute: ~1 hr.

### E9. Probe-then-online-ALS HYBRID [P0] [REVIVE/synthesis]
- Question: can we close the only gap where PTF beats us (final policy at rho=1)
  WITHOUT losing masking-robustness or the anytime lead?
- Design: short UCB/random probe -> SVD warm-start -> our ONLINE weighted-ALS (not
  SGD); compare to RewardCF, PTF, ESTR across the fine rho grid and anytime.
- Metrics: unseen (vs rho) + anytime. Expected: matches PTF at rho=1, keeps our
  flat unseen curve and anytime dominance => dominant EVERYWHERE. If so this
  becomes the recommended method. Compute: ~45 min.

### E10. Precision-gated fusion (fix BothCFGated) [P1] [REVIVE]
- Question: does gating the choice channel by reward PRECISION (1/sigma^2, not
  raw count) make BothCF strictly dominate RewardCF and ChoiceCF across the noise
  grid?
- Design: BothCF with gate weight = reward-precision; sweep sigma_obs; 15 seeds.
- Metrics: skill + unseen across sigma_obs. Expected: erases the reward-clean
  penalty AND keeps the choice benefit under noise (the earlier count-based gate
  failed here). Compute: ~30 min.

### E11. Calibrated exploration under choice masking (C3d / Motif A) [P2] [REVIVE]
- Question: does dual-source confidence (factor precision + decision-alignment)
  beat fixed-eps when only a fraction rho of CHOICES are observed?
- Design: exploration rate = f(1 - confidence); confidence from precision and from
  alignment of teammates' observed choices with predictions; sweep rho. Use
  alignment as an AGGREGATE signal only (the per-observation gate is the known
  cycle-9 deadlock). 15 seeds.
- Metrics: anytime + calibration (M5). Expected: calibrated exploration matches or
  beats tuned fixed-eps without per-instance tuning; calibration improves with
  more data. Compute: ~45 min.

## 4. Ablation studies (attribute the win)
- A1. Channel ablation: Tabular vs RewardCF vs ChoiceCF vs BothCF across sigma_obs
  (isolates each observation channel's contribution). [folds into E3]
- A2. Precision weighting on/off: replace 1/sigma^2 weights with uniform; expect
  masking-robustness to drop toward the batch-SVD curve (tests the proposed
  mechanism for F5). [P1]
- A3. Online vs batch: refit_every=1 (fully online) vs a single refit at the end
  (batch); isolates the ANYTIME mechanism behind F6. [P1]
- A4. Warm-start on/off: our online ALS with vs without SVD warm-start (the E9
  hybrid is warm-on); isolates warm-start's contribution at dense rho. [P1]
- A5. d_hat over/under: covered by E6.
- A6. Reward model: signed vs nonneg, bilinear vs mild-nonlinear (does CF survive
  a slightly nonlinear link / higher effective rank?). [P2]

## 5. Statistical protocol
- Seeds: 15 for sweeps, 20 for the four headline panels (E1). Same seed set across
  methods (paired comparisons).
- Report mean with bootstrap 95% CI (10k resamples) on the per-seed skill values.
- Significance: paired test (ours vs each baseline) per operating point; report
  the effect size (skill difference) and a Holm correction across the sweep.
- Always plot error bands; never a bare point. Keep full per-seed JSON (catalogue).
- Sanity gates every run: Random ~ 0; Oracle = 1 by definition; dynamic range
  (oracle - random) healthy; effective rank ~ min(d,K1,K2).

## 6. Priority and schedule (CPU-parallel; one GPU job is irrelevant here)
- WAVE 1 (P0, do first): E1 (CIs), E3 (rho x sigma grid, fills the channel gap),
  E9 (hybrid, closes the PTF gap), E2/E4/E6 (rank, starvation, d_hat scaling).
  These convert the qualitative story into a defensible, curve-based evidence base
  and resolve the only competitor advantage.
- WAVE 2 (P1, strengthen method): E7 (newcomer categorical), E8 (active
  exploration), E10 (precision-gated fusion), E5 (cluster onboarding); ablations
  A2, A3, A4.
- WAVE 3 (P2, nuance/robustness): E11 (calibrated exploration), A6 (reward model),
  m and cand sweeps, p_share.
- Parallelism: each suite is independent cells (method x knob x seed); run with a
  ProcessPoolExecutor at 4 workers (keeps C: temp peak low). A whole wave is a few
  hours of wall-clock. Save complete data and catalogue every run; commit per
  suite.
- Estimated total: Wave 1 ~ half a day of compute, Wave 2 ~ half a day, Wave 3 ~
  few hours, all unattended and parallelized.

## 7. Revived directions (what we are bringing back, and why)
Brought back to STRENGTHEN the method and paper (see Section 3):
- C10 active uncertainty-reducing exploration [E8]: leverages the broadcast for
  COLLECTIVE sample efficiency; sharpens the two metrics we already win (anytime,
  onboarding); principled (estimation/decision separation). Was demoted as
  "polish"; with the anytime result it is now a natural amplifier.
- Probe-then-online-ALS hybrid [E9]: synthesizes the PTF lesson; aims to make our
  method dominant at EVERY density (removes the lone caveat).
- M2 newcomer cold-start [E7]: a SECOND categorical result (tabular newcomer ~
  random) at low cost.
- Precision-gated fusion [E10]: fixes the count-based gate that failed under noise;
  targets a strictly-dominant BothCF.
- Dual-source calibrated exploration [E11] and hierarchical/type priors (D7, in
  E7): the confidence story and cold-start shrinkage, in-scope and cheap.
KEPT PARKED (genuine scope drift; at most a one-line remark):
- D1 Byzantine/faulty teammates (malicious agents) is OUT of the honest-agent
  setting; the competence-weighting mechanism it produced is already salvaged in
  ChoiceCF.
- D5 RANSAC robust factorization is subsumed by precision weighting unless we
  demonstrate genuine outliers; revisit only then.
- D6 contention/assignment (Hungarian) changes the problem (matching, where
  coordination value appears); it is a strong FUTURE paper, not a revival for this
  one.
```
