# Zero-Knowledge / decentralization compliance audit

Goal: verify our approach falls strictly inside the stated setting:
ZERO PRIOR KNOWLEDGE, ZERO COMMUNICATION, PARTIAL + NOISY OBSERVATION, with
decentralized decisions arising from the absence of communication. This document
audits the generative model, the observation channel, every method, and the
evaluation, and states the one item that needs an explicit modeling convention.

## The three assumptions, made precise

1. ZERO PRIOR KNOWLEDGE. No learner may use the latent factors P, U, the true rank
   d, the cluster/type assignments, or any reward labels. A learner may know only
   the OBSERVABLE action-space dimensions (how many drones m, how many targets n,
   which targets are offered this round) and a GUESSED rank d_hat for its own
   factorization.
2. ZERO COMMUNICATION. No agent transmits any message or shares any parameter with
   any other agent; there is no coordinator, no consensus/gossip, no joint policy.
   Agents only PASSIVELY OBSERVE the public consequences of actions in the shared
   environment (an engagement and its outcome are publicly visible), subject to
   partial and noisy sensing. Observation of a shared environment is NOT
   communication: no agent chooses to send information, there is no protocol, no
   addressing, no parameter exchange.
3. PARTIAL + NOISY OBSERVATION. Each agent senses its own outcome cleanly (small
   sigma_own) and a per-agent-limited, noisy subset of other agents' public
   outcomes (masking rho, noise sigma_obs). Masking models limited
   detection/sensing (range, orientation, dropout), NOT radio packet loss (which
   would presuppose transmission, i.e. communication).

Decentralization is then a CONSEQUENCE: with no communication and heterogeneous
partial sensing, each agent forms a different internal state and decides
independently (formalized in THEORY_FORMAL.md Theorem 4).

## Component-by-component audit

### Generative world (experiments/core.py make_world)
Returns (P, U, R, meta). These are the GROUND TRUTH used ONLY by the experiment
harness to (i) generate observed rewards R[i, a] + noise and (ii) compute the skill
metric. They are NEVER passed to any learner. PASS.

### Observation channel (run loops in pilot_*.py)
Each round: drone i selects a_i from its offered set; the harness computes the true
reward R[i, a_i], adds noise, applies the per-agent mask, and delivers to drone i:
its own (a_i, R[i,a_i]+noise) and, for each unmasked teammate k, (a_k,
R[k,a_k]+noise). This is passive public-outcome sensing. No learner sends anything.
PASS (with the masking = sensing convention of assumption 3).

### Methods
| method | observes (cross-agent) | uses true d / P / U ? | shares params ? | strict ZK |
|---|---|---|---|---|
| Tabular | own outcome only | no (d_hat) | no | YES |
| UCBIndep | own outcome only | no | no | YES |
| UCBHomo | pooled outcomes (action+reward) | no | no | YES |
| MFSGD | action+reward | no (d_hat) | no | YES |
| ESTR | action+reward | no (d_hat) | no | YES |
| BPMF | action+reward | no (d_hat) | no | YES |
| RewardCF (ours) | teammates' action+reward | no (d_hat) | no | YES |
| HybridCF (ours) | teammates' action+reward | no (d_hat) | no | YES |
| ChoiceCF (ours) | teammates' action + OFFERED MENU | no (d_hat) | no | see note |
| BothCF (ours) | action+reward + OFFERED MENU | no (d_hat) | no | see note |

Key points verified in code:
- Every learner is constructed as Cls(m, n, d_hat, idx, seed, ...): it receives the
  GUESSED rank d_hat (= 8), never the true d (= 5), and never P, U, R, or meta.
- Factor initialisations are random (rng.normal); there is no warm start from
  ground truth. (HybridCF/PTF warm-start from an SVD of their OWN observed
  empirical matrix, not from the truth.)
- Each drone holds its OWN learner instance with its OWN P, U estimates. No method
  reads, averages, or receives another drone's parameters. There is no coordinator.
- RewardCF / HybridCF (our HEADLINE methods) consume ONLY teammates' (action,
  reward) outcomes. They are strictly ZK and strictly communication-free.

### The setting holds UNIFORMLY for ALL methods (apples-to-apples)

The answer to "does the setting hold for our AND all baseline methods?" is YES, by
construction of the harness, which treats every method identically:
- ZERO PRIOR KNOWLEDGE: no method receives P, U, R, the true rank d, or type/labels.
  Structured methods (MFSGD, ESTR, PTF, BPMF, RewardCF, HybridCF) all get the SAME
  guessed rank d_hat=8; tabular methods (Random, UCBIndep, UCBHomo, Tabular) carry
  no rank at all. None is warm-started from ground truth.
- ZERO COMMUNICATION / FULLY DISTRIBUTED: every method (baselines included) is
  instantiated as one INDEPENDENT per-drone learner; the run loop never shares,
  averages, or routes one drone's parameters to another, and there is no
  coordinator. (ESTR's literature default is a single centralized estimator; our
  port runs it PER DRONE so the comparison is fully distributed and fair.)
- PARTIAL + NOISY BROADCAST ONLY: every method receives the identical per-drone
  masked, noisy outcome stream (own clean-ish, teammates masked at rho and noised
  at sigma); no method gets a privileged or denoised view.
- The ORACLE is the only centralized/complete-information object, used solely to
  NORMALISE skill; it is never a competing method.
So all reported gaps are within one setting: zero prior knowledge, zero
communication, partial+noisy passive observation, fully distributed.

### Evaluation
skill = (method - random)/(oracle - random) uses the true R, but only in the
experimenter's metric, never inside any learner. Oracle is a CEILING baseline
(centralized + complete information); it is reported for normalisation, never used
as a method. PASS.

## The offered menu: RESOLVED (every drone may choose any active target)

Earlier we worried that ChoiceCF/BothCF read teammates' per-drone offered menus
(cand_sets[k]) for exposure-debiased negative sampling. We RESOLVE this by adopting
the natural model: there is NO per-drone private menu. Every drone may choose ANY
currently ACTIVE target, and the active-target set is PUBLIC (everyone sees which
targets exist). Consequences:
- The "menu" a teammate chose from is the public active set, so observing it is
  passive public observation, not communication. The exposure debiasing samples
  negatives from this public set. ZK holds.
- Equivalently, the choice channel can sample negatives GLOBALLY from all targets
  using only the observed chosen action a_k (within=False). This is the canonical
  choice channel we now use (ChoiceZK; StackCF's choice sub-estimator). It observes
  NOTHING beyond teammates' actions, identical to RewardCF's footprint.

EVIDENCE it costs nothing: the choice-only ablation (E13) shows ChoiceZK (global
negatives, no per-drone menu) matches ChoiceCF (per-drone menu) on every metric at
every rho (gap <= 0.03, within noise). So the choice channel's value is NOT an
artifact of menu observation. (The pilot still draws per-round size-c offers as a
stand-in for limited per-round availability; a drone always knows its OWN offer,
which is unproblematic, and never needs a teammate's private menu.)

CANONICAL METHODS are therefore all strictly ZK with an action+outcome observation
footprint: RewardCF, HybridCF (rewards only), ChoiceZK (actions only), StackCF
(adaptively selects between them by self-validation; global negatives).

## Conclusion

- ZERO PRIOR KNOWLEDGE: satisfied by all methods (guessed rank, random init, no
  ground-truth factors/types/labels). PASS.
- ZERO COMMUNICATION: satisfied. The broadcast is passive public-outcome sensing,
  not message passing or parameter sharing; each agent decides independently with
  no coordinator. PASS, under the masking-as-sensing convention (assumption 3).
- PARTIAL + NOISY OBSERVATION: satisfied by masking rho and noise sigma. PASS.
- OFFERED MENU: RESOLVED. Every drone may choose any active target; the active set
  is public; the canonical choice channel (ChoiceZK / StackCF) uses global negatives
  and observes only teammates' actions. E13 confirms this costs nothing (ChoiceZK ~=
  ChoiceCF). No method needs a teammate's private menu. PASS.

CONCLUSION: all canonical methods (RewardCF, HybridCF, ChoiceZK, StackCF) are
strictly ZK and communication-free: guessed rank, random init, independent per-drone
estimators, no parameter sharing, no coordinator, and an action+outcome observation
footprint over a passively-sensed public outcome stream.

## Harness / evaluation-side audit (cycle 62)

The audit above covers the METHOD classes. We separately audited every EXPERIMENT
HARNESS: all 46 `experiments/pilot_*.py` plus every shared run loop (`run_episode`
families, `run_masked`, `run_anytime_clshp`, `run_iid`, `run_contention`, `run_2ch`,
`run_param`, `run_robust`, `run_stress`, `run_newcomer`, `run_2ch`). The question:
does the harness ever leak a prior into a learner, or route one learner's parameters
to another (a back-door for communication)? Method: enumerate every use of P, U, R,
true rank d, type labels, and every `.copy()` / cross-learner attribute read
(`learners[i].U/.P`, `muU/muP`, `P_hat/U_hat`).

GENERAL RESULT: across all harnesses, R/P/U are used ONLY to (a) generate observed
outcomes and (b) compute oracle/random/popularity REFERENCES for the skill metric.
Learners always select / observe / predict from their own broadcast. The shared
compliant cores are `run_masked` (per-drone persistent mask), `run_anytime_clshp`,
`run_iid` (persistent + iid masks, both channels masked consistently), `run_2ch`
(dual channel), and the `run_episode` families.

Two findings, both now resolved:

1. VIOLATION (cross-learner parameter copy), FIXED this cycle.
   `pilot_e7_newcomer.py` (old lines 63-64) handed the late-joining newcomer a peer's
   learned factors directly: `U_hat = learners[0].U.copy()` and `p_pop =
   mean(learners[0].P)`. That is exact only at full broadcast (rho=1) and is a
   parameter copy, not passive observation. FIX: the newcomer is now a PASSIVE
   `RewardCF` listener that hears the public broadcast under its OWN persistent
   rho-mask and recovers U_hat by its OWN weighted-ALS (its population prior is the
   mean of the teammate factors IT recovered). Incumbents are masked at the same rho.
   Re-run sweeps rho in {1.0, 0.5, 0.25}; the categorical CF-vs-Tabular gap is
   re-stated as holding under masking (it survives; the probe-efficiency slope
   flattens as rho falls because the self-recovered U becomes the bottleneck).
   This was the ONLY cross-learner parameter copy in the codebase.

2. IDEALIZATION (oracle rank), DOCUMENTED + made overridable.
   The true-d `run_episode` family (`pilot_noise.run_episode`, `pilot_refit.run_episode`,
   `pilot_choice_only.py`) set the learner factor dim to the TRUE rank `d = P.shape[1]`,
   so the older reward-class diagnostics built on them (pilot_structure, _bootstrap,
   _confirm, _bakeoff, _em, _rank, _mfaudit, _starvation, _trust, _choice_only) are
   "fair" only when d_hat = d. This is BENIGN: every paper-headline result uses the
   d_hat harnesses (`run_masked` / `run_anytime_clshp`), never this path. FIX:
   `run_episode` now takes an optional `d_hat=` (defaults to true d for backward
   compatibility) and is documented in-code as an ORACLE-RANK DIAGNOSTIC, not the fair
   ZK setting. `pilot_rank.py` deliberately uses true d ("oracle rank") and is labeled
   as such. No re-run needed (no headline depends on it).

No other violations: no learner is ever constructed with or assigned true P/U/R or
type labels; every other `P_hat/U_hat/.copy()` is a learner reading its OWN state or
an offline-recovery diagnostic output; channel typing is correct throughout (reward
channels add noise, choice channels mask to -1, dual-channel harnesses mask both
consistently); the `OracleMate` actor in the heterogeneous-teammate sanity is an
ENVIRONMENT actor (excluded from learning and from all metrics), used only to shape
the public broadcast. HARNESS SIDE: CLEAN after the E7 fix.

## Baseline-fairness audit (cycle 64): do the COMPETITORS cheat?

Question: is any baseline exposed to information it should not have (true rank,
ground-truth factors/reward matrix, an unmasked or de-noised broadcast, a centralized
pool), which would make it UNFAIRLY strong? Audited every class in pilot_baselines.py
(Random, UCBIndep, UCBHomo, MFSGD, ESTR, PTF, BPMF, BiasModel, KNNCF, SoftImpute) plus
Tabular (pilot_noise) and the new CBBALite, against how the headline harnesses build
and feed them.

THE INTERFACE CLOSES THE LEAK BY CONSTRUCTION. Every learner is created as
`Cls(m, n, d_hat, idx, seed, **hp)` and thereafter receives data ONLY through
`observe(t, choices, revealed, cand_sets, rvar)`. The true `R`, `P`, `U`, rank `d`, and
type labels are NEVER passed to any constructor or method. `revealed` is the
harness-masked, noise-added outcome stream. So a learner structurally CANNOT read the
ground truth; the only fairness lever is the single scalar `d_hat`.

| baseline | factor dim | learns from | reads true R/P/U? | verdict |
|---|---|---|---|---|
| Random | n/a | nothing (rng scores) | no | fair floor |
| UCBIndep / UCBHomo | n/a (tabular) | observed `revealed` (own + broadcast) | no | fair; floor on unseen by design |
| Tabular | n/a | own observed reward only | no | fair |
| MFSGD | d_hat | observed `revealed` (SGD-MF) | no | fair |
| ESTR | d_hat | SVD of its OWN observed R_hat (=sum/cnt) | no (R_hat is observed, not true) | fair; PER-DRONE (not centralized) |
| PTF | d_hat | own-row UCB probe -> SVD of observed R_hat -> online SGD | no | fair (see clip note) |
| BPMF | d_hat | conjugate posterior from `revealed` + `rvar` | no | fair |
| BiasModel | rank<=2 | observed mu+b_i+c_j | no | fair (additive-only by design) |
| KNNCF | n/a (memory) | observed user-user similarity | no | fair |
| SoftImpute | ~d_hat | nuclear-norm completion of observed matrix | no | fair |
| CBBALite | d_hat | RewardCF utility (observed) + reactive backoff | no | fair (same CF utility as ours) |

Key confirmations:
- RANK IS d_hat=8 FOR ALL, IN EVERY HEADLINE. run_masked (pilot_c11_masking:32),
  run_anytime_clshp (pilot_anytime:44,51), pilot_compare (D_HAT=8 -> run_masked),
  pilot_crossover, and pilot_contention (pc.D_HAT) all pass d_hat, never the true d=5.
  No structured baseline ever gets the true rank in a reported result.
- LOW-RANK BASELINES FACTORIZE THEIR OWN OBSERVED MATRIX, not the truth: every SVD/fit
  is over `R_hat = sum/cnt` accumulated from the masked broadcast (ESTR._svd, PTF._warm,
  SoftImpute._impute, KNNCF, BiasModel._fit). Grep confirms `R` (the true matrix) is
  never in scope inside any baseline.
- SAME MASKED BROADCAST: every baseline's observe loops `if not np.isnan(revealed[k])`,
  i.e. consumes exactly the harness-masked, noised stream our methods get. No baseline
  sees an unmasked or de-noised view.
- ESTR IS DECENTRALIZED HERE: each ESTR instance builds its own R_hat from its own
  observed broadcast and scores its own row; the "centralized" in its docstring is the
  literature default, which our per-drone port does NOT use (so our ESTR is, if anything,
  WEAKER than the canonical one -- conservative for our claims).

Minor, non-cheating notes (flagged for completeness; all are either neutral or GENEROUS
to the baseline, i.e. conservative for our wins):
- Exploration schedules are per-method (MFSGD/PTF eps=0.15, BiasModel/KNN/SoftImpute
  eps=0.2, UCB c=2.0, ours eps0=0.5 decaying). This is hyperparameter heterogeneity, not
  information access; the earlier "same schedule" wording referred to our own family.
- PTF/ESTR clip observed R_hat to the known reward range [-1,1] before SVD. This uses the
  structural bound (cosine reward in [-1,1]), not ground-truth values; it mildly DENOISES
  the baseline (helps it), so it cannot inflate our advantage.
- BPMF seeds its factor init from idx only (not the per-run seed), so its init is fixed
  across seeds. Random (not ground-truth) init; a reproducibility quirk, not a cheat.
- The true-d run_episode diagnostics give ALL methods (ours + baselines) the true rank;
  symmetric within those (oracle-rank) diagnostics and NOT used in any headline (see the
  harness audit above; now documented + d_hat-overridable).

CONCLUSION (baseline side): no competitor is exposed to privileged information. Rank is
the guessed d_hat for all; low-rank baselines complete their own observed matrix; the
broadcast is identically masked and noised; ESTR is decentralized. The few asymmetries
that exist are GENEROUS to the baselines, so our reported gaps are conservative.

## Noise-variance exposure (cycle 68): is sigma known to learners?

Question (raised in review): during estimation, is the OBSERVATION-NOISE LEVEL exposed to
the drones? Answer: YES, the per-observation variance is passed to observe() as `rvar`
(=sigma^2: sigma_own^2 for own outcome, sigma_obs^2 for a sensed teammate), and the
precision-weighting methods USE it (RewardCF/ChoiceCF/BothCF: weight = 1/rvar; BPMF: rvar
as the likelihood variance). Honest assessment:
- This is NOT a ground-truth prior in the forbidden sense (no P/U/R/true-rank/types). It is
  the OBSERVATION-MODEL noise level, i.e. SENSOR CALIBRATION, exactly the known measurement-
  noise covariance that a Kalman filter assumes. A robot knowing its own sensor's noise (and,
  in the sensing-grounded model, sigma(d) from its own range estimate) is admissible self-
  knowledge, not knowledge of the hidden reward structure.
- The HEADLINE does NOT depend on it: the `precision=False` (uniform-weighting) ablation
  IGNORES sigma entirely, and uniform weighting is competitive-or-better in most regimes
  (PRECISION_SWEEP: uniform wins unseen at ALL noise; PRECISION_HETERO: bounded precision
  helps ONLY under heterogeneous noise). So the categorical result holds with NO knowledge of
  sigma.
- Which methods ASSUME sigma known (use rvar=sigma^2): RewardCF(precision=True, default), ChoiceCF,
  BothCF, BPMF, AND EMCF (correction: EMCF uses the TRUE likelihood precision 1/sigma^2; it estimates
  the FACTOR posterior, NOT the observation noise), plus all descendants (ContentionCF/AdaCF, UnifiedCF,
  CBBALite, MusicalChairs, CoordCF). sigma-AGNOSTIC (never use sigma): RewardCF(precision=False, uniform),
  MFSGD, ESTR, PTF, SoftImpute, KNNCF, BiasModel, Tabular/UCB*/Random; and the CHOICE channel
  (ChoiceZK/ChoiceCF cross-agent side) is inherently sigma-free (choices are noise-free).
- Strictly-sigma-agnostic path: the uniform variant ignores sigma and is competitive-or-better in most
  regimes, so the categorical headline does NOT require known sigma. NO method currently ESTIMATES sigma.
- DONE (cycle 69): RewardCFEstSigma -- estimates per-source sigma^2 from prediction residuals (empirical
  Bayes), removing the known-sigma assumption; validated to match the known-sigma version (see EST_SIGMA).
  TODO: state the sensor-calibration assumption explicitly in the paper's setting.
