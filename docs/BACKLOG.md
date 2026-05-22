# Experiment Backlog (not-yet-explored / in-progress ideas)

Living list. Priorities (P0 highest) re-ranked whenever a finding lands.

## SCOPE ANCHOR (do not drift)
SETTING: a swarm of drones + targets with an UNKNOWN but existing LATENT
STRUCTURE; LIMITED + NOISY OBSERVABILITY; NO COMMUNICATION. Agents are HONEST
LEARNERS (cold-start = not-yet-learned, NOT malicious).
METHOD: collaborative filtering via latent-space MATRIX DECOMPOSITION (+ variants).
DESIGN SPACE = two axes:
  (a) LATENT STRUCTURE: true rank, #clusters, cluster tightness, sharpness /
      effective rank, anisotropy, approximate low-rank, spread.
  (b) OBSERVABILITY TYPE & NOISE: what is broadcast (rewards / decisions / both),
      reward-sharing fraction p, observation noise (effect vs per-observer),
      sample-starvation / changing availability.
METHOD VARIANTS to compare: SGD-MF, ALS, closed-form/RLS, BPMF (Bayesian),
  BPR / implicit-feedback, dual-likelihood (rewards+choices), confidence-weighted.
METRICS: skill (converged), AUC / cumulative (anytime), targets-destroyed@K,
  regret. GOAL: the CLEAR, SIMPLE CF variant that dominates the design space.

## TWO CENTRAL METHOD MOTIFS (in-scope; honest agents, limited observability)
MOTIF A -- CONFIDENCE, from TWO sources (combine, they catch each other's
  failures):
  (a) FACTORIZATION confidence = posterior precision over p_i,u_j (Bayesian PMF).
      Internal certainty; high with plentiful consistent data.
  (b) DECISION-ALIGNMENT confidence = do OTHER drones' observed choices match my
      model's prediction argmax<p_k,u_hat>? External corroboration / decentralized
      SELF-VALIDATION against swarm behaviour (no ground truth, no comms).
  Combined confidence = f(precision, alignment): precise-but-wrong (overfit) ->
  alignment flags it; aligned-by-luck -> precision flags it.
  USE (b) as an AGGREGATE model-quality signal (exploration vs exploitation;
  trust unseen-pair predictions; per-teammate trust) -- NOT as a per-observation
  learning gate (that is the cycle-9 deadlock: half-trained model down-weights
  good choices it does not yet agree with).
  (Also: confidence of a DECISION = choice sharpness/consistency = observable
  proxy for a teammate's own certainty.)
MOTIF B -- WEIGHTING teammates' observations by informativeness/reliability
  (driven by A, or behavioural consistency / recency / residual fit). HKV
  confidence-weighted implicit feedback.
KEY QUESTION: unify A+B in one EM/VB (weight = posterior responsibility, derived
  from confidence) OR keep them as two mechanisms? Prior signal: model-agreement
  EM FAILS (cycle 9, deadlock); BEHAVIOURAL weighting WORKS. So test: a
  BEHAVIOURAL-confidence EM unification vs the two-mechanism heuristic.

## GROUNDBREAKING BAR: novel AND DEFINITELY outperforming (categorical, not %).
Aim metrics/knobs at regimes where the trivial baseline is at the FLOOR BY
CONSTRUCTION -> any CF win is decisive. The categorical edge of matrix
decomposition: it PREDICTS UNSEEN agent-task pairs; tabular cannot.

## CREATIVE METRICS (relevant to our setting; highlight the structural edge)
- M1. UNSEEN-PAIR reward: restrict to targets a drone NEVER pulled. Tabular ~
  floor by construction; CF predicts. CATEGORICAL.
- M2. COLD-START NEWCOMER reward: late-joining drone, zero own history -> act
  from the broadcast alone. Tabular ~ random; CF warm-starts from swarm latent.
- M3. ROUNDS-TO-X%-oracle (sample efficiency).
- M4. COLLABORATIVE GAIN = (CF-with-broadcast - CF-solo)/(oracle - solo).
- M5. CONFIDENCE CALIBRATION (Bayesian variants; Motif A): predicted uncertainty
  vs actual error.
- (have: skill [converged], AUC [anytime], Urec [factor recovery].)

## CREATIVE KNOBS (dials that CREATE the categorical-win regime)
- K1. FORCED-NOVELTY rate: offer only never-pulled targets -> forces
  generalization -> tabular fails.
- K2. NEWCOMER INJECTION: agents join over time -> cold-start from broadcast.
- K3. TARGET CHURN / LATENT DRIFT: constant novelty / non-stationarity.
- K4. REWARD BINARIZATION & BROADCAST SPARSITY: lower info/obs -> amplifies CF
  pooling advantage.
- K5. ANISOTROPY / effective-rank decay (spectral knob).
- K6. CHOICE-OBSERVATION MASKING (rho): each observer sees only a fraction rho of
  others' CHOICES, with a HETEROGENEOUS mask per drone (partial, decentralized
  view of the decision channel). The realistic decision-channel noise model
  (discrete: a choice is seen-cleanly or missed, not value-corrupted). Pairs with
  Motif A: low rho -> lower factorization (a) AND alignment (b) confidence ->
  calibrated exploration. RecSys analog: exposure / MAR.
- (have: rank d, #clusters, tightness, sharpness, n/T starvation, p reward-share,
  sigma_obs noise.)

## P0 -- do next (within scope)
- [TODO] C8. *** GROUNDBREAKING CANDIDATE *** Generalization / cold-start
  NEWCOMER (K1/K2 + M1/M2): in a decentralized swarm with NO communication, can
  a CF agent generalize to UNSEEN targets, and can a NEWCOMER act well from the
  broadcast alone, CATEGORICALLY beating independent learning (at floor on unseen
  pairs)? Novel framing + definite outperformance. Cheapest gate: forced-novelty
  subsets; measure unseen-pair reward CF vs tabular.
- [TODO] C1. CF VARIANT BAKE-OFF across the (structure x observability) grid:
  which matrix-decomposition variant wins where; find the simple dominant method.
  Baselines: random, tabular/independent (UCB-Indep), homogeneous.
- [TODO] C2. METRICS upgrade: AUC / cumulative reward (anytime, highlights the
  transient CF advantage) + targets-destroyed@K (needs a depletion task model).
- [TODO] C3a. MOTIF A: latent-confidence beta model -- confidence of decisions
  (sharpness) and estimates (posterior precision); Bayesian PMF + Thompson.
- [TODO] C3b. MOTIF B: teammate-observation weighting (behavioural consistency,
  recency, residual). 
- [TODO] C3c. UNIFY vs SEPARATE: behavioural-confidence EM/VB (weight derived
  from confidence) vs two-mechanism heuristic -- which wins, and is unification
  worth it? (model-agreement EM already known to fail -> use behavioural signal.)
- [TODO] C3d. DUAL-SOURCE confidence (factorization precision + decision-
  alignment) driving exploration, under CHOICE MASKING (K6): does
  confidence-calibrated exploration beat fixed-eps when only a fraction rho of
  choices are observed (heterogeneous masks)? Tests Motif A's two sources + the
  masking knob together; the calibrated-exploration story.

## P1 -- structure & observability nuances
- [TODO] C4. Latent-structure nuances: anisotropy (skewed singular values),
  approximate / soft low-rank, continuous vs clustered, spread; effect on the
  CF advantage and on the right rank d_hat.
- [TODO] C5. Observability nuances: partial reward-sharing p in [0,1] sweep;
  asymmetric observation (see choices but not outcomes); two-stage noise (single
  effect noise + per-observer noise); availability/starvation sweeps.
- [TODO] C6. BPMF + posterior-uncertainty exploration (Thompson) within the core
  -- principled CF variant; uncertainty drives exploration and confidence.
- [TODO] C7. Rank selection / ARD (unknown d) -- the structure is UNKNOWN, so
  inferring rank is squarely in scope.

## P2 / PARKED (drift or later)
- [PARKED] D1. Byzantine / FAULTY-teammate robustness (cycles 12-16). DRIFT:
  introduces malicious agent-reliability heterogeneity, NOT the core setting
  (honest agents under limited observability). Keep as at most a one-line
  robustness remark; do not headline. The competence-weighting MECHANISM it
  produced is salvaged for C3 (cold-start of honest learners).
- [TODO] D2. Connect to real ZK-MRTA env + real policies (BPMF, IQL-ZK, etc.).
- [TODO] D3. Theory pack (separation, sample complexity) for the core claims.
- [TODO] D4. Cold-start warm-start from broadcast U; convergence dynamics.
- [PARKED] D5. RANSAC/consensus robust factorization (only if a robustness
  extension is pursued; tied to D1).

## Priority log (re-rank events)
- 2026-05-22 init: P0 = B1,B3,B6,B5 (robustness spine).
- 2026-05-22 RE-ANCHOR (user): research had drifted to Byzantine/faulty
  robustness (D1). Re-centered on the CORE setting (latent structure x
  observability/noise, honest agents, CF via matrix decomposition). New P0 =
  C1 (variant bake-off), C2 (metrics), C3 (decision-only cold-start, in-scope).
  Faulty/Byzantine (D1) and RANSAC (D5) PARKED. Done (on-target): cycles 1-11
  (characterization, structure, rank, noise) in PROJECT_LOG.
