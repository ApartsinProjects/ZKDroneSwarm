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
MOTIF A -- CONFIDENCE: (i) confidence of an ESTIMATE = posterior precision over
  factors (Bayesian PMF) -> exploration + self-trust; (ii) confidence of a
  DECISION = choice sharpness (latent inverse-temperature beta). Decision-
  confidence is the OBSERVABLE proxy for a teammate's estimate-confidence.
MOTIF B -- WEIGHTING teammates' observations by informativeness/reliability
  (driven by A, or behavioural consistency / recency / residual fit). HKV
  confidence-weighted implicit feedback.
KEY QUESTION: unify A+B in one EM/VB (weight = posterior responsibility, derived
  from confidence) OR keep them as two mechanisms? Prior signal: model-agreement
  EM FAILS (cycle 9, deadlock); BEHAVIOURAL weighting WORKS. So test: a
  BEHAVIOURAL-confidence EM unification vs the two-mechanism heuristic.

## P0 -- do next (within scope)
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
