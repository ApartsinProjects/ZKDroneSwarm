# Experiment Backlog (not-yet-explored / in-progress ideas)

Living list. Priorities (P0 highest) re-ranked whenever a finding lands.

## SCOPE ANCHOR (do not drift)
SETTING: a swarm of drones + targets with an UNKNOWN but existing LATENT
STRUCTURE; LIMITED + NOISY OBSERVABILITY; NO COMMUNICATION. Agents are HONEST
LEARNERS (cold-start = not-yet-learned, NOT malicious).

VALIDITY FIX (critical): FULL broadcast -> every drone sees the same stream ->
identical models -> decentralization is COSMETIC (one centralized learner x m).
"No communication" is vacuous under full observability. FIX: observability must
be LIMITED + HETEROGENEOUS so each drone has a UNIQUE state, but DECOUPLED FROM
ACTION (any drone can still engage any target). Spatial sensing is DEFERRED
(future) because sensing range would also imply ATTACK range -> constrained-
assignment confound. PRIMARY mechanism = per-drone HETEROGENEOUS degradation of
the OBSERVATION CHANNEL:
  (1) MASKING (preferred): drone i registers only fraction rho_i of broadcast
      events; PERSISTENT blind spots -> DURABLY unique states; targets i never
      observes/pulls -> unseen -> must COMPLETE via CF (tabular at floor).
      Motivation: lossy radios / packet loss / limited bandwidth / detection
      dropout.
  (2) ADDITIVE per-drone reward NOISE sigma_i: heterogeneous sensor quality;
      transiently unique; noisy drones lean on structure (CF denoises).
Makes decentralization REAL and motivates CF, without action coupling.

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

## P0 -- GROUNDBREAKING SPINE (reprioritized 2026-05-22 for best categorical-win odds)
ORDER: (1) C11 = run the C8 unseen-pair CATEGORICAL win UNDER heterogeneous
masking -- the NATURAL operating regime (each drone has durable blind spots), so
the win is everyday behaviour, not a contrived eval, AND it fixes decentralization
validity. (2) C12 dynamic target onboarding. (3) D3 theory of the separation.
Report ORACLE (centralized+complete-info ceiling) + RANDOM (floor) explicitly in
every table; use the continuous EXPLORE-KNOB (schedule/adaptive).
DONE infra (core.py): signed-cosine reward, K1xK2 block model, OraclePolicy,
explore_knob. C1 + C8 DONE (below). Method work (C6/C10/C3/C2) demoted to P1
(strengthens methods but is not a categorical win on its own).

- [P1 -- needs C6] C10. *** GROUNDBREAKING CANDIDATE *** COLLECTIVE UNCERTAINTY-REDUCING
  EXPLORATION. Decouple ESTIMATION (posterior over latents -> mu, Sigma) from
  DECISION (policy). Exploration probes the target with highest predictive
  variance / info-gain on the FACTOR posterior (active learning / optimal design
  in latent space), not eps-random. Every probe is BROADCAST -> collective
  benefit (one drone's uncertainty-reducing probe improves ALL estimates), no
  communication. Should beat passive eps-greedy on SAMPLE EFFICIENCY (where CF's
  edge lives) -> categorical-win candidate. Needs the posterior-tracking
  (Bayesian/RLS-with-Sigma) estimator (C6). Subsumes/elevates old B14
  (info-directed sampling). Diversify across drones via the broadcast (avoid
  re-probing what was just probed) to keep it decentralized.
- [TODO] C12. *** GROUNDBREAKING (reward-observable) *** TWO-PHASE: (1) DRONE-
  SIMILARITY learning -- all drones learn P (who is similar) from the shared
  reward broadcast; (2) TARGET ONBOARDING -- a NEW target is probed by a few
  diverse drones (quick joint sample), u_j fit by d-dim ridge/WALS projection
  given known P, then ALL drones predict it. Separation: tabular needs EVERY
  drone to probe EVERY new target (Theta(m)/target); CF needs ~Theta(d) shared
  probes/target. Grounded in WALS/fold-in item cold-start; novelty = decentral.
  + online + broadcast-only. Metric: reward on freshly-injected targets vs #probes.
- [TODO] C11. *** CORE VALIDITY + GROUNDBREAKING *** HETEROGENEOUS OBSERVATION
  CHANNEL (masking / additive noise; NOT spatial -> decoupled from action). Each
  drone perceives a different masked/noisier slice of the broadcast (persistent
  per-drone mask rho_i and/or sensor noise sigma_i) -> UNIQUE per-drone states
  (decentralization REAL); cells a drone never observes -> unseen -> must
  COMPLETE via CF (tabular at floor). Sweep rho_i / sigma_i. Fixes the
  broadcast=shared hole AND is the strongest CF motivation; structurally forces
  C8's unseen pairs. (Spatial observability-graph = FUTURE/parked: sensing range
  would couple to attack range -> constrained-assignment confound.)
- [DONE cycle 18] C8 UNSEEN-PAIR generalization: CATEGORICAL win (RewardCF 0.496
  vs Tabular 0.006, fair guessed rank d_hat=8). => fold into C11 to run it in the
  NATURAL masking regime (the H1 headline). Newcomer-cold-start variant (M2) TODO.
- [DONE cycle 17] C1 bake-off: BothCF (fuse reward+choice) dominates the
  (structure x observability) grid. Confidence-gated BothCF -> C3.
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
- [DRAFTED docs/THEORY.md] D3. THEORY pack (elevated toward the spine): the unseen-pair /
  Theta(d)-per-drone vs Theta(n)-tabular separation. Basis: MC needs O(d(m+n))
  obs (Candes-Recht/Keshavan) vs mn; block structure lowers it further (arXiv
  1912.04099). NOVELTY = adapt to the DECENTRALIZED broadcast + per-drone masking,
  no comms (tabular has ZERO info on unobserved pairs -> floor by construction).
- [TODO] D4. Cold-start warm-start from broadcast U; convergence dynamics.
- [TODO] D6. CONTENTION / ASSIGNMENT setting (targets deplete / each-once) ->
  centralized optimum = Hungarian matching; centralized DECISION (coordination)
  starts to matter (operational-assumption theme; the regime where coordination
  value appears). Currently NON-contention -> oracle = independent best-in-subset.
- [TODO] D7. Hierarchical priors / borrowing strength (cold-start shrinkage to a
  population/type prior; ties to the K1xK2 type structure).
- [PARKED] D5. RANSAC/consensus robust factorization (only if a robustness
  extension is pursued; tied to D1).

## Literature notes (web search, 2026-05-22; agents were API-blocked)
- ITEM COLD-START via WALS/fold-in projection: known user factors -> new item
  embedding from a few interactions, no retrain. Grounds C12 (target onboarding).
- CO-CLUSTERING / bipartite MIXED-MEMBERSHIP SBM for CF (VB inference, 2023):
  grounds the K1xK2 block generative model + type-based cold-start. Caveat:
  co-clustering trails neural/graph CF on accuracy -> use for generative model +
  type-assignment, not raw accuracy.
- DECENTRALIZED/FEDERATED CF (FCMF, DPMF, gossip MF) SHARE item latent vectors /
  aggregate params. OUR broadcast-only, NO-parameter-sharing setting is stricter
  = a genuine novelty axis.
- EXPOSURE BIAS / discrete-choice (MNL) models that use the FULL CHOICE SET are
  the PRINCIPLED debiasing of implicit feedback -> ChoiceCF observing the offered
  menu is CORRECT (debiases exposure), not a fairness violation.
- THEORY -- MC sample complexity: rank-r recoverable from O(r n log n) (Candes-
  Recht 2009; Recht 2011) or O(rn) (Keshavan OptSpace) vs n^2. Basis for CF's
  O(d(m+n)) vs tabular O(mn) = the unseen-pair / Theta(d)-vs-Theta(n) separation.
  Standard bounds are CENTRALIZED; our novelty = decentralized per-drone masked
  obs, no comms.
- THEORY -- clustering HELPS: block/community structure PROVABLY lowers MC sample
  complexity (community detection + MC with similarity-graph side info, arXiv
  1912.04099; hierarchical similarity graphs 2023; two-sided synergy). Supports
  K1xK2 block model; grounds D3.
- COLD-START META-LEARNING: MeLU (2019) -- new user from few items via meta-learn
  + ACTIVE 'evidence candidate selection' (= which drones probe a new target);
  M2EU/MWUF/CoMeta warm-embed cold ITEMS. Grounds C12.
- BANDITS: LinUCB (Li 2010), Thompson (Chapelle-Li 2011, often > UCB), low-rank/
  sparse contextual bandits. Grounds C6 (Bayesian/Thompson) + C10 (active explore).

## Priority log (re-rank events)
- 2026-05-22 init: P0 = B1,B3,B6,B5 (robustness spine).
- 2026-05-22 RE-ANCHOR (user): research had drifted to Byzantine/faulty
  robustness (D1). Re-centered on the CORE setting (latent structure x
  observability/noise, honest agents, CF via matrix decomposition). New P0 =
  C1 (variant bake-off), C2 (metrics), C3 (decision-only cold-start, in-scope).
  Faulty/Byzantine (D1) and RANSAC (D5) PARKED. Done (on-target): cycles 1-11
  (characterization, structure, rank, noise) in PROJECT_LOG.
- 2026-05-22 REPRIORITIZE for groundbreaking (user review): SPINE = C11
  (natural-regime unseen-pair win under masking + validity) > C12 (dynamic target
  onboarding) > D3 (theory of Theta(d)-vs-Theta(m) / unseen-pair separation). C8
  + C1 marked DONE (categorical win + bake-off proven). Method work
  (C6 Bayesian -> C10 active-explore, C3 confidence/weighting, C2 metrics)
  demoted to P1 (strengthens, not categorical alone). Added: continuous
  explore-knob + ORACLE baseline (infra, core.py), D6 contention/assignment axis,
  D7 hierarchical priors. All discussed ideas now captured.
- 2026-05-22 COMPARISON DONE (cycles 23-26): full method bake-off vs UCBIndep/
  UCBHomo/Tabular/MFSGD/ESTR/PTF/BPMF on three metrics (final-policy unseen,
  masking-robustness, anytime/AUC). Findings: (1) unseen-pair categorical win is
  ESTIMATOR-INDEPENDENT (all 5 low-rank methods clear the no-structure floor);
  (2) our online weighted-ALS is masking-robust + anytime-optimal -> dominates
  cumulative reward at every horizon and every rho<1; (3) PTF (probe-then-fit)
  beats us on FINAL policy only at rho=1 (no-limit case). Figures F5,F6 + Table 1.
- 2026-05-22 PLAN + REVIVE (user review): wrote docs/EXPERIMENT_PLAN.md (parameter
  taxonomy, 11 suites E1-E11, ablations A1-A6, stats protocol, 3-wave schedule).
  REVIVED (moved toward P0/P1): E9 probe-then-online-ALS HYBRID (close the only
  rho=1 gap -> dominate everywhere); E8=C10 active uncertainty-reducing exploration
  (collective via broadcast; amplifies anytime+onboarding); E7=M2 newcomer
  cold-start (2nd categorical result); E10 precision-gated fusion (fix the count-
  based gate); E11=C3d dual-source calibrated exploration; D7 hierarchical priors.
  NEW P0 = E1 (CIs/20 seeds) + E3 (rho x sigma_obs grid, fills the channel gap +
  ChoiceCF) + E9 (hybrid) + E2/E4/E6 (rank, starvation, d_hat scaling). KEPT
  PARKED: D1 Byzantine (out of scope), D5 RANSAC (subsumed by precision weights),
  D6 contention (future paper). Tutorial: docs/tutorial.html.
- 2026-05-22 THEORY + MASKING-MODEL (user): wrote docs/THEORY_FORMAL.md with detailed
  proofs: T1 tabular unseen floor (exact), T2 CF row recovery O(d) given U (exact),
  T3 ANYTIME separation under starvation (structure-free skill <= g(cT/n) -> 0; CF
  >= 1-O(d/T); matches C16), T4 persistent-vs-iid masking dichotomy (iid -> state-
  uniqueness -> 0 transient; persistent -> durable; categorical results invariant).
  NEW EXPERIMENT E12 [P1]: re-run headline panels (unseen, anytime, state-uniqueness
  vs rho) under IID per-round masking to confirm T4(c) empirically and measure the
  durable-vs-transient state-uniqueness gap. Motivation: iid (packet loss) is equally
  realistic; persistent (fixed topology/sensor) chosen for DURABLE decentralization.

## STATE REVIEW + NEXT DIRECTIONS (2026-05-23)
DONE this arc: 20-seed headline CIs; CONFIDENCE bake-off (EM/Bayesian factorization
with predictive-interval UCB DOMINATES uniform; EMshrink best-unseen Pareto);
precision-vs-noise crossover (coverage, not noise, is the binding constraint);
ChoiceEM (HONEST NEGATIVE: cold-start deadlock + choice channel weaker than rewards);
CONTENTION WIN (ContentionCF = CF estimate + fixed private per-target offset; ~2x at
severe contention; de-confliction needs PRIVATE FIXED randomness); ARD rank
self-determination (IN PROGRESS); full formal tutorial+paper (KaTeX), 3-condition
scope iff. Positioning scout: NO direct competitor for decentralized broadcast-only
no-comms online CF for MRTA (BanditMF = centralized; decentralized MRTA = auction/GA,
need comms). 

IMPROVEMENT HYPOTHESES / DIRECTIONS (general; ranked):
- [P0] H1 = C10 COLLECTIVE INFO-DIRECTED EXPLORATION. Use EMCF's factor posterior:
  probe argmax predictive-variance / info-gain (D-optimal / IDS in latent space),
  broadcast-shared so one probe helps all. HYP: beats eps-greedy + the count-bonus on
  rounds-to-X%-oracle (M3) and anytime, esp. when the explored latent slice is
  low-dim. Grounded: Russo-VanRoy IDS; info-guided low-rank MC sampling (1706.08037).
- [P1] H2 = ADAPTIVE ContentionCF. Scale the fixed-offset eps_break by each drone's
  OBSERVED collision rate -> dominate ALL contention levels (currently regime-dep).
  HYP: single adaptive policy >= max(argmax-CF, fixed-offset) everywhere.
- [P1] H3 = UNIFIED RECOMMENDED METHOD. Fold EMCF (confidence) + info-directed
  exploration (H1) + symmetry-breaking decision (contention) into ONE estimator+policy
  -> kills the "method zoo", one method dominating the whole design space.
- [P1] H4 = CALIBRATION (M5). Reliability diagram: EMCF predictive intervals vs actual
  error. HYP: EMCF well-calibrated, naive-precision mis-calibrated -> justifies the
  UCB/Thompson use of the posterior.
- [P1] H5 = THEORY for the new wins. (a) fixed-private-offset de-confliction =
  decentralized symmetry-breaking / matching-without-comms mini-result; (b)
  predictive-variance-UCB regret (why EM-confidence helps); (c) ARD rank-recovery.
- [DONE -- HONEST NEGATIVE] H6 = NON-STATIONARITY / churn (K3). HYP was: CF fold-in
  re-adapts in O(d) -> a 3rd categorical result. TESTED (pilot_churn.py, continuous
  turnover, n=600 world / 200 active / 8 depart+arrive per 5 rounds, 8 seeds): NOT a clean
  categorical win. CF beats Tabular on the active set (0.632 vs 0.448) but only TIES the
  optimistic UCBIndep (0.619); on FRESH arrivals CF TRAILS UCBIndep (0.074 vs 0.132,
  non-overlapping). Root cause: collective fold-in needs ~d probes to pin a newcomer, a
  LATENCY that fast churn outpaces, while UCBIndep optimism probes the new targets directly.
  So PLAIN CF's categorical edge is a STATIC sample-starved property. -> docs/CHURN.md.
- [DONE -- WIN] H6b = CONFIDENCE-DIRECTED CF wins under churn (the fix). Added ActiveCFconv
  (broadcast count-bonus) and EMCF (predictive-variance UCB) to the churn sweep (8 seeds).
  Both UNITE low-rank fold-in WITH directed probing of the uncertain (fresh) targets, and
  DOMINATE: active set EMCF 0.842 vs UCBIndep 0.619 / RewardCF 0.632; FRESH arrivals
  ActiveCFconv 0.363 and EMCF 0.371 vs UCBIndep 0.132 (all non-overlapping CIs). So
  non-stationarity IS handled categorically, but only by the variant combining fold-in with
  confidence-directed newcomer-probing (neither structure-free optimism nor exploitative CF
  alone suffices). A clean negative->diagnosis->win arc. NEXT: fold into tutorial/paper as a
  non-stationarity result (the collective-exploration extension also wins under churn).
- [DROPPED 2026-05-23 per user] H7 = NON-GAUSSIAN rewards (K4): logistic/GLM-link
  weighted-ALS for BINARY outcomes. A scope extension, not a categorical-win direction;
  out of scope for this work.
- [P2] H8 = TYPE-PRIOR shrinkage (D7): newcomer cold-start shrinks to its TYPE prior
  (not just popularity). HYP: faster newcomer warm-up than popularity shrinkage.
- [P0/RUNNING] H9 = HELD-OUT choice-informativeness (Prop 9). In-sample ChoiceEM gamma
  CANNOT down-weight uniform-random teammates (E[r]=gamma fixed point) and overfits a factor
  to their choices (inflates gamma ~0.70 vs prior 0.1). FIX: predictive responsibility (score
  each choice once vs the model BEFORE the refit sees it). Smoke PASSES the oracle-vs-random
  sanity (gamma 0.48 vs 0.11). DONE: ChoiceEM(predictive=True) in pilot_noise; pilot_choicehetero.py
  (real learners + ORACLE/RANDOM special teammates, gamma-separation diagnostic). 8-seed run in
  progress. Win condition: predictive beats fixed-ramp ChoiceCF as RANDOM teammates grow + leverages ORACLE ones.
- [P1] H10 = CONSENSUS-GROUNDED informativeness (refined user reward-gradient idea; complements
  H9). Held-out gamma catches RANDOM teammates (unpredictable) but still trusts CONSISTENTLY-WRONG
  ones (predictable). KEY INSIGHT: a teammate's "improvement" / value CANNOT be judged from its OWN
  choices, that is circular (a wrong-objective teammate looks just as consistent under a factor fit
  to its own choices as a right one). The choice-value trend s_k^t=<P_k,U[c_k^t]> with self-fit P_k
  is therefore uninformative for right-vs-wrong. To work it must GROUND value in something INDEPENDENT
  of k's choices: (a) k's reliable TYPE-MATES' reward-grounded factors (consensus: is k picking
  high-value targets for the type its type-mates reveal?), or (b) k's rewards (but then use RewardCF,
  already robust). So: estimate k's factor from consensus, score k's choices against it, trust the
  improving + consensus-consistent ones. Realizes "some drones discover structure earlier; estimate
  that (via consensus, not self-report) and bootstrap from them." Needs type structure + >=1 reliable
  type-mate per teammate; out of scope for the pure choice-only no-structure-prior setting.
- [P1] H11 = SANITY-CHECK SUITE for negatives/weak results (methodology). Construct
  obvious-expected experiments and assert the output: (a) ChoiceEM oracle>>random gamma (DONE,
  running); (b) PRECISION weighting under HETEROGENEOUS-noise teammates (half extreme-noise,
  half clean) MUST beat uniform; (c) ARD on KNOWN true rank d=3 then 8 must track 3->8;
  (d) EMCF interval CALIBRATION (X% intervals contain truth X%) = H4; (e) contention identical-vs-
  distinct types isolates de-confliction; (f) d=1 popularity makes CF's unseen edge vanish (Thm 5).
NEXT EXPERIMENTS (queued): H9 held-out gamma (RUNNING); H10 reward-gradient ChoiceEM-grad;
H11(b) precision heterogeneous-noise sanity; eps-greedy contention unification (H2 follow-up);
H4 calibration.

## NEGATIVE / WEAK RESULTS: root cause + fix (2026-05-23 rigor review)
All results below use >=6 seeds + bootstrap 95% CIs (single-seed numbers were only
SMOKE PREVIEWS, never reported). Confirmed not ALS/convergence artifacts: the precision
and ablation findings used the CONVERGED config (als_sweeps=20, refit_every=1).

NEGATIVES:
- ChoiceEM (choice-EM < ChoiceCF). ROOT CAUSE (code validated, NOT a bug): gamma_init=0.5
  + no warm-up -> choice channel weighted ~0.5/s2c from round 1 when choices are RANDOM
  (early exploration) -> noise injection -> model corruption -> responsibilities stay
  uniform -> deadlock (the cold-start gate the backlog predicted). FIX (testing):
  gamma_init=0.1 + warm-up (warm_em=0.3, build the model on rewards first), at HIGH
  sigma_obs (the choice channel's only winning regime). DEEPER: choice channel is a
  weaker signal than rewards until sigma_obs >> 1.
- Full precision weighting (1/sigma^2) < uniform on unseen. ROOT CAUSE: regularization-
  scale confound (weights 11-100 >> prior 1.0 -> under-regularizes) + over-weights own
  data irrelevant to the unseen u_j. FIX = relcap (bounded+normalized) ~ uniform; EM
  (Bayesian) is the principled answer (won the bake-off). CI-confirmed (precision sweep,
  12 seeds), NOT a convergence artifact.
- Posterior-UCB exploration (big beta) over-explores. ROOT CAUSE: own-factor uncertainty
  term uniformly large early -> bonus dominates -> charged by anytime. FIX = collective
  term only + SMALL beta (0.3) -> best final anytime (shown).
WEAK POSITIVES (improvements):
- ContentionCF regime-dependent (loses pool>=60). FIX = H2 collision-rate-adaptive offset.
- ARD recovers rank ~3.2 < true 5. ROOT CAUSE: weak directions unidentifiable under
  masking+spread; ARD prior b0 sets aggressiveness. FIX (optional): sweep b0 (less
  pruning -> closer to 5, at cost of keeping noise dims). The recovered rank IS the
  identifiable rank (honest).
- collective-UCB(0.3) best final anytime but slow early. FIX: anneal beta, or fuse with
  the count-bonus (fast early + structure-info late).
MORE BASELINES (user: "surprised there aren't more"): we already run 11 (Random,
UCBIndep, UCBHomo, Tabular, MFSGD, ESTR, PTF, BPMF, SoftImpute, kNN-CF, BiasModel)
under identical limits. Methods NOT included are either INADMISSIBLE (MAPPO/QMIX/IPPO
need centralized training/comms; LinUCB needs context features we lack) or REDUNDANT
(Thompson-MF/BanditMF ~ our BPMF; IQL ~ UCBIndep). Candidate ADDITIONS if useful:
item-kNN (we have user-kNN), a GLM/logistic-link MF (for binary rewards, H7), and an
explicit oracle-rank CF upper bound (bracket).
