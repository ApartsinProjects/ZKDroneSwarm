# Experiment Backlog (not-yet-explored / in-progress ideas)

Living list. Priorities (P0 highest) re-ranked whenever a finding lands.

## ===== ACTIVE WORKLOG (RESUME HERE; survives session compaction) =====
Goal (user, cycle 64+): DO ALL the pending items, parallelize where possible, and
keep this checklist updated so a fresh/compacted session can continue cold. Mark each
[TODO]/[BUILDING]/[RUNNING]/[DONE]. Heavy experiments: 8 logical CPUs, each ProcessPool
uses max_workers=4, so run AT MOST 2 pools at once (3+ thrashes). Python: /c/Python314/python.
Run generators/experiments from REPO ROOT (E:\Projects\ColabDroneSwarm). After any
experiment: add a DATA_CATALOGUE row + update this worklog. Recommended commit per item or small batch.

WRITING / non-CPU (do interleaved while experiments run):
- [DONE cycle 65] W1 MARL framing paragraph: added to paper (after Related-work honest-positioning) +
  tutorial (§9 novelty, plain callout). Covers indep-MARL=floor, CTDE/comms inadmissible, SIC-MMAB =
  the de-confliction baselines we beat, broadcast-CF = decentralized model-based MARL, unseen-generalization
  = the differentiator.
- [DONE cycle 65] W2 de-confliction FIGURE F15 (make_figures.py) from contention8_155056; embedded in
  tutorial (§9). Shows ours (private offset) beats CBBA-backoff + MAB re-seating at severe contention.
- [ ] W1-ORIG MARL framing paragraph -> paper (make_paper_v2.py, after Related work) + tutorial. Content:
      independent-MARL (IQL/indep-UCB)=our UCBIndep=floor on unseen (Thm1); broadcast-CF=decentralized
      MODEL-BASED MARL (shared world model = low-rank R, estimated privately); broadcast=passive
      sensing NOT learned messaging (vs CommNet/TarMAC/DIAL); ChoiceEM=decentralized teammate modeling;
      T7 offset=symmetry-breaking in a congestion/matching game. (Full notes in "MARL PERSPECTIVE" below.)
- [ ] W2 De-confliction FIGURE (make_figures.py new F15): earned reward vs pool {240,60,30,15} for
      ContentionAdaCF/ContentionCF (ours) vs CBBAlite vs MusicalChairs vs greedy RewardCFconv vs PTF,
      from latest results/pilots/contention8_*.json (the 9-method one, _155056). Embed in paper+tutorial 8.13.
- [DONE cycle 66, focused] W3 LaTeX main.tex sync: added the 3 highest-value items -- per-observer
  independent noise (formal model), the CBBA-lite + SIC-MMAB/musical-chairs de-confliction comparison
  (we win), and the UnifiedCF capstone (best-or-tied everywhere). NOTE: this is a FOCUSED sync of the
  new headline content, not a full line-by-line reconciliation; a deeper pass (figures, ablation table,
  anisotropy/CLUB rows) remains if a camera-ready is imminent.
- [ ] W3-ORIG LaTeX main.tex sync with the HTML (docs/paper_aamas/main.tex): the HTML moved ahead -- add the
      UnifiedCF capstone (best-or-tied everywhere via abundance gate), the CBBA/MusicalChairs
      de-confliction comparison, the per-observer-independent noise model statement, strict-ZK newcomer.
      Careful manual sync; verify it still reads.
- [DONE cycle 66] W4 Theory: P11-P15 promoted to full Propositions in THEORY_FORMAL.md (P11 crossover
  existence EXACT; P12 churn latency ORDER; P13 loss+abundance envelope EXACT at corners, theorizes
  UnifiedCF+ab; P14 VI discriminative-but-anti-conservative; P15 keystone = partial(rho=1 exact) +
  precise conjecture(persistent rho<1), honestly OPEN). Original sketch note kept below:
- [ ] W4-ORIG Theory writeups in THEORY_FORMAL.md (proposed P11-P15 already sketched there, ~line 465+):
      P13 adaptive-envelope (NOW partly realized by the abundance gate -- formalize: exploration should
      vanish when offer abundant OR loss high); P11 choice-vs-reward crossover sigma*; P12 churn fold-in
      latency; P14 calibration/VI; P15 KEYSTONE (decentralized masked low-rank U-recovery -- the central
      open problem; write the strongest partial result + precise conjecture + cite Nagaraj/Jain).

EXPERIMENTS / CPU (build first = non-CPU, then run <=2 pools at once):
- [DONE cycle 65] E-CLUB CLUB baseline: CLUB(_AccBase) in pilot_baselines.py (hard drone clustering),
      pilot_club.py runner -> docs/CLUB.md, catalogue row 58. FINDING: continuous low-rank (RewardCF) is
      the MOST masking-robust on unseen (rho=0.25 0.337 > CLUB 0.257, non-overlapping); cluster/memory
      edge ahead only at full broadcast; all structured >> floor. CLUB credible, not a strawman.
- [DONE cycle 65] E-COORD CoordCF negative-correlation exploration. Result row 59: FASTEST early
      (round-10 0.084 highest, rounds-to-half 21.2) -> explicit division-of-labor helps EARLY sample
      efficiency; collective-UCB (EMCF b=0.3) still wins FINAL (0.356 vs 0.336). Honest early-coverage
      win, not a final breakthrough. CoordCF in pilot_noise.py, pilot_explore.py REG.
- [DONE cycle 66] E-H11types: pilot_h11types.py (k1 sweep) -> docs/H11TYPES.md, catalogue row 61. HYP
  CONFIRMED: de-confliction value (AdaCF-greedy) +0.112 at K1=1 (identical) -> +0.035 at K1=30 (distinct);
  AdaCF beats greedy + both field primitives at every K1.

## ===== RAS PUBLISHING TRACK (user pivot cycle 67: target Robotics & Autonomous Systems) =====
- [DONE cycle 67] RAS grounding #1 -- SENSING-grounded observability: pilot_sensing.py places robots+tasks
  in a 2-D arena and DERIVES masking+noise from sensing radius R + distance-noise (sigma(d)=sigma0(1+d/R)),
  so rho/sigma are EMERGENT physics, not parameters. Catalogue row 62, docs/SENSING.md. RESULT: categorical
  unseen win SURVIVES geometry-limited sensing once coverage>=~0.3 (RewardCF unseen 0.15-0.30, structure-free
  ~0 at every radius); degrades only at ~10% coverage. Folded into paper (Robotics grounding + Sensing-grounded
  result) + tutorial (§8.17). This is the single biggest RAS-fit improvement (physical observability).
- [TODO RAS] Save docs/PUBLICATION_ASSESSMENT.md (T-RO + RAS honest assessments from this session).
- [TODO RAS] Deeper grounding for a stronger RAS submission: (a) a concrete SCENARIO (search/coverage/
  inspection) with robot capability vs task requirement semantics in a sim; (b) feature the tabula_drone
  PettingZoo validation (row 38) as a named environment, maybe add one more; (c) a sensing-grounded FIGURE
  (unseen skill vs coverage). (d) reformat main.tex to RAS (Elsevier elsarticle) class when submitting.
- [TODO RAS] (stretch, shared with T-RO) close/bound P15 keystone; small hardware or high-fidelity (Gazebo/
  Isaac/AirSim) demo would lift toward T-RO too.
## ----- BATCH cycle 68 (user: do D2, D4, D7/H8, E-CTDE, RAS grounding) -----
- [DONE cycle 68] D2 real ZK-MRTA env + real policies: tabula_bench.py re-run in the REAL PettingZoo
  tabula_drone env (m=9, n=27, spatial, HP depletion, episodic). Our weighted-ALS (=RewardCF) skill
  0.806 +-0.016 = BEST, approaches oracle, beats UCBIndep 0.721 and the env's own MF 0.251. The headline
  advantage TRANSFERS to the real env. tabula_bench_real.json (catalogue row 38). RAS: feature this as the
  named real-environment validation. (Further D2 extension if wanted: port UnifiedCF/EMCF as env policies.)
- [DONE cycle 68] E-CTDE / Rank5 ceiling: pilot_ctde.py, catalogue row 64, docs/CTDE.md. PRICE OF ZERO
  COMMUNICATION (8 seeds): CTDE-ceiling 0.553/0.489/0.434/0.271 vs AdaCF 0.448/0.205/0.153/0.100
  (pool 240/60/30/15). Comms-free recovers 81% of the comms-full ceiling at no contention, ~35-42% under
  contention (within-round coordination needs comms). All CTDE < oracle. Quantifies the no-comms cost.
- [DONE cycle 68, HONEST NEGATIVE] D7/H8 type-prior shrinkage: pilot_d7.py, catalogue row 63, docs/D7.md.
  10-seed result REVERSES the 1-seed smoke: type-prior UNDERPERFORMS population-prior at small k
  (type-minus-pop k=1 -0.329, k=3 -0.161, k=5 +0.012, k=16 -0.018). Inferring the type from <d probes is
  unreliable; a wrong centroid is a worse prior than the safe global mean. Population-prior (E7) is the
  robust cold-start choice. (D7/H8 closed: honest negative, hierarchical prior needs known/reliable type.)
- [RESOLVED-COVERED] D4 cold-start warm-start CONVERGENCE DYNAMICS: characterized by EXISTING results --
  E7 (catalogue row 55: unseen skill vs #own probes = the cold-start convergence curve, incl. strict-ZK
  recovery of U from the broadcast) + the anytime rho-sweep (row 30/F6: skill vs rounds, broadcast
  warm-start acceleration vs isolated). No separate run needed; the two axes (own-data, rounds x rho)
  jointly give the warm-start convergence dynamics. (If a dedicated single-drone factor-error-vs-round
  curve is wanted later, it is a thin wrapper over run_anytime tracking ||U_hat-U|| per round.)
## ----- BATCH cycle 69 (noise-assumption + foundational theory + small items) -----
- [DONE] Skill-score lineage cited in paper (§2) + tutorial glossary (Murphy skill score / RL normalized score).
- [DONE] Residual-sigma variant: RewardCFEstSigma (pilot_noise.py) estimates per-source sigma^2 from
  residuals. est-sigma study (pilot_estsigma.py, catalogue row 65): the 'noise known' assumption is NOT
  load-bearing -- est-sigma BEATS known-sigma (+0.101 hetero unseen), uniform best of all. EMCF noise-
  exposure correction in ZK_COMPLIANCE (EMCF uses true 1/sigma^2, does NOT estimate sigma).
- [DONE] FOUNDATIONAL THEORY (user "last effort"): THEORY_FORMAL.md T10 (fold-in perturbation bound,
  EXACT -- 3-source cold-start error, explains the sensing F16 curve), T11 (collective broadcast speedup,
  Theta(m) faster + impossible-alone -- the theory behind rho>0), P17 (minimax Omega(d) lower bound ->
  Theta(d)-vs-Theta(n) tight on both sides). Headlines folded into paper §4.
- [DONE cycle 70] OPERATIONAL MISSION (the clean applicative win): pilot_strike.py = target-servicing/
  dispatch on OUR STANDARD reward+skill (no metric drift), FULL field (RewardCF/EMCF vs structured
  PTF/ESTR/BPMF/SoftImpute/MFSGD vs structure-free), masked rho-sweep. WIN at rho=0.25: EMCF 0.360 /
  RewardCF 0.348 > best-of-field SoftImpute 0.289 (non-overlapping CIs), structure-free ~0; rho=1.0 EMCF
  0.485 leads. catalogue row 66, docs/STRIKE.md. Folded into paper + tutorial (8.18) with the TRAIT
  interpretation (latent = capability/requirement traits; extends trait-based MRTA to UNKNOWN traits).
  Lesson: COUNT/coverage metrics reward spreading (CF loses); REWARD/skill (act-well) is the win metric.
- [DONE cycle 70, companion] concrete SCENARIO sim: pilot_mission.py = AREA-INSPECTION mission (capability-vs-
  requirement quality, coverage with depletion, range-limited sensing). HONEST result (quality-vs-coverage
  tradeoff): CF wins MEAN INSPECTION QUALITY per engagement decisively (~0.25 vs ~0 structure-free) -- it
  dispatches the right drone to the right target; but pure COVERAGE breadth is an EXPLORATION objective
  that broad explorers (UCB/Random) brute-force, and exploit-CF does not cover faster. So CF's mission
  edge is DISPATCH QUALITY/EFFICIENCY, not blanket search. -> docs/MISSION.md (catalogue row pending run).
- [TODO RAS] reformat docs/paper_aamas/main.tex -> Elsevier elsarticle class (do at submission time).
## ===== END RAS TRACK =====
- [SUPERSEDED] E-H11types-ORIG contention with IDENTICAL vs DISTINCT drone types: does de-confliction depend on type
      homogeneity? vary K (clusters) in the contention world; if all drones same type they all want the
      same targets (max contention) -> private offset should matter MORE. Reuse pilot_contention with a
      type-controlled make_world.
- [DONE cycle 66] E-C4 anisotropy: pilot_c4.py (make_world_aniso, decay sweep) -> docs/C4.md, catalogue
  row 60. FINDING: low-rank unseen win ROBUST to anisotropy and GROWS as spectrum concentrates (RewardCF
  0.394->0.491), stays >> popularity (BiasModel 0.13-0.16) and floor (~0) at every decay. Honest:
  hypothesized shrink-to-rank-1, actual = robust/grows (collapse needs near-pure rank-1).
- [DONE-by-decision cycle 66] OPTIONAL fairness-hardening: intentionally NOT changed. The per-method eps
  schedules + PTF/ESTR clip + BPMF init are all GENEROUS to baselines (conservative for our claims), so
  matching them could only weaken baselines (wrong direction) and forces a full bake-off re-run. Left as
  documented in ZK_COMPLIANCE; no action.
- [DONE cycle 66] HOUSEKEEPING: removed the 4 orphan uncatalogued result JSONs.
- [ ] E-C4-ORIG anisotropy (skewed singular values / heavy-tailed factor spectrum): does the low-rank win
      survive non-uniform factor importance? Add an anisotropic make_world variant; reuse run_masked.
- [RESOLVED cycle 65, SUBSUMED] E-C12: C12 dynamic target onboarding was already run (catalogue row 19,
      cycle 19, c12_onboard) and the harness audit (ZK_COMPLIANCE) confirmed it COMPLIES (centralized
      als_fit over pooled OBSERVED tuples + ridge fold-in for the new target, uses d_hat, no true
      factors). E7 (rows 32/55) is the strict-ZK transpose (new DRONE). The "two-phase" framing is the
      same fold-in result viewed twice; no new run needed. If a paper wants a single onboarding figure,
      reuse F3 (target onboarding) + F10 (newcomer). DONE.

STILL TODO (next turn / post-compaction): W3 (LaTeX main.tex sync), W4 (theory P11-P15, esp P13
abundance-gate envelope now realizable + P15 keystone), E-H11types (contention K-sweep: vary type
homogeneity; HYP private-offset matters MORE when all drones share a type -> recommend: add Kover param
to run_contention, sweep K in {1,3,10,30} at pool=15, compare AdaCF/CBBAlite/MusicalChairs/greedy),
E-C4 (anisotropy: needs an anisotropic make_world variant -- skew the factor singular values -- then
reuse run_masked; HYP low-rank win survives non-uniform factor importance).
ALSO TODO (from audits/session, do not forget):
- E-CTDE / Rank5 BRACKET: centralized matcher WITH a shared model (Hungarian on a pooled CF estimate),
  reported like the ORACLE (not a competitor), to quantify the cost of the no-comms constraint.
- AUDIT-FIX (low pri, no headline depends on it): thread an explicit d_hat through the older true-d
  run_episode diagnostics (pilot_structure/_bootstrap/_confirm/_bakeoff/_em/_rank/_mfaudit/_starvation/
  _trust/_choice_only) if any ever feeds a paper claim; run_episode already accepts d_hat= (default true d,
  documented as oracle-rank). pilot_rank stays true-d intentionally (label it oracle-rank reference).
- OPTIONAL fairness-hardening (audit, conservative-for-us so not urgent): match exploration schedules
  across baselines (currently per-method eps); note PTF/ESTR clip-to-[-1,1] and BPMF idx-only init seed.
- HOUSEKEEPING: remove orphan uncatalogued result files (contention8_111224/_152650 [superseded 8-method],
  explore8_095853/_100154 [old smokes]) -- safe to git-clean; not referenced by any catalogue row.
- PUSH: local was ~113 commits ahead of origin/main; push when the user asks (DONE this cycle if pushed).

STATUS LINE (update me): cycles 62-65 committed (HEAD a61dfcb). DONE: H3 capstone+WIN (abundance gate,
best-or-tied everywhere), strict-ZK newcomer, harness+baseline ZK audits, per-observer noise clarified,
CBBA+MusicalChairs de-confliction baselines (we win T7), CLUB baseline, CoordCF, MARL framing, F15,
W1, W2, E-C12. REMAINING: W3 LaTeX sync, W4 theory, E-H11types, E-C4. ~113 commits ahead of origin (push is the user's call).
STALE-TAG RECONCILIATION (cycle 66 -- these older entries below are tagged [TODO]/[P*] but are
ACTUALLY DONE; treat as COMPLETE):
- C2 anytime/AUC metric = DONE (catalogue row 26, F6_anytime). C6 BPMF + Thompson = DONE (BPMF
  baseline, row 24). C7 rank/ARD = DONE (rows 45/50/57). C11 heterogeneous masking = DONE (the
  HEADLINE, rows 20+). C12 two-phase onboarding = DONE/subsumed (row 19 + E7 rows 32/55). C3a-d
  confidence motifs = DONE via EMCF + the confidence bake-off (row 46). C5 partial-sharing rho sweep
  = DONE (crossover rows 25/27/29). D6 contention/assignment = DONE (rows 42/44/48/57).
- H1 collective exploration = DONE (rows 47/52). H2 adaptive ContentionCF = DONE (rows 48/53).
  H4 calibration = DONE (row 51). H5 theory = MOSTLY DONE (T7/T8/P16/P10-P15). H11 sanity suite =
  DONE (H11b row 50, H11c row 52, E-H11types row 61 this cycle).
- GENUINELY STILL OPEN: E-H11types (running this cycle), E-CTDE/Rank5 ceiling bracket, W3 deeper
  LaTeX sync, D2 real-env+real-policies (partial via tabula_drone row 38), D4 convergence dynamics,
  D7/H8 hierarchical type priors, H10 (POSTPONED per user), D1/D5 (PARKED: robustness).

## ===== END ACTIVE WORKLOG =====

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
- [VALIDATED (8 seeds), honest residual] H3 = UNIFIED RECOMMENDED METHOD. UnifiedCF
  (pilot_noise.py): EMCF (confidence + predictive-variance UCB + ARD) UNITED with a loss-
  self-gating de-confliction offset (=PURE EMCF until the drone actually loses) AND a loss-gated
  exploration ANNEAL. Full 8-seed validation (pilot_unified.py -> UNIFIED.md): TIES the per-regime
  specialist in 4 of 5 -- standard anytime 0.437 vs EMCF 0.433; churn active 0.851 vs 0.842,
  recent 0.347 vs 0.371; contention pool=15 0.104 vs ContentionAdaCF 0.100 (both ~2x greedy 0.059).
  HONEST RESIDUAL: at pool=240 (no contention) earned reward 0.344 TRAILS ContentionAdaCF/greedy
  (~0.44) -- the explore/exploit tension: loss is low there so the anneal does not fire and EMCF's
  UCB exploration costs earned reward under capacity-1. So "ONE method best-or-tied EVERYWHERE" holds
  except no-contention earned reward; report honestly. FIX idea (P13 envelope): also gate exploration
  by the scarcity signal (offer<=k*m), not only by loss. TODO: fold UnifiedCF + this honest result
  into paper/tutorial as the single recommended method (choice channel stays a separate module).
- [DONE (cycle 63), gap SURVIVES] STRICT-ZK NEWCOMER. The row-32 E7 copied a peer's factors
  (U_hat = learners[0].U.copy(), p_pop = mean peers' P; exact only at rho=1). FIXED: the newcomer
  is now a PASSIVE RewardCF listener that recovers U from its OWN persistent-masked broadcast and
  folds in (population prior = mean of the teammate factors IT recovered). Incumbents masked at the
  same rho; rho swept {1.0,0.5,0.25}, 10 seeds (results row 55, F10 regenerated). RESULT: the
  categorical O(d)-vs-O(n) gap SURVIVES strict ZK at every rho -- Tabular ~0 everywhere; CF >>
  Tabular. rho=1.0 fold-in 0.28->0.57 >> popularity 0.27; rho=0.25 CF ~ pop ~ 0.30 (the
  probe-efficiency SLOPE flattens because the self-recovered U is the bottleneck under heavy
  masking) but both still categorically beat Tabular. Harness audit (ZK_COMPLIANCE.md) confirms
  this was the ONLY cross-learner parameter copy; C12 onboarding already fits from pooled
  observations (compliant). save_results() ROOT-anchored so future runs don't misplace JSONs.
  TODO(optional): re-state the E7 claim in paper/tutorial as "recovers U from own broadcast"
  (not "given U") and note the slope-flattens-under-masking caveat.
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
- [POSTPONED 2026-05-23 per user] H10 = CONSENSUS-GROUNDED informativeness (refined user reward-gradient idea; complements
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

## AUDITS (2026-05-23, two background agents)
- ZK-COMPLIANCE AUDIT: CLEAN. All methods/variants (pilot_noise + pilot_baselines) and all
  harnesses respect no-prior-knowledge (only d_hat, never P/U/R/true-rank/types), no-communication
  (private per-instance state, no shared/global mutable, no learner reads another's state), and
  partial-noisy-broadcast-only. Borderline-but-fine: OracleMate Rrow is env-only (commented);
  contention loss-signal choices[idx]==-1 is own public outcome; ChoiceCF within=True negatives use
  the PUBLIC offer set (broadcast), within=False is the strict-ZK variant. No code change needed.
- NOVELTY / PRIOR-WORK: the niche is OPEN but a niche (novelty is COMPOSITIONAL: all 4 constraints
  together, decentralized+private-params + broadcast-only + online/starved + unseen-pair categorical).
  CLOSEST neighbor = Multi-User RL with low-rank rewards (Nagaraj/Agarwal 2022, arXiv:2210.05355),
  which CENTRALIZES trajectories where we stay broadcast-only. Genuinely-new mechanisms: masking-
  robust zero-WEIGHT estimator; anytime-under-starvation separation; comms-free symmetry-breaking on
  a LEARNED low-rank preference (cf. multiplayer-MAB no-comms tradition); held-out choice-gamma
  (Dawid-Skene ported online). Weakest: bare T1-T5 (decentralized restatement of completion bounds).
  DONE: added closest-prior-work cites (nagaraj2022multiuser, katariya2017rank1, kang2024lowrank)
  + amato2024ctde to main.tex related-work + references.bib (replaced the PLACEHOLDER); softened to
  a COMPOSITIONAL-niche framing. TODO: fold the same cites into paper_v2/literature-review.

## BENCHMARK CANDIDATES (scouted; ranked) -- non-trivial MRTA/MARL baselines to add
[STATUS cycle 65 -- entries below are the original scout notes; current done-state:]
  Rank1 CBBA-lite = DONE (cat row 57, we win T7). Rank2 CLUB = DONE (cat row 58). Rank3 UCBIndep = HAVE.
  Rank4 SIC-MMAB/musical-chairs = DONE as MusicalChairs (cat row 57). Rank5 Hungarian coordination-CEILING
  bracket = STILL TODO (a centralized-with-shared-model ceiling, reported like the oracle; see CTDE bracket
  in MARL section). So only the Rank5 ceiling bracket remains unbuilt.
- [DONE (cycle 64), WE WIN at severe contention] CBBA-lite (broadcast-bid greedy/market auction):
  canonical MRTA baseline, consensus/comms step REMOVED. Implemented as CBBALite(RewardCF) in
  pilot_noise.py (same CF utility; reactive public-loss BACKOFF) + pilot_contention.py 9-method sweep
  (results row 57). RESULT (pool=15 earned): ours 0.100-0.105 > CBBAlite 0.064 (non-overlapping CIs);
  CBBAlite competitive at no/low contention (pool=240 0.464, lowest collisions). MusicalChairs
  (SIC-MMAB re-seating, scout Rank4) also DONE: 0.028 at pool=15 (worst; re-seating adds collisions).
  Both field primitives beaten by the proactive-static private offset (T7). Folded into paper §5 menu
  + related-work. (Pure tabular multiplayer-MAB = UCBIndep, already present.)
- [Rank2] Per-agent CLUB/COFIBA (clustering-of-bandits run LOCALLY on the broadcast, cluster on the
  CF-estimated latent factors since we lack context features): strongest structure-exploiting non-MF
  baseline; tests clustering(discrete) vs factorization(continuous) in our regime.
- [Rank3, MOSTLY HAVE] Broadcast-fed structure-free learner = our UCBIndep (per-arm table updated
  from the broadcast, floor on unseen) -> already empirically backs Thm 1 ("broadcast useless to
  tabular"); make this explicit in the paper.
- [Rank4] Multiplayer-MAB no-comms matcher (SIC-MMAB / musical-chairs) on top of CF-predicted prefs:
  the correct prior-art comparison for the contention symmetry-breaking claim.
- [Rank5, BRACKET] Distributed/greedy Hungarian with learned utilities: a coordination-CEILING
  bracket (NOT ZK-admissible), shows how much the no-comms constraint costs.
- INADMISSIBLE (state+dismiss): MAPPO/QMIX/VDN (centralized training), LinUCB/COFIBA-with-features
  (need context features), BanditMF (~= our BPMF), CBBA-with-consensus (needs comms).

## MARL PERSPECTIVE (analysis 2026-05-23; framing + baselines + improvements)
[STATUS cycle 65: PAPER FRAMING = DONE (W1, paper Related-work + tutorial §9). SIC-MMAB = DONE
 (MusicalChairs). Coordinated/negative-correlation exploration = DONE (CoordCF, cat row 59, honest
 early-coverage win). STILL TODO: the CTDE "ceiling" bracket (centralized matcher w/ shared model,
 reported like the oracle to quantify the no-comms cost) = same as Rank5 above.]
WHAT THIS IS, in MARL terms: a COOPERATIVE, DECENTRALIZED, COMMUNICATION-FREE multi-agent
BANDIT with low-rank structure (a "multiplayer matrix / low-rank bandit"), NOT a sequential
Markov game (reward R[i,j]=<p_i,u_j> is stateless, no transitions). Family = multi-agent MAB +
matrix factorization, under the Dec-POMDP umbrella but degenerate in the transition dim. State
this crisply in the paper to preempt "why not MAPPO?".
- [PAPER FRAMING, TODO] Add a short "MARL view" paragraph: (a) Independent learners (IQL /
  independent UCB) = our UCBIndep/Tabular = NO shared structure -> provably floor on unseen (Thm 1);
  so "independent MARL = structure-free = categorical floor". (b) Broadcast-CF = DECENTRALIZED
  MODEL-BASED MARL whose shared "world model" is the low-rank reward matrix, learned collectively but
  estimated independently = emergent coordination WITHOUT communication. (c) Our broadcast is PASSIVE
  OBSERVATION, not learned messaging (contrast CommNet/TarMAC/DIAL). (d) ChoiceEM = decentralized
  TEAMMATE MODELING ("what do others know?"); held-out gamma = teammate-competence estimator.
  (e) Fixed-private-offset (T7) = decentralized symmetry-breaking in a congestion/matching game.
- [BASELINE, Rank4 = TOP MARL] SIC-MMAB / musical-chairs: the MARL-NATIVE no-comms multiplayer-MAB
  matcher; pairs with CBBA-lite (auction side) to bracket the field's two standard comms-free
  de-confliction primitives. Build next after CBBA-lite folds in.
- [BASELINE, BRACKET] CTDE "ceiling": a centralized matcher WITH a shared model, reported like the
  oracle (NOT a competitor), to quantify the cost of the no-comms constraint from the MARL side.
  (Overlaps Rank5 Hungarian bracket; can be the same row.)
- [IMPROVEMENT, MARL-flavored] Coordinated / negative-correlation exploration: down-weight exploring
  a target you have SEEN teammates already probe in the broadcast, so the swarm DIVIDES exploration
  without comms. We partly have this (H1 collective-UCB / count-bonus); a cleaner version is a
  genuinely MARL coordination upgrade. Test vs eps-greedy + count-bonus on anytime + fresh-arrival.

## *** [DONE cycle 64] WIN ACHIEVED *** UnifiedCF+ab best-or-tied EVERYWHERE
RESULT (8 seeds, results row 56): the ABUNDANCE GATE (abundance_k=4: damp UCB when offer>4m, opt-in)
CLOSES the only residual. pool=240 earned 0.344 -> 0.425 [0.406,0.444] (ties greedy 0.439 / AdaCF
0.448, overlapping CIs) WHILE the 4 small-offer regimes are byte-identical (gate fires only at
offer>120): anytime 0.437, churn active 0.851 / recent 0.347, contention pool=15 0.104. UnifiedCF+ab
is best-or-statistically-tied in ALL 5 regime-metrics -> the design space collapses to ONE method,
no per-regime tuning. (The recommended UnifiedCF config sets abundance_k=4; default stays None so the
pilot_unified ablation keeps a clean with/without column.) TODO: fold this into paper/tutorial.
Original PLAN (kept for the record):
The ONLY loss is pool=240 no-contention EARNED reward (UnifiedCF 0.344 vs greedy/ContentionAdaCF
~0.44). Cause: exploration anneal is gated on LOSS only, so at ~0 loss the UCB bonus never anneals
and finite-horizon exploration spends earned reward it cannot cash back. PLAN (implement+test in
pilot_unified.py / UnifiedCF in pilot_noise.py), keep the 4 existing ties:
- (a) FINITE-HORIZON anneal (recommended, principled): beta_eff *= sqrt(max(T-t,0)/T) so late-episode
  exploration vanishes (value of information -> 0 near the horizon). Needs T threaded to the learner.
- (b) SCARCITY gate (P13 envelope): also reduce exploration when offer is abundant (offer >> m); best
  combined with (a) since scarcity alone cannot separate pool=240 from the anytime-shared regime.
- (c) CONFIDENCE gate: drop the UCB bonus once the top candidate's EMCF predictive sd < threshold
  ("explore until confident, then exploit"). Mild unseen-tail risk; test.
- WIN CONDITION: pool=240 earned -> ~0.44 (matches greedy) WHILE standard anytime ~0.437, churn
  active ~0.851 / recent ~0.347, contention pool=15 ~0.104 all HOLD (non-overlapping-CI ties). If so,
  upgrade the H3 claim to "one method, best-or-tied EVERYWHERE" (drop the honest-residual caveat).

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
