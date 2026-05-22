# Paper Reframing Plan v2: Choice-Only Observation + Dual-Likelihood BPMF

**Status**: FINALIZED DIRECTION. EXECUTING PHASE A (identifiability pilot).
Paper writing remains gated on the Phase A decision gate (see §7, §8).
**Created**: 2026-05-22. **Finalized**: 2026-05-22.
**Supersedes (in direction)**: REFRAMING_PLAN.md, whose structural-substitution
thesis was empirically falsified (see that file's §4b).

---

## FINALIZED THESIS: observation topology, not reward structure

> Whether collaborative filtering helps in decentralized multi-agent learning is
> NOT determined by whether the reward matrix is low-rank. It is determined by
> what agents observe about each other. When agents observe each other's
> OUTCOMES, latent structure is worthless and a trivial tabular learner matches
> everything. When agents observe each other's DECISIONS but not outcomes,
> latent structure becomes indispensable: independent learners are provably
> incapable of cross-agent transfer. We characterize this transition, prove the
> separation, and give the method that exploits it.

The paper's new axis is **observation topology**, not reward structure.

**Why this is groundbreaking, not just publishable:**
1. It converts every prior negative result into positive evidence. Sets M and N
   are the reward-observable side of the transition, where CF correctly offers
   no advantage (ratio ~1.0). They become the controlled baseline that proves
   the transition is real. Nothing was wasted.
2. It is a phase transition, not a percentage. In the decision-only regime,
   tabular methods do not get "a bit worse," they become structurally unable to
   learn from teammates. The empirical gap is unambiguous and grows with n.
3. It reverses conventional wisdom (share more to learn faster). The regime
   where collaboration REQUIRES structure is the one where agents share LESS
   (decisions only).
4. It is provable and honest. Showing both sides of the transition defuses the
   "you rigged the baseline" critique before a reviewer can raise it.

**Working title**: "Learning from decisions, not outcomes: a phase transition in
the value of collaborative inference for decentralized task allocation."

**The centerpiece figure (Phase B)**: sweep the observation model from full
reward-broadcast (p=1) to decision-only (p=0); the CF-vs-tabular advantage
crosses from ~1.0 to a large, n-growing gap as p falls.

**The single highest-value next action (Phase A)**: the identifiability pilot.
The whole edifice rests on recovering target factors U from choices; it is cheap
to test; no paper writing until that gate passes.

---

## 0. Why v1 failed (one paragraph)

v1 claimed "a single structural assumption (low-rank compatibility) can
substitute for all operational assumptions (coordination, communication,
priors)." Set N falsified this: the CF/tabular avg-steps ratio sat within
+/-3.4% of 1.0 at every true rank. Root cause: the public broadcast carried
**(action, reward)** for every drone, so every drone could reconstruct the
entire observed reward matrix. With that much shared information plus a
sample-rich budget, tabular methods (UCB-Indep, IQL-ZK) had no reason to lose.
The reward broadcast was leaking the whole table. The decentralization was
cosmetic.

---

## 1. The new core idea: asymmetric, choice-only observation

Change the observation model so the broadcast carries **choices but not
rewards**:

- A drone observes the **target selection** (target id) of every other drone,
  at every step, with **no reward / no effect** attached.
- A drone observes only the **(noisy) effect of its own** choice.

So drone i's evidence splits into:

- **Own row, with rewards**: r ~ N(<p_i, u_j>, sigma^2) for the targets i
  personally engaged.
- **All other rows, choices only**: i sees that drone k engaged target j, but
  never learns the reward k received.

This is *less* shared information than v1 (choices only, not rewards), so it is
a stronger, not weaker, ZK story. Observing where a teammate flew or which
target it engaged is a public action, not a message, not coordination, not a
shared parameter. It is also realistic: in real swarms you can see what a
teammate did but not the private reward it computed.

**Composition note**: the effect must stay genuinely private, so this pairs
naturally with the repeated-assignment regime (target HP huge, targets never
visibly deplete). In that regime there is no public HP-drop that would leak the
reward back through the side channel.

---

## 2. Why this makes CF shine (and cripples tabular)

### 2.1 Tabular is provably stuck on its own row

A tabular/independent learner for drone i keeps a value table Q_i[j]. Its only
update signal is its own reward on its own pulls. Another drone's choice j_k
carries no reward, so it produces no update to Q_i. Therefore Q_i[j] for any
target i has not personally pulled stays at the prior forever. Drone i must
personally pull every target to know anything: Theta(n) exploration per drone,
Theta(mn) across the team. This is not a tuning weakness; it is structural.
There is no mechanism in an independent learner to convert "drone k chose j"
into knowledge about my value for j.

### 2.2 CF turns others' choices into information

Under R[i,j] = <p_i, u_j>:

- Drone i's own row localizes p_i and pins down u_j for the few targets i
  pulled.
- Other drones' choices are **revealed preferences**. If drone k keeps choosing
  target j, that reveals <p_k, u_j> is large relative to k's other options,
  which constrains the latent geometry of u_j even with zero reward
  information. A model that ties drones and targets through shared latent
  factors converts "who chose what" into constraints on the unseen cells.

Result: CF needs roughly Theta(d) well-chosen own-pulls per drone to localize
itself, then predicts the rest. Tabular needs Theta(n). This is a clean,
provable per-drone separation, and unlike v1's Theorem 6' it would actually be
TRUE in this model.

### 2.3 Bonus: kills the "ZK redundancy" objection

Earlier critique of IQL/ESTR: all drones compute the same function of the same
broadcast, so decentralization is cosmetic. Here each drone has a *different
private row* of reward evidence, so per-drone computations are genuinely
distinct. Decentralization becomes real.

---

## 3. The method extension: dual-likelihood BPMF

Extend the existing per-drone Bayesian PMF
(`tabula_drone/policies/bayesian_pmf_policy.py`) to ingest two evidence types:

1. **Gaussian likelihood on own rewards** (already implemented): for i's own
   pulls, r ~ N(<p_i, u_j>, sigma^2). Conjugate closed-form update on p_i and
   on the pulled u_j.
2. **Categorical / softmax "choice likelihood" on others' selections** (new):
   for each observed selection k -> j, model
   P(k chooses j) proportional to exp(<p_k, u_j> / tau) over k's option set.
   This couples p_k and all candidate u_j.

Each drone maintains posteriors over **all** drones' factors p_1..p_m and all
targets u_1..u_n, with dense reward evidence on its own row and choice-only
evidence on every other row.

The softmax term is not Gaussian-conjugate. Options (decide during prototyping):
- Laplace approximation (one Newton step per update),
- Polya-Gamma augmentation (the clean Bayesian route that keeps it
  conjugate-ish and "proper PMF" in spirit),
- a small variational step.

"Per-agent Bayesian PMF under a choice-only broadcast, with a Gaussian
likelihood on own rewards and a categorical likelihood on others' choices" is,
to current knowledge, novel. It is exactly the mechanism the separation theorem
rewards.

---

## 4. The separation theorem (to be stated formally and proven)

**Informal statement**: Under the choice-only observation model, any
ZK-compliant policy whose per-drone estimator depends only on that drone's own
reward history (the tabular/independent class) requires Omega(n) own-pulls per
drone to achieve sublinear regret, whereas a latent-factor policy under the
rank-d assumption achieves the same with O(d * polylog) own-pulls per drone,
given identifiability of the factorization from the combined own-reward and
revealed-choice evidence.

This replaces v1's falsified Theorem 6'. Unlike 6', the gap here is real
because the information asymmetry is real: reward evidence for off-row cells
literally does not exist for an independent learner.

**Identifiability lemma (the crux to prove or assume)**: own-row rewards fix
the scale and i's own factor; revealed choices (argmax constraints) fix the
directions of the u_j up to the resolution of the choice set. State the
conditions under which these combine to recover U well enough for prediction.

---

## 5. Honest difficulties / open questions (we are NOT done thinking)

1. **Identifiability from choices.** A choice reveals an argmax, which
   constrains latent vectors to cones, not exact values. Own-row rewards anchor
   the scale. Whether the combination recovers u_j well enough is an empirical
   question; revealed-preference / inverse-choice / social-learning literatures
   suggest it is possible but data-hungry. MUST test in a pilot before
   committing.
2. **Coupled non-stationarity.** Each drone's choices reflect its evolving
   posterior, which depends on others' past choices. It is a multi-agent
   inference loop that could be slow or unstable. If it converges cleanly, that
   is itself a result; if it oscillates, we need a damping / stale-posterior
   scheme.
3. **Exploration coupling toward "operational."** We must keep choice
   observation strictly passive (drones see public actions). No agreed
   schedules, no messages, or we drift back into operational assumptions.
4. **Temperature tau identifiability.** The softmax temperature trades off how
   sharply choices reveal preferences; may need to be inferred or fixed by
   design.
5. **Does it still beat tabular if we also give tabular the choice stream but
   it cannot use it?** Yes by construction, but we should run that as the clean
   control to make the "tabular structurally cannot use choices" point
   airtight.

---

## 6. Supporting levers (keep in reserve)

Regime levers that push toward "structure binds":
- Repeated assignment with a hard pull budget (already validated; BPMF wins).
- Large matrix, small budget: m*n big relative to T.
- Binary / sparse 0-1 rewards (low information per pull).
- Slow latent drift (fatigue/weather): tabular relearns each cell, CF tracks a
  low-dim drift.
- Target churn / cold start: one pull from one drone places a target in latent
  space for everyone (CF), tabular needs each drone to pull it.

Method levers beyond the dual-likelihood:
- Hierarchical Gaussian-Wishart hyperpriors (full Salakhutdinov-Mnih BPMF).
- ARD / automatic rank selection (robustness to wrong d).
- Information-directed sampling instead of Thompson/UCB.
- D-optimal / leverage-score probe design.
- Cross-episode target-factor warm start.
- Student-t likelihood for outlier robustness.

Note on "large noise instead of scarce observations": noise amplifies
sample-starvation but does not substitute for a hard budget. The structural win
is the ratio mn / d(m+n), and sigma cancels out of that ratio. Use noise
together with a hard budget and a large matrix, not alone.

---

## 7. Proposed execution plan (GATED, do not start without confirmation)

### Phase A: Prototype + pilot (highest risk first)
- A.1 Add a choice-only observation mode to the env (broadcast carries target
  ids of all drones; reward delivered only to the acting drone). First-class
  config flag.
- A.2 Implement dual-likelihood BPMF (start with Laplace approx for the choice
  term; Polya-Gamma later if needed).
- A.3 Pilot: small matrix where Theta(n) vs Theta(d) should bite (e.g. n large,
  hard pull budget). Policies: UCB-Indep, IQL-ZK (both with and without access
  to the unusable choice stream), MF-CF, BPMF-dual. 3 seeds.
- **Decision gate**: does tabular collapse (flat per-drone learning off its own
  row) and BPMF-dual win clearly? If yes, proceed. If no, diagnose
  identifiability before scaling.

### Phase B: Full study (only if Phase A passes)
- B.1 Sweep n in {large set}, hard budget, choice-only model, all ZK policies,
  5 seeds.
- B.2 Ablations: choice stream on/off; reward-broadcast (v1) vs choice-only;
  vary tau; vary true rank d.
- B.3 Confirm the separation empirically (own-pulls-to-target-accuracy curves).

### Phase C: Theory + writing (only if Phase B confirms)
- C.1 State and prove the separation theorem + identifiability lemma.
- C.2 Reframe abstract/intro/conclusion around the choice-only observation
  model and the provable separation.
- C.3 Add method section for dual-likelihood BPMF.
- C.4 Results section. Rebuild .docx. Commit/push.

---

## 8. Acceptance criteria

The v2 direction is validated if:
1. In the choice-only model, tabular per-drone learning is flat for targets the
   drone has not personally pulled (the structural-impossibility control).
2. BPMF-dual achieves target-prediction accuracy with O(d) own-pulls per drone
   while tabular needs O(n).
3. BPMF-dual beats all ZK baselines on cumulative reward under a hard budget,
   by a margin that does NOT shrink to within noise (unlike v1).
4. The separation theorem holds under stated identifiability conditions.
5. The coupled multi-agent inference is stable (converges, does not oscillate).

If 4 of 5 hold, this is a clean, defensible, novel thesis. If fewer, return to
ideation (we expect to iterate).

---

## 9. Still open (because we are not done)

- Exact identifiability conditions for U from choices + own-row rewards.
- Whether to infer tau or fix it.
- Whether the coupled inference needs damping.
- Whether to also add a small "I can see my own choice in the public stream"
  consistency term.
- Alternative: model others' choices as soft evidence (probabilities) vs hard
  argmax constraints.
- Whether any of the reserve regime levers (binary rewards, drift, churn)
  should be combined with the choice-only model for an even sharper story.

---

## 10. Refinement (2026-05-22): the separation needs BOTH levers, and the
## make-or-break is U-recovery from choices

While wiring Phase A I sharpened the mechanism. Two corrections to the naive
story:

1. **Tabular cannot use others' rewards anyway** (heterogeneous rows: R[i,j] !=
   R[k,j]), so the reward-broadcast does NOT directly help an independent
   tabular learner. The earlier "broadcast leaks the whole matrix to tabular"
   intuition was loose. What the broadcast helps is CF (which estimates the
   shared U from all rewards).

2. **The separation requires sample-starvation AND choice-only together.**
   - Sample-rich (any observation model): every method estimates its own row
     directly; tie. (This is Sets M/N: no separation.)
   - Sample-starved + reward-observable: CF estimates U from ALL drones'
     rewards and wins; ordinary reward-only CF already beats tabular.
   - Sample-starved + decision-only (rewards private): reward-only CF collapses
     to own-row learning (no U signal off its own pulls), dropping toward
     tabular. ONLY the dual-likelihood method, which reads others' CHOICES,
     still recovers U and stays high.

So the centerpiece is the DIFFERENTIAL: as reward-share p goes 1 -> 0,
reward-only CF degrades to tabular while dual-likelihood CF stays high. The gap
that opens up is the value of the choice likelihood, and it is the paper's
unique claim.

**The make-or-break diagnostic (Phase A gate #1)**: U-recovery. For each drone,
Spearman correlation between predicted score <p_i, u_j> and true reward R[i,j]
over targets the drone NEVER pulled. Under decision-only:
- dual-likelihood: should be high (recovered U from choices),
- reward-only CF and tabular: should be ~0 (no off-row signal).
If dual-likelihood cannot recover U from choices (low Spearman), the thesis is
dead regardless of reward numbers. Test this FIRST.

**Pilot regime**: self-contained synthetic harness (no env), m drones, n
targets, rank d, mode-structured P/U, sample-starved changing candidate subsets
(T < n pulls/drone, random subset offered each step) so generalisation is
forced. Scripts: experiments/pilot_identifiability.py (offline gate),
experiments/pilot_choice_only.py (online, conjugate/RLS core).

---

## 11. RESULTS LOG (autonomous cycles, 2026-05-22)

### Cycle 1: identifiability gate (offline, competent choosers)
m=30, n=120, d=5, subset=12. Recovery quality (Spearman of predicted vs true
row; greedy reward; oracle greedy=0.836, random floor=0.145):

| obs (choices) | REWARD-ALS greedy | BPR-within greedy | BPR Spearman |
|---|---|---|---|
| 750  | 0.356 | **0.448** | 0.337 |
| 1500 | 0.512 | 0.453 | 0.371 |
| 3000 | 0.520 | 0.439 | 0.350 |
| 6000 | 0.679 | 0.449 | 0.374 |

FINDINGS:
- U IS recoverable from choices alone (far above random: 0.45 vs 0.145 greedy).
  Thesis NOT dead.
- At LOW data choices BEAT rewards (750 obs: 0.448 vs 0.356). A single
  argmax-over-12 choice carries more signal than a single noisy reward. This is
  the sample-starved advantage we are targeting.
- Choices SATURATE (~0.45 greedy, ~0.37 Spearman); rewards keep climbing and
  overtake by ~1500 obs. Choice recovery covers ~44% of the random->oracle gap.
- Within-subset negatives > global negatives at scale (0.374 vs 0.236), so
  observing offered sets helps; choices-only (global) still works at low data.
- Implication: the choice advantage is REGIME-SPECIFIC (very sample-starved).
  The online win must come from that regime. Competent-chooser gate is an UPPER
  bound; online (noisy learners) will be at or below this.

### Cycle 2: online single-pass RLS (conjugate core)
RLS fixed factor RECOVERY (Urec 0.06 [SGD] -> 0.25 [RLS]) but still lost to
Tabular on reward, because single-pass plug-in underfits the bilinear problem
(offline ALS on the same data reached Spearman 0.6). Conclusion: the conjugate
update is correct but needs multi-sweep convergence online.

### Cycle 3: warm-started batch refit (vectorised ALS)
n=120, m=20, T=50, changing subsets. Metric = final-model greedy (exploitation)
and online avg.

| policy | online p=1 | greedy p=1 | frac-oracle p=1 |
|---|---|---|---|
| Tabular  | 0.317 | 0.485 | 0.59 |
| RewardMF | 0.391 | **0.687** | **0.83** |

**GATE 3a PASSED. The reward-observable separation is SOLID**: in the
sample-starved regime, CF pooling the public reward broadcast beats independent
tabular by +42% greedy (0.687 vs 0.485). This is the result that explains every
prior negative (Sets M/N were sample-RICH; there CF and tabular tie). The
binding variable is sample-starvation, which prior ZK-MRTA benchmarks never
exercised.

### Cycle 4: decision-only (choices) -- still open
At p=0 (rewards private), getting choices to help online is hard:
- Separated (U from choices only): BROKE (Urec 0.017). Cold-start bootstrap
  chicken-and-egg: untrained models -> uninformative choices -> U never forms.
- Implicit-feedback DualBoot (choices as pseudo-rewards fused in one weighted
  ALS, anchored by own-reward scale): STABLE, ties RewardMF at p=1 (0.688), but
  choices add ~nothing at p=0 under MILD starvation (n=120: DualBoot 0.446 vs
  RewardMF 0.451). Reason: 42% own-pull coverage already fits the own row, so
  little room for choices. Testing harder starvation (n=200, m=40, T=40 -> 20%
  coverage) to see if choices help when own-data is genuinely insufficient.

### Cycle 4d: harder starvation (n=200, m=40, T=40) -- DECISION-ONLY FALSIFIED
| policy | greedy p=1 | greedy p=0 |
|---|---|---|
| Tabular  | 0.479 | 0.460 |
| RewardMF | **0.735** (+53%) | 0.447 |
| DualBoot | 0.752 | 0.344 (choices HURT) |

Reward-observable win STRENGTHENS with starvation (+42% -> +53%). But
decision-only choices got WORSE (0.344 < tabular), Urec dropped (0.111 < 0.158).

**Decision-only choice-transfer thesis FALSIFIED online.** Three reasons, now
understood:
1. Choice-recovery CEILING ~= tabular. Offline gate best-case greedy ~0.45;
   tabular gets 0.46-0.48. Even perfect online choice recovery only ties
   tabular. No room for a decision-only WIN in these worlds.
2. Online bootstrap deadlock + false positives: at p=0, sparse own-data ->
   incompetent early choices -> implicit feedback treats bad picks as positives
   -> corrupts U. Worse with more starvation.
3. Asymmetry: a reward is informative even from a random explorer (true value,
   poolable); a choice is informative only from a competent chooser (argmax,
   corruptible). Pooling rewards >> pooling choices.

### CONSOLIDATED THESIS (solid, true, defensible)
The CF advantage over independent tabular learning is governed by TWO knobs
TOGETHER: sample-starvation AND reward-sharing. CF wins (+40-53%) ONLY in the
corner (starved AND rewards shared). It ties tabular when sample-rich (any
sharing) OR when rewards are private (choices too weak). This:
- explains ALL prior negatives (Sets M/N were sample-rich -> tie),
- gives a clean 2D phase-diagram result (centerpiece: pilot_starvation.py),
- is honest about the observation-topology idea (the decision-only corner does
  NOT yield a CF win; only the reward-shared corner does).

The mechanism CF exploits: pooling the public REWARD broadcast across agents +
low-rank generalisation to unpulled arms under changing availability. Tabular
cannot pool (heterogeneous rows) and cannot generalise to unpulled arms; under
changing candidate sets it needs Omega(n) own pulls, CF needs O(d).

NOTE: the offline finding "a decision is more informative than a noisy reward at
very low data" survives as a SECONDARY information-content result (idealised /
competent choosers), but does not translate to an online decision-only win.

### Cycle 5: 2D PHASE DIAGRAM (starvation x reward-sharing) -- the centerpiece
m=30, d=5, T=50, cand=15, 5 seeds. CF/Tab = ratio of final-greedy reward.

| n | coverage | p=1 CF/Tab | p=0 CF/Tab |
|---|---|---|---|
| 30  | 100% | 1.16 | 0.92 |
| 60  | 83%  | 1.28 | 0.93 |
| 120 | 42%  | 1.38 | 0.90 |
| 240 | 21%  | **1.51** | 0.94 |
| 480 | 10%  | 1.33 | 0.93 |

CLEAN, PUBLISHABLE RESULT:
1. CF advantage is NON-MONOTONIC in starvation: rises to +51% at ~21% coverage,
   then declines at extreme starvation (CF can no longer fit U either). A "sweet
   spot," more interesting than a monotone trend.
2. Reward-sharing is NECESSARY: at p=0, CF/Tab ~ 0.90-0.94 flat across all
   starvation. Without shared rewards CF never beats tabular.
The advantage lives strictly in the corner (starved AND reward-shared).

### Cycle 6: LATENT STRUCTURE sweep (p=1, starved, skill metric)
CF-minus-Tabular skill gap (CF wins for ALL structures; gap 0.10-0.34):

| structure | Tab | CF | gap |
|---|---|---|---|
| onehot eps.15 nc5 | 0.57 | 0.84 | 0.27 |
| onehot eps.35 nc5 | 0.55 | 0.86 | 0.31 |
| clustG eps.10 nc5 | 0.63 | 0.92 | 0.29 |
| clustG eps.10 nc15 | 0.52 | 0.86 | **0.34** |
| clustG eps.10 nc40 | 0.53 | 0.86 | 0.33 |
| gauss continuous | 0.56 | 0.88 | 0.32 |
| onehot sharp3 | 0.56 | 0.75 | 0.19 |
| clustG sharp3 | 0.57 | 0.83 | 0.26 |
| gauss sharp3 | 0.40 | 0.49 | **0.10** |

FINDINGS:
1. CF advantage is ROBUST: positive for every latent geometry. Main result is
   NOT an artifact of the one-hot structure used in cycles 1-5.
2. More clusters / rarer good targets WIDEN the gap (nc15-40 > nc5): tabular
   can't stumble onto rare good targets, CF predicts them.
3. Cluster tightness (eps) has small effect.
4. KEY: reward SHARPENING (elementwise nonlinearity) collapses the CF advantage
   (gauss 0.32 -> 0.10). An elementwise power of a rank-d matrix is HIGH rank,
   so CF's rank-d model can no longer fit it. CF's edge REQUIRES genuinely
   low-rank reward. This reconnects to v1's "low-rank necessary" thesis, in the
   correct (sample-starved) regime that Set N never tested.

### Cycle 7: TRUE-RANK sweep (p=1, starved). CF given oracle rank.
CF-minus-Tabular skill gap by true rank d (m=30, n=120):

| d | Tab | CF | gap | CF/Tab |
|---|---|---|---|---|
| 1 | 0.82 | 0.86 | 0.04 | 1.05 |
| 2 | 0.65 | 0.94 | 0.29 | 1.44 |
| 3 | 0.60 | 0.94 | 0.33 | 1.54 |
| 5 | 0.56 | 0.88 | 0.32 | 1.58 |
| 8 | 0.51 | 0.60 | 0.09 | 1.17 |
| 15 | 0.49 | 0.62 | 0.14 | 1.28 |
| 30 (full) | 0.46 | 0.47 | 0.01 | 1.02 |

FINDINGS (low-rank is load-bearing, with a Goldilocks shape):
1. Full rank (d=30): CF advantage vanishes (1.02). LOW-RANK IS NECESSARY.
   Confirms v1's thesis in the correct (starved) regime; Set N missed it because
   it was sample-rich (rank never bound). At d=2 starved CF/Tab=1.44, vs Set N's
   sample-rich d=2 ratio 1.034 -- consistent: the rank effect needs starvation.
2. d=1: advantage also collapses (gap 0.04). Rank-1 => all drones rank targets
   identically (one universally-best target) => no PERSONALISATION => tabular
   just finds the one good target. CF needs rank >1.
3. Sweet spot at intermediate rank d~3-5.

### CONSOLIDATED NECESSARY CONDITIONS for CF >> tabular (all with sweet spots)
1. Reward genuinely LOW-RANK but PERSONALISED: 1 < d << min(m,n). (cycles 6,7)
2. SAMPLE-STARVED + changing availability; non-monotonic, peak ~21% coverage.
   (cycle 5)
3. REWARD-SHARING (public broadcast poolable across agents). (cycle 5)
Robust across cluster geometry; destroyed by reward nonlinearity (raises rank).

### Cycle 8: competence-weighted bootstrap (user's RS idea) -- FIRST p=0 WIN
DualConf = decaying eps + per-drone competence gamma_k (time ramp x choice
consistency) + gamma-weighted folding of choices. p=0 skill, clustG nc15:

| n | cover | Tab | RewMF | Boot(naive) | DualConf |
|---|---|---|---|---|---|
| 120 | 42% | 0.554 | 0.527 | 0.354 | **0.598** |
| 240 | 21% | 0.411 | 0.401 | 0.220 | 0.383 |
| 480 | 10% | 0.284 | 0.269 | 0.187 | 0.253 |

BREAKTHROUGH (modest but real): DualConf BEATS tabular at p=0, n=120 (0.598 vs
0.554, +8%). FIRST decision-only win: choices transfer knowledge with NO reward
sharing, beating independent tabular. Competence weighting is essential
(DualConf 0.598 >> naive Boot 0.354). Win is at MODERATE starvation; at extreme
starvation (n=480) drones lack own-data to become competent choosers -> gamma
low -> signal dries up.

### Cycle 9: principled EM joint model -- INSTRUCTIVE NEGATIVE
EM (mixture of rational-softmax / uniform-random) p=0 skill:

| n | Tab0 | Conf0 | EM0 | Rew1 | EM1 |
|---|---|---|---|---|---|
| 120 | 0.554 | **0.598** | 0.351 | 0.863 | 0.883 |
| 240 | 0.411 | 0.383 | 0.234 | 0.780 | 0.741 |

EM works at p=1 (EM1 ~ reward ceiling) but FAILS at p=0 (EM0 ~ naive Boot,
0.351). Reason: EM's responsibility ties trust to MODEL-AGREEMENT p_soft(c),
which reintroduces the deadlock -- early, when U is wrong, genuinely-good choices
get low p_soft -> low responsibility -> down-weighted -> never learned. The
fallback-to-gamma only rescues the exactly-uniform case, not the model-is-wrong
case.

KEY METHODOLOGICAL FINDING: competence must be inferred from OBSERVABLE BEHAVIOUR
(choice consistency/entropy over time), NOT from whether the half-trained model
agrees. DualConf wins because it does the former; model-agreement EM fails
because it does the latter. The correct probabilistic framing: gamma_k is a
latent variable with a BEHAVIOURAL observation model, inferred separately, then
used as a confidence weight in weighted MF. (User's original RS intuition was
right; model-agreement EM is the instructive wrong turn.)

### Cycle 10: multi-seed confirmation (5 seeds, error bars)
| structure | n | CF/Tab p=1 | DualConf vs Tab p=0 |
|---|---|---|---|
| onehot | 120 | 1.44 | -0% (0.598+-.059 vs 0.599) |
| onehot | 240 | 1.69 | -3% |
| clustG | 120 | 1.57 | +6% (0.586+-.042 vs 0.552) |
| clustG | 240 | 1.89 | -10% |

1. REWARD-OBSERVABLE: ROCK-SOLID. CF beats tabular +44% to +89%, tight error
   bars, both structures, both starvations. This is the real result.
2. DECISION-ONLY: the earlier +8% win was a 3-seed fluke. With 5 seeds it is
   parity AT BEST (clustG n=120 +6% but std 0.042 swamps it; rest ~0 or
   negative). Competence weighting beats naive folding but only reaches tabular
   parity, exactly as the offline choice-recovery ceiling (~= tabular)
   predicted. THE DECISION-ONLY THESIS DOES NOT SURVIVE MULTI-SEED SCRUTINY.

## FINAL HONEST STANDING (after 10 cycles)
SOLID: a multi-seed phase characterization of WHEN collaborative filtering beats
independent tabular learning in decentralized MRTA -- CF wins (+44-89%) iff the
reward is low-rank-but-personalised (1<d<<min(m,n)), the regime is sample-starved
with changing availability, AND rewards are shared; non-monotonic sweet spots in
both starvation and rank; robust across cluster geometry; destroyed by reward
nonlinearity. Explains ALL prior negatives (Sets M/N were sample-rich).

WASHED OUT: decision-only choice transfer. Naive folding and model-agreement EM
fail; behavioural competence weighting (DualConf) reaches tabular PARITY but no
reliable win. Choices cannot beat shared rewards; the choice-recovery ceiling
~= tabular.

ASSESSMENT: solid, honest, complete CHARACTERISATION paper (JAAMAS-appropriate).
NOT clearly "groundbreaking": the strong result's mechanism is close to known
cooperative-bandit / matrix-completion territory (contribution = the
characterisation + explaining prior MRTA negatives); the novel decision-only
angle did not yield a positive result. Decision point for direction surfaced to
user. USER CHOSE: one more decision-only push.

### Cycle 11 (running): decisions are NOISE-IMMUNE (the realistic decision win)
Reframe: choices win not because rewards are ABSENT but because observed rewards
are NOISY while choices are CLEAN. Two-stage observation: acting drone gets clean
own reward (sigma_own=0.1) -> good choices; observers see others' rewards with
HIGH noise sigma_obs (swept) -> reward-pooling corrupted; choice (target id)
observed cleanly by all. Fair NOISE-AWARE RewardMFN (weights obs by 1/sigma^2, so
it optimally discounts noisy others -> falls back to ~tabular, not strawmanned).
Hypothesis: as sigma_obs rises, RewardMFN -> tabular while choice-based DualConfN
stays robust and OVERTAKES both. If so, a genuine, realistic decision-based win:
"when observing outcomes is noisy but observing decisions is clean, decision CF
dominates." 5 seeds.

CORRECTION (user, apples-to-apples): the first cut let the choice method ALSO eat
others' noisy rewards (strictly more info than the reward method) -> not a fair
channel comparison. Fixed: hold OWN info identical (every drone has own clean
reward) and vary ONLY the cross-agent channel.
  Tabular  : own reward only (no transfer)            [reward-class]
  RewardCF : own + others' NOISY rewards (noise-aware)[reward-class, transfer]
  ChoiceCF : own reward + others' CLEAN choices ONLY  [choice-class, transfer]
HEADLINE metric = ChoiceCF_comp - RewardCF (same own-info, differ only in
cross-agent observation type). Positive at high sigma_obs => clean choices beat
noisy rewards. Also within-class: Tabular vs RewardCF; ChoiceCF naive vs comp.

### Cycle 12 (running): WHICH teammates -- per-teammate, per-channel trust
User's idea generalised: each drone infers per-teammate reliability and folds
teammate info proportionally, for BOTH channels.
  - REWARD: RewardCFRobust = EM infers per-teammate precision tau_k from
    residuals (Student-t / IRLS). vs uniform RewardCF.
  - CHOICE: ChoiceCF comp (per-teammate gamma_k from behavioural consistency) vs
    naive (trust all choices).
Setting: heterogeneous teammates, a fraction FAULTY (random choices + garbage
reward broadcast). Drones do NOT know who is faulty. Measure RELIABLE drones.
Claim: as faulty% rises, inferring WHICH teammates to trust robustly beats
uniform pooling, both channels (Byzantine-robust decentralized low-rank CF).
This is the most likely route to a DECISIVE, novel positive result.

### Cycle 11 RESULT: decisions are NOISE-IMMUNE (apples-to-apples)
clustG nc15, n=240, own clean (sigma_own=0.1), sweep sigma_obs:
| sigma_obs | Tabular(solo) | RewardCF | ChoiceCF_comp | comp-Rew |
|---|---|---|---|---|
| 0.10 | 0.395 | 0.728 | 0.423 | -0.305 |
| 0.30 | 0.395 | 0.591 | 0.423 | -0.168 |
| 0.60 | 0.395 | 0.491 | 0.423 | -0.068 |
| 1.00 | 0.395 | 0.451 | 0.423 | -0.028 |
| 2.00 | 0.395 | 0.368 | 0.423 | +0.055 |
FINDINGS: choice channel is FLAT (0.423) regardless of obs-noise (choices carry
no observation noise); reward channel degrades 0.728->0.368 and DROPS BELOW solo
at sigma_obs=2.0 (pooling noisy outcomes becomes HARMFUL). Crossover: decisions
beat outcomes once outcomes are noisy enough. Real but needs high noise; modest
margin. Note ChoiceCF_comp (0.423) > solo (0.395) robustly.

### Cycle 12 RESULT: choice-trust WORKS; reward-trust was a BUG
| faulty% | RewardCF | RewRobust | Ch_naive | Ch_comp | C-gain |
|---|---|---|---|---|---|
| 0 | 0.636 | 0.636 | 0.276 | 0.435 | +0.159 |
| 20 | 0.527 | 0.527 | 0.251 | 0.424 | +0.173 |
| 40 | 0.390 | 0.390 | 0.225 | 0.405 | +0.180 |
- CHOICE channel: competence weighting ROBUST. Ch_comp ~flat (0.435->0.405)
  while Ch_naive collapses (0.276->0.225); gap GROWS with faulty% (+0.159 ->
  +0.180). Confirms "which teammates" for choices.
- REWARD channel: R-gain = +0.000 = BUG (RewardCF.observe called _als directly,
  bypassing the RewardCFRobust._refit override -> robust EM never ran). FIXED
  (observe now calls self._refit()).

### Cycle 13 (running) = P1 strengthened: collaboration-harm threshold
Fixed reward-robust + explicit SOLO baseline + faulty% up to 0.5. Tests the
KILLER claim: naive pooling drops BELOW solo as faulty% rises (collaboration
HARMFUL) while trust-aware stays >= solo, BOTH channels. If it holds: the
groundbreaking spine ("how to collaborate SAFELY under unreliable peers").

NOTE: research agent 1 (latent-confidence) failed on connection (0 tokens);
re-dispatched. Agent 2 (cold-start dynamics) still running.
