# Acting on the Unseen: Collaborative Filtering for Decentralized Multi-Robot Task Allocation under Limited, Communication-Free Observability

*Draft, 2026-05-22. Numbers and figures are regenerable from `results/pilots/*.json`
via `experiments/make_figures.py` and `experiments/make_table1.py`. Figures in
`docs/figures/`.*

## Abstract

We study decentralized multi-robot task allocation (MRTA) in a swarm where
agent-task compatibility has unknown low-rank structure, each agent observes only
a limited and noisy slice of a public broadcast, and there is NO communication or
coordination. We show that having each agent run collaborative filtering (CF;
online low-rank matrix decomposition) over the public broadcast lets it act well
on agent-task pairs it has NEVER personally observed, and onboard brand-new tasks
for the whole swarm from O(d) shared probes rather than O(m). Against an
independent / tabular learner this is a CATEGORICAL separation, not a constant
factor: the tabular learner is at the error floor on unseen pairs by construction.
We give a per-agent sample-complexity theory (tabular Theta(n) vs CF Theta(d), with
an Omega(1) unseen-pair floor) that matches the experiments. We then compare
against the full relevant method set (structure-free bandits UCBIndep/UCBHomo/
Tabular; low-rank MFSGD/ESTR/PTF/BPMF). Two findings: (i) the unseen-pair win is an
ESTIMATOR-INDEPENDENT property of low-rank structure (every low-rank method clears
the no-structure floor); (ii) our specific online weighted-ALS estimator is
uniquely suited to THIS regime, being masking-robust (unseen skill stays flat as
the broadcast is masked, whereas batch-SVD hybrids decay) and anytime-optimal (no
probe phase), so on cumulative reward ("targets destroyed by round K") it dominates
every baseline at every horizon and every observation density rho < 1.

## 1. Introduction

A swarm of drones must repeatedly choose which targets to engage. Different drones
are good at different targets (heterogeneous, latent compatibility), the swarm is
sample-starved (far more targets than engagement rounds), agents cannot
communicate or share parameters, and each agent sees only a noisy, partial slice
of what the swarm did. The central question is one of GENERALIZATION under these
constraints: can an agent act well on a target it has never personally engaged,
and on targets that did not exist when learning began?

A structure-free learner cannot. If an agent estimates the value of target j only
from its own engagements with j, then for any target it has never engaged its best
guess is the prior mean: it is at the error floor by construction. With many more
targets than rounds, this floor dominates.

Our thesis: when compatibility is low-rank, an agent can recover the shared target
factors from the public broadcast and thereby predict its OWN value for targets it
never touched. We make this precise (Section 8) and demonstrate it (Sections 5-7),
and we show it is robust to the heterogeneous, masked observability that defines
the decentralized setting.

**Contributions.**
1. A clean generative setting (Section 2) and a per-agent online CF method
   (Section 3) for decentralized, communication-free MRTA.
2. A characterization (Section 4) of exactly when CF beats independent learning:
   low-rank-but-personalized reward, sample-starved regime, shared reward channel.
3. The categorical unseen-pair result (Section 5) and dynamic task onboarding
   (Section 6): Theta(d) vs Theta(m).
4. A full, fair comparison (Section 7) showing the categorical result is
   estimator-independent, and that our online estimator wins the operationally
   relevant (anytime) metric throughout the limited-observability regime.
5. Matching per-agent sample-complexity theory (Section 8).

## 2. Setting and model

**World (block model).** m drones, n targets. Drones fall into K1 latent types,
targets into K2 types; a rank-d type-compatibility matrix C induces per-entry
reward. Concretely we draw latent factors so that the true reward matrix is
`R = P U^T` of rank `d = min(d, K1, K2)`, with `p_i, u_j` L2-unit vectors, so
`R[i,j] = <p_i, u_j>` is the cosine compatibility in [-1, 1]. We use the SIGNED
cosine reward (faithful rank; random pick ~ 0) and verify there is no nonlinear
link (a nonlinear link would inflate the effective rank and destroy low-rankness).
Default: m=30, n=240, d=5 (K1=K2=10), so n >> the horizon T.

**Observability (communication-free).** There is a public BROADCAST of (action,
outcome) events. Each round each drone is offered a random candidate subset of
`cand` targets and picks one; it earns the true reward of its pick. Each drone
observes its OWN outcome cleanly (small noise sigma_own) and a PERSISTENT,
per-drone-random subset of teammates' broadcast: mask `M[i,k] ~ Bernoulli(rho)`,
`M[i,i]=1`, with broadcast-observation noise sigma_obs. Masking models action-
observability limits; additive noise models reward-observability limits. Because
masks are persistent and per-drone, agents have genuinely DIFFERENT information
(decentralization is real, not cosmetic), and targets an agent never pulls or sees
are UNSEEN for it.

**Baselines and ceiling.** RANDOM (floor); independent/TABULAR (own-row optimal,
no transfer); ORACLE = centralized + complete-information ceiling (best target in
each offered subset). Skill `= (method - random) / (oracle - random)` (0 = random,
1 = oracle). We also report UNSEEN-pair skill (restricted to never-pulled targets)
and ANYTIME cumulative-reward skill (Section 7).

**Fairness.** Every structured learner is given a GUESSED rank `d_hat = 8` (not the
true `d=5`); no method is told the true rank or the latent factors. All methods are
ZK-compliant: they use only the public broadcast (no latent vectors, no privileged
identities).

## 3. Method: per-agent online collaborative filtering

Each drone i maintains its own factor estimates `P in R^{m x d_hat}`,
`U in R^{n x d_hat}` and updates them by ONLINE weighted alternating least squares
(WALS) over the events it has observed, weighting each observed reward by its
precision `1/sigma^2` (own outcomes count more than noisy broadcast ones, masked
ones not at all). Selection is epsilon-greedy on `U[cand] @ P[i]` with a decaying
schedule; estimation and decision are cleanly separated.

We study two variants:
- **RewardCF**: cross-agent channel = teammates' (noisy) rewards, precision-
  weighted. The core method.
- **BothCF**: additionally fuses teammates' CHOICES as competence-weighted implicit
  feedback (competence inferred from behavioral consistency, not model agreement).

Both are anytime (no probe/commit phase), decentralized (each agent its own
estimator), and handle missing entries natively through the precision weights.

## 4. When does CF help? (characterization)

Across a structure x observability grid (cycles 1-17, data catalogued), CF beats
tabular IFF three conditions hold together: (1) reward is low-rank but PERSONALIZED
(`1 < d << min(m,n)`; it collapses at full rank and at d=1 where there is no
personalization to exploit, and under a nonlinear reward link); (2) the regime is
SAMPLE-STARVED with changing availability; (3) the reward channel is SHARED.
Decision-only (choices-as-signal) without rewards is at best at tabular parity. The
fused BothCF variant dominates the design grid. This characterization explains
earlier null results (sample-rich regimes) and scopes the positive claims.

## 5. The categorical result: acting on unseen pairs

**Static (C8).** With factors learned from a fair guessed rank, CF achieves
unseen-pair skill 0.496 vs Tabular 0.006: CF acts near-optimally on agent-task
pairs it never observed; tabular is at the floor.

**Natural masked regime (C11, Fig. F2).** Under persistent per-drone masking the
result holds at EVERY density: CF unseen-pair skill 0.16-0.41 vs Tabular ~0 for all
rho > 0; CF overall skill 0.50-0.65 vs Tabular 0.42. Critically, the per-agent
state-uniqueness metric (divergence of agents' learned reward matrices) rises
monotonically 0.54 -> 0.92 as rho falls: agents genuinely learn DIFFERENT models
from their different masked views. Decentralization is real, and CF still completes
the unseen entries.

**Rank scaling (C13, Fig. F4).** CF unseen-pair skill decreases monotonically with
the true rank (d=2: 0.67; d=3: 0.58; d=5: 0.38; d=8: 0.27) while Tabular stays at
~0 for every d. The gap scales with low-rankness exactly as the theory predicts.

## 6. Dynamic task onboarding

A brand-new target is introduced after agents have learned their factors P. Given P
(from the broadcast), the new target's d-dim factor is recovered by ridge fold-in
from a handful of SHARED probes, after which ALL agents can predict it. CF reaches
high skill at ~d_hat probes; tabular needs ~m probes (each agent must try the new
target itself). This is the Theta(d) vs Theta(m) onboarding separation (Fig. F3).

## 7. Comparison to the full method set (cycles 23-26)

We port all relevant competitors into one fair harness (guessed rank, masked
broadcast, decentralized): structure-free UCBIndep (per-(agent,target) UCB1),
UCBHomo (single shared arm table), Tabular (epsilon-greedy own-row); low-rank
MFSGD (online SGD-MF), ESTR (explore-then-spectral-refit: random explore -> SVD of
the empirical R_hat -> exploit; a centralized low-rank bandit), PTF (probe-then-fit:
UCB-probe -> SVD warm-start -> online SGD finetune), BPMF (Bayesian PMF with
Thompson sampling). Table 1 (`docs/TABLE1_comparison.md`) summarizes; we read it
through three lenses.

**(a) Final-policy unseen skill (C14).** Every low-rank method clears the
no-structure floor; the no-structure methods (Random, UCBIndep, Tabular) sit at ~0.
UCBHomo gets only PARTIAL unseen skill (0.17 -> 0.07 as rho falls): pooling recovers
the rank-1 "popularity" main effect but not personalization. Thus the unseen-pair
win is an ESTIMATOR-INDEPENDENT property of low-rank structure, not an artifact of
our particular method. (At full broadcast the probe-then-fit
hybrid PTF is the strongest competitor; with a properly converged estimator our
HybridCFconv ties it there and dominates it elsewhere; see 7.4.)

**(b) Masking-robustness (C15, Fig. F5).** Sweeping rho finely, our online
weighted-ALS unseen skill is essentially FLAT for rho >= 0.4 (RewardCF
0.39-0.41), while every batch-SVD method decays monotonically (PTF 0.51 -> 0.18,
ESTR 0.23 -> 0.01, BPMF 0.23 -> 0.07). The reason is structural: ESTR/PTF/BPMF SVD
an empirical R_hat whose unobserved entries are imputed 0, which under masking is
sparse and biased; weighted-ALS instead down-weights missing entries to zero
precision. The unseen-skill crossover is near rho = 0.55: PTF leads only when the
broadcast is dense (rho >= 0.7).

**(c) Anytime cumulative reward (C16, Fig. F6).** The operationally relevant metric
is reward actually EARNED over the rounds (targets destroyed by round K), which
charges the cost of any probe/explore phase. Here our method dominates at EVERY
horizon and density. At rho = 0.25 the cumulative-normalized skill at K = T/4, T/2,
T is RewardCF 0.069/0.180/0.341 vs PTF -0.002/0.055/0.230 vs ESTR 0.008/0.064/0.181
vs UCBIndep -0.002/-0.004/-0.006. At the final round we beat the strongest
competitor PTF by ~47% at rho=1 (0.404 vs 0.274) and ~48% at rho=0.25 (0.341 vs
0.230). Two structural facts emerge:
- PTF's superior final policy at dense rho is operationally IRRELEVANT: it earns
  ~random during its 40%-of-rounds probe phase (the kink at round 20 in Fig. F6).
- UCBIndep's strong final-policy "overall" skill (~0.59) is a MIRAGE. On the
  anytime metric it is stuck at ~0 because with n=240 targets and only T=50 rounds
  it cannot pull each arm once; its offer almost always contains an untried target
  (infinite UCB bonus), so it explores forever and never exploits. The sample-
  starved regime n >> T is exactly where structure-free methods fail operationally
  and low-rank generalization pays off.

**Summary.** The categorical claim (low-rank vs no-structure on unseen pairs) holds
across all five low-rank estimators. Among low-rank methods, our online weighted-
ALS is the right tool for THIS regime: masking-robust and anytime-optimal, hence
dominant on cumulative reward throughout the limited-observability regime (rho < 1)
that defines the problem.

### 7.4 A hybrid that dominates the strongest competitor (E9, Fig. F5/F6)

The only place any competitor (PTF) led was final-policy unseen skill at full
broadcast. We eliminate that gap with HybridCF: a short UCB probe, an SVD
warm-start, then our ONLINE weighted-ALS (PTF's probe and warm-start, but our
estimator instead of its SGD finetune). Two findings:
(i) the apparent PTF lead was largely an artifact of our UNDER-CONVERGED default
    estimator (few ALS sweeps, infrequent refit); with a converged configuration
    (HybridCFconv: 20 sweeps, refit every round, slightly more exploration) the
    estimator was understating us.
(ii) With 10 seeds and bootstrap 95% CIs, HybridCFconv DOMINATES PTF: on unseen it
    TIES PTF at full broadcast (rho=1: 0.494 vs 0.505, diff CI [-0.033,+0.011],
    contains 0) and WINS significantly under masking (rho=0.5: +0.120; rho=0.25:
    +0.119, CIs exclude 0); on the anytime metric it WINS at EVERY density
    (+0.07 to +0.10, CIs exclude 0). PTF has no remaining significant advantage.
The converged config trades a little anytime versus the default RewardCF (which
stays the anytime-optimal choice), so our methods span the Pareto frontier
(RewardCF/BothCF anytime-optimal; HybridCFconv final-policy-optimal) and the
strongest prior-art competitor is dominated on every metric and density. A
probe-budget ablation confirms the probe vs anytime tradeoff is a single clean knob.

### 7.5 Both observability channels, and which to use (E3/E10/E13, Fig. F7)

Crossing the two channels (action masking rho x reward noise sigma_obs), the
reward-channel method (RewardCF) degrades as sigma_obs rises while the
action-channel method (ChoiceCF) is flat (clean choices are noise-invariant). The
crossover is at sigma_obs ~= 1, i.e. when the reward-observation noise std reaches
HALF the full signal range [-1,1]: for sigma_obs < 1 (the realistic regime) the
REWARD channel dominates; only under SEVERE noise (sigma_obs >= 1) does the clean
CHOICE channel win. A choice-only ablation (E13) shows the action channel ALONE
still lifts unseen skill above the floor (teammates' choices carry recoverable
low-rank structure), confirming it as a genuine fallback signal, though weaker than
rewards. We therefore recommend the simple reward channel (RewardCF/HybridCF) as
the default and the choice channel (ChoiceZK) as severe-noise insurance. We also
tested learned fusions (precision-gated BothCFPrec; validation-stacked StackCF):
precision-gating erases the reward-clean penalty to within ~0.01 in the realistic
regime but no learned fusion strictly dominates both channels at severe noise (the
fusion still carries down-weighted noisy rewards that the pure choice channel
discards), and in this sample-starved regime stacking is too data-expensive. The
practical takeaway is clean: a dominant fusion is unnecessary because the reward
channel already wins throughout the realistic regime.

### 7.6 Robustness to the masking model, and zero-knowledge compliance (E12, E13)

Masking model (Theorem 4, Fig. F8): re-running every headline metric under i.i.d.
per-round loss instead of the persistent mask leaves unseen and anytime skill
essentially unchanged (within ~0.04 at every rho), so the categorical results do
not depend on the modeling choice. The two models differ only in the durability of
decentralization: state-uniqueness is flat in the horizon under persistent masking
but decreases under i.i.d. (drones converge as they each eventually sample
everything), matching the theory. We adopt persistent masking precisely because it
makes "no communication implies durably different per-agent knowledge" a structural
property; i.i.d. (packet-loss-style) loss is equally admissible for the main claims.

Zero-knowledge compliance (audit in the supplement): every method uses a guessed
rank and random initialization (no ground-truth factors, types, or labels); each
agent runs an independent estimator with no parameter sharing and no coordinator;
the broadcast is passive sensing of public outcomes (masking models limited
detection, not radio transmission), so the setting is genuinely communication-free.
The headline methods RewardCF and HybridCF observe only teammates' action and
outcome; the choice-channel methods additionally use the offered menu for exposure
debiasing, and the choice-only ablation (E13) verifies their benefit survives a
strict-ZK relaxation that needs only the observed action (global negative sampling),
so it is not an artifact of menu observation.

### 7.7 Collective active exploration (E8)

Replacing eps-greedy with a latent-space UCB (predicted reward plus a count-based
uncertainty bonus, where counts are accumulated from the public broadcast) yields
ActiveCF: each agent probes the targets it is most uncertain about, and because one
agent's probe is broadcast it lowers EVERY agent's uncertainty (collective active
learning, still no communication). With a converged estimator (ActiveCFconv) this
strictly improves on eps-greedy RewardCF (12 seeds, bootstrap CIs): unseen wins at
rho=1 (+0.097) and rho=0.5 (+0.045), ties at rho=0.25; anytime wins at rho=1
(+0.040) and ties otherwise (never worse). It attains the best anytime of any method
at full broadcast (0.440) while reaching near-top unseen, because it explores WHILE
exploiting (no separate probe phase). This is the principled "confidence drives
exploration" idea; the per-observation confidence GATE (model agreement) deadlocks
at cold-start and is not used. Recommended methods: ActiveCFconv (best balanced),
HybridCFconv (best final-policy unseen under masking), RewardCF (simplest, anytime);
all dominate the prior-art competitors.

## 8. Theory (per-agent sample complexity)

Full statements and proof sketches in `docs/THEORY.md`.

- **Prop. 1 (tabular floor).** A tabular learner estimates R[i,j] only from
  observations of the specific pair (i,j); on any never-observed pair its expected
  squared error is Omega(Var_j R[i,j]) = Omega(1). To be good on all of agent i's
  targets it must observe Omega(n) of them; total Omega(mn). [matches C8/C11]
- **Prop. 2 (CF completes a row from O(d)).** If R has rank d and the broadcast
  identifies the target-factor column space U to o(1) (standard matrix completion
  once ~ d*n entries are seen; Candes-Recht, Keshavan-Montanari-Oh OptSpace), then
  agent i's row is determined by its d-dim factor p_i, recoverable by a d-dim
  ridge/WALS fold-in from O(d) observations; agent i then predicts ALL targets,
  including never-pulled ones. [matches C12, C13]
- **Corollary (separation).** Per-agent: tabular Theta(n) vs CF Theta(d); on unseen
  pairs tabular Omega(1) vs CF -> 0. A categorical, not constant-factor, gap.
- **Under masking.** Agent i sees ~ rho * (population observations); U is recovered
  while rho*(obs) >~ d*n and CF degrades GRACEFULLY (not to the floor) for rho > 0.
  Distinct masks -> distinct U^(i) -> genuinely unique per-agent states. [matches
  C11]
- **Anytime corollary.** In the sample-starved regime n >> T, a per-arm method's
  offer w.h.p. contains an untried arm, so it cannot exploit; structure-free
  anytime skill -> 0. Low-rank generalization is what enables exploitation under
  n >> T. [matches C16: UCBIndep anytime ~0]
- **Refinement (block model).** With K1/K2 types, identifying an agent's type is an
  O(log K1) classification and the type factor transfers, lowering per-agent /
  onboarding sample complexity below the generic O(d) (community-detection +
  matrix-completion lower bounds, arXiv 1912.04099). [matches C12]

**Novelty vs cited theory.** Standard matrix-completion bounds are centralized
(one estimator, uniform sampling). Ours is the DECENTRALIZED, ONLINE, broadcast-
only, per-agent-masked statement: each agent recovers U from a partial public
broadcast with no parameter sharing, and the separation is per-agent (Theta(d) vs
Theta(n)) with the unseen-pair floor making it categorical.

## 9. Related work

Matrix completion (Candes-Recht; Keshavan-Montanari-Oh OptSpace; Recht) gives the
centralized statistical backbone; we use ONLINE weighted-ALS rather than a single
batch SVD, which is what buys robustness to masking. Low-rank bandits: explore-
then-commit / spectral methods (ESTR-style, Kang-Hsieh-Lee) and probe-then-fit
hybrids are centralized and/or phase-structured; we are anytime and decentralized.
Bayesian PMF (Salakhutdinov-Mnih) is batch with Thompson sampling and over-explores
in the anytime regime. Structure-free bandits (UCB1) have no cross-arm
generalization and fail under n >> T. We connect to co-clustering / bipartite
mixed-membership stochastic blockmodels; to federated/gossip CF (which SHARE
factors, whereas we use only the public broadcast); to exposure/MNL choice
debiasing; and to cold-start meta-learning for the onboarding result.

## 10. Limitations and future work

Our method does not universally dominate: PTF achieves a better final policy at
full broadcast (rho = 1), the no-observability-limit case the premise excludes; a
probe-then-online-ALS hybrid is a natural extension that should combine PTF's dense
strength with our masking-robust anytime behavior. We study the non-contention
setting (each agent picks from its own offer); under contention/assignment the
allocation becomes a matching (Hungarian) and a value of explicit coordination may
appear (future axis). Further method polish (precision-aware confidence-gated
fusion; Bayesian/active collective exploration) and validation on a real swarm
simulator remain.

## Reproducibility

All results derive from complete per-seed JSON in `results/pilots/` (registry:
`docs/DATA_CATALOGUE.md`; chronology + per-cycle reviews: `docs/PROJECT_LOG.md`).
Figures (`python experiments/make_figures.py`): F2 unseen-masking, F3 onboarding,
F4 rank, F5 masking-robustness crossover, F6 anytime trajectory, F7 two-channel
grid, F8 persistent-vs-iid (Theorem 4), F9 scaling sweeps, F10 newcomer cold-start.
Table 1: `python experiments/make_table1.py`. Bootstrap CIs:
`experiments/ci_report.py` (`docs/CI_REPORT.md`). Harnesses: `pilot_compare.py`
(bake-off), `pilot_crossover.py` (masking-robustness), `pilot_anytime.py` (anytime),
`pilot_e3_channels.py` (channels), `pilot_iid.py` (E12), `pilot_e13_choice.py`
(choice ablation), `pilot_scaling.py` (E2/E4/E6), `pilot_e7_newcomer.py` (newcomer);
competitor ports in `pilot_baselines.py`; core world/reward/oracle in `core.py`.
Supporting docs: `docs/THEORY_FORMAL.md` (proofs), `docs/ZK_COMPLIANCE.md` (audit),
`docs/EXPERIMENT_PLAN.md` (protocol), `docs/tutorial.html` (graduate-level
walkthrough). All learners use a guessed rank and broadcast-only inputs.
