# Experiment Program: Confidence-Aware Decentralized CF (toward a groundbreaking result)

**Created**: 2026-05-22. **Status**: prioritized plan; P1/P2 are the spine.

## The candidate groundbreaking thesis

After 12 cycles, the strongest novel-AND-true claim is NOT "CF beats tabular when
starved" (real but close to known cooperative-bandit territory). It is:

> In decentralized multi-agent learning where agents observe each other's
> DECISIONS (not private outcomes), the key to robust collaboration is inferring
> per-agent, per-decision CONFIDENCE/RELIABILITY. Naive pooling of peer signals
> is fragile, beyond a threshold of unreliable peers it becomes WORSE than not
> collaborating at all. A latent-confidence model that infers whom and when to
> trust makes collaboration provably safe (never worse than solo) and robust to
> cold-start, noise, faulty, and adversarial agents. Moreover, DECISIONS are a
> more robust collaborative signal than OUTCOMES under observation noise.

Why this can be groundbreaking: (a) it answers "when is collaboration SAFE, and
how to guarantee it" - novel, timely (Byzantine/federated robustness); (b) it
gives CLEAN, decisive wins (naive fails catastrophically, confidence-aware is
robust), not marginal effects; (c) it pairs with theory (influence bounds,
collaboration-safety monotonicity); (d) it reframes our whole body of results
(including the negatives) into one coherent story about confidence.

Fixed setting for all experiments (adapted to our pilots): m=30 drones, n in
{120,240}, rank d=5, changing candidate subsets (cand=15-20), T=50, structure
cluster_gauss nc15 (low-rank, robust), 5+ seeds, skill=(greedy-rand)/(oracle-rand).
ZK: no parameter sharing, public broadcast only, online.

---

## Prioritized experiments

### P1 (HIGHEST): the "collaboration-harm threshold" -- Byzantine robustness
Extends cycle 12. Vary faulty fraction f in [0,0.1,0.2,0.3,0.4,0.5] x faulty
type {random, lazy(stuck on one bad target), adversarial(inverted preference)}.
Methods: SOLO (tabular, no collaboration), NAIVE-POOL CF (trust all peers),
CONFIDENCE-AWARE CF (per-teammate trust via EM: tau_k rewards, gamma_k choices).
KILLER RESULT to establish: a threshold f* beyond which NAIVE-POOL < SOLO
(collaboration becomes HARMFUL), while CONFIDENCE-AWARE >= SOLO for ALL f and
>= NAIVE everywhere. The crossover figure is the centerpiece.
Success: confidence-aware never below solo; naive dips below solo past f*; gap
grows with f. Groundbreaking if the crossover is sharp and robust.

### P2 (METHOD): continuous latent-confidence EM/VB model
Implements the user's idea principled. Decision model: P(c|O) = softmax(beta_{k,t}
<p_k,u_o>), beta = latent confidence. Sources to ablate: (i) behavioural
consistency, (ii) chooser posterior precision (Bayesian drones), (iii) HKV-style
preference/confidence split, (iv) time/experience ramp. Infer beta jointly with
factors via EM (behavioural E-step, NOT model-agreement which deadlocked).
Test: beats heuristic DualConf and naive in P1's settings; which confidence
source matters most. This is the methodological contribution that powers P1/P3.

### P3: adversarial / Byzantine extreme + theory
Adversaries actively corrupt (coordinated inverted choices, factor-poisoning).
Show trust-inference isolates them. Pair with THEORY: bound the influence of an
eps-fraction of arbitrary agents on the trust-weighted factor estimate
(robust-aggregation O(eps/(1-eps)) style bound) and a collaboration-safety lemma
(confidence-weighted pooling >= solo under bounded reliability). Theory+empirics
is what elevates to top-venue.

### P4: decisions vs outcomes under observation noise (channel robustness)
Extends cycle 11. Map crossover over sigma_obs x starvation x structure: clean
CHOICE channel vs noisy REWARD channel (identical own-info). Establish the regime
where decisions dominate outcomes. Novel secondary claim: "share decisions, not
outcomes, when outcomes are noisy to observe."

### P5: trust-weighted HYBRID fusion of both channels
Each drone fuses own rewards + others' (noisy, trust-weighted) rewards + others'
(confidence-weighted) choices in ONE model. Show hybrid dominates either channel
alone across the noise/faulty grid. "Optimal fusion of decision and outcome
signals under uncertainty."

### P6: cold-start warm-start + convergence dynamics
Late-joining drones (no own data) warm-start from broadcast-derived U vs random.
Measure cold-start regret and time-to-competence. Track system convergence:
does confidence-aware exploration avoid premature convergence / filter-bubble
pathologies that naive greedy suffers? (RecSys feedback-loop literature.)

---

## Sequencing and rationale

1. P1 first: highest groundbreaking potential x highest success probability;
   extends already-running cycle 12; robustness wins are decisive not marginal.
2. P2 in parallel/next: the method that makes P1 work best and is the
   methodological novelty.
3. P3 (adversarial + theory): converts P1 into a top-venue contribution.
4. P4, P5, P6: strengthen and broaden; secondary novelty (decisions-as-signal,
   fusion, cold-start).

## What would make us DECLARE groundbreaking
- P1 crossover holds, multi-seed, across faulty types and structures.
- P2 confidence model robustly best; ablation shows posterior-precision and/or
  consistency is the key confidence source.
- P3 theory bound matches empirics.
If P1 fails to show naive<solo (i.e., naive pooling is already robust), the
Byzantine angle weakens -> fall back to P4 (decisions-vs-outcomes) as the novel
thread, plus the solid reward-observable characterization.

## Risks
- P1 may show naive pooling is already robust (population averaging dilutes a few
  faulty agents) -> need high faulty fraction or adversarial (coordinated) agents
  to break it. Mitigation: include adversarial/coordinated faulty in P1 from the
  start, and push f to 0.5.
- Confidence inference may not beat heuristic by much -> still fine if it powers
  robustness; the story is robustness, not raw accuracy.
