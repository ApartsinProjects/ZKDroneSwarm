# Paper Outline (draft, 2026-05-22)

Working title: **"Acting on the Unseen: Collaborative Filtering for Decentralized
Multi-Robot Task Allocation under Limited, Communication-Free Observability."**

## Thesis (one sentence)
In a swarm with unknown low-rank agent-task compatibility, limited & heterogeneous
observability, and NO communication, collaborative filtering over the public
broadcast lets each agent act well on agent-task pairs it has NEVER observed and
onboard new tasks with O(d) shared probes, CATEGORICALLY beating independent
learning (which is at the floor by construction), with matching theory.

## Abstract (bullets)
- Problem: decentralized MRTA, no comms, partial/noisy observability, unknown
  low-rank compatibility.
- Method: per-agent CF (matrix decomposition) over the public broadcast.
- Result 1 (characterization): CF beats independent/tabular IFF low-rank-but-
  personalised reward + sample-starved/changing-availability + reward-sharing;
  ties otherwise (explains prior null results).
- Result 2 (categorical): CF acts on UNSEEN agent-task pairs (skill ~0.5 vs
  tabular ~0, floor by construction), and this holds in the NATURAL heterogeneous-
  masking regime where per-agent states are genuinely unique.
- Result 3 (dynamic): new tasks onboarded for ALL agents from ~d shared probes
  vs tabular's ~m (Theta(d) vs Theta(m)).
- Result 4 (comparison vs the field): against the full relevant method set
  (no-structure bandits UCBIndep/UCBHomo/Tabular; low-rank MFSGD/ESTR/PTF/BPMF),
  the unseen-pair win is an ESTIMATOR-INDEPENDENT property of low-rank structure
  (all 5 low-rank methods clear the no-structure floor). Our specific online
  weighted-ALS adds two edges that matter in THIS regime: (a) masking-robustness
  (unseen skill stays flat as the broadcast is masked, while batch-SVD hybrids
  ESTR/PTF/BPMF decay because they SVD an R_hat with unobserved entries imputed
  0), and (b) anytime optimality (no probe phase) -- on cumulative reward
  ("targets destroyed @K") ours dominates at EVERY horizon and density (+~48% over
  the strongest competitor PTF at the final round), while UCBIndep is stuck ~random
  because n>>T keeps it perpetually exploring. The only metric a competitor wins is
  final-policy quality at FULL broadcast (rho=1), which is the no-observability-
  limit case the problem excludes.
- Theory: per-agent Theta(d) vs Theta(n) sample-complexity separation; categorical
  on unseen pairs.

## Sections
1. **Introduction.** Decentralized MRTA without communication; the generalization
   question (act beyond what you personally observed). Contributions list.
2. **Setting & model.** K1xK2 block model (types), rank-d cosine reward (unit
   factors; no nonlinear link). Observability: public broadcast; CHOICES with
   masking, REWARDS with additive noise; reward-sharing fraction p. Baselines:
   RANDOM (floor), TABULAR/independent (own-row optimal), ORACLE (centralized +
   complete information ceiling). Metric: skill=(method-random)/(oracle-random);
   AUC; unseen-pair skill. Fairness: CF uses a GUESSED rank (no oracle rank).
3. **When does CF help? (characterization).** Phase structure over (latent
   structure x observability): non-monotonic sweet spots in starvation and rank;
   collapses at full rank, at d=1 (no personalisation), and under reward
   nonlinearity. Decision-only choices ~ tabular parity. Bake-off: the fused
   variant (BothCF / confidence-gated) dominates the grid. [cycles 1-17]
4. **The categorical result: acting on unseen pairs.** C8 (static, 0.496 vs
   0.006, fair guessed rank); C11 (NATURAL masked regime: holds at every rho;
   per-agent states genuinely unique, state-divergence rises with masking ->
   decentralization is real); C13 (the gap scales with low-rankness).
5. **Dynamic task onboarding.** Two-phase: learn agent factors P, then onboard a
   new task via d-dim ridge/fold-in given P from ~d shared probes; all agents
   then predict it. Theta(d) vs Theta(m). [C12]
6. **Comparison to relevant methods (cycles 23-26).** ALL competitors in ONE fair
   harness (guessed rank d_hat, masked broadcast, decentralized): no-structure
   bandits (UCBIndep per-(agent,task) UCB1; UCBHomo shared arm table; Tabular
   eps-greedy) and low-rank methods (MFSGD online SGD-MF; ESTR explore-then-
   spectral-refit, the centralized low-rank bandit of Kang-Hsieh-Lee'22 style;
   PTF probe-then-fit = UCB-probe -> SVD warm-start -> SGD finetune; BPMF Bayesian
   PMF with Thompson). Three lenses:
   (a) Final-policy unseen skill [C14]: all 5 low-rank methods clear the
       no-structure floor -> the categorical result is estimator-independent;
       UCBHomo gets only partial unseen (rank-1 "popularity", no personalization).
   (b) Masking-robustness [C15, F5]: ours stays flat in rho; ESTR/PTF/BPMF decay
       (they SVD an R_hat with unobserved entries imputed 0). Crossover ~rho=0.55:
       PTF leads only at dense broadcast (rho>=0.7).
   (c) Anytime cumulative reward [C16, F6]: ours wins at EVERY horizon and density
       (final-round +~48% over PTF); UCBIndep stuck ~random (n>>T: never stops
       exploring); ESTR/PTF pay a probe phase. This is the operationally decisive
       result; it shows PTF's dense-rho final-policy edge is irrelevant in
       practice (it earns ~random while probing).
7. **Theory (docs/THEORY.md).** Prop 1 (tabular floor), Prop 2 (CF O(d) row
   completion), corollary (separation, categorical), decentralization+masking,
   block-model refinement. Novelty vs centralized MC bounds. Anytime corollary:
   in the n>>T sample-starved regime, per-arm methods cannot exploit (offer
   contains untried arms w.h.p.) -> structure-free anytime skill ~0.
8. **Related work.** Matrix completion (Candes-Recht, Keshavan OptSpace) -- we use
   ONLINE weighted-ALS, not a single batch SVD, hence robustness to masking.
   Low-rank bandits: explore-then-commit / spectral (ESTR/Kang-Hsieh-Lee) and
   probe-then-fit hybrids -- centralized and/or phase-structured; we are anytime &
   decentralized. Bayesian PMF (Salakhutdinov-Mnih) -- batch/Thompson, over-
   explores in the anytime regime. Structure-free bandits (UCB1) -- no cross-arm
   generalization, fail under n>>T. Co-clustering / bipartite MMSBM; federated/
   gossip CF (they share factors -- we do not, only the public broadcast);
   exposure/MNL choice debiasing; cold-start meta-learning; bandits-for-rec.
9. **Limitations & future.** PTF beats us on final-policy at rho=1 (full broadcast
   = no observability limit, excluded by premise) -- a probe-then-online-ALS
   hybrid is a natural extension. Non-contention only (contention/assignment ->
   Hungarian, where coordination value appears: future axis D6); method polish
   (precision-gated fusion, Bayesian/active exploration); real-env validation.

## Headline figures/tables (data in results/pilots/, catalogued; PNGs in docs/figures/)
GENERATED (regenerable via experiments/make_figures.py from saved JSON):
- F2_unseen_masking.png: C11 unseen-pair skill vs rho, CF vs Tabular -- the
  categorical win under masking. [cycle 20]
- F3_onboard.png: onboarding curve, skill vs #probes, CF vs Tabular (Theta(d) vs
  Theta(m)). [cycle 19]
- F4_rank.png: unseen-pair skill vs true rank d (CF) with Tabular floor. [cycle 21]
- F5_crossover.png: masking-robustness -- unseen skill vs rho for ours vs
  ESTR/PTF/BPMF vs UCBIndep floor (ours flat, batch-SVD decays). [cycle 25]
- F6_anytime.png: anytime cumulative reward trajectory (rho=0.25) -- ours earns
  from round 1; ESTR/PTF pay a probe phase; UCBIndep stuck (n>>T). [cycle 26]
PLANNED:
- F1 phase diagram: CF/Tab vs (starvation x reward-sharing). [cycle 5 data]
- T1 comparison table: all 10 methods x {final-unseen, masking-slope, anytime-AUC}.
  [C14/C15/C16 data]
- Box: the three propositions + the Theta(d)-vs-Theta(n) corollary.

## Status
Spine empirics DONE + multi-seed + complete data saved; theory drafted; FULL
method comparison DONE (cycles 23-26: vs UCBIndep/UCBHomo/Tabular/MFSGD/ESTR/PTF/
BPMF on final-policy, masking-robustness, and anytime metrics; 5 figures
generated). Remaining before submission: T1 comparison table render, F1 phase
diagram render, prose draft. Venue target: JAAMAS / AAMAS (decentralized MARL /
MRTA).

## Honest positioning (for the rebuttal file)
We do NOT claim universal dominance. PTF (probe-then-fit) achieves a better FINAL
policy at full broadcast (rho=1). Our claims, all evidenced: (1) the unseen-pair /
onboarding categorical separation over NO-STRUCTURE methods (the spine); (2) among
low-rank methods, our online weighted-ALS is uniquely masking-robust and anytime-
optimal, so it dominates on the operational metric (cumulative reward) at every
horizon and at every density rho<1 -- i.e. throughout the limited-observability
regime that defines the problem.
