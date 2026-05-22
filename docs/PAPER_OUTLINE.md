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
6. **Theory (docs/THEORY.md).** Prop 1 (tabular floor), Prop 2 (CF O(d) row
   completion), corollary (separation, categorical), decentralization+masking,
   block-model refinement. Novelty vs centralized MC bounds.
7. **Related work.** Matrix completion (Candes-Recht, Keshavan); co-clustering /
   bipartite MMSBM; federated/gossip CF (they share factors -- we do not);
   exposure/MNL choice debiasing; cold-start meta-learning; bandits-for-rec.
8. **Limitations & future.** Non-contention only (contention/assignment ->
   Hungarian, where coordination value appears: future axis D6); method polish
   (confidence-gated fusion, Bayesian/active exploration); real-env validation.

## Headline figures/tables (data in results/pilots/, catalogued)
- F1 phase diagram: CF/Tab vs (starvation x reward-sharing). [cycle 5]
- F2 unseen-pair table: C8 (static) + C11 (masked, vs rho) -- the categorical win
  + state-uniqueness. [cycles 18,20]
- F3 onboarding curve: skill vs #probes, CF vs Tabular. [cycle 19]
- F4 rank scaling: unseen-pair skill vs true d (CF) with Tabular floor. [cycle 21]
- F5 bake-off grid: variant x (structure x observability). [cycles 17 + gated]
- Box: the three propositions + the Theta(d)-vs-Theta(n) corollary.

## Status
Spine empirics DONE + multi-seed + complete data saved; theory drafted. Remaining
before submission: confidence-gated bake-off (running), final multi-seed
confirmation runs at paper settings, figure generation, prose. Venue target:
JAAMAS / AAMAS (decentralized MARL / MRTA).
