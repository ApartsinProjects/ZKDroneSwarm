# Next Steps — Toward a Publishable Experimental Evaluation

This document outlines a phased roadmap for turning the current single-configuration preliminary result into a statistically interpretable and publishable experimental evaluation. The first phase establishes reproducibility across scenario seeds; later phases test whether the method's design choices, capacity assumptions, noise robustness, and scaling behavior hold beyond the initial configuration.

---

## The Core Problem

The current paper reports one scenario seed, one policy configuration, and a 35-episode training run. This is sufficient for a preliminary proof of intent, but not yet enough for a complete experimental claim. A reviewer will first ask:

1. **Is the result reproducible across independently generated scenarios?**
2. **How much variation exists across latent worlds?**

Phase 1 addresses this immediate evidence gap. The remaining phases build on that foundation by testing supervision mode, factorization capacity, noise sensitivity, and swarm scale.

---

## Phase 1 — Multi-Seed Paired Benchmark Evaluation

**Goal:** Convert the current single-seed result into a statistically interpretable benchmark by evaluating the same experimental configuration across multiple independently generated scenario seeds.

**Design:** Run the random baseline, oracle benchmark, and matrix-factorization policy on the same set of scenario seeds, e.g. $S = \{42, 43, \dots, 61\}$ with $|S| = 20$. Each seed generates a distinct latent world: new drone latent vectors, target latent vectors, and target-mode assignments. Using the same seed set for all policies creates a paired comparison, so MF performance can be compared against random and oracle within each scenario rather than only in aggregate.

**What this tests:** This phase asks whether the seed-42 result is reproducible across problem instances. It does not yet test supervision modes, latent-dimension mismatch, noise robustness, or scaling. Its purpose is to establish the baseline effect size and variance of the current method.

**What to report:**
- Mean and standard deviation across seeds for all primary metrics.
- Paired improvement of MF over random for each seed.
- Paired gap between MF and oracle for each seed.
- Learning curves with mean and uncertainty band across seeds.
- Final-episode, best-episode, and last-$k$ average performance, where last-$k$ summarizes late-training stability.

**Expected interpretation:** If MF consistently outperforms random across most seeds and remains closer to oracle than random on efficiency and match-quality metrics, the current result becomes a reproducible preliminary finding rather than a single illustrative run. If performance varies strongly by seed, that variance becomes a useful result and motivates the later ablations.

---

## Priority 2 — Supervision Mode Ablation

**Goal:** Test whether integration-matrix supervision improves learning relative to direct supervision from immediate rewards.

**Design:** Using the same seed set and benchmark configuration from Phase 1, compare two matrix-factorization variants: direct supervision and integration-matrix supervision. In direct mode, the learner updates from the immediate observed reward for each interaction. In integration-matrix mode, the learner first accumulates a running estimate of drone-target interaction quality and uses that estimate as the supervision target. Random and oracle baselines from Phase 1 remain as reference points.

**What this tests:** This phase isolates one architectural choice in the proposed policy: whether smoothing interaction evidence through the integration matrix produces more stable or effective latent-structure recovery than learning directly from noisy rewards.

**What to report:**
- Learning curves for direct vs. integration-matrix supervision across the same seeds.
- Final-episode, best-episode, and last-$k$ average performance for both modes.
- Match quality and latent mismatch, because these directly reflect whether the learned compatibility model improves.
- Efficiency metrics such as steps and ammo, because these show whether better compatibility learning translates into task performance.
- Coordination diagnostics such as overkill and collisions, because smoothing may improve pairing while still increasing crowding.

**Expected interpretation:** If integration-matrix supervision consistently improves match quality, reduces latent mismatch, or stabilizes learning across seeds, it supports the design choice used in the main method. If direct supervision performs similarly, the paper should present integration-matrix as an implementation variant rather than a necessary component. If direct supervision outperforms it, the method section should be revised around the simpler learning signal.

**Config change required:** Toggle `use_integration_matrix` in the policy config.

### Backlog Table

| Field | Detail |
|---|---|
| **Parameter** | Supervision mode: `integration_matrix` vs. `direct` |
| **Motivation / Story** | The integration-matrix mode accumulates a running-mean interaction history before computing gradients, smoothing out reward noise. The direct mode regresses against raw immediate rewards. The story is: *does smoothing the supervision signal matter under noisy conditions, and if so, how much?* |
| **Values to try** | `direct`, `integration_matrix` (current) |
| **How to present** | Side-by-side learning curves (episodes 1–35) for each mode, mean ± std band over N seeds. Bar chart or paired table for final, best, and last-$k$ performance. |
| **Expected results** | Integration-matrix should outperform direct if smoothing the supervision signal improves latent-structure recovery under reward noise. Direct may perform similarly if immediate rewards are already informative enough. |
| **Implementation notes** | `use_integration_matrix` flag in the policy config controls the switch. `direct` mode is already coded. Defer hybrid variants until after the core two-way comparison is understood. |

---

## Priority 3 — Latent-Dimension Sensitivity

**Goal:** Test how sensitive the matrix-factorization policy is to the choice of factorization dimension $d_f$.

**Design:** Hold the benchmark environment fixed at latent dimension $d = 3$ and run the integration-matrix MF policy with $d_f \in \{1, 2, 3, 5, 8\}$ using the same seed set from Phase 1. The environment latent dimension remains hidden from the policy; only the policy's internal representational capacity changes.

**What this tests:** This phase evaluates whether the method depends on correctly matching the true latent dimension. Under-specifying $d_f$ may prevent the policy from representing the full compatibility structure, while over-specifying $d_f$ may add unnecessary capacity, slow learning, or increase instability.

**What to report:**
- Learning curves for each $d_f$ value.
- Final-episode, best-episode, and last-$k$ average performance.
- Match quality and latent mismatch as direct indicators of compatibility recovery.
- Efficiency metrics such as steps and ammo to show whether representation quality translates into task performance.
- Variance across seeds, because some dimensions may be less stable even if their mean performance is similar.

**Expected interpretation:** If $d_f = 3$ performs best, the current configuration is well matched but somewhat favorable. If lower dimensions perform nearly as well, the hidden structure may be simpler than the nominal dimension suggests. If larger dimensions improve or destabilize performance, the method's robustness depends on capacity selection and should be discussed as a hyperparameter sensitivity.

---

## Priority 4 — Noise Sensitivity

**Goal:** Characterize how robust the matrix-factorization policy is to corruption in the information used for learning.

This phase separates two noise sources that affect different parts of the learning process. Observation noise corrupts *which interaction* an agent thinks occurred, while reward noise corrupts *how valuable* the interaction appears to be. They should be swept separately so the effect of each channel remains interpretable.

### 4A — Shared-Observation Noise

**Design:** Hold reward noise fixed at the current value and vary observation noise, e.g. $\sigma_{\text{obs}} \in \{0.0, 0.05, 0.1, 0.2, 0.4\}$, using the same seed set from Phase 1. Observation noise corrupts the swarm-history channel, meaning agents may observe incorrect target selections for other drones. Random and oracle remain reference policies: random does not use the history channel, while oracle has privileged access to the true latent structure and HP state.

**What this tests:** Whether the collaborative component of the method depends on accurate observation of other agents' actions. When observation noise is low, shared interaction traces should help each local model infer drone-target compatibility. As observation noise increases, the integration matrix receives corrupted interaction evidence, so the benefit of decentralized shared learning should weaken.

**Expected interpretation:** Degradation under observation noise indicates sensitivity to corrupted swarm-history data. A gradual decline suggests robustness; a sharp drop identifies the operating threshold where decentralized shared learning breaks down.

### 4B — Reward-Noise Sensitivity

**Design:** Hold observation noise fixed at the current value and vary reward noise, e.g. $\sigma_{\text{reward}} \in \{0.0, 0.05, 0.1, 0.2, 0.4\}$, using the same seed set from Phase 1. Reward noise corrupts the scalar feedback signal while preserving the identity of the interaction.

**What this tests:** Whether the method can still recover useful compatibility structure when the scalar feedback signal is noisy. This subphase connects directly to Priority 2: if integration-matrix supervision remains stable under reward noise, it supports the claim that running-mean supervision smooths noisy interaction feedback.

**Expected interpretation:** Degradation under reward noise indicates sensitivity to noisy utility estimates. If performance remains stable across increasing reward noise, the learned structure is robust to imperfect scalar feedback; if it degrades sharply, the method depends on relatively clean reward observations.

**What to report:**
- Learning curves at each noise level.
- Final-episode, best-episode, and last-$k$ average performance.
- Match quality and latent mismatch, because these show whether corrupted information damages compatibility learning.
- Steps and ammo, because these show whether degraded learning affects task efficiency.
- Collisions and overkill, because noise may affect crowding and target contention.

**Implementation notes:** `observation_noise` and `reward_noise` are top-level fields in `config/scenario.json`. Run each sweep independently over the same seed set as Priority 1, changing only one noise source at a time.

---

## Priority 5 — Swarm-Scale Sensitivity

**Goal:** Test whether the matrix-factorization policy remains effective as the number of drones and targets increases.

**Design:** Vary swarm and task-set size while preserving the 3:1 target-to-drone ratio, e.g. $(m,n) \in \{(3,9), (6,18), (9,27), (15,45)\}$. Use the same latent dimension, noise settings, supervision mode, and seed protocol as the main benchmark. Because larger swarms create more interactions and more policy parameters, training horizon may need to be scaled or reported carefully.

**What this tests:** This phase evaluates whether the learned compatibility mechanism scales beyond the current 9-drone, 27-target benchmark. It also tests whether coordination costs, especially collisions and overkill, grow with swarm size.

**What to report:**
- Steps and ammo as a function of swarm size.
- Match quality and latent mismatch as indicators of whether compatibility learning remains effective.
- Collisions and overkill as indicators of coordination pressure.
- Runtime or wall-clock cost, because scalability includes computational burden.
- Late-training performance, using final, best, and last-$k$ summaries.

**Expected interpretation:** If match quality remains strong while steps and ammo scale reasonably, the method shows evidence of scalability. If match quality remains good but collisions and overkill grow, the policy learns compatibility but struggles with coordination at larger swarm sizes. If learning degrades sharply, scaling exposes a representational or exploration limitation.

**Note:** This is more expensive to run than the earlier phases. Do this after Priorities 1–4 are complete unless compute is abundant.

---

## Figures to Produce

| Figure | What it shows | Needed for |
|---|---|---|
| Learning curve (mean ± std band) | MF improvement over 35 episodes, with confidence interval | Priority 1 |
| Cross-policy bar chart (episode 35) | MF vs. random vs. oracle on all metrics, with error bars | Priority 1 |
| Supervision mode comparison | Integration-matrix vs. direct learning curves | Priority 2 |
| $d_f$ sensitivity line plot | Performance vs. policy factorization dimension | Priority 3 |
| Noise sweep plot | Performance vs. observation noise level | Priority 4 |

---

## Revised Paper Structure (Experiments + Results)

Once the above experiments are run, Sections 6 and 7 should be restructured:

**Section 6 (Experiments):**
- 6.1 Benchmark Scenario and Compared Policies
- 6.2 Multi-Seed Paired Evaluation Protocol *(new)*
- 6.3 Training and Reporting Protocol *(new — final, best, and last-$k$ summaries)*
- 6.4 Experimental Conditions *(new — supervision mode, $d_f$, observation noise, reward noise, and swarm scale)*
- 6.5 Evaluation Metrics

**Section 7 (Results):**
- 7.1 Multi-Seed Baseline Comparison *(MF vs. random vs. oracle, with uncertainty)*
- 7.2 Learning Dynamics and Convergence *(mean curves, final/best/last-$k$ summaries)*
- 7.3 Supervision Mode Ablation *(integration-matrix vs. direct)*
- 7.4 Latent-Dimension Sensitivity *($d_f$ sweep)*
- 7.5 Noise Sensitivity *(shared-observation noise and reward noise)*
- 7.6 Swarm-Scale Sensitivity *(if Priority 5 is run)*
- 7.7 Latent-Structure and Coordination Diagnostics
- 7.8 Summary and Limitations

---

## Recommended Sequence

```
Step 1: Build multi-seed runner → rerun current config (N=20 seeds)
Step 2: Run supervision mode ablation (same N seeds)
Step 3: Run d_f sweep (same N seeds)
Step 4: Run noise sweep (same N seeds)
Step 5: Run swarm composition scaling (if time permits)
Step 6: Revise Sections 6 and 7 with new results
Step 7: Update figures
```

Steps 1–4 are independent once the multi-seed infrastructure exists. Steps 2–4 can run in parallel if compute allows.

---

## Executive Summary

### Phase 1 — Multi-Seed Paired Benchmark Evaluation

Run the current benchmark configuration across a shared set of scenario seeds for random, oracle, and MF, so each policy is compared on the same latent worlds.

**Tests:** Whether the current seed-42 result is reproducible across problem instances and how much variance exists across latent scenarios.

### Phase 2 — Supervision Mode Ablation

Compare direct supervision against integration-matrix supervision using the same benchmark configuration and seed set, with random and oracle retained as reference policies.

**Tests:** Whether smoothing interaction evidence through the integration matrix improves latent-structure recovery and task efficiency.

### Phase 3 — Latent-Dimension Sensitivity

Hold the environment latent dimension fixed at $d = 3$ and vary the policy factorization dimension $d_f$ to test under- and over-specified learner capacity.

**Tests:** Whether the method depends on correctly matching the true latent dimension, and how robust it is to factorization-capacity choices.

### Phase 4 — Noise Sensitivity

Run separate sweeps for shared-observation noise and reward noise, changing one noise source at a time while holding the rest of the benchmark fixed.

**Tests:** Whether MF remains effective when the interaction identity signal or scalar reward signal is corrupted.

### Phase 5 — Swarm-Scale Sensitivity

Increase the number of drones and targets while preserving the 3:1 target-to-drone ratio, using the same benchmark assumptions and seed protocol where feasible.

**Tests:** Whether compatibility learning and coordination quality remain stable as the allocation problem grows.
