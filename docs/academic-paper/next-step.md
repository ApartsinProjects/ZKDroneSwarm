# Next Steps — Toward a Publishable Experimental Evaluation

This document outlines a phased roadmap for turning the current single-configuration preliminary result into a statistically interpretable and publishable experimental evaluation. The first phase establishes reproducibility across scenario seeds; later phases test whether the method's design choices, capacity assumptions, noise robustness, and scaling behavior hold beyond the initial configuration.

---

## The Core Problem

The current paper reports one scenario seed, one policy configuration, and a 35-episode training run. This is sufficient for a preliminary proof of intent, but not yet enough for a complete experimental claim. A reviewer will first ask:

1. **Is the result reproducible across independently generated scenarios?**
2. **How much variation exists across latent worlds?**

Phase 1 addresses this immediate evidence gap. The remaining phases build on that foundation by testing supervision mode, factorization capacity, noise sensitivity, and swarm scale.

---

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
