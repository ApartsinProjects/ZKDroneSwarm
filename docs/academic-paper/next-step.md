# Next Steps — Toward Real Academic Experiments

This document outlines the prioritized roadmap for upgrading the current single-configuration experiment into a credible, publishable experimental evaluation.

---

## The Core Problem

The current paper reports one scenario (seed=42), one policy configuration, 35 episodes. A reviewer will ask two questions that cannot be answered yet:

1. **Is this result statistically reproducible?** (No — one run, no variance estimate)
2. **Does the method generalize beyond one lucky configuration?** (Unknown — one scenario)

Both must be addressed before the Experiments and Results sections can be considered complete.

---

## Priority 1 — Multi-Seed Evaluation (Statistical Validity)

**What:** Run the current configuration across N independent scenario seeds (e.g., seed ∈ {42, 43, ..., 61}, N=20). Each seed generates a different latent structure (different drone/target latent vectors), so this tests generalization across problem instances.

**Why this first:** It is infrastructure. Every other ablation also needs multi-seed runs. Build this once, reuse everywhere.

**What to report:**
- Mean ± std of each metric (steps, targets neutralized, latent match quality, collisions, overkill) across seeds, at episode 35
- Learning curve: mean ± std band across episodes 1–35
- Fraction of seeds where MF outperforms random (robustness claim)

**Implementation note:** `config/scenario.json` already has `"seed": 42`. The runner at `main_zk_mrta.py` passes it through the full stack. A simple outer loop over seed values — writing per-seed result CSVs and aggregating — is sufficient.

**Expected result:** MF should consistently outperform random across seeds, though the margin will vary. High variance across seeds would be an interesting finding in itself.

---

## Priority 2 — Supervision Mode Comparison (Core Ablation)

**What:** Compare `integration_matrix` mode (current) against `direct` supervision mode. Both are already implemented (§4.5.1). This is currently noted as "future work" in §6.4 — but it is actually the most important ablation because integration-matrix is one of the paper's design choices.

**Why second:** It directly validates a specific architectural decision in the methods section. Without this comparison, the choice of integration-matrix mode is unjustified.

**What to report:**
- Learning curves for both modes (mean ± std over N seeds)
- Final-episode metric comparison (table or bar chart)
- Whether integration-matrix converges faster or to a higher ceiling

**Config change required:** Toggle `use_integration_matrix` in policy config.

### Backlog Table

| Field | Detail |
|---|---|
| **Parameter** | Supervision mode: `integration_matrix` vs. `direct` vs. `hybrid` |
| **Motivation / Story** | The integration-matrix mode accumulates a running-mean interaction history before computing gradients, smoothing out reward noise. The direct mode regresses against raw immediate rewards. The hybrid takes a running mean of the new and current embeddings. The story is: *does smoothing the supervision signal matter under noisy conditions, and if so, how much?* |
| **Values to try** | `direct`, `integration_matrix` (current), `hybrid` (if implemented) |
| **How to present** | Side-by-side learning curves (episodes 1–35) for each mode, mean ± std band over N seeds. Bar chart of final-episode metrics. |
| **Expected results** | Integration-matrix should outperform direct, especially on latent match quality and overkill, because reward noise is averaged out before gradient updates. Hybrid may converge faster but reach a similar ceiling. Direct may plateau earlier or oscillate. |
| **Implementation notes** | `use_integration_matrix` flag in policy config controls the switch. `direct` mode is already coded. Hybrid requires a running mean over embedding matrices — not yet implemented. Run integration-matrix and direct first; add hybrid only if the gap is interesting. |

---

## Priority 3 — Latent Dimension Mismatch ($d_f$ vs. $d$)

**What:** Fix environment dimension $d = 3$ and vary policy factorization dimension $d_f \in \{1, 2, 3, 5, 8\}$.

**Why:** The current setup sets $d_f = d = 3$, giving the policy "just enough" capacity. This is a favorable condition. A real evaluation should test what happens when the policy under-specifies ($d_f < d$) or over-specifies ($d_f > d$) the latent structure.

**What to report:**
- Final-episode performance vs. $d_f$ (line plot)
- Whether $d_f = d$ is actually the optimal choice, or whether more capacity helps/hurts

**Story this tells:** Robustness of the method to hyperparameter mismatch.

---

## Priority 4 — Observation Noise Sweep (Collaborativeness Regime)

**What:** Vary observation noise $\sigma_{\text{noise}} \in \{0.0, 0.1, 0.2, 0.4\}$ while holding all else fixed.

**Why:** Observation noise controls how accurately agents observe each other's actions — i.e., the degree of implicit coordination possible. At $\sigma = 0$, agents have perfect knowledge of swarm history. At $\sigma = 0.4$, the history channel is nearly uninformative. This sweeps across the "collaborativeness" spectrum.

**What to report:**
- Performance degradation curve as noise increases
- Whether the MF policy degrades gracefully or collapses at high noise

**Story this tells:** Characterizes the operating regime of the method and its sensitivity to the quality of the shared observation channel.

### Backlog Table

| Field | Detail |
|---|---|
| **Parameter** | `observation_noise` (and optionally `reward_noise` as a separate axis) |
| **Motivation / Story** | Observation noise corrupts the swarm-history channel — the signal agents use to infer what their teammates are doing. Sweeping this parameter simulates a spectrum from *fully collaborative* (noise=0, agents see accurate peer actions) to *effectively isolated* (noise=0.4, history channel is nearly random). The story is: *at what noise level does the benefit of collaboration break down, and does MF degrade gracefully or catastrophically?* |
| **Values to try** | `observation_noise` ∈ {0.0, 0.05, 0.1, 0.2, 0.4}; `reward_noise` can be swept independently at ∈ {0.0, 0.1, 0.2, 0.4} |
| **How to present** | Line plot: x-axis = noise level, y-axis = metric (latent match quality, targets neutralized, collisions). One line per policy (MF, random, oracle). Oracle and random serve as invariant reference lines since they do not use the history channel. |
| **Expected results** | Oracle performance is unaffected (uses privileged model, not observations). Random is unaffected (uniform sampling). MF degrades as noise increases — the integration matrix accumulates corrupted observations, weakening the learned compatibility signal. At noise=0, MF should approach oracle closely. At noise=0.4, MF may approach random. The crossover point is the key result. |
| **Implementation notes** | `observation_noise` and `reward_noise` are top-level fields in `config/scenario.json`. No code changes needed — pure config sweep. Run each noise level over N seeds (same seed set as Priority 1). Report both metrics separately to distinguish the effect of observation corruption from reward corruption. |

---

## Priority 5 — Swarm Composition Scaling

**What:** Vary $(m, n)$ while holding the 3:1 ratio fixed: $(3, 9)$, $(6, 18)$, $(9, 27)$, $(15, 45)$.

**Why:** The current 9-drone, 27-target setup is one point on the scaling curve. A paper claiming practical relevance should show the method scales (or identify where it breaks down).

**What to report:**
- Steps-to-completion and latent match quality vs. swarm size
- Whether coordination efficiency (collisions, overkill) degrades with scale

**Note:** This is more expensive to run (larger environments, more policy parameters). Do this after Priorities 1–3 are complete.

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
- 6.1 Benchmark Scenario (keep as-is)
- 6.2 Multi-Seed Evaluation Protocol *(new)*
- 6.3 Ablation Conditions *(new — supervision mode, $d_f$, noise, composition)*
- 6.4 Evaluation Metrics (keep as-is)

**Section 7 (Results):**
- 7.1 Baseline Comparison (multi-seed, with error bars)
- 7.2 Learning Dynamics (curve with confidence band)
- 7.3 Supervision Mode Ablation *(new)*
- 7.4 Latent Dimension Sensitivity *(new)*
- 7.5 Noise Sensitivity *(new)*
- 7.6 Summary

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
