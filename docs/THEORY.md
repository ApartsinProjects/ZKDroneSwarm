# Theory: per-drone sample-complexity separation (D3)

The empirical spine (C8, C11, C12, C13) is matched by a clean separation. Proof
SKETCHES below; each maps to an experiment.

## Setup
- m drones, n targets. True reward R = P Uᵀ, rank d: R[i,j] = ⟨p_i, u_j⟩, with
  p_i, u_j unit vectors (cosine compatibility). The factors are UNKNOWN.
- NO communication. A public BROADCAST carries actions and (possibly masked /
  noisy) outcomes. Each drone i independently forms an estimate from what IT
  observes (own outcomes clean; teammates' via the broadcast, masked w.p. 1-ρ and
  noisy with σ). No parameter sharing.
- A learner is GOOD if it achieves ε expected error on (the relevant entries of)
  drone i's row R[i,:].

## Proposition 1 (tabular floor on unseen pairs)
An independent / tabular learner estimates R[i,j] only from observations of the
specific pair (i,j). For any pair (i,j) it has never observed (neither an own
pull nor a received broadcast event on j), its estimate is the prior mean; its
expected squared error is Ω(Var_j R[i,j]) = Ω(1) (a constant FLOOR).
=> To be GOOD on ALL of drone i's targets, tabular must observe Ω(n) distinct
targets for drone i (each at least once). Per-drone sample complexity Ω(n);
total Ω(mn).
EMPIRICS: C8/C11 Tabular unseen-pair skill ≈ 0 at every ρ (the floor).

## Proposition 2 (CF completes a row from O(d) observations)
Assume R has rank d and the broadcast lets drone i estimate the target factor
matrix U (its column space) to accuracy o(1). [Standard matrix completion: the
population's observations identify a rank-d nxd' factor space once the number of
observed entries is Õ(d·n) under incoherence (Candès-Recht 2009; Keshavan-
Montanari-Oh OptSpace O(dn); Recht 2011).] Then drone i's row R[i,:] = p_iᵀ Uᵀ is
determined by the d-dim factor p_i, and
    p_i = argmin_p Σ_{j ∈ obs_i} (r_ij - ⟨p, u_j⟩)²
is identifiable once |obs_i| ≥ d and the observed {u_j} span R^d (a d-dim ridge /
WALS fold-in). With O(d) own observations, drone i predicts ALL targets,
including never-pulled ones, via ⟨p_i, u_j⟩, error -> 0.
=> Per-drone sample complexity O(d) (given U from the broadcast).
EMPIRICS: C12 (a new target is onboarded for ALL drones from ~d shared probes via
exactly this fold-in ridge); C13 (unseen-pair skill degrades as the true d rises,
i.e. more to complete from the same data).

## Corollary (the separation)
Per-drone observations for a GOOD row: tabular Θ(n) vs CF Θ(d). For d ≪ n this is
a Θ(n/d) separation. On UNSEEN pairs specifically: tabular error Ω(1) (floor) vs
CF error -> 0. This is the CATEGORICAL (not constant-factor) gap.
EMPIRICS: C8 0.496 vs 0.006; C11 0.16-0.41 vs ~0.

## Decentralization + masking (Proposition 2 under partial observability)
Each drone estimates U independently from the broadcast it can see; under
per-drone masking ρ, drone i's visible events ≈ ρ·(population observations). U's
column space is recovered while ρ·(obs) ≳ Õ(d·n); below this threshold U is only
partially recovered and CF degrades GRACEFULLY (not to the floor, as long as
ρ>0). Distinct masks => distinct visible data => distinct estimates U^(i) =>
genuinely UNIQUE per-drone states (decentralization is real, not cosmetic).
EMPIRICS: C11 CF unseen 0.41 -> 0.16 as ρ: 0.5 -> 0.1 (graceful), and the
state-uniqueness metric rises monotonically 0.54 -> 0.92 as ρ falls.

## Refinement: clustering lowers it further (block model)
If drones/targets fall into K1/K2 TYPES (block model, R[i,j] ≈ C[t_i, s_j], C
rank d), identifying drone i's TYPE is a K1-way classification needing O(log K1)
discriminative observations; the type's shared factor then transfers. So
clustering reduces per-drone/onboarding sample complexity below the generic O(d).
GROUNDING: community-detection + matrix-completion lower bounds show block / SBM
structure reduces sample complexity (arXiv 1912.04099; hierarchical similarity
graphs 2023). EMPIRICS: C12 onboards from very few probes; C13.

## What is NOVEL vs the cited theory
Standard MC bounds are CENTRALIZED (one estimator, uniform sampling). Our
contribution is the DECENTRALIZED, ONLINE, broadcast-only, per-drone-masked
adaptation: each agent recovers U from a partial public broadcast (no parameter
sharing, no communication) and the separation is stated PER DRONE (Θ(d) vs Θ(n))
with the unseen-pair floor making it categorical.
