# Formal theory: rigorous statements and proofs

Companion to `THEORY.md` (overview/sketches). Here every result is stated with
explicit assumptions and a detailed proof. We mark each result as EXACT (pure
linear algebra / probability), ORDER (an order bound, constants stated), or CITED
(reduces to a standard matrix-completion result we invoke). Nothing here is
hand-waved; where a step relies on an external theorem we name it.

## Setup and notation

- m drones, n targets. True reward matrix R = P Uᵀ has rank d, with rows
  p_i ∈ ℝ^d and u_j ∈ ℝ^d unit vectors, so R_{ij} = ⟨p_i, u_j⟩ ∈ [-1, 1]. Factors
  are UNKNOWN to every learner.
- For drone i write the reward marginal F_i = empirical distribution of
  {R_{ij}}_{j=1..n}; let μ_i = (1/n) Σ_j R_{ij} = E_{J∼Unif[n]} R_{iJ} and assume
  the row is non-degenerate (Var_{J} R_{iJ} = v_i > 0).
- Each round t, drone i is offered a uniformly random size-c subset S_t ⊆ [n]
  (without replacement), selects a_t ∈ S_t, and earns R_{i,a_t}.
- Reference values for drone i: random ρ^rand_i = μ_i; oracle
  ρ^orac_i = E_S[ max_{j∈S} R_{ij} ]. For a policy with expected earned reward ρ,
  skill(ρ) = (ρ − μ_i)/(ρ^orac_i − μ_i) ∈ (−∞, 1], with 0 = random, 1 = oracle.
- Observation channel: a public broadcast of events (k, a, R_{ka}+noise). Drone i
  observes its own outcome (noise sd σ_own) and, subject to a mask, teammates'
  outcomes (noise sd σ_obs). Masking models are defined in Section 4.

**Definition 1 (structure-free learner).** A learner for drone i is structure-free
if, for every target j, its estimate R̂_{ij} is a measurable function ONLY of drone
i's own past observations of target j (the rewards it received on its own pulls of
j), and equals a fixed prior constant b for any j drone i has never pulled. (This is
the per-(drone,target) tabular class: UCB-Indep, ε-greedy-own-row. It has no model
linking different targets or different drones.)

---

## Theorem 1 (tabular unseen-pair floor). EXACT.

Let L be any structure-free learner for drone i. (a) For any target j that drone i
has never pulled, the squared error of its estimate obeys
  E[(R̂_{ij} − R_{ij})²] ≥ v_i = Var_J(R_{iJ}) > 0,
a constant floor independent of how long the swarm has run or how much was
broadcast. (b) On an offer drawn uniformly from the targets drone i has never
pulled, L's expected earned reward is exactly the unseen-set mean, so its
unseen-pair skill is 0.

**Proof.** (a) By Definition 1, on an unpulled target j the estimate R̂_{ij} = b is
a constant chosen before seeing j. For any constant predictor b,
E_J[(b − R_{iJ})²] = (b − μ_i)² + v_i ≥ v_i, minimized at b = μ_i. Restricting to
the never-pulled targets only changes μ_i, v_i to the unseen-subset moments, still
a positive constant for a non-degenerate row. Hence the per-pair MSE is ≥ v_i =
Ω(1). The broadcast cannot reduce it: R̂_{ij} is by definition not a function of any
event (k, ·) with k ≠ i, nor of events on targets j' ≠ j.

(b) Let A be the (random) set of never-pulled targets at the evaluation round and
S ⊆ A the offer. Conditioned on the learner's information, the rewards
{R_{ij}}_{j∈A} are unknown and, by Definition 1, the selection rule restricted to A
cannot depend on them; therefore a_t is (conditionally) independent of
{R_{ij}}_{j∈A}, and by exchangeability E[R_{i,a_t}] = (1/|A|) Σ_{j∈A} R_{ij} =:
μ^A_i. Skill = (μ^A_i − μ^A_i)/(ρ^orac − μ^A_i) = 0. ∎

**Remark (pooling recovers only the rank-1 popularity term).** A learner that POOLS
all drones' outcomes on target j (UCB-Homogeneous) estimates
(1/m) Σ_k R_{kj} = ⟨ p̄, u_j ⟩ with p̄ = (1/m) Σ_k p_k. This is the rank-1 projection
of R onto the population-mean drone direction; it predicts the SAME target ranking
for every drone (a "popularity" order). Its unseen-pair skill equals the alignment
between that shared ranking and drone i's true ranking, which is positive but < 1
whenever d > 1 (there is personalization beyond popularity). This matches the
measured UCB-Homogeneous partial unseen skill (≈ 0.17 at d = 5).

---

## Theorem 2 (CF recovers an entire row from O(d) observations given U). EXACT
(noiseless) / CITED-rate (noisy).

Suppose the target-factor matrix U is known and has rank d. If drone i observes its
true rewards on a set Ω ⊆ [n] with |Ω| ≥ d such that the vectors {u_j}_{j∈Ω} span
ℝ^d, then:
(a) p_i is the unique solution of R_{i,Ω} = U_Ω p_i, namely
  p_i = (U_Ωᵀ U_Ω)^{-1} U_Ωᵀ R_{i,Ω},
and therefore R̂_{ij} = ⟨p_i, u_j⟩ = R_{ij} EXACTLY for ALL j ∈ [n], including every
target drone i never pulled.
(b) If the observed rewards carry independent noise of variance σ², the ridge
estimate p̂_i = (U_Ωᵀ U_Ω + λI)^{-1} U_Ωᵀ y has, for the prediction at any target j,
  E[(⟨p̂_i, u_j⟩ − R_{ij})²] ≤ σ² · uⱼᵀ (U_ΩᵀU_Ω + λI)^{-1} u_j + bias(λ)²
  = O( σ² d / λ_min(U_Ωᵀ U_Ω) ) uniformly over unit u_j, with bias → 0 as λ → 0.

**Proof.** (a) By definition R_{ij} = ⟨p_i, u_j⟩, so stacking j ∈ Ω gives the linear
system R_{i,Ω} = U_Ω p_i. Since {u_j}_{j∈Ω} span ℝ^d, U_Ω (an |Ω|×d matrix) has rank
d, hence U_Ωᵀ U_Ω is invertible and the least-squares solution is unique and equals
the true p_i (the system is consistent because R was generated by this p_i). Then
⟨p_i, u_j⟩ reproduces R_{ij} for every j. (b) Standard ridge/Gauss-Markov:
Cov(p̂_i) = σ² (U_ΩᵀU_Ω+λI)^{-1} U_Ωᵀ U_Ω (U_ΩᵀU_Ω+λI)^{-1} ⪯ σ²(U_ΩᵀU_Ω+λI)^{-1};
projecting onto u_j gives the stated variance, bounded by σ²/λ_min(U_ΩᵀU_Ω) for
||u_j||=1; the ridge bias vanishes as λ→0. ∎

**Where U comes from (CITED).** U (its column space) is identified from the swarm's
pooled broadcast by matrix completion: a rank-d n×m matrix is recovered from
Õ(d(m+n)) observed entries under standard incoherence (Candès-Recht 2009;
Keshavan-Montanari-Oh OptSpace, O(d(m+n)); Recht 2011). Theorem 2 is the per-drone
"fold-in" step that turns a known U into an O(d)-sample row completion. The
combination gives per-drone sample complexity O(d) versus the tabular Ω(n) of
Theorem 1: a Θ(n/d) separation, categorical on unseen pairs (Ω(1) error vs 0).

---

## Theorem 3 (anytime separation under sample starvation). ORDER.

Rewards in [0,1] (rescale otherwise); offers uniform size c i.i.d. per round;
horizon T. Define g_i(x) = E[ max of ⌈x⌉ i.i.d. draws from F_i ] − μ_i, an
increasing, concave function with g_i(0) = 0 and g_i(c) = ρ^orac_i − μ_i.

(a) [Structure-free anytime ceiling] For ANY structure-free learner of drone i,
  E[ Σ_{t=1}^{T} R_{i,a_t} ] ≤ T μ_i + Σ_{t=1}^{T} g_i( c·(t−1)/n ),
and therefore
  skill_anytime(L) ≤ (1/T) Σ_{t=1}^T g_i(c(t−1)/n) / (ρ^orac_i − μ_i)
                  ≤ g_i(cT/n) / (ρ^orac_i − μ_i).
In particular, in the starved regime cT = o(n) we have g_i(cT/n) → 0, so the
structure-free anytime skill → 0. This holds EVEN with full broadcast.

(b) [CF attainment] A CF learner that has recovered U (Theorem 2) and fixed p̂_i
from O(d) own pulls selects a_t = argmax_{j∈S_t} ⟨p̂_i, u_j⟩ and thereafter earns
ρ^orac_i per round (exactly, in the noiseless case). Hence
  skill_anytime(CF) ≥ 1 − O(d/T) − O(σ-prediction-error).

Consequently, when cT = o(n) and T ≫ d, the anytime-skill gap (CF minus
structure-free) is 1 − o(1).

**Proof of (a).** Fix drone i; drop the subscript. Let K_{t−1} be the set of
distinct targets pulled before round t (|K_{t−1}| ≤ t−1, one pull per round). On
round t the learner picks a_t ∈ S_t. Split on B_t = {S_t ∩ K_{t−1} ≠ ∅}:
- On B_t^c (offer disjoint from everything pulled), every offered target is
  unpulled, so by Definition 1 the selection is independent of those targets'
  rewards; by exchangeability E[R_{i,a_t} | B_t^c, K_{t−1}] = the mean reward of the
  unpulled targets, μ^{unp}(K_{t−1}). Because the pulled set K_{t−1} is reward-blind
  (established below), every target is equally likely to lie in K_{t−1}, so taking
  expectation over K_{t−1} gives E[μ^{unp}(K_{t−1})] = μ. Hence the B_t^c rounds
  contribute exactly 0 to E[R_{i,a_t} − μ]; we drop them.
- On B_t the best the learner can do is take the maximum reward among the
  offered-AND-pulled targets: R_{i,a_t} ≤ max_{j ∈ S_t ∩ K_{t−1}} R_{ij}. Hence
  E[R_{i,a_t} − μ] ≤ E[ ( max_{j ∈ S_t ∩ K_{t−1}} R_{ij} − μ )_+ ].

Now the key structural fact: a structure-free learner chooses which NEW (never
pulled) target to add to K based only on the prior (identical for all unpulled
targets) and offer identities, hence INDEPENDENTLY of the to-be-revealed reward of
that target. Therefore K_{t−1} is a reward-blind subset, i.e. the multiset
{R_{ij} : j ∈ K_{t−1}} is distributed as |K_{t−1}| i.i.d. draws from F_i. Given
|K_{t−1}| = k, the intersection S_t ∩ K_{t−1} has size N ∼ Hypergeometric, with
E[N] = c·k/n, and the rewards in it are N i.i.d. draws from F_i. Writing
h(N) = E[max of N draws] (with h(0)=μ), h is concave in N (the increments
h(N)−h(N−1) = E[max of N] − E[max of N−1] are non-increasing, a standard property
of expected order statistics). By Jensen,
  E[ (max_{S_t∩K_{t−1}} R − μ)_+ ] = E[ h(N) − μ ] ≤ h(E[N]) − μ
     = h(c k/n) − μ = g_i(c·(t−1)/n) (using k ≤ t−1 and g increasing).
Summing over t gives the displayed bound; bounding each term by g_i(cT/n) (g
increasing) gives the closed form. Since g_i is continuous with g_i(0)=0, cT/n→0
implies the ceiling → 0. The broadcast does not change K_{t−1} (Definition 1: a
structure-free learner's own-row estimates ignore teammates' events), so the bound
is unchanged under any broadcast. ∎

**Proof of (b).** After Theorem 2's conditions hold, ⟨p̂_i, u_j⟩ = R_{ij} for all j
(noiseless), so a_t = argmax_{j∈S_t} R_{ij} and E[R_{i,a_t}] = ρ^orac_i for each of
the ≥ T − O(d) post-recovery rounds; the ≤ O(d) recovery rounds lose at most
(ρ^orac − μ) each, costing O(d)·(ρ^orac − μ). Dividing by T(ρ^orac − μ) gives
skill ≥ 1 − O(d/T). With noise, replace exact equality by Theorem 2(b), adding an
O(σ-error) term that is uniform over targets. ∎

**Remark (why UCB is even lower than the ceiling).** Theorem 3(a) bounds the BEST
structure-free anytime skill. A UCB-Indep learner is strictly below it: an untried
arm carries an infinite exploration bonus, so whenever the offer contains any
never-pulled target (probability ≈ 1 − (1 − k/n)^c, which stays near 1 while k ≪ n)
UCB pulls the untried arm instead of re-exploiting a known-good one. It therefore
earns ≈ μ every round until it has tried most arms, i.e. skill ≈ 0 throughout the
starved regime, exactly as observed. This is the precise sense in which "n ≫ T
makes per-arm bandits explore forever".

---

## Theorem 4 (masking model: durable vs transient heterogeneity; robustness of the
categorical results). EXACT (limits) + reduction to Theorems 1-3.

Two masking models, both with marginal observation rate ρ ∈ (0,1):
- PERSISTENT: M_{ik} ∼ Bernoulli(ρ) drawn once; drone i observes teammate k's
  events at ALL rounds iff M_{ik} = 1. Let V_i = {k : M_{ik} = 1} (its fixed
  visible sub-population), U_i = {k : M_{ik} = 0} (its permanent blind set).
- IID: M^t_{ik} ∼ Bernoulli(ρ) independently each round; drone i observes
  teammate k's round-t event iff M^t_{ik} = 1.

Claims:
(a) [IID heterogeneity is transient] Under IID, as T → ∞ each drone's empirical
measure over observed events converges almost surely to ρ × (the population event
measure), the SAME limit for every drone. Hence any estimator that is a continuous
functional of the observed event measure (in particular the completed model R̂^{(i)})
satisfies R̂^{(i)} → R̂^{(i')} for all i, i': the per-drone states converge and the
state-uniqueness metric → 0.
(b) [Persistent heterogeneity is durable] Under PERSISTENT masking, drone i obtains
ZERO observations of any teammate k ∈ U_i at every round, so it can never estimate
the factor p_k for k ∈ U_i (its data are orthogonal to that row). For i ≠ i',
U_i ≠ U_{i'} almost surely (they differ on a Binomial(m−1, 2ρ(1−ρ)) set of
teammates, nonempty w.h.p. for moderate m), so the drones' completed FULL models
R̂^{(i)} = P^{(i)} (U^{(i)})ᵀ differ on the teammate rows each can fill; the
state-uniqueness metric is bounded away from 0 uniformly in T.
(c) [the categorical results are invariant to the model] In BOTH models, in the
starved regime n ≫ cT: Theorem 1 (tabular unseen floor) and Theorem 3 (anytime
ceiling) hold verbatim, because a structure-free learner uses only its own pulls
(≤ T targets) and ignores the broadcast regardless of masking (Definition 1); and
Theorem 2 (CF row recovery) holds for drone i as soon as the events it observes
identify U, which both models achieve for ρ > 0 once the observed entry count
exceeds the completion threshold. Therefore the unseen-pair floor and the anytime
separation do NOT depend on the masking model; only the DURABILITY of
decentralization (state-uniqueness as T → ∞) distinguishes persistent (durable)
from IID (transient).

**Proof.** (a) The observed events under IID are an independent ρ-thinning of the
common event stream; by the strong law of large numbers the per-drone empirical
event measure converges a.s. to ρ times the population measure, identically across
drones. A completion estimator that depends continuously on this measure (ALS/SVD
limits are continuous in the observed-entry empirical averages on their support)
therefore has the same limit for every drone. (b) Under persistent masking the
rows {p_k : k ∈ U_i} appear in NONE of drone i's observations; the design matrix for
those rows is identically zero, so they are non-identifiable for drone i and remain
at the prior for all T. Distinct blind sets U_i ≠ U_{i'} thus yield models that
differ on at least the symmetric-difference rows, a set of size ≥ 1 w.h.p.; the
frame-invariant distance between R̂^{(i)} and R̂^{(i')} is bounded below by the
contribution of those rows, uniformly in T. (c) Tabular: by Definition 1 its
own-row estimates are a function of own pulls only, identical under both masks; its
pulled set has ≤ T targets, so Theorems 1 and 3(a) apply unchanged. CF: Theorem 2
needs only that the OBSERVED events identify U and that drone i has ≥ d own pulls
spanning the relevant directions; for ρ > 0 both masks deliver Θ(ρ · population)
observations, exceeding the completion threshold for T large enough, after which
the per-drone own-row recovery is identical in form. ∎

**Reading.** This is the precise answer to "why persistent and not i.i.d.": i.i.d.
loss (packet drop) is equally admissible and the headline categorical results hold
under it; we adopt persistent masking because it makes "no communication implies
durably different per-drone knowledge" a STRUCTURAL property (state-uniqueness
bounded away from 0 as T → ∞) instead of a finite-sample transient that i.i.d.
averages away. Experiment E12 re-runs the headline panels under i.i.d. masking to
confirm (c) empirically and to measure the predicted state-uniqueness gap (a).

---

## What is novel here (versus the cited literature)

Matrix-completion sample-complexity bounds (Candès-Recht; Keshavan et al.; Recht)
are CENTRALIZED (one estimator, uniform random sampling) and are about ESTIMATION
error, not online decision reward. Our contributions are: (i) the per-drone,
broadcast-only, masked statement with the unseen-pair floor making the gap
categorical (Theorems 1-2); (ii) the ANYTIME (cumulative-reward) separation under
sample starvation, which charges exploration cost and shows structure-free skill
→ 0 while CF → 1 (Theorem 3), explaining the operational result; (iii) the
masking-model dichotomy (Theorem 4) characterizing when decentralization is durable
versus transient, and showing the main results are invariant to that choice. These
are decentralized and online, which the centralized completion theory does not
cover.
```
