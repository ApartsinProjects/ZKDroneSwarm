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
(a) [IID heterogeneity is transient] Under IID, every teammate k is observed by
drone i on each round independently with probability ρ > 0, so by Borel-Cantelli
(divergent independent events) drone i observes k infinitely often almost surely,
and likewise observes every target pulled. Hence as T → ∞ drone i accumulates
unboundedly many observations of every row and column and (under the same
identifiability conditions used for any single estimator) recovers the full true
model R̂^{(i)} → R. The limit R is common to all drones, so R̂^{(i)} → R̂^{(i')} for
all i, i': the per-drone states converge and the state-uniqueness metric → 0.
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

## Theorem 5 (additive-model rank ceiling). EXACT.

Let an ADDITIVE predictor be R_hat[i,j] = a + b_i + c_j (a global level, a per-drone
bias, a per-target bias; the BiasModel baseline). (a) Its prediction matrix has rank
at most 2: writing it as (a 1 + b) 1^T + 1 c^T, it is a sum of two rank-1 terms. (b)
For RANKING targets within a fixed drone i, the term a + b_i is constant in j, so the
induced order is exactly the per-target order of c_j (a popularity order, shared by
all drones). Hence an additive model's unseen-pair skill equals the popularity-
ranking skill, which is positive but strictly below CF whenever d > 1 (there is
personalization, the interaction orthogonal to the additive subspace, that an
additive model cannot represent). Pooling (UCB-Homogeneous, Theorem 1 remark) is the
special case b_i = 0; it ranks by the same c_j.

**Proof.** (a) a + b_i + c_j = ((a 1 + b) 1^T + 1 c^T)[i,j], a sum of two rank-1
matrices, so rank <= 2. (b) For fixed i, argsort_j (a + b_i + c_j) = argsort_j c_j.
The reward R[i,j] = <p_i, u_j> has, beyond its rank-<=2 additive projection, an
interaction component that varies the per-drone target order; for d > 1 this
component is nonzero for generic factors, so the popularity order is not the
per-drone-optimal order and the additive skill is bounded below CF's. EMPIRICS:
BiasModel unseen ~0.12 and UCBHomo ~0.17 (both additive/popularity) vs CF ~0.49 at
d = 5 (E15, C14); the additive ceiling is real and well below CF.

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

## Experimental alignment (theory vs measured)

Each result mapped to its quantitative confirmation; tensions noted honestly.

| theorem | prediction | experiment(s) | measured | aligned? |
|---|---|---|---|---|
| T1 tabular floor | unseen error Omega(1), unseen skill = 0, broadcast useless to per-arm tabular | C8, C11, C14, C15, E13, scaling[d] | Tabular/UCBIndep unseen ~0 at EVERY rho, d, sigma (e.g. C8 0.006; scaling[d] ~0 for d=1..20) | YES |
| pooling lemma | UCBHomo recovers only rank-1 popularity -> partial unseen, < CF | C14, E7 | UCBHomo unseen 0.17->0.07; E7 popularity prior 0.28 < CF | YES |
| T2 CF row from O(d) given U; skill scales with low-rankness | onboarding/newcomer at ~d probes; unseen falls as true d rises | C12, E7, C13, scaling[d] | C12 knee ~d_hat; E7 personalizes by ~d probes; unseen 0.96@d1 -> 0.10@d20 | YES (trend); finite-data U recovery degrades at large d (idealization is "U known") |
| T3 anytime: structure-free skill <= g(cT/n) -> 0; CF -> 1 | per-arm bandits stuck ~0 under n>>T; CF earns from round 1 | C16, scaling[T], scaling[n] | UCBIndep anytime ~0 EVEN at T=200 (n=240>T); RewardCF 0.21->0.65 as T grows; structure-free worsens with n | YES (mechanism exact). NOTE: the closed-form bound is an ORDER bound, loose at our params (cT/n~4>1 -> vacuous); it bites for cT<<n, while the UCB "stuck" mechanism is what the data show directly |
| T4 masking model | unseen/anytime invariant to persistent vs iid; uniq durable (flat in T) vs transient (decays) | E12 | unseen/anytime within ~0.04 across modes; uniq persistent flat 0.86->0.80, iid decays 0.90->0.51 over T=25..200 | YES (quantitative) |
| T2 fairness corollary | robust to guessed rank d_hat | scaling[d_hat] | RewardCF/HybridCF unseen stable/improving over d_hat=2..20 (true d=5) | YES |

TENSIONS (stated for honesty):
1. T3's algebraic skill <= g(cT/n) is loose at the default operating point (it is
   only informative for cT<<n); the EMPIRICAL anytime-0 of UCBIndep is driven by the
   sharper "untried-arm bonus" mechanism (the T3 remark), which the data confirm
   exactly. Both point the same way; the closed form is a conservative envelope.
2. T2 assumes U is known exactly; in experiments U is recovered with finite, masked
   data, so CF unseen is BELOW the idealized 1.0 and falls with d as U-recovery
   degrades. The theory's monotone "gap scales with low-rankness" matches; the
   absolute level reflects finite-sample U recovery (and our ALS was under-converged
   at default settings, understating CF; see the convergence finding).

Net: every theorem's qualitative prediction is confirmed; T4 and the rank/floor
results are confirmed quantitatively; T3 is confirmed via its mechanism (the
order-bound constant is loose by design).
```

---

## Results on the recent findings (confidence, contention, rank)

These formalize the cycle 40-45 empirical results.

### Proposition 6 (confidence: inverse-variance fit-weighting is suboptimal; the Bayesian posterior is the right object). REASONED.

Consider the unseen-pair prediction R̂_{ij} = ⟨p_i, û_j⟩ for a target j drone i never
pulled, so û_j is identified ONLY from teammates' broadcast observations of j.
(a) Weighting the FIT by inverse own-noise ("precision", own events weight
1/σ_own² ≫ teammate events 1/σ_obs²) over-weights drone i's own rewards, which carry
ZERO information about û_j (i never observed j); relative to uniform weighting it
shrinks the broadcast's contribution to û_j and thus INFLATES Var(û_j) and the unseen
prediction error. Hence uniform (coverage-preserving) weighting weakly dominates
precision weighting for unseen prediction whenever the broadcast is the only source of
û_j. (b) The variational-Bayes posterior (EMCF) is the consistent full-information
estimator: each observation enters with its likelihood precision INSIDE the model while
the prior fixes the data-vs-regularizer scale; the predictive variance
Var(R̂_{ij}) = p_i^T Σ_{u_j} p_i + u_j^T Σ_{p_i} u_j + tr(Σ_{p_i}Σ_{u_j}) is a valid
interval, so a UCB on it is optimism under (calibrated) uncertainty.
**Confirmed by** E15/§8.12 (uniform/EM > precision on unseen at default noise),
PRECISION_SWEEP (uniform wins unseen at every σ_obs), CONFIDENCE bake-off (EM dominates
uniform). **Honest caveat (H1):** the FULL predictive-variance UCB OVER-explores early
(the own-factor term u_j^T Σ_{p_i} u_j is uniformly large for all targets while p_i is
unknown), so for exploration use only the COLLECTIVE term p_i^T Σ_{u_j} p_i (the shared
û_j uncertainty), which is target-specific and anneals as the swarm pins down U.

### Theorem 7 (decentralized symmetry-breaking under contention). EXACT.

m drones in K types (within-type spread → same-type drones share the estimate R̂ up to
o(1)); a SHARED offer pool S each round; capacity-1 matching; NO communication.
(a) [argmax collides] Deterministic a_i = argmax_{j∈S} R̂_{ij}: every same-type group
whose top target lies in S has all its members pick that one target, so the expected
number of lost (collided) engagements is ≥ Σ_{types k} (g_k − 1)·Pr(top_k ∈ S) =
Θ(m − K) when S covers the type-tops, the matching floor.
(b) [fixed private offset de-conflicts] Give drone i a FIXED offset h_i ∈ ℝ^n with
i.i.d. continuous entries and select argmax_{j∈S}(R̂_{ij} + ε h_i[j]). Within a same-type
group the perturbed argmaxes are a.s. distinct (continuous ties have measure zero), so
same-type collisions on a g-member group vanish once its top-g targets lie in S; and any
target with reward margin > 2ε‖h‖_∞ over its runner-up is unchanged, so value is
preserved up to O(ε).
(c) [why fixed AND private] A RE-randomized per-round offset gives the same per-round
collision probability in expectation (no stable assignment); a SHARED-signal offset
(popularity / collective count) shifts every drone identically and RE-synchronizes.
Only a FIXED, PRIVATE offset both de-conflicts and is stable.
**Confirmed by** §8.13: ContentionCF (fixed private offset) earns ~2× at severe
contention (pool=15: 0.105 vs ~0.05), non-overlapping CIs; per-round-softmax and
shared-popularity routing both backfire, exactly as (c) predicts.

### Theorem 8 (ARD recovers the identifiable rank). EXACT (identifiability) + REASONED (level).

Under variational PMF with ARD (per-column prior precision α_r, VB update
α_r = (m+n)/(E‖P_{·r}‖² + E‖U_{·r}‖² + 2b_0)), a latent column r is RETAINED (α_r
bounded) iff the observed, masked design excites direction r with second-moment energy
above the prior/noise floor; otherwise α_r → ∞ and column r is pruned. Therefore the
recovered effective rank equals the number of latent directions IDENTIFIABLE from the
observed design, which is ≤ the generative rank d, with equality iff every direction is
both sufficiently excited (factor variance) and sufficiently observed. Under masking +
within-type spread the weakest directions fall below the floor, so the recovered rank is
< d. Crucially the retained set does not depend on the guessed d̂ (extra columns are all
pruned), so ARD removes the rank hyperparameter.
**Confirmed by** §5.7 ARD: recovered effective rank ≈ 3.2 (< generative d=5), IDENTICAL
at d̂=8 and d̂=20, with no accuracy loss and improved anytime.

### Proposition 9 (choice-informativeness is identifiable only HELD-OUT). EXACT (random fixed point) + REASONED.

Setup (ChoiceEM, §5.6): teammate k's choice a in offer S is a mixture: with prob γ_k a
Boltzmann-rational pick (softmax on ⟨p_k,u⟩/τ), else uniform. The EM responsibility is
r = γ·s / (γ·s + (1−γ)/|S|), where s is the model's softmax probability of the OBSERVED
choice; the M-step sets γ_k ← mean_t r.

(a) Random fixed point (EXACT). If teammate k chooses UNIFORMLY AT RANDOM and the scoring
model is INDEPENDENT of that choice, then E[s] = (1/|S|)·Σ_j softmax_j = 1/|S| (softmax sums
to 1), so E[r] = γ·(1/|S|)/(γ/|S| + (1−γ)/|S|) = γ. Thus γ_k is a fixed point at ANY value:
a uniform-random teammate's informativeness is NOT driven down by the data, it just sits at
its prior. (This is why a uniform-random chooser, the "almost-random due to low confidence"
teammate, is the HARDEST case: maximally ambiguous between the two mixture components.)

(b) In-sample INFLATION (the failure mode). If s is computed from a factor p̂_k FIT to k's
OWN choices (in-sample), the model is positively correlated with the choice it scores, so
E[s] > 1/|S| and E[r] > γ: the estimator OVERFITS a random teammate's choices and spuriously
INFLATES γ_k above its prior, wrongly trusting noise. This is the mechanism behind the
homogeneous-world null (learned γ ties the fixed ramp because neither discriminates).

(c) Held-out IDENTIFIABILITY (the fix). Score each choice ONCE against the model BEFORE the
refit incorporates it (predictive responsibility: choices added since the last refit are not
yet in p̂_k). Then s is independent of that choice, so by (a) a uniform-random teammate's γ
stays at its LOW prior, while a genuinely informative teammate has E[s] > 1/|S| EVEN held-out
(its choices concentrate on model-preferred targets), driving γ above the prior. With a low
prior γ_0, informative and uninformative teammates SEPARATE. Caveat: a CONSISTENTLY-WRONG
(stable but off-objective) teammate is predictable, so held-out γ still trusts it; detecting
that needs a reward-IMPROVEMENT (choice-value gradient) signal, not just predictability.
**Confirmed by** §5.6 heterogeneous-teammate SANITY (oracle vs random teammates): predictive
γ(oracle) ≈ 0.48 ≫ γ(random) ≈ 0.11 (sanity PASSES), vs in-sample γ(oracle) ≈ 0.95 vs
γ(random) ≈ 0.70 (sanity FAILS, the inflation of (b)); 1-seed smoke, full 8-seed run pending.
