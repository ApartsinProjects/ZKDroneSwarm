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
- Observation channel: each round drone i privately SENSES a (masked) subset of the
  round's engagements. Sensing is INDEPENDENT PER OBSERVER: drone i's record of drone
  k's engagement is r̃^(i)_k = R_{k,a_k} + η^(i)_k, with η^(i)_k ~ N(0, σ²) drawn FRESH
  per observer i, per observed k, per round (σ = σ_own for k=i, σ = σ_obs for k≠i,
  subject to the mask M_{ik}). So the SAME action a_k is recorded with INDEPENDENT noise
  by different observers, η^(i)_k ⟂ η^(i')_k for i≠i'; there is NO single shared
  broadcast value (that would be a transmitted measurement, i.e. communication). This
  per-observer independence (together with the per-observer mask) is exactly what yields
  genuinely distinct per-drone internal states (state-uniqueness; Theorem 4). Masking
  models are defined in Section 4.

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
measured UCB-Homogeneous partial unseen skill (≈ 0.17 at d = 5 in pre-catalogue
cycle-23/E15 runs; qualitative, not yet in a committed-data table, regenerate before citing).

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
d = 5 (pre-catalogue E15/C14 runs; these specific numbers are not in a committed-data
table and should be regenerated before citing in the paper, the QUALITATIVE additive
ceiling < CF is sound and matches the §8.9 broader bake-off).

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
û_j AND the broadcast sources are HOMOGENEOUSLY noisy. **Scope caveat (PRECISION_HETERO,
H11b):** when the teammate sources DIFFER in reliability, noise-aware weighting helps, but
only in a RATIO-BOUNDED form: bounded precision (relcap) beats uniform under heterogeneous
noise (+0.09 unseen, non-overlapping CIs) while UNBOUNDED 1/σ² loses everywhere by
over-concentrating on the few clean rows and starving coverage. So the correct statement is
"uniform dominates UNBOUNDED precision; bounded precision dominates both iff sources differ
in reliability" (see proposed Proposition 10). (b) The variational-Bayes posterior (EMCF) is the consistent full-information
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

### Theorem 7 (decentralized symmetry-breaking under contention). EXACT (a, b) + REASONED (c).

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
**Caveat (T7 covers only the FIXED offset):** the fixed offset HURTS preference quality
when contention is absent (CONTENTION: ContentionCF unseen 0.023 at pool=240 vs RewardCF
0.323), because a constant offset is value-preserving only up to O(ε) and at no-contention
even that costs the categorical metric. The headline method is the ADAPTIVE, loss-gated,
scarcity-gated ContentionAdaCF (unseen recovers to ~0.32, earned reward best-or-tied at
every pool), which T7 does NOT theorize. The proposed "adaptive-offset envelope" (backlog)
would cover it: scale→0 at no contention (reduces to greedy, value-preserving) and →fixed
offset at saturation (T7b). This is an honest gap between the proven theorem and the
deployed method.

### Theorem 8 (ARD recovers the identifiable rank). REASONED (identifiability "iff", no VB fixed-point proof) + CONFIRMED (d̂-invariance).
<!-- Label corrected (theory audit): the "retained iff excited above floor" iff is REASONED
(standard ARD intuition, not a fixed-point proof for the coupled masked-online VB updates),
not EXACT. The solidly-supported claim is d̂-INVARIANCE (recovered ~3.2-3.3 at d̂=8 and 20).
And the recovered rank is NON-monotone in the true rank (ARDRANK: 2.00/2.35/2.13/1.73 for
d=2/3/5/8) due to an SNR confound, so "recovers the rank" means the identifiable rank, not
the raw true rank. -->


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
**Confirmed by** §5.6 heterogeneous-teammate SANITY (oracle vs random teammates), 8 seeds,
bootstrap CIs (CHOICEHETERO): predictive γ(oracle) 0.29-0.49 ≫ γ(random) ≈ 0.10 (sits at the
prior, EXACTLY as (a) predicts), and good-drone unseen improves; vs in-sample γ(random)
0.67-0.72 (the inflation of (b), sanity FAILS). Held-out is the best-confirmed of the recent
propositions.

---

## Theory audit (2026-05-23): keystone gap + proposed new results

A critical pass over T1-T8/P6/P9 (verified against the code and committed data). Honest verdict:
the MOST-CITED results (T1, T2's math, T3's closed-form bound, T5) are textbook OR rest on an
unproven cited keystone; the genuinely NOVEL and well-supported results are T4 (masking
dichotomy) and P9 (held-out choice identifiability + the random fixed point P9a). Labels
corrected this pass: T3(a) (the displayed positive-part equality is loose, see TENSIONS #1; the
bound still holds via the B_t-indicator decomposition), T7 (EXACT a,b + REASONED c), T8
(REASONED "iff" + CONFIRMED d_hat-invariance). Stale inline numbers (UCBHomo ~0.17, BiasModel
~0.12) flagged as pre-catalogue. P6 scoped (bounded precision wins under heterogeneous noise).
T7 caveat added (covers only the FIXED offset; the deployed adaptive ContentionAdaCF/UnifiedCF
is untheorized).

KEYSTONE GAP (the central open problem). Every "Theta(d) vs Theta(n)" / "exact row recovery"
claim (T2, T4c) silently CITES decentralized masked U-recovery (Candes-Recht style). Under
PERSISTENT per-drone masking the sampling is STRUCTURED / non-uniform, exactly where
incoherence-based uniform-sampling completion does not directly apply. So none of T1-T8 actually
PROVE decentralized masked U-recovery; they invoke centralized uniform-sampling theory for it.
Closing this (P15 below) would make the categorical claim self-contained.

PROPOSED new results (statement + sketch + rigor flag; NOT yet proven or integrated):
- P10 [HIGH, CLEAN] Bounded-precision dominance under heterogeneous reliability. Ridge fold-in
  with w_k proportional to min(1/sigma_k^2, kappa*min_j 1/sigma_j^2) beats uniform iff
  Var_k(sigma_k^2)>0, and beats unbounded 1/sigma^2 when the high-precision sources fail to span
  R^d. Sketch: minimize the sigma-weighted prediction variance under a coverage (span)
  constraint; uniform ignores heterogeneity, unbounded 1/sigma^2 drives lambda_min(design)->0.
  Fixes P6's overreach; explains PRECISION_HETERO + PRECISION_SWEEP.
- T9 [HIGH, MEDIUM/assembly] The 3-condition scope as a formal iff. CF unseen-skill >
  structure-free + Omega(1) iff (i) d>1, (ii) cT=o(n), (iii) rho>0. (<=) = T1+T2. Necessity:
  not(i)=>T5; not(ii)=>tabular measures Omega(n) entries, floor->0; not(iii)=>CF=Definition-1.
  Promotes the boxed paper scope claim to a theorem; (ii) needs a short tabular-coverage lemma.
- P11 [HIGH, CLEAN-ish] Choice-vs-reward crossover sigma*. Exists sigma* s.t. for sigma_obs>
  sigma* an argmax-choice estimator (sigma-independent Fisher information) beats a
  noisy-cardinal-reward one (Fisher proportional to 1/sigma^2); sigma* grows with offer size c,
  shrinks with d. Explains CHOICEEM's sigma=2.0 niche.
- P12 [HIGH, MEDIUM] Churn fold-in latency / recovery condition. Under turnover eta/round with
  greedy exploitation, expected probes-before-first-pull of a fresh target is Theta(n/c) (->
  never if dominated) => steady-state fresh-skill->0; a predictive-variance UCB gives O(d) =>
  fresh-skill bounded below iff eta*d < c. Theorizes CHURN / H6b.
- P13 [MEDIUM, HEURISTIC] Adaptive-offset envelope. A loss-gated scale s(l)=eps_hi*l^p reduces
  to greedy (value-preserving, T7-margin) as realized loss l->0 and to the T7(b) fixed offset as
  l->1, so it is best-or-tied at both extremes (interior is empirical). Theorizes ContentionAdaCF
  and UnifiedCF, which T7 omits.
- P14 [MEDIUM] Mean-field VI is discriminative but anti-conservative. VB-PMF predictive variance
  is monotone in the true conditional error (discrimination) but under-estimates it (coverage <
  nominal) by the dropped cross-covariance Cov(p_i,u_j). Explains CALIBRATION.
- P15 [HIGHEST value, HARD] Decentralized masked U-recovery (the keystone). Under persistent
  Bernoulli(rho) per-drone masking each drone recovers col-space(U) to o(1) once its visible
  entry count exceeds Otilde(d(m+n)/rho), under incoherence + a rho-effective-sampling condition.
  The honest version of the clause T2/T4c currently cite; persistent masking is non-uniform
  sampling so off-the-shelf Candes-Recht does not apply. Closing this makes the categorical
  claim self-contained.

Most valuable to ADD next: P10 (clean, fixes a live inconsistency), T9 (clean assembly), and
P15 (the hard keystone). The rest are explanatory and honestly heuristic where flagged.

---

## Theorem 9 (scope of the categorical advantage: a PRECISE iff). EXACT given T1/T2/T5/T3 + a coverage count.

This promotes the boxed "3-condition" scope claim to a precise statement and CORRECTS it: the box
("CF beats tabular iff d>1 AND sample-starved AND shared") conflates three gaps with three
DIFFERENT roles. Skill is on UNSEEN pairs (targets a drone never pulled). Conditions: (i)
PERSONALIZED rank d>1; (ii) SAMPLE-STARVED T = o(n) (a drone pulls one target/round, so it engages
≤ T of n targets; for the anytime corollary the relevant quantity is cT = o(n), per T3); (iii)
SHARED channel ρ > 0.

(A) [vs structure-free TABULAR: the categorical mechanism] By T1 a structure-free learner has
unseen-pair skill = 0 in EVERY regime. A CF learner attains Ω(1) unseen-pair skill iff (iii) ρ > 0
AND R is structured (rank ≥ 1): then U is identified from the broadcast (KEYSTONE clause) and the
row is completed from O(d) own pulls (T2), so CF predicts unseen pairs with o(1) error. Hence the
CATEGORICAL (zero-vs-nonzero) unseen gap vs tabular needs ONLY (iii) + structure, and holds even at
d = 1 (CF recovers the rank-1 popularity that tabular cannot) and even when sample-rich.

(B) [vs the POPULARITY / additive baseline: the role of d>1] Against the rank-1 popularity / additive
model (UCBHomo / BiasModel, which already attains the rank-1 part), the unseen gap Δ_pop is Ω(1) iff
(i) d > 1. At d = 1, R = popularity and CF's per-drone ranking coincides with the shared popularity
ranking (T5), so Δ_pop = 0; for d > 1 the personalization (the interaction orthogonal to the additive
subspace, T5) is recoverable by CF but not by the additive model, giving Δ_pop = Ω(1).

(C) [role of starvation: OPERATIONAL relevance] (A)/(B) concern the unseen-RESTRICTED skill, which is
regime-independent (T1 holds in any regime; CF's recovery needs only ρ>0). Starvation (ii) is what
makes the unseen advantage drive the OVERALL / ANYTIME metric: the fraction of a drone's targets that
stay unseen is 1 − E|pulled|/n = 1 − Θ(T/n), which → 1 iff T = o(n). When sample-rich (T = Ω(n)) the
per-pair gap persists but unseen pairs are rare, so by T3 the anytime gap closes.

**Necessity (summary).** ¬(iii) [ρ=0] ⇒ CF sees only own pulls = Definition 1 ⇒ CF unseen skill = 0
(no gap, T1). ¬(i) [d=1] ⇒ Δ_pop = 0 (T5), though Δ_tabular can still be > 0. ¬(ii) [sample-rich]
⇒ the unseen advantage is operationally negligible (T3).

**Proof.** (A) T1 gives the 0 floor for tabular; T2 gives CF's o(1) unseen error for ρ>0 (via the
U-identification clause; see KEYSTONE GAP for the one cited step). (B) T5. (C) unseen-fraction:
each round adds ≤ 1 distinct pulled target, so |pulled| ≤ T and, when T = o(n) (collisions rare),
E|pulled| = Θ(T); hence unseen-fraction = 1 − Θ(T/n). T3 converts that fraction into the anytime
gap. ∎

**Reading (the correction).** The honest scope is NOT a single 3-way "iff." (iii)+structure already
give the CATEGORICAL unseen gap vs tabular (the headline). (i) d>1 is what's additionally needed to
beat the POPULARITY baseline. (ii) starvation is what makes the unseen advantage OPERATIONALLY
dominant. Three conditions, three distinct roles, the paper/tutorial scope box should say so rather
than ANDing them against one baseline. Promotes proposed-T9 to a stated result.

---

## Proposition 10 (bounded precision dominates under HETEROGENEOUS source noise). EXACT (a) + REASONED (b,c).

Fixes P6's overreach and explains PRECISION_HETERO. Estimate an UNSEEN target's factor by ridge
fold-in from the teammates k that engaged it, û_j = (Σ_k w_k p_k p_k^T + λI)^{-1} Σ_k w_k r_{kj} p_k,
with r_{kj} = ⟨p_k,u_j⟩ + N(0,σ_k²) and weights w_k > 0; predict R̂_{ij} = ⟨p_i, û_j⟩.

(a) [Homogeneous noise: weighting only rescales λ.] If all σ_k = σ (so the variance-optimal weights
are uniform), any constant weight w gives û_j = (Σ p_k p_k^T + (λ/w)I)^{-1} Σ r_{kj} p_k: changing w
only rescales the regularizer, not the relative source weighting. So under homogeneous teammate
noise, "precision" buys nothing over uniform. (This is the regime of P6: there, naive precision
still LOSES because it over-weights the drone's OWN clean row, σ_own ≪ σ_obs, which carries zero
information about an unseen u_j and shrinks the broadcast that does, inflating Var(û_j). P10 is the
complementary teammate-vs-teammate story; P6 is the own-vs-teammate story.)

(b) [Heterogeneous noise: noise-aware weighting helps.] If Var_k(σ_k²) > 0, the
variance-minimizing (Gauss-Markov / BLUE) weighting is w_k ∝ 1/σ_k², PROVIDED the weighted design
G(w) = Σ_k w_k p_k p_k^T stays full-rank. Uniform weighting ignores the heterogeneity (it weights a
σ=1.9 source like a σ=0.1 one), so it is strictly suboptimal whenever the variances differ.

(c) [Why it must be BOUNDED.] Unbounded w_k = 1/σ_k² concentrates almost all mass on the lowest-noise
sources. If those clean sources do NOT span ℝ^d (their {p_k} lie in a proper subspace), then in the
unspanned directions λ_min(G(w)) collapses to the prior λ, so û_j there is dominated by the prior
(bias), and the prediction error is LARGE for any p_i with a component in that subspace. A
ratio-bounded weight w_k ∝ min(1/σ_k², κ·min_{k'} 1/σ_{k'}²) caps the concentration: it still
down-weights the noisiest sources (noise-awareness) but keeps every direction excited (coverage),
so G(w) stays full-rank. Hence bounded precision DOMINATES BOTH uniform (it uses the noise
information) AND unbounded precision (it preserves coverage) exactly when sources are heterogeneous
and the clean ones are coverage-deficient (the generic case under sparse per-target engagement).

**Confirmed by** PRECISION_HETERO: ratio-capped "relcap" beats uniform by +0.093 unseen / +0.089
anytime under heterogeneous noise (non-overlapping CIs) and loses under homogeneous; UNBOUNDED
"full" 1/σ² loses in BOTH (it over-concentrates and starves coverage), exactly as (a)/(c) predict.
**Correction to P6:** the right statement is "uniform dominates UNBOUNDED precision for unseen; but
BOUNDED precision dominates both iff teammate sources differ in reliability." Rigor: (a) exact;
(b) Gauss-Markov is standard, the full-rank proviso is the operative new condition; (c) is the
coverage/eigenvalue mechanism, correct but stated rather than fully derived (REASONED).

---

## Proposition 16 (H5b: predictive-variance UCB regret). EXACT given U (reduction to LinUCB) + OPEN (joint).

This is the requested H5(b): why confidence-DIRECTED exploration (EMCF) beats a fixed ε-schedule.
GIVEN the target factors U (or after U is recovered; see KEYSTONE/P15), each drone's per-round
problem is a LINEAR bandit in the known feature u_j with unknown parameter p_i: it earns
⟨p_i,u_j⟩ + noise and updates the Gaussian posterior Σ_{p_i} = (Σ_j β_{ij} u_j u_j^T + λI)^{-1}.
EMCF's index, score(j) = ⟨p̂_i,u_j⟩ + β·sqrt(u_j^T Σ_{p_i} u_j), is EXACTLY the LinUCB / OFUL
optimistic index with that posterior confidence ellipsoid. Therefore, by the standard
self-normalized / elliptical-potential argument (Abbasi-Yadkori, Pál, Szepesvári 2011), the
per-drone cumulative regret of learning its own factor is Õ(d√T) w.h.p., and the predictive-variance
bonus is precisely the term that drives probes to the LEAST-explored latent directions, which is why
directed exploration beats a fixed ε-schedule on sample efficiency (confirmed: H1 collective
exploration best final anytime; H6b EMCF dominates under churn by probing fresh arrivals).
**Caveat (P6/H1):** for the bonus use the COLLECTIVE term p_i^T Σ_{u_j} p_i (shared-U uncertainty,
target-specific, anneals as U is pinned down), NOT the full predictive variance, whose own-factor
term u_j^T Σ_{p_i} u_j is uniformly large early and over-explores.
**OPEN (the hard part, tied to P15):** the JOINT regret when U is learned SIMULTANEOUSLY from the
masked broadcast is a bilinear/low-rank bandit under STRUCTURED masking; LinUCB does not cover it
and the centralized phase-structured low-rank-bandit bounds (ESTR, Jun et al. 2019) are not anytime
or decentralized. A self-contained joint, decentralized, masked regret bound is open and would build
on P15 (decentralized masked U-recovery). Rigor: the given-U reduction is EXACT (it IS LinUCB); the
joint, U-learning case is OPEN.

---

## Propositions 11-15 (promoted from the proposed-results block; cycle 66)

These flesh out the P11-P15 sketches into standalone statements with the best available proof and an
HONEST rigor flag. P11/P12/P14 are clean-to-medium; P13 now also covers the cycle-64 abundance gate;
P15 is the keystone and is stated as a partial result + precise conjecture (it is genuinely open).

### Proposition 11 (choice-vs-reward crossover sigma*). REASONED (existence EXACT, monotonicity heuristic).

Setup: a teammate k chooses a_k = argmax_{j in S} R_{k,j} from a clean offer S of size c (the CHOICE
channel observes a_k exactly; only the cardinal REWARD channel is noised by sigma_obs). Consider two
estimators of u_j-direction information from one observed engagement: (R) the noisy reward
R_{k,a_k}+N(0,sigma_obs^2); (C) the argmax event {a_k = argmax}.

Claim: there is a sigma* > 0 such that for sigma_obs > sigma* the Fisher information the CHOICE
channel carries about the relevant factor direction exceeds that of the REWARD channel.

Proof (existence). The Gaussian reward likelihood gives per-observation Fisher information
I_R(theta) = (d mu/d theta)^2 / sigma_obs^2, strictly DECREASING in sigma_obs and -> 0 as
sigma_obs -> inf. The choice event is generated from the teammate's CLEAN rewards, so its likelihood
(a max-of-c ordering probability) does NOT depend on sigma_obs at all: I_C(theta) is a constant
I_C0 > 0 (positive because the argmax is informative about the ordering of the <p_k,u_j>). Since
I_R -> 0 while I_C = I_C0 > 0, by continuity there is a unique crossover sigma* with
I_R(sigma*) = I_C0, and I_C > I_R for all sigma_obs > sigma*. QED (existence).

Monotonicity (heuristic): I_C0 increases with c (a max over more alternatives pins the ordering more
tightly) and the per-direction signal (d mu/d theta)^2 shrinks as the d directions share a unit-norm
budget (~1/d), so sigma* shifts with c and d; the precise law is not derived here. EXPLAINS the
CHOICEEM niche: at sigma_obs = 2.0 the noise-immune choice channel overtakes the reward channel on
BOTH unseen and anytime (catalogue rows 43/48), exactly the sigma_obs > sigma* regime.

### Proposition 12 (churn fold-in latency / recovery condition). ORDER (balance constant heuristic).

Setup: continuous turnover of eta fresh targets per round; each drone is offered c of n targets and
pulls one. A FRESH target has no observations, so a low-rank model predicts it at its prior mean.

Claim (a, negative): a purely GREEDY (exploitative) policy pulls a fresh target only if its
prior-mean prediction exceeds the running best; the probability of that -> 0 as the model sharpens,
so the expected number of rounds before a given fresh target is first pulled is Theta(n/c) when it is
ever competitive and unbounded when it is dominated. Hence steady-state skill on the RECENT-arrival
set -> 0 (matches CHURN: plain RewardCF fresh-skill 0.074, catalogue row 53).

Claim (b, fix): a predictive-variance UCB adds a bonus that is largest for unpulled targets and
decays like 1/sqrt(n_pulls); a fresh target's bonus exceeds the value gap until it has been pulled
O(d) times (enough to fold its factor in). With eta fresh targets/round and c slots, the swarm can
service the fresh backlog iff eta * d < c (each arrival needs ~d collective probes, c are available
per round); under this balance steady-state fresh-skill is bounded below by a constant. EXPLAINS H6b
(ActiveCF/EMCF fresh-skill 0.36-0.37 vs UCBIndep 0.13). Rigor: ORDER; the iff is a flow-balance
heuristic (constants not pinned).

### Proposition 13 (loss-and-abundance-gated envelope). EXACT at the extremes, interior empirical.

This now theorizes the deployed ContentionAdaCF AND the cycle-64 UnifiedCF+ab, both omitted by T7.
Policy: offset scale s(l) = eps_hi * l^p (l = realized loss-rate EMA) AND a UCB-exploration gate that
is damped when l is high OR when the offer is abundant (|S| > k*m).

Claim: the policy is best-or-statistically-tied to the per-regime specialist at all four corners of
(loss, abundance):
- l -> 0, |S| > k*m (no contention, plentiful): s -> 0 and the abundance gate zeroes exploration, so
  the policy reduces to GREEDY exploitation, which is optimal for one-shot earned reward when there is
  nothing to learn-vs-exploit and nothing to de-conflict (matches greedy at pool=240).
- l -> 1, |S| <= k*m (severe contention): s -> eps_hi and the gate is ON, recovering the T7(b) FIXED
  private offset, which beats greedy by the T7 collision-probability margin (~2x at pool=15).
- l -> 0, |S| <= k*m (scarce but winning): exploration ON (UCB), offset ~0 = EMCF, the learning regime.
- l -> 1, |S| > k*m: cannot co-occur for long (losing implies contention implies not-abundant), so the
  fourth corner is transient.
Proof: each corner is a reduction to an already-characterized policy (greedy / T7(b) / EMCF) by taking
the stated limit of s(l) and the gate; EXACT at the limits. The interior trajectory is the
empirically-validated best-or-tied result (catalogue rows 54/56: UnifiedCF+ab ties or wins every
regime). Rigor: EXACT corner reductions; interior is empirical (the honest envelope claim).

### Proposition 14 (mean-field VI: discriminative but anti-conservative). REASONED + CITED.

Claim: the VB-PMF predictive standard deviation sd_ij = sqrt(p_i^T Sigma_{u_j} p_i + u_j^T Sigma_{p_i}
u_j) is (a) MONOTONE in the true conditional RMSE (discrimination), but (b) UNDER-estimates it
(empirical coverage < nominal).
(a) Each posterior precision Lambda_{u_j} grows by the (precision-weighted) outer products of the
observers of j, so Sigma_{u_j} shrinks monotonically in j's effective observation count, which is
exactly what drives down the true error; hence sd and RMSE move together (matches CALIBRATION: RMSE
rises monotonically across sd quintiles, Q1 0.231 -> Q5 0.492).
(b) Mean-field factorizes q(P,U) = q(P)q(U), DROPPING the posterior cross-covariance Cov(p_i,u_j); the
true predictive variance of <p_i,u_j> includes a positive cross term that the factorized sd omits, so
sd is biased LOW and nominal intervals under-cover (50% -> 40%, 95% -> 92%). This is the standard
mean-field variance-underestimation (CITED: Blei-Kucukelbir-McAuliffe 2017; Wang-Titterington). Use:
the posterior is sound for RELATIVE uncertainty (UCB/shrinkage ordering), not exact coverage.

### Proposition 15 (decentralized masked U-recovery, THE KEYSTONE). BOUNDED: deterministic sufficient + necessary condition (exact noiseless, bounded noisy) + coverage-time for the non-adaptive policy; residual = the adaptive finite-time coverage rate.

This is the clause T2/T4c cite. Goal: each drone i recovers col-space(U) (so its row fold-in
generalizes to UNSEEN targets) from its OWN persistently-masked, noisy broadcast. Previously stated as
fully OPEN; we now CLOSE it to an explicit, checkable per-drone condition with an exact (noiseless) /
bounded (noisy) recovery guarantee and a coverage-time bound, leaving only the adaptive finite-time
coverage rate open.

Setup (exact, matching the harness run_masked). R = P U^T, P in R^{m x d}, U in R^{n x d}, rank d,
mu-incoherent. The persistent mask is over DRONE PAIRS: M in {0,1}^{m x m}, M_{ik} ~ Bern(rho) i.i.d.,
M_{ii}=1; drone i observes EVERY engagement of its fixed visible-teammate set N_i = {k : M_{ik}=1} (and
none of the others) for all rounds, each value noised by sigma_obs. Let G_i = (N_i ∪ [n], E_i) be drone
i's observation bipartite graph, edge (k,j) in E_i iff teammate k engaged target j at some round; for a
target j let its visible-engager set be E_i(j) = {k in N_i : (k,j) in E_i} with degree deg_i(j)=|E_i(j)|.

SUFFICIENT CONDITION (recovery is EXACT, noiseless). Suppose:
  (i) [Anchor / frame] G_i contains a d x d fully observed block R[A, J_0], A in N_i, J_0 in [n],
      |A|=|J_0|=d, with rank(P[A,:]) = rank(U[J_0,:]) = d (an invertible anchor block); AND
  (ii) [Per-target spanning coverage] target j has rank(P[E_i(j),:]) = d (its visible engagers'
      factor rows span R^d).
Then drone i recovers P[N_i,:] and every row {u_j : (ii) holds} up to a SINGLE common invertible frame
G in R^{d x d} (pinned by the anchor block), EXACTLY. Folding in drone i's own p_i from >= d of its own
probes on recovered targets (Theorem 10 with eps=sigma=0) recovers p_i in the dual frame, and the
prediction R_hat[i,j] = p_i^T u_j is frame-invariant and EXACT for every j satisfying (ii).

PROOF. (Frame) R[A,J_0] = P[A,:] U[J_0,:]^T is d x d and invertible by (i); any factorization
consistent with the observed entries must agree with it up to a common G (a fully observed invertible
d x d block pins the factorization frame; standard low-rank identifiability). (Per-target) the observed
column entries satisfy R[E_i(j), j] = P[E_i(j),:] u_j; by (ii) P[E_i(j),:] has full column rank d, so
u_j = (P[E_i(j),:]^T P[E_i(j),:])^{-1} P[E_i(j),:]^T R[E_i(j),j] is the UNIQUE least-squares solution and
equals the true u_j (in the anchor frame). (Fold-in) Theorem 10 at eps=sigma=0, lam->0, k>=d, rank d
returns p_i exactly; the bilinear product p_i^T u_j cancels G. QED. (rho=1 full broadcast is the special
case N_i=[m]: every entry observed, the anchor and coverage conditions hold trivially, recovering the
earlier centralized-completion result.)

NECESSITY (per-target floor). If deg_i(j) < d or rank(P[E_i(j),:]) < d, the system
R[E_i(j),j] = P[E_i(j),:] u_j is underdetermined: u_j has a non-trivial component free in the nullspace
of P[E_i(j),:]^T, so R[i,j] = p_i^T u_j is NON-IDENTIFIABLE from drone i's broadcast and its
Bayes-optimal estimate is the prior mean. This is the per-target analogue of the tabular floor
(Theorem 1): per-target spanning coverage is NECESSARY as well as sufficient.

NOISY BOUND. With per-observation noise sigma, the least-squares solve of (ii) gives
E||u_j_hat - u_j|| <= sigma sqrt(d) / sigma_min(P[E_i(j),:]) ~ sigma sqrt(d / deg_i(j)) for incoherent
factors; composing with the fold-in bound (Theorem 10) gives end-to-end
  E|R_hat[i,j] - R[i,j]| <= C ( eps_anchor ||u_j|| + sigma sqrt(d / deg_i(j)) + ridge ),
eps_anchor the anchor-block conditioning error: the SAME three-source structure as Theorem 10, now
per target with the coverage degree deg_i(j) controlling the noise term.

COVERAGE TIME (non-adaptive policy, makes the condition self-achieving). Under uniform exploration with
size-c offers, each visible teammate engages each target with probability 1/n per round, so a teammate
has engaged target j at least once after T rounds with probability 1 - (1-1/n)^T ~ 1 - e^{-T/n}. The
number of distinct visible engagers of j is then ~ Binomial(|N_i|, 1 - e^{-T/n}) with |N_i| ~ rho m, and
a union bound over the n targets gives: every target is spanning-covered and the anchor block exists,
w.h.p., once
  T  =  O( (n d / (rho m)) log n )  rounds.
The rate IMPROVES with the visible-teammate count rho m (more teammates cover targets faster): the
coverage-side face of the collective speedup (Theorem 11). It requires rho m > d, below which condition
(i) can never hold, recovering the IMPOSSIBLE-ALONE boundary (Theorem 11a) at rho m = O(1).

RESIDUAL (the remaining open part, now precisely scoped). Under an ADAPTIVE (exploiting) policy,
engagements concentrate on high-reward targets, so low-reward targets may sit below the deg_i(j) >= d
threshold longer; the finite-time coverage rate then COUPLES the policy to the coverage process and is
not given by the clean coupon-collector bound above. eps-greedy exploration keeps every target's
per-round engagement probability >= eps/n > 0, so by Borel-Cantelli every target is eventually
spanning-covered almost surely (asymptotic recovery holds unconditionally); the open quantity is only the
FINITE-TIME adaptive coverage rate and its dependence on the reward gap, a self-contained
random-bipartite-coverage question.

STATUS: BOUNDED (was OPEN). Recovery is EXACT under an explicit, checkable spanning-coverage condition,
provably IMPOSSIBLE without it, error-BOUNDED under noise, and self-achieving in O((n d/(rho m)) log n)
rounds under non-adaptive exploration. T2/T4c now rest on a self-contained per-drone condition (anchor
frame + per-target spanning coverage) rather than an imported uniform-sampling assumption; the only
residual is the adaptive finite-time coverage rate. The EMPIRICAL masked results (C11, catalogue rows
20+) remain the evidence that the condition is met in practice.

EMPIRICAL VALIDATION (pilot_p15.py, docs/P15_VALIDATION.md). On the harness's ACTUAL coverage patterns
(m=30, n=240, d=5, rho=0.5, 6 seeds, noiseless to isolate identifiability), oracle reconstruction from
the observed entries recovers the unseen pair (i,j) to mean error 0.0000 EXACTLY when p_i lies in the
span of its visible engagers' factors (5609 pairs, the full-rank-d cases), and sits at the prior floor
0.30 otherwise (30238 pairs), with GRACEFUL partial recovery as the spanning rank rises toward d
(rank 0/1/2/3/4 non-recoverable error 0.36/0.31/0.24/0.17/0.096). The identifiability threshold is
EXACTLY the proposition's spanning condition, sufficiency and necessity both confirmed on the coverage
the swarm actually produces.

---

## Foundational results (cycle 69): the fold-in error bound, the collective speedup, and minimax tightness

Three new results that DIRECTLY underpin the approach, are self-contained where flagged, and tie the
theory to the experiments (cold-start, the broadcast's value, and the categorical separation's tightness).

### Theorem 10 (fold-in perturbation bound -- cold-start / onboarding error). EXACT (linear algebra).

Unifies E7 (new drone), C12 (new target), and the sensing degradation. A new entity has hidden factor
x_* in R^d (a newcomer's own p_*, or a new target's u_*). It is probed against k known cross-factors
stacked as B in R^{k x d} (for a new target: the d-dim factors of the k probing drones; for a new
drone: the factors of the k probed targets), observing y = B x_* + eta, with mean-0 per-entry noise of
variance sigma^2. The learner holds an ESTIMATE B_hat = B + Delta of that basis (||Delta||_op <= eps,
the swarm's basis-recovery error) and ridge-folds-in x_hat = (B_hat^T B_hat + lam I)^{-1} B_hat^T y. For
any other cross-factor b (estimate b_hat = b + delta, ||delta|| <= eps) the predicted reward is
r_hat = b_hat^T x_hat; the truth is r = b^T x_*. Let s = sigma_min(B) (> 0 iff k >= d and B has rank d).

CLAIM: there are constants C1,C2,C3 (absolute) with
  E|r_hat - r|  <=  C1 * eps * ||x_*|| * (1 + ||b||/s)            [basis-recovery error]
                 +  C2 * ||b|| * sigma * sqrt(d) / s             [own-probe noise]
                 +  C3 * lam * ||x_*|| * ||b|| / s^2             [ridge bias],
and r_hat = r EXACTLY when eps = 0, sigma = 0, lam -> 0, k >= d and rank(B) = d.

PROOF. Write x_hat - x_* = (B_hat^T B_hat + lam I)^{-1} [ B_hat^T eta - lam x_* + (B^T B - B_hat^T B_hat) x_* ].
By Weyl, ||(B_hat^T B_hat + lam I)^{-1}||_op <= 1/(max(s^2 - 2 eps ||B|| - eps^2, 0) + lam); for eps < s/2
this is <= 2/s^2. Then ||B_hat^T eta|| has mean <= ||B_hat||_F sigma <= sqrt(d) ||B_hat||_op sigma (each
of the d coordinates is a noise projection of variance <= ||B_hat||_op^2 sigma^2); ||(B^T B - B_hat^T B_hat) x_*||
<= (2 eps ||B|| + eps^2) ||x_*||; ||lam x_*|| = lam ||x_*||. Collect to bound ||x_hat - x_*||. Finally
|r_hat - r| = |b_hat^T x_hat - b^T x_*| <= |b_hat^T (x_hat - x_*)| + |(b_hat - b)^T x_*|
<= ||b_hat|| ||x_hat - x_*|| + eps ||x_*||, and ||b_hat|| <= ||b|| + eps; substituting gives the three
terms. EXACT case: eps = sigma = lam = 0 makes B_hat = B and x_hat = (B^T B)^{-1} B^T (B x_*) = x_* (rank d),
so r_hat = b^T x_* = r. QED.

WHY IT MATTERS. (i) It is the first SELF-CONTAINED error bound for our cold-start/onboarding fold-in,
and it SEPARATES the three error sources. (ii) It makes the Theta(d) probe complexity rigorous: s > 0
requires k >= d, and under sub-Gaussian unit factors s grows like sqrt(k), so the noise term ~ sigma
sqrt(d/k) -- the own-data rate. (iii) It QUANTITATIVELY EXPLAINS the sensing experiment (F16): sparser
sensing -> larger swarm basis-recovery error eps -> larger term (i) -> lower cold-start skill, exactly
the observed degradation; and at full coverage eps -> 0 the bound collapses to the noise/ridge terms.

### Theorem 11 (collective broadcast speedup -- why sharing is the crux). ORDER (coverage EXACT; recovery CITED).

Setup of Section "Setup", m drones, n targets, rank d, persistent Bernoulli(rho) broadcast, one
engagement per drone per round, T rounds, n >> T (sample-starved).

(a) IMPOSSIBLE ALONE. An ISOLATED drone (rho = 0) observes only its own T engagements, all in ONE row of
R. A single row cannot identify a rank-d > 1 column space (the other rows are entirely unconstrained),
so its estimate on any UNSEEN target is the prior mean and its unseen MSE stays at the floor v_i
(this is exactly Definition 1 / Theorem 1). Sharing is therefore NECESSARY, not merely helpful.

(b) COLLECTIVE THRESHOLD. With the broadcast, the swarm's pooled observed support is the union of all
drones' sensed engagements; since every engagement is seen by at least its own drone, |Omega_swarm| =
Theta(mT) distinct entries (n >> T, so few re-engagements). Noisy low-rank completion recovers
col-space(U) once |Omega_swarm| = Otilde(d(m+n)) (CITED: Candes-Plan; with the persistent-masking
caveat of P15). Hence the swarm crosses the recovery threshold after
  T_*  =  Otilde( d (m + n) / m )  =  Otilde( d (1 + n/m) )  rounds,
after which EACH drone, folding-in on the collectively-recovered basis (Theorem 10), generalizes to
unseen pairs.

(c) SPEEDUP. A hypothetical single agent allowed to probe ANY entry needs Otilde(d(m+n)) rounds (one
probe/round) to reach the same threshold; the m-drone swarm reaches it in Otilde(d(m+n)/m), a
Theta(m)-fold COLLECTIVE SPEEDUP, achieved with NO communication (only passive observation of the public
outcome stream). Equivalently, per-drone effective sample complexity drops from Theta(n) (measure your
own row) to Theta(d(1 + n/m)) -> Theta(d) as m grows.

Rigor: (a) EXACT; coverage counting in (b) EXACT; "recover from Otilde(d(m+n)) entries" CITED
(completion) and inherits P15's persistent-masking caveat; the speedup ratio in (c) is exact given (b).
WHY IT MATTERS. Formalizes the central intuition: the broadcast is not a convenience, it is what makes
decentralized recovery POSSIBLE, and it yields an explicit m-fold collective speedup. This is the
theoretical content behind the rho > 0 condition (Theorem 9-iii) and the "broadcast is useless to
tabular but essential to CF" dichotomy, and it predicts the n/m scaling seen in the E2/E4 sweeps.

### Proposition 17 (minimax tightness: Omega(d) probes are necessary). EXACT.

CLAIM: even GIVEN the exact basis, any learner needs k >= d probes to predict a new entity's reward on a
generic unseen cross-factor below the prior variance, in the worst case over x_*. PROOF: with k < d, the
system B x_* = y constrains x_* only on the k-dimensional row-space of B; the orthogonal (d-k)-dimensional
component x_*^perp is unobserved. For an unseen cross-factor b with a nonzero component along that
subspace, b^T x_* depends on x_*^perp, which the data does not constrain; under any rotationally-symmetric
prior on x_* the Bayes-optimal estimate sets that component to its mean and incurs squared error
E[(b^perp . x_*^perp)^2] = Omega(||b^perp||^2 Var(x_*)) > 0, i.e. no better than the prior. Hence k >= d
is NECESSARY. QED.

Combined with Theorem 10 (k >= d SUFFICIENT) and Theorem 1 (a structure-free learner needs Omega(n),
one probe per unseen target), the Theta(d) vs Theta(n) separation is MINIMAX-TIGHT on both sides: no
communication-free learner beats Theta(d) own probes, and no structure-free learner beats Theta(n).
