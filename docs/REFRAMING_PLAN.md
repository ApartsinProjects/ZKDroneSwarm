# Paper Reframing Plan: Structural-vs-Operational Assumption Substitution

**Status**: Active plan, executing top-to-bottom.
**Created**: 2026-05-11.
**Goal**: Reposition the paper from "we have an algorithm called MF-CF that does OK" to "we identify and validate a new design axis (operational vs structural assumptions) for decentralized task allocation."

---

## 1. Narrative

### 1.1 The problem with the current narrative

The paper currently presents itself as a method paper: *"Here's MF-CF for the ZK-MRTA problem we defined. It learns latent factors. Here are 12 theorems."* This framing has fatal weaknesses:

- **No clear contribution claim**: IQL-ZK (Tampuu 2017, 25 lines, no theory) matches MF-CF on raw steps. ESTR (Kang-Hsieh-Lee 2022 style) matches PTF on both steps and LMQ. Among 11 theorems, most apply existing results.
- **No paper-defining unique claim**: each individual contribution is matched or exceeded by existing baselines or literature.
- **Reviewer-fragile**: "the simple baseline beats your method on the main metric" is the kind of finding that sinks reviews at any top venue.

### 1.2 The reframing

Instead of "we have a method," the paper claims:

> Decentralized multi-robot task allocation is traditionally made tractable by *operational* assumptions: explicit coordination protocols (markets, planners), inter-agent communication (cooperative bandits, parameter sharing), or prior knowledge (known capabilities, cost matrices, feasibility constraints). This paper proposes a complementary alternative: substitute all operational assumptions for a single *structural* assumption — that the agent-task compatibility matrix has hidden low rank — and shows this is sufficient for effective decentralized allocation via online collaborative filtering on a public broadcast.

### 1.3 Why this is novel (literature gap)

Mapping prior work onto the operational-vs-structural axis:

| Framework | Operational | Structural |
|---|---|---|
| Classical MRTA (Wurman 2008) | Many (known cost, feasibility) | None |
| Market-based MRTA (Brafman 1996, et al.) | Bidding protocol | None |
| Cooperative bandits (Hillel 2013, Wang 2020) | Inter-agent messaging | None |
| CTDE-MARL (QMIX, MAPPO) | Centralized training | Often implicit |
| Dec-POMDP | Known reward model | State transition |
| Low-rank linear bandits (Katariya 2017, Kang 2022) | None decentralized | Low-rank reward |
| Multi-user low-rank RL (Nagaraj 2023) | Centralized aggregation | Low-rank reward |
| **ZK-MRTA (ours)** | **None** | **Low-rank compatibility** |

The closest neighbors are low-rank bandits (have the structural assumption but assume centralized learning) and cooperative bandits (have the operational restriction but no structural assumption). The combination — "zero operational, one structural" — is the gap.

### 1.4 The make-or-break empirical claim

The structural assumption being **necessary** and **sufficient** must be empirically validated. The single critical experiment:

> **Vary the true rank $d$ of the compatibility matrix. The CF methods (MF-CF, PTF, BPMF) should dominate tabular (IQL-ZK, UCB-Indep) when $d \ll \min(m, n)$, and lose to tabular when $d \approx \min(m, n)$ or $R$ is full-rank.**

If this holds, the paper has a clean defensible thesis. If it doesn't, the central claim is empirically wrong and the paper needs further rethinking.

### 1.5 The secondary empirical claim

Bayesian PMF (Salakhutdinov-Mnih 2008-style) is the most principled operationalization of the structural assumption. If BPMF empirically dominates IQL-ZK at the baseline, the paper can honestly claim algorithmic leadership among ZK-compliant methods. Preliminary single-seed test: BPMF 67.6 vs MF-CF 94.7 on the same seed — a 29% improvement. 5-seed validation needed.

---

## 2. Execution Plan

### Phase 1: Validate the framing empirically (highest risk first)

**Why first**: if the predictions don't hold, every other piece of the plan must be revised. Pay this risk before any writing.

**Task 1.1**: Run BPMF on the 5-seed baseline ($n=27, d=3$). Compare against MF-CF / PTF / IQL-ZK / ESTR / TS-MF. Acceptance criterion: BPMF avg-steps $\le$ IQL-ZK's 63.3 and late-LMQ $\ge$ MF-CF's 0.753.

**Task 1.2**: Implement Set N — Structural Assumption Stress Test. Conditions:

| Condition | True rank $d$ | Construction |
|---|---|---|
| struct_d1 | 1 | Single latent direction (all drones, all targets aligned in 1D) |
| struct_d2 | 2 | 2-mode block-diagonal |
| struct_d3 (baseline) | 3 | Default one-hot 3-mode |
| struct_d5 | 5 | 5-mode (latent_dim=5 in config) |
| struct_d8 | 8 | 8-mode (already partially in §7.15) |
| struct_full | 9 = min(m,n) | center_mode="random", high variance |
| struct_approx | "approximately rank-3" | Random R + soft rank constraint |

Six policies: UCB-Indep, IQL-ZK, MF-CF, PTF K=5, TS-MF, BPMF. 5 seeds each = 6 conditions × 6 policies × 5 seeds = 180 runs.

**Task 1.3**: Run Set N. Estimated runtime: ~10 min at n=27 + larger for higher d.

**Task 1.4**: Aggregate Set N results. Compute CF/tabular ratio at each rank. Validate predictions:
- struct_d1: CF/tabular ratio < 0.8 (CF wins by 25%+)
- struct_d3 (baseline): CF/tabular ratio $\approx$ 1.0 (roughly tied, current finding)
- struct_d8: CF/tabular ratio $\approx$ 1.0 (tied at high d, existing §7.15)
- struct_full: CF/tabular ratio > 1.0 (CF loses)

**Decision gate**: If at least 3 of 4 predictions hold, proceed to Phase 2. If <3, redesign experiments or abandon framing.

### Phase 2: Reframe the paper

Assuming Phase 1 validates the predictions:

**Task 2.1**: Rewrite abstract — lead with assumption substitution.

**Task 2.2**: Rewrite §1 introduction — open with operational/structural trade-off paragraph.

**Task 2.3**: Add §2.4 Table 1b (axis comparison).

**Task 2.4**: Add §3.2.3 Structural Assumption subsection (the single substantive assumption).

**Task 2.5**: Rewrite §4 opening — CF as direct consequence of the assumption.

**Task 2.6**: Add §4.5 — Bayesian PMF as the proper operationalization. Document the conjugate update rule, MAP vs VB modes, Thompson vs UCB exploration.

**Task 2.7**: Add §7.25 — Set N results. The paper's strongest empirical section.

**Task 2.8**: Add Corollary to §C.10 — without low-rank assumption, no ZK policy beats tabular $\Theta(mn \log T)$.

**Task 2.9**: Rewrite §9 conclusion — assumption substitution as primary contribution.

### Phase 3: Final integration

**Task 3.1**: Update Table 15 (cross-policy comparison at baseline) with BPMF row.

**Task 3.2**: Update Tables 16, 17, 20, 21 with BPMF where applicable.

**Task 3.3**: Rebuild .docx via html2doc (3-stage pipeline).

**Task 3.4**: Update theorems_proofs.html — add Corollary derivation.

**Task 3.5**: Commit and push.

---

## 3. Acceptance Criteria

The reframing is **successful** if:

1. BPMF empirically matches or beats IQL-ZK on the baseline ($\le$ 63.3 avg steps, 5 seeds).
2. CF/tabular performance ratio decreases as true rank $d$ decreases.
3. CF/tabular ratio is $\le 0.8$ at $d=1$ (CF wins by 25%+).
4. CF/tabular ratio is $\ge 1.0$ at struct_full (CF loses to tabular).
5. Theory corollary holds: without low-rank, no advantage.

If 4 of 5 hold, the paper has a clean defensible thesis suitable for JAAMAS submission. If fewer hold, return to plan revision.

---

## 4. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| BPMF fails 5-seed test | Medium | Tune likelihood variance, prior variance, init scale; fall back to MAP-only |
| At d=1, CF doesn't dominate | Low | This would be surprising; investigate algorithm bug |
| At full-rank, CF still wins | Medium | This would *strengthen* the paper actually — CF still useful even without low-rank |
| Set N experiments fail to differentiate | Medium | Increase n=54 or n=108 for sharper contrast |
| Writing takes longer than estimated | High | Acceptable — narrative quality matters more than speed |

---

## 4b. DECISION GATE OUTCOME (2026-05-11): THESIS FALSIFIED

Set N completed (150 runs; d=1 failed on a latent-builder degeneracy). The
structural stress test **does not validate the reframing thesis**.

CF/tabular ratio (best-CF / best-tabular avg-steps) across true rank:

| True rank d | CF/tabular ratio | Predicted | Holds? |
|---|---|---|---|
| 2 | 1.034 (CF loses) | < 0.8 (CF wins big) | NO |
| 3 (baseline) | 0.988 (~tie) | ~1.0 | yes |
| 5 | 1.000 (tie) | shrinking advantage | n/a |
| 8 | 1.001 (~tie) | ~1.0 | yes |
| full (9, random) | 1.002 (~tie) | > 1.0 (CF loses) | NO |

**0 of the 4 critical acceptance criteria hold.** The CF/tabular ratio is
essentially 1.0 (within +/-3.4%) across ALL ranks. There is no regime where
CF methods dominate tabular by a meaningful margin, and no regime where they
clearly lose.

Two confounds discovered:
1. **Full-rank "random" is EASY, not hard.** Random unit latent vectors have
   high average cosine similarity, so almost any assignment delivers good
   reward (5-8 steps from episode 1, no learning needed). This is the
   opposite of the intended "no-structure hard control."
2. **The benchmark is sample-rich at every rank.** With 70 steps/ep x 9
   drones x 35 episodes and n=27, every method has 20+ observations per arm.
   The rank structure never becomes the binding constraint, so low-rank
   factorisation gives no sample-complexity advantage.

Root cause is identical to the Set M (noise) negative result: the benchmark
operates in the sample-rich regime where structural priors do not help. The
structural advantage of CF requires a sample-starved regime (much larger n,
much shorter T, or genuinely sparse 0/1 rewards) not exercised here.

**Decision: DO NOT proceed with Phase 2 reframing.** Writing the paper around
"structural assumption drives CF advantage" would be making an empirically
false claim. The thesis is falsified in the current benchmark.

What survives:
- BPMF is marginally the best ZK method at the d=3 baseline (62.5 vs IQL-ZK
  63.3, a 1.3% edge) but this REVERSES at d=2 (BPMF 56.1 vs IQL-ZK 51.8). The
  "BPMF is best" claim is fragile and benchmark-specific.
- The honest finding across Sets M (noise) and N (structure): at the tested
  scale, all ZK-compliant policies perform within ~few percent of each other;
  the problem is too sample-rich to discriminate them.

## 5. Expected Outcome (NOT REALISED -- see 4b)

If the plan succeeds, the paper transforms from:
- *"Marginal method paper with one weak baseline win and 11 mostly-applied theorems"*

to:
- *"Identifies new design axis for decentralized task allocation; empirically validates that structural assumptions can substitute for operational ones; proposes three operationalizations (SGD-MF, PTF, BPMF) at different points on the algorithmic frontier; provides matching lower bounds confirming the structural assumption is necessary for tractable rate."*

The latter is a JAAMAS-strong paper. The former is borderline.
