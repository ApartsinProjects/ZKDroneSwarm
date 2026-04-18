# 7. Results

This section reports the results of the evaluation described in Section 6. It presents task-completion outcomes, cross-policy comparisons, learning dynamics, latent-structure recovery, coordination effects, and the convergence assessment for the decentralized matrix-factorization policy.

## 7.1 Task Completion

All three policies — random, oracle, and matrix-factorization — successfully neutralized all 27 targets in their respective evaluation episodes. The task-completion criterion is therefore not a differentiating signal in this experiment: 100% success was achieved by every policy at every episode across the 35-episode training horizon.

This result is consistent with the scenario design. With 9 drones engaging 27 targets over a maximum of 250 steps, even a uniformly random policy accumulates sufficient damage to neutralize all targets before the step budget is exhausted. The informative comparison across policies lies entirely in efficiency: how quickly and with how much total effort was full task completion achieved. All subsequent analysis therefore focuses on efficiency and coordination quality metrics rather than completion rate.

## 7.2 Cross-Policy Comparison

The table below reports each metric for the random baseline, the oracle benchmark, and the matrix-factorization policy at its final episode (episode 35), which is also the best episode in this run. The baseline and oracle are each evaluated on a single episode; the matrix-factorization result reflects accumulated learning over 35 training episodes.

| Metric | Random | MF (ep. 35) | Oracle | MF vs. Random | MF vs. Oracle |
|---|---|---|---|---|---|
| Steps ↓ | 126 | **69** | 62 | −45.2% | +11.3% |
| Total ammo ↓ | 1,134 | **621** | 558 | −45.2% | +11.3% |
| Shots per target ↓ | 42.0 | **23.0** | 20.7 | −45.2% | +11.3% |
| Avg. match quality ↑ | 0.313 | **0.568** | 0.657 | +81.7% | −13.6% |
| Total latent mismatch (HP) ↓ | 619.8 | **235.8** | 143.6 | −62.0% | +64.3% |
| Total overkill (HP) ↓ | 7.0 | 10.7 | **3.65** | +53.5% | +194.1% |
| Total collisions | 225 | 295 | 382 | +31.1% | −22.8% |
| Total net damage (HP) | 270.0 | 270.0 | 270.0 | 0% | 0% |
| Targets neutralized ↑ | 27 | 27 | 27 | 0% | 0% |

*Arrows indicate preferred direction (↑ higher is better, ↓ lower is better). In the comparison columns, positive values indicate that MF is numerically higher than the reference policy on that metric.*

The matrix-factorization policy closes approximately 45% of the step and ammo gap between random and oracle. Starting from a policy that required 126 steps and 1,134 shots to complete the scenario, learning reduces these figures to 69 steps and 621 shots — within 11% of the oracle on both efficiency metrics. Average match quality improves from 0.313 (random) to 0.568 (MF), closing 60% of the gap to the oracle value of 0.657. Total latent mismatch is reduced by 62% relative to random, from 619.8 HP to 235.8 HP, though a substantial residual gap to the oracle (143.6 HP) remains.

Net damage is identical across all three policies, as all episodes result in full target neutralization. This confirms that the relevant variation is not in whether damage is applied, but in how efficiently it is allocated.

Two metrics move against the direction of improvement. Total overkill increases from 4.6 HP in episode 1 to 10.7 HP in episode 35, and is substantially higher under MF (10.7 HP) than under random (7.0 HP) or oracle (3.65 HP). Total collisions under MF (295) exceed the random baseline (225). Both findings are analyzed in Section 7.6.

## 7.3 Episode Engagement Profiles

A complementary view of policy behavior is provided by per-episode engagement profiles, which plot the fraction of total HP remaining (blue) and the fraction of active targets remaining (orange) over each timestep of the best episode for each policy. These curves encode the engagement *strategy* of each policy as a geometric relationship: the gap between the HP curve and the Active Targets curve at any point in the episode measures how much cumulative damage has been distributed across still-active targets without yet eliminating them.

**Figure 1. Engagement profiles by policy — Total HP (blue) and Active Targets (orange) as a percentage of initial values, plotted over episode timesteps.**

| MF Policy (ep. 35, 69 steps) | Random Baseline (126 steps) | Oracle Benchmark (62 steps) |
|:---:|:---:|:---:|
| ![MF engagement profile](figures/fig-engagement-profile-mf.png) | ![Random engagement profile](figures/fig-engagement-profile-random.png) | ![Oracle engagement profile](figures/fig-engagement-profile-oracle.png) |

---

**Random baseline (126 steps).** The random profile exhibits the widest and most persistent gap between the two curves. HP declines approximately linearly while the Active Targets curve remains near 100% until approximately step 40, when the first eliminations begin. The two curves converge only in the final 20–30 steps. This pattern is the structural signature of **distributed spray damage**: shots are spread across all targets without concentrating enough fire to eliminate any single target promptly. The large enclosed area between the two curves corresponds directly to the high total latent mismatch (619.8 HP) reported in Section 7.2.

**Matrix-factorization policy (69 steps).** The MF profile shows a substantially reduced HP-Active Targets gap. Target eliminations begin noticeably earlier — within approximately the first seven to eight steps — and the Active Targets curve descends in a more regular staircase pattern throughout the episode rather than clustering at the end. The two curves track each other more closely than in the random case, and the episode concludes in 69 steps. The gap is not eliminated, however: the HP curve consistently leads the Active Targets curve, indicating that damage is still partially distributed across multiple targets before each elimination occurs. This reflects the learned but imperfect focus-fire behavior of the policy: having inferred the dominant latent compatibility structure, it concentrates fire on high-affinity targets, but without access to remaining HP values it cannot schedule shots to eliminate targets sequentially in the way an HP-aware planner would.

**Oracle benchmark (62 steps).** The oracle profile exhibits the tightest coupling between the two curves. Target eliminations begin from the very first steps, and the Active Targets curve descends in a consistent staircase that closely parallels the HP decline throughout, terminating together at step 62. The minimal enclosed area reflects the oracle's HP-aware marginal scheduling: it assigns drones to targets near elimination, producing a nearly sequential target-elimination pattern and the lowest total latent mismatch (143.6 HP) and overkill (3.65 HP) of the three policies.

**Structural interpretation.** The progression from random to MF to oracle in these profiles illustrates the fundamental trade-off at the core of the ZK-MRTA problem. Under random assignment, all damage potential is wasted on spread: HP drains uniformly but no target is eliminated until very late. Under oracle assignment, damage potential is maximally converted into eliminations through privileged HP awareness. The MF policy occupies an intermediate position determined by what can be inferred from latent compatibility structure alone, without access to HP state. The shrinking of the HP-Active Targets gap across the three policies is a direct qualitative expression of the same progression captured quantitatively by the latent mismatch and match quality metrics in Section 7.2.

## 7.4 Learning Dynamics Across 35 Episodes

The matrix-factorization policy's performance trajectory across 35 episodes is characterized by three broadly distinguishable phases: rapid early convergence, a mid-training plateau with crowding, and a slow late refinement.

**Phase 1 — Rapid Convergence (episodes 1–9).** The largest performance gains occur in the first nine episodes. Episode duration drops from 184 steps (episode 1) to 108 steps (episode 9), a reduction of 41%. Average match quality doubles from 0.207 to 0.411 over the same window. Total latent mismatch falls from 1,063.6 HP to 433.1 HP, a 59% reduction. These gains reflect the rapid accumulation of coverage in the integration matrix and the early recovery of gross compatibility structure. At episode 1, agents have explored 94.2% of drone-target pairs; by episode 18, coverage reaches 100%, meaning the running-mean estimates are fully populated.

**Phase 2 — Mid-Training Plateau with Crowding (episodes 9–21).** Beginning around episode 9, the rate of improvement in step count and match quality slows substantially. Steps stabilize near 75 per episode between episodes 17 and 21; match quality oscillates in the range 0.53–0.57. The learning state data reveals that this period coincides with a target-crowding phenomenon: by episode 18, the number of unique most-preferred targets across the 9 agents has collapsed from 9 (at episode 1) to 4. Multiple agents have converged on the same small set of best-predicted targets. This concentration leads to high contention — total collisions peak at 629–630 in episodes 10–11 — and temporarily limits further efficiency gains, since many agents are competing for the same targets rather than distributing across the task set.

This crowding pattern is a structural consequence of the learning mechanism. As the integration matrix fills and predictions improve, agents converge on a shared estimate of which targets yield the highest utility for each drone type. Because the policy is decentralized and agents cannot communicate to divide targets, they independently arrive at similar greedy choices, producing emergent contention.

**Phase 3 — Slow Late Refinement (episodes 21–35).** After episode 21, efficiency metrics resume a slow but sustained improvement. Steps decline from 75 to 69 over the final 14 episodes; shots per target fall from 25 to 23. Match quality reaches its training peak of 0.590 at episode 31 before settling at 0.568 in the final episode. Crucially, the number of unique preferred targets recovers to 8 by episode 35, indicating that as epsilon continues to decay and the policy's internal model becomes more accurate, agents begin to differentiate their preferences more clearly — consistent with the latent structure being gradually resolved from a previously crowded shared representation.

## 7.5 Latent Structure Recovery

The core question motivating this evaluation is whether decentralized matrix factorization can recover the hidden compatibility structure of the environment from interaction outcomes alone. Two direct indicators are available: average match quality and the internal agreement gap.

**Match quality progression.** Average match quality rises from 0.207 at episode 1 to a training-high of 0.590 at episode 31, settling at 0.568 in episode 35. This represents a 174% increase over the training horizon, indicating substantial latent structure recovery. The policy begins near the expected value for a uniformly random assignment policy and progressively approaches the oracle's 0.657. By the final episode, approximately 60% of the structural gap between uninformed random assignment and privileged oracle allocation has been closed.

**Integration matrix agreement gap.** The agreement gap, defined as the mean absolute discrepancy between the policy's predicted utilities and the entries of the integration matrix, declines monotonically across the three checkpoints: 0.365 at episode 1, 0.232 at episode 18, and 0.106 at episode 35. This threefold reduction confirms that the policy's internal model is progressively converging toward the accumulated empirical interaction estimates. The parallel rise in mean top-1 predicted reward — from 0.0006 at episode 1 to 0.731 at episode 18 and 0.922 at episode 35 — further supports this interpretation: the policy's predicted utility for its most-preferred target has converged to a high-confidence value, indicating that embedding representations have stabilized around the dominant compatibility signals in the interaction history.

**Geometric structure in the learned embeddings.** The numerical indicators above establish that the policy's model converges toward the true interaction structure. Figure 2 provides direct geometric evidence of this recovery. It shows a t-SNE projection of the P (drone) and U (target) embedding vectors from drone 1's local model at the end of training (episode 35), with points colored by ground-truth latent mode.

![Figure 2: t-SNE projection of drone 1's learned P and U embeddings at episode 35, colored by ground-truth latent mode.](figures/fig-tsne-drone1-embeddings.png)

*Figure 2. t-SNE projection of drone 1's learned P and U embedding matrices at episode 35. Each point represents either a drone (P-row) or a target (U-column) in the 3-dimensional factorization space. Colors correspond to ground-truth latent mode assignments (red, green, blue), which are unknown to the policy and applied here post-hoc for interpretability.*

Three spatially separated clusters have formed, each aligned with one of the three ground-truth latent modes. Red-mode entities are grouped in one region, green-mode entities in another, and blue-mode entities occupy a distinct third region. This separation arises entirely from learned interaction outcomes: the policy was never given mode labels, latent vectors, or any explicit structural signal. The embedding geometry reflects what was inferrable from the public reward signal and swarm interaction history alone.

This is a direct and interpretable manifestation of the structure recovery quantified above. The agreement gap measures how close the model's predictions are to empirical interaction estimates; this visualization shows that the underlying embedding geometry has organized itself to be consistent with the hidden latent structure of the environment. The two measures are complementary: the agreement gap is a scalar summary, and the t-SNE projection is a geometric witness to the same underlying recovery process.

Taken together, these indicators confirm that the decentralized matrix-factorization policy does recover meaningful latent structure from ZK-constrained interaction outcomes. The recovery is substantial, progressive, and consistent across independent convergence indicators, though it remains incomplete by the end of the 35-episode training run.

## 7.6 Coordination Dynamics

**Collision trajectory.** Total collisions per episode follow a non-monotonic trajectory that closely mirrors the learning curve. In episode 1, 318 collisions occur — a moderate baseline consistent with partially random target selection. Collisions rise sharply through episodes 6–11, peaking at 630 in episode 11, as the policy rapidly improves match quality but has not yet differentiated individual drone preferences. Following the crowding peak, collisions gradually decline, reaching 295 in the final episode. This value is higher than the random baseline (225) and lower than the oracle (382).

The fact that the oracle produces more collisions (382) than the random baseline (225) is notable. It reflects the oracle's deliberate multi-drone focus-fire strategy: the oracle explicitly assigns multiple drones to the same high-value target when doing so minimizes total steps. MF collisions (295) are between these two values, consistent with partial focus-fire behavior that emerges from learned latent preference without explicit coordination logic.

**Overkill dynamics.** Total overkill per episode increases over training, from 4.6 HP in episode 1 to a range of 8–13 HP across the plateau and refinement phases, settling at 10.7 HP in episode 35. This pattern is the inverse of what might be expected if overkill were a simple inefficiency indicator. In fact, overkill is a structural side-effect of focus-fire convergence: as the policy learns to route multiple agents to high-compatibility targets, shots continue landing on targets whose HP has already been reduced to zero within the same timestep. Since the policy has no access to remaining HP values (a core ZK constraint), it cannot schedule its fire to avoid overkill in the way the oracle does.

The oracle's overkill is exceptionally low (3.65 HP) precisely because it incorporates remaining HP into its marginal-allocation scoring. The MF policy's inability to do so is not a failure of learning but a fundamental consequence of the ZK constraint. The increase in overkill over training is therefore best interpreted as an indirect indicator of improving match quality: as more agents converge on compatible targets, coordinated focus-fire increases, and unavoidable overkill rises with it.

## 7.7 Convergence Assessment

The training run is assessed as **potentially undertrained**. The best episode across all 35 training episodes is episode 35, the final episode of the run. Neither efficiency nor match quality shows signs of stabilization at the end of training: episode duration continued to decline across the final phase, while match quality remained near its peak but did not settle to a stable plateau (0.590 at episode 31 and 0.568 at episode 35). The exploration rate at the end of training ($\varepsilon = 0.052$) remains substantially above the specified minimum of $\varepsilon_{\min} = 0.02$, indicating that the policy had not yet entered its fully exploitative regime.

The average performance over the full 35-episode training run ($\bar{T} = 99.6$ steps with standard deviation 39.4 across episodes, $\bar{A} = 896.7$ shots with standard deviation 354.8) is considerably below the best-episode results (69 steps, 621 shots), reflecting the large variance during the early rapid-convergence phase. Cross-policy comparisons using the best episode therefore represent the ceiling of what the policy achieved under the given training budget, not its steady-state behavior.

These observations suggest that extending the training horizon would likely yield further efficiency gains, particularly in match quality and latent mismatch. The policy has not saturated, and the structural dynamics — declining agreement gap, recovering preference diversity — indicate that learning is ongoing. Convergence behavior under extended training is identified as a primary question for future investigation.
