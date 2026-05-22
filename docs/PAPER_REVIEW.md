# Critical paper review (toward AAMAS / JAAMAS)

A self-review of the current draft (PAPER_DRAFT.md / paper_v2.html), written as a
skeptical reviewer, with a prioritized fix list. Goal: turn a research-complete body
of work into a submittable paper.

## Strengths (keep and lead with)
- A genuinely minimal-assumption setting (no prior knowledge, zero communication,
  partial/noisy observation, fully distributed) that is cleanly motivated.
- A CATEGORICAL result (acting on unseen pairs; onboarding; cold-start) with matching
  per-agent theory (Theta(d) vs Theta(n)), not a few-percent improvement.
- An operational ANYTIME separation that explains real, early reward, and exposes a
  failure mode of structure-free bandits under n>>T.
- A broad, FAIR bake-off (UCBIndep/UCBHomo/Tabular; MFSGD/ESTR/PTF/BPMF; SoftImpute/
  kNN-CF/BiasModel) under identical limits, with bootstrap CIs.
- Robustness: masking-model invariance (T4/E12), scaling (E2/E4/E6), generality
  (E14), assumption stress (approx low-rank + nonlinear link), and real-simulator
  validation (tabula_drone).
- Five theorems with proofs, each mapped to a confirming experiment.

## Weaknesses / gaps (a reviewer will flag these)
1. CITATIONS. Related work names ideas but has no real bibliography. AAMAS needs a
   proper, cited related-work (matrix completion; low-rank/linear bandits; federated/
   gossip CF; MRTA; decentralized MARL; exposure-debiased implicit feedback).
   PRIORITY P0.
2. FORMAL PROBLEM STATEMENT. The setting, observation model, and metrics are spread
   across sections. Add a single crisp problem-definition section with one notation
   block and formal definitions of skill, unseen-pair skill, and anytime skill.
   P0.
3. STATISTICAL RIGOR. Headline numbers use 8-12 seeds; a reviewer will want >=20 for
   the main claims, with the CI methodology stated once. P1.
4. FIGURE QUALITY. Figures are PNG (raster); a camera-ready needs vector PDFs with
   readable fonts and consistent styling. P1.
5. METRIC JUSTIFICATION. "Skill" is non-standard. Relate it explicitly to regret /
   normalized reward, and define the anytime metric precisely (cumulative normalized
   reward), so reviewers map it to familiar quantities. P1.
6. CONTENTION. We study non-contention only; a reviewer may ask whether the result
   survives when targets deplete (matching). State this as scope + future work, and
   ideally a short experiment (Wave (a)). P1.
7. THEOREM PRECISION. T3 is an order bound whose constant is loose at the default
   operating point; state the regime where it bites and lean on the mechanism. Make
   assumptions (incoherence for U-identification) explicit in T2. P1.
8. METHOD CONSOLIDATION. We present several variants (RewardCF, BothCF, ChoiceZK,
   HybridCF/conv, ActiveCFconv). The paper should foreground ONE recommended method
   (ActiveCFconv) and present the others as ablations, to avoid a "zoo". P1.
9. REPRODUCIBILITY STATEMENT + LIMITATIONS as explicit sections. P2.
10. ABLATION TABLE for our own method (precision weighting on/off; online vs batch;
    warm-start; active vs eps-greedy; d_hat sensitivity) in one place. P2.

## Anticipated reviewer objections and our answers (rebuttal file)
- "Is low-rank realistic?" -> approximate-low-rank + nonlinear-link stress test
  (graceful degradation) and real-simulator validation.
- "Is the broadcast really 'no communication'?" -> ZK_COMPLIANCE: passive sensing of
  public outcomes; masking = detection dropout, not transmission; no parameter
  sharing; oracle only normalizes.
- "Is persistent masking a cheat?" -> T4/E12: i.i.d. gives the same headline results;
  persistence only affects durability of decentralization.
- "Are the baselines fair / strong?" -> all run per-drone, broadcast-only, guessed
  rank; we include the strongest (PTF, SoftImpute) and beat them where it matters.
- "Does the method need the true rank?" -> E6: robust across guessed rank 2..20.
- "Why not just a bias/popularity model?" -> Theorem 5 + BiasModel: additive is rank
  <=2 and cannot personalize.

## Prioritized fix list (execution order)
P0 (blockers for submission):
  1. Add a cited related-work section (BibTeX) and inline citations.
  2. Add a formal problem-statement + notation + metric-definitions section.
  3. Build the AAMAS LaTeX skeleton (sigconf) and port the streamlined narrative.
P1 (strongly expected):
  4. 20-seed confirmation of the 3-4 headline panels; state CI methodology once.
  5. Vector-PDF figures with consistent styling.
  6. Foreground ActiveCFconv as THE method; others as ablations; one ablation table.
  7. Tighten theorem statements (assumptions, regime of T3); relate skill to regret.
  8. Add a short contention/matching experiment or a crisp scope+future-work note.
P2 (polish):
  9. Explicit reproducibility + limitations sections.
  10. Consolidated method-ablation table and hyperparameter appendix.

## Status of evidence already in hand (what the fixes can draw on)
- Real-env validation: tabula_bench (ours skill ~0.83 vs SGD-MF ~0.35, UCBIndep
  ~0.68, approaching oracle).
- Assumption stress: pilot_stress (approx low-rank + nonlinear link).
- All headline data saved per-seed in results/pilots/ (catalogued).
- Theory with proofs: THEORY_FORMAL.md (T1-T5 + alignment table).
