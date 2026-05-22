# Experiment Backlog (not-yet-explored / in-progress ideas)

Living list. Priorities (P0 highest) re-ranked whenever a finding lands. Goal:
CLEAR, SIMPLE framing/method that beats all baselines and is groundbreaking;
complex extensions only if the simple option can't carry it.

How to read: [status] id. idea -- why/impact (depends-on).
status: TODO / IN-PROGRESS / DONE (moved to PROJECT_LOG) / PARKED.

## P0 -- do next (highest groundbreaking x probability)
- [IN-PROGRESS] B1. Extended faulty fraction 50->80%: is competence-weighted
  CHOICE pooling MAJORITY-robust? Per-teammate consistency is an absolute signal
  (not majority consensus) -> may tolerate arbitrary faulty fraction = RANSAC-level
  robustness WITHOUT RANSAC machinery. If holds -> simple+groundbreaking. (staged
  in pilot_trust.py faulty up to 0.8)
- [TODO] B3. Trust-weighted HYBRID fusion: each drone fuses own rewards + others'
  reward (reward-trust weighted) + others' choices (competence weighted) in ONE
  weighted ALS. Should DOMINATE across faulty%: rewards when teammates reliable,
  choices when not. The "one method that wins everywhere."
- [TODO] B6. BETTER METRICS (user): AUC / cumulative reward (anytime), normalized
  cumulative regret, anytime reward-vs-round curves; and targets-destroyed@K
  (needs a depletion task model). Highlights CF's transient advantage and the
  naive-collaboration-degrades-over-time effect. Quick win for AUC; depletion is
  a task-model change.
- [TODO] B5. Adversarial / Byzantine teammates (coordinated, inverted-preference,
  consistent-but-wrong) + THEORY (influence bound O(eps/(1-eps)); breakdown
  point; collaboration-safety lemma trust-weighted >= solo). Top-venue elevator.

## P1
- [TODO] B2. RANSAC / seed-grow CONSENSUS robust factorization (anchored on self
  or on choice-consensus; absolute residual threshold) -- the COMPLEX extension
  for >50% faulty, ONLY if B1 shows the simple method breaks past 50%.
- [TODO] B4. Latent-confidence EM/VB model: continuous beta_{k,t} (HKV
  preference/confidence split + posterior precision + consistency). Principled
  method that powers robustness; ablate which confidence source matters. (NOTE:
  model-agreement EM failed cycle 9; use BEHAVIOURAL inference.)
- [TODO] B7. Posterior-precision-driven confidence: Bayesian drones (BPMF /
  Thompson); confidence = own posterior precision; observers infer it. Ties
  exploration to uncertainty; elegant beta source.
- [TODO] B9. Detectability formalization: reliability is inferable from CHOICES
  (random = inconsistent) but NOT from in-range garbage REWARDS. "Decisions make
  trust inferable" -- a second reason decisions beat outcomes.
- [TODO] B10. Reward-side robustness with clearer-outlier faults so reward-trust
  is a fair test (current in-range garbage is hard to detect).

## P2
- [TODO] B8. Cold-start warm-start from broadcast-derived U + convergence
  dynamics (late-joining drones; consensus expands as agents converge).
- [TODO] B12. Connect to REAL ZK-MRTA env + real policies (BPMF, IQL-ZK,
  UCB-Indep, MF-CF) -- ecological validity; port the pilot characterization.
- [TODO] B11. Theory pack: collaboration-safety, influence/breakdown bounds,
  Theta(d) vs Theta(n) per-drone separation (correct-regime version).
- [TODO] B13. Hierarchical priors / borrowing strength (cold-start shrinkage to
  population prior).
- [TODO] B16. Re-run literature research agents (all 3 failed on API 529/ECONNRESET).

## P3 -- parked (revisit if a thread needs it)
- [TODO] B14. Information-directed sampling / active probing for the explore phase.
- [TODO] B15. ARD / automatic rank selection (robustness to unknown d).
- [TODO] B17. Two-stage noise theory (effect vs observation).
- [TODO] B18. Heavy-tailed (Student-t) likelihood for robustness.
- [TODO] B19. Non-stationary latent drift (fatigue/weather).
- [TODO] B20. Target churn / cold-start TARGETS.

## Priority log (re-rank events)
- 2026-05-22 init: P0 = B1, B3, B6, B5 (robustness spine + metrics). B2/B4 to P1
  (complex; only if simple breaks). Reward-observable characterization + decision
  parity + noise-immunity + collaboration-harm threshold already DONE (see
  PROJECT_LOG cycles 1-13).
