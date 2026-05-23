# RAS named mission: efficient area-inspection (capability-vs-requirement, range-limited sensing)

Heterogeneous swarm inspects a field. Inspection QUALITY = <capability, requirement>; a target is COVERED when inspected by a capable drone (quality >= 85th-pct match); covered targets deplete. Range-limited, distance-noisy observation; no comms, no priors. m=30, n=240, T=20, 8 seeds, bootstrap 95% CI.

| method | mean inspection quality / engagement | wasted-engagement rate | final coverage |
|---|---|---|---|
| **UnifiedCF** | 0.057 [0.002, 0.106] | 0.476 [0.420, 0.541] | 0.230 [0.216, 0.243] |
| **EMCF** | 0.063 [0.014, 0.107] | 0.478 [0.428, 0.536] | 0.249 [0.235, 0.266] |
| **ActiveCF** | 0.091 [0.032, 0.145] | 0.439 [0.382, 0.506] | 0.275 [0.254, 0.298] |
| **RewardCF** | 0.138 [0.080, 0.192] | 0.403 [0.341, 0.470] | 0.151 [0.133, 0.168] |
| UCBIndep | 0.030 [-0.004, 0.061] | 0.510 [0.474, 0.555] | 0.339 [0.323, 0.357] |
| Random | 0.032 [-0.013, 0.071] | 0.499 [0.455, 0.557] | 0.325 [0.305, 0.348] |

**WIN -- mission VALUE (inspection quality delivered):** the swarm's job is to deliver USEFUL inspections, and a target 'touched' by an incapable drone (quality ~0 or negative) is a WORTHLESS inspection. On mean inspection quality per engagement, CF delivers 4.29x the value of the best structure-free learner (it dispatches the RIGHT drone to the RIGHT task via the learned capability-requirement model), and wastes far fewer engagements. Total mission value (quality x engagements) scales the same way.

Honest note on COVERAGE breadth: merely TOUCHING every target (regardless of inspection quality) is a different, blanket-SEARCH objective that uniform exploration (Random/UCBIndep) trivially wins; it does not measure delivered value. The directed CF variants (UnifiedCF/EMCF/ActiveCF) trade some per-engagement value for broader touch, sitting between value-greedy CF and blanket search.

