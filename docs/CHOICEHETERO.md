# Choice channel under heterogeneous teammate competence (+ sanity check)

Own reward + teammates' masked CHOICES only (no teammate rewards). A `frac` fraction of the 30-drone swarm are SPECIAL teammates: either RANDOM choosers (uninformative) or ORACLE choosers (true best-in-offer, maximally informative). We report the GOOD (method) drones. Does learned per-teammate informativeness (gamma_k) tell them apart and help? rho=1.0, 8 seeds, bootstrap 95% CI.

## Teammate kind = RANDOM

### Unseen-pair skill (good drones)

| method | special=0% | special=33% | special=50% |
|---|---|---|---|
| RewardCF | 0.443 [0.398, 0.481] | 0.489 [0.453, 0.519] | 0.530 [0.502, 0.560] |
| ChoiceCF | 0.089 [0.057, 0.124] | 0.083 [0.046, 0.127] | 0.015 [-0.016, 0.042] |
| **ChoiceEM** | 0.031 [0.000, 0.060] | 0.012 [-0.019, 0.041] | 0.008 [-0.022, 0.039] |
| **ChoiceEM-pred** | 0.066 [0.036, 0.099] | 0.040 [0.008, 0.076] | 0.033 [-0.007, 0.069] |

### Anytime earned skill (good drones)

| method | special=0% | special=33% | special=50% |
|---|---|---|---|
| RewardCF | 0.374 [0.356, 0.389] | 0.371 [0.351, 0.389] | 0.393 [0.373, 0.416] |
| ChoiceCF | 0.219 [0.196, 0.241] | 0.224 [0.206, 0.244] | 0.218 [0.195, 0.243] |
| **ChoiceEM** | 0.217 [0.199, 0.235] | 0.200 [0.182, 0.215] | 0.205 [0.191, 0.221] |
| **ChoiceEM-pred** | 0.235 [0.215, 0.256] | 0.215 [0.194, 0.234] | 0.223 [0.202, 0.245] |

### Learned gamma separation (good vs random teammates)

| method / teammate | special=0% | special=33% | special=50% |
|---|---|---|---|
| ChoiceEM : gamma(good) | n/a | 0.825 [0.762, 0.870] | 0.871 [0.822, 0.916] |
| ChoiceEM : gamma(random) | n/a | 0.671 [0.601, 0.728] | 0.719 [0.666, 0.770] |
| ChoiceEM-pred : gamma(good) | n/a | 0.290 [0.249, 0.328] | 0.275 [0.230, 0.318] |
| ChoiceEM-pred : gamma(random) | n/a | 0.099 [0.095, 0.103] | 0.100 [0.096, 0.104] |

## Teammate kind = ORACLE

### Unseen-pair skill (good drones)

| method | special=0% | special=33% | special=50% |
|---|---|---|---|
| RewardCF | 0.443 [0.399, 0.481] | 0.601 [0.576, 0.628] | 0.617 [0.588, 0.647] |
| ChoiceCF | 0.089 [0.057, 0.124] | 0.460 [0.410, 0.513] | 0.547 [0.503, 0.593] |
| **ChoiceEM** | 0.031 [0.001, 0.060] | 0.332 [0.295, 0.372] | 0.417 [0.373, 0.465] |
| **ChoiceEM-pred** | 0.066 [0.036, 0.099] | 0.410 [0.374, 0.450] | 0.487 [0.432, 0.547] |

### Anytime earned skill (good drones)

| method | special=0% | special=33% | special=50% |
|---|---|---|---|
| RewardCF | 0.374 [0.355, 0.389] | 0.440 [0.422, 0.457] | 0.470 [0.446, 0.496] |
| ChoiceCF | 0.219 [0.196, 0.241] | 0.287 [0.272, 0.302] | 0.311 [0.283, 0.340] |
| **ChoiceEM** | 0.217 [0.199, 0.235] | 0.281 [0.252, 0.306] | 0.317 [0.290, 0.345] |
| **ChoiceEM-pred** | 0.235 [0.214, 0.256] | 0.289 [0.261, 0.313] | 0.325 [0.299, 0.353] |

### Learned gamma separation (good vs oracle teammates)

| method / teammate | special=0% | special=33% | special=50% |
|---|---|---|---|
| ChoiceEM : gamma(good) | n/a | 0.887 [0.854, 0.921] | 0.926 [0.896, 0.956] |
| ChoiceEM : gamma(oracle) | n/a | 0.924 [0.895, 0.949] | 0.957 [0.936, 0.978] |
| ChoiceEM-pred : gamma(good) | n/a | 0.350 [0.309, 0.393] | 0.380 [0.330, 0.433] |
| ChoiceEM-pred : gamma(oracle) | n/a | 0.406 [0.375, 0.435] | 0.493 [0.445, 0.546] |

Sanity: a working estimator must drive gamma(oracle) HIGH and gamma(random) LOW. Win condition: ChoiceEM-pred separates them more sharply than in-sample ChoiceEM and keeps the good drones' skill robust as RANDOM teammates grow, while leveraging ORACLE teammates.

