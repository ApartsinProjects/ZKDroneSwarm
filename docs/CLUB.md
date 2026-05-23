# CLUB (clustering of bandits) vs continuous low-rank, under masking

Does DISCRETE agent-clustering match CONTINUOUS low-rank factorization for unseen-pair generalization? Same masked harness, guessed d_hat=8, 8 seeds, bootstrap 95% CI.

## Unseen-pair skill (the categorical claim)

| method | rho=1.00 | rho=0.50 | rho=0.25 |
|---|---|---|---|
| **RewardCF** | 0.388 [0.361, 0.414] | 0.410 [0.385, 0.434] | 0.337 [0.308, 0.377] |
| CLUB | 0.440 [0.404, 0.480] | 0.359 [0.314, 0.407] | 0.257 [0.234, 0.283] |
| KNNCF | 0.482 [0.447, 0.523] | 0.385 [0.358, 0.416] | 0.308 [0.274, 0.346] |
| Tabular | 0.001 [-0.005, 0.007] | 0.002 [-0.005, 0.009] | 0.005 [-0.000, 0.011] |
| UCBIndep | 0.002 [-0.002, 0.006] | 0.003 [-0.004, 0.009] | 0.005 [-0.002, 0.011] |

## Overall skill

| method | rho=1.00 | rho=0.50 | rho=0.25 |
|---|---|---|---|
| **RewardCF** | 0.653 [0.637, 0.671] | 0.651 [0.635, 0.665] | 0.607 [0.591, 0.624] |
| CLUB | 0.552 [0.515, 0.600] | 0.537 [0.500, 0.577] | 0.521 [0.507, 0.535] |
| KNNCF | 0.564 [0.527, 0.605] | 0.466 [0.430, 0.503] | 0.401 [0.363, 0.440] |
| Tabular | 0.430 [0.415, 0.445] | 0.429 [0.404, 0.451] | 0.435 [0.417, 0.452] |
| UCBIndep | 0.601 [0.569, 0.626] | 0.593 [0.565, 0.614] | 0.594 [0.571, 0.614] |

Read: if CLUB (hard clusters) trails RewardCF (continuous low-rank) on unseen, the personalization lives in CONTINUOUS factor directions that discrete grouping coarsens away; if it matches KNNCF, hard-vs-soft grouping is a wash. Both clustering methods should still beat the structure-free floor (Tabular/UCBIndep ~ 0 on unseen).

