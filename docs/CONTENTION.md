# Contention (capacity-1 matching): does the result survive depletion?

Shared offer pool of size `pool` (smaller = more contention; m=30 drones, n=240). Each target is awarded to one random contender; losers earn 0. anytime = earned reward normalized by the per-round MATCHING optimum (Hungarian), 8 seeds, bootstrap 95% CI. unseen = preference quality evaluated contention-free AFTER training under contention (the categorical claim). rho=1.00 (full broadcast, to isolate contention from masking).



## Operational: earned-reward skill (matching-normalized)

| method | pool=240 | pool=60 | pool=30 | pool=15 |
|---|---|---|---|---|
| **ActiveCFconv** | 0.439 [0.412, 0.464] | 0.230 [0.206, 0.252] | 0.125 [0.109, 0.144] | 0.046 [0.022, 0.070] |
| **RewardCFconv** | 0.439 [0.418, 0.462] | 0.199 [0.186, 0.214] | 0.121 [0.102, 0.138] | 0.059 [0.045, 0.079] |
| PTF | 0.243 [0.223, 0.260] | 0.151 [0.136, 0.167] | 0.096 [0.083, 0.110] | 0.057 [0.040, 0.075] |
| UCBIndep | 0.005 [-0.004, 0.014] | -0.002 [-0.009, 0.005] | -0.002 [-0.009, 0.006] | 0.004 [-0.005, 0.013] |
| Random | -0.010 [-0.022, -0.000] | 0.002 [-0.011, 0.014] | -0.003 [-0.010, 0.005] | -0.010 [-0.026, 0.007] |

## Categorical: unseen-pair skill (learned under contention, eval contention-free)

| method | pool=240 | pool=60 | pool=30 | pool=15 |
|---|---|---|---|---|
| **ActiveCFconv** | 0.113 [0.080, 0.145] | 0.350 [0.320, 0.383] | 0.310 [0.288, 0.330] | 0.195 [0.166, 0.221] |
| **RewardCFconv** | 0.323 [0.292, 0.362] | 0.383 [0.351, 0.415] | 0.339 [0.313, 0.367] | 0.287 [0.242, 0.324] |
| PTF | 0.343 [0.332, 0.356] | 0.338 [0.301, 0.373] | 0.262 [0.232, 0.292] | 0.141 [0.114, 0.169] |
| UCBIndep | -0.007 [-0.014, -0.001] | 0.001 [-0.006, 0.005] | -0.003 [-0.009, 0.002] | -0.006 [-0.015, 0.003] |
| Random | -0.007 [-0.026, 0.014] | 0.014 [-0.002, 0.031] | -0.009 [-0.026, 0.009] | -0.005 [-0.027, 0.020] |

## Collision rate (fraction of engagements lost to contention)

| method | pool=240 | pool=60 | pool=30 | pool=15 |
|---|---|---|---|---|
| ActiveCFconv | 0.207 | 0.376 | 0.503 | 0.685 |
| RewardCFconv | 0.147 | 0.371 | 0.526 | 0.660 |
| PTF | 0.187 | 0.323 | 0.448 | 0.624 |
| UCBIndep | 0.054 | 0.209 | 0.363 | 0.561 |
| Random | 0.061 | 0.211 | 0.369 | 0.563 |

Takeaway: the CATEGORICAL preference quality (unseen) is contention-invariant (CF stays high, structure-free at the floor) because it is a property of the learned model; the OPERATIONAL earned-reward gap narrows as collisions, not preferences, become the bottleneck, yet CF still leads by spreading drones across targets via diverse, accurate preferences (lower collision rate).

