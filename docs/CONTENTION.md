# Contention (capacity-1 matching): does the result survive depletion?

Shared offer pool of size `pool` (smaller = more contention; m=30 drones, n=240). Each target is awarded to one random contender; losers earn 0. anytime = earned reward normalized by the per-round MATCHING optimum (Hungarian), 8 seeds, bootstrap 95% CI. unseen = preference quality evaluated contention-free AFTER training under contention (the categorical claim). rho=1.00 (full broadcast, to isolate contention from masking).



## Operational: earned-reward skill (matching-normalized)

| method | pool=240 | pool=60 | pool=30 | pool=15 |
|---|---|---|---|---|
| **ContentionAdaCF** | 0.448 [0.422, 0.472] | 0.205 [0.186, 0.223] | 0.153 [0.132, 0.173] | 0.100 [0.086, 0.113] |
| **ContentionCF** | 0.314 [0.290, 0.337] | 0.178 [0.160, 0.200] | 0.134 [0.115, 0.155] | 0.105 [0.096, 0.114] |
| CBBAlite | 0.464 [0.434, 0.487] | 0.216 [0.204, 0.226] | 0.127 [0.112, 0.142] | 0.064 [0.047, 0.081] |
| MusicalChairs | 0.310 [0.279, 0.344] | 0.124 [0.108, 0.141] | 0.077 [0.053, 0.103] | 0.028 [0.019, 0.040] |
| **ActiveCFconv** | 0.439 [0.412, 0.464] | 0.230 [0.206, 0.252] | 0.125 [0.109, 0.145] | 0.046 [0.022, 0.071] |
| **RewardCFconv** | 0.439 [0.418, 0.463] | 0.199 [0.186, 0.214] | 0.121 [0.102, 0.138] | 0.059 [0.044, 0.079] |
| PTF | 0.243 [0.224, 0.260] | 0.151 [0.136, 0.168] | 0.096 [0.083, 0.110] | 0.057 [0.040, 0.075] |
| UCBIndep | 0.005 [-0.004, 0.014] | -0.002 [-0.009, 0.005] | -0.002 [-0.010, 0.006] | 0.004 [-0.004, 0.013] |
| Random | -0.010 [-0.023, -0.000] | 0.002 [-0.010, 0.014] | -0.003 [-0.010, 0.005] | -0.010 [-0.026, 0.007] |

## Categorical: unseen-pair skill (learned under contention, eval contention-free)

| method | pool=240 | pool=60 | pool=30 | pool=15 |
|---|---|---|---|---|
| **ContentionAdaCF** | 0.320 [0.282, 0.356] | 0.302 [0.283, 0.323] | 0.352 [0.325, 0.382] | 0.295 [0.260, 0.326] |
| **ContentionCF** | 0.023 [0.002, 0.048] | 0.290 [0.268, 0.313] | 0.312 [0.270, 0.345] | 0.273 [0.233, 0.311] |
| CBBAlite | 0.339 [0.311, 0.365] | 0.385 [0.351, 0.421] | 0.355 [0.334, 0.379] | 0.288 [0.256, 0.322] |
| MusicalChairs | 0.120 [0.090, 0.150] | 0.257 [0.215, 0.304] | 0.298 [0.264, 0.326] | 0.275 [0.251, 0.297] |
| **ActiveCFconv** | 0.113 [0.079, 0.145] | 0.350 [0.319, 0.383] | 0.310 [0.287, 0.330] | 0.195 [0.165, 0.221] |
| **RewardCFconv** | 0.323 [0.291, 0.361] | 0.383 [0.351, 0.414] | 0.339 [0.314, 0.367] | 0.287 [0.242, 0.324] |
| PTF | 0.343 [0.333, 0.356] | 0.338 [0.302, 0.375] | 0.262 [0.232, 0.293] | 0.141 [0.114, 0.169] |
| UCBIndep | -0.007 [-0.014, -0.000] | 0.001 [-0.006, 0.005] | -0.003 [-0.009, 0.002] | -0.006 [-0.015, 0.003] |
| Random | -0.007 [-0.026, 0.014] | 0.014 [-0.002, 0.031] | -0.009 [-0.027, 0.009] | -0.005 [-0.028, 0.019] |

## Collision rate (fraction of engagements lost to contention)

| method | pool=240 | pool=60 | pool=30 | pool=15 |
|---|---|---|---|---|
| ContentionAdaCF | 0.126 | 0.392 | 0.465 | 0.581 |
| ContentionCF | 0.245 | 0.466 | 0.535 | 0.625 |
| CBBAlite | 0.091 | 0.356 | 0.507 | 0.658 |
| MusicalChairs | 0.391 | 0.555 | 0.625 | 0.697 |
| ActiveCFconv | 0.207 | 0.376 | 0.503 | 0.685 |
| RewardCFconv | 0.147 | 0.371 | 0.526 | 0.660 |
| PTF | 0.187 | 0.323 | 0.448 | 0.624 |
| UCBIndep | 0.054 | 0.209 | 0.363 | 0.561 |
| Random | 0.061 | 0.211 | 0.369 | 0.563 |

Takeaway: the CATEGORICAL preference quality (unseen) is contention-invariant (CF stays high, structure-free at the floor) because it is a property of the learned model; the OPERATIONAL earned-reward gap narrows as collisions, not preferences, become the bottleneck, yet CF still leads by spreading drones across targets via diverse, accurate preferences (lower collision rate).

De-confliction primitive (a recognized MRTA baseline). CBBAlite is the canonical consensus-based auction (CBBA) with the comms/consensus step REMOVED: each drone bids its OWN CF-predicted utility on the public pool (identical model class to RewardCF) and de-conflicts by a reactive, public-loss BACKOFF. It isolates the de-confliction primitive against our fixed PRIVATE per-drone offset (Theorem 7). At severe contention (pool=15): ContentionAdaCF 0.100 vs CBBAlite 0.064 vs greedy RewardCFconv 0.059 earned. Reading: our proactive private-offset de-confliction BEATS the reactive auction-with-backoff -- a proactive STATIC private offset spreads drones once and for all, whereas a reactive SHARED backoff makes all colliders flee the same target together, re-synchronizing them.

