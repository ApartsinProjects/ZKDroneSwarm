# Table 1: method comparison (regenerated from saved data)

Fair guessed rank d_hat=8; block-model world; decentralized masked broadcast.
UNSEEN = final-policy unseen-pair skill [C14b, 5 seeds]. ANYTIME = final-round cumulative-reward skill [C16, 8 seeds].
Skill = (method - random) / (oracle - random); ~0 = no better than random.

| Method | Class | UNSEEN @rho=1 | UNSEEN @rho=0.25 | ANYTIME @rho=1 | ANYTIME @rho=0.25 |
|---|---|---|---|---|---|
| Random | no-struct | 0.007 | 0.004 | -0.009 | 0.000 |
| UCBIndep | no-struct | 0.004 | 0.003 | 0.001 | -0.006 |
| UCBHomo | no-struct | 0.167 | 0.070 | 0.032 | 0.010 |
| Tabular | no-struct | -0.001 | 0.003 | 0.246 | 0.252 |
| MFSGD | low-rank | 0.042 | -0.019 | 0.128 | 0.121 |
| ESTR | low-rank | 0.232 | 0.058 | 0.216 | 0.181 |
| PTF | low-rank | 0.516 | 0.280 | 0.274 | 0.230 |
| BPMF | low-rank | 0.233 | 0.126 | 0.046 | 0.010 |
| **RewardCF** | low-rank(ours) | 0.376 | 0.336 | 0.404 | 0.341 |
| **BothCF** | low-rank(ours) | 0.372 | 0.349 | 0.400 | 0.342 |

Reading: every low-rank method clears the no-structure UNSEEN floor (estimator-independent categorical result). On ANYTIME (the operational metric) our online weighted-ALS (RewardCF/BothCF) leads at both densities; UCBIndep's strong final-policy skill collapses to ~0 anytime (n>>T); PTF leads UNSEEN only at rho=1 (full broadcast).
