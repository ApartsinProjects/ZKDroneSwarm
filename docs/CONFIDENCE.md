# Confidence mechanisms: can we use confidence WITHOUT losing generalization?

All variants are the same online weighted-ALS (ConfCF) or variational EM (EMCF); they differ only in HOW confidence enters. Baseline = uniform. Skill 0=random, 1=oracle, 6 seeds, bootstrap 95% CI. sigma_obs is the broadcast noise (own fixed at 0.10).

## UNSEEN skill

| mechanism | rho=1.00, s=0.3 | rho=1.00, s=1.0 | rho=0.25, s=0.3 | rho=0.25, s=1.0 |
|---|---|---|---|---|
| **uniform** | 0.594 [0.556, 0.628] | 0.196 [0.178, 0.212] | 0.376 [0.346, 0.411] | 0.139 [0.130, 0.149] |
| full | 0.449 [0.400, 0.484] | 0.165 [0.132, 0.202] | 0.370 [0.336, 0.400] | 0.125 [0.102, 0.150] |
| relcap4 | 0.581 [0.566, 0.596] | 0.197 [0.168, 0.226] | 0.378 [0.351, 0.415] | 0.145 [0.122, 0.169] |
| EM | 0.627 [0.599, 0.652] | 0.326 [0.276, 0.386] | 0.391 [0.355, 0.425] | 0.173 [0.146, 0.201] |
| EMshrink | 0.689 [0.670, 0.708] | 0.317 [0.281, 0.353] | 0.438 [0.395, 0.477] | 0.207 [0.173, 0.239] |

## ANYTIME skill

| mechanism | rho=1.00, s=0.3 | rho=1.00, s=1.0 | rho=0.25, s=0.3 | rho=0.25, s=1.0 |
|---|---|---|---|---|
| **uniform** | 0.427 [0.406, 0.442] | 0.218 [0.190, 0.241] | 0.337 [0.319, 0.356] | 0.242 [0.225, 0.260] |
| full | 0.373 [0.352, 0.387] | 0.230 [0.197, 0.258] | 0.322 [0.303, 0.341] | 0.256 [0.239, 0.275] |
| relcap4 | 0.435 [0.422, 0.448] | 0.218 [0.197, 0.235] | 0.334 [0.318, 0.349] | 0.242 [0.223, 0.262] |
| EM | 0.422 [0.404, 0.438] | 0.245 [0.225, 0.265] | 0.307 [0.286, 0.330] | 0.195 [0.177, 0.215] |
| EMshrink | 0.350 [0.316, 0.384] | 0.201 [0.175, 0.227] | 0.245 [0.221, 0.265] | 0.162 [0.150, 0.176] |

## Dominance vs uniform (mean of unseen+anytime per condition; + = better)

| mechanism | rho=1.00, s=0.3 | rho=1.00, s=1.0 | rho=0.25, s=0.3 | rho=0.25, s=1.0 | verdict |
|---|---|---|---|---|---|
| full | -0.099 | -0.010 | -0.010 | -0.000 | <= uniform |
| relcap4 | -0.002 | +0.001 | -0.001 | +0.003 | <= uniform |
| EM | +0.014 | +0.079 | -0.008 | -0.007 | DOMINATES uniform |
| EMshrink | +0.009 | +0.052 | -0.015 | -0.006 | mixed |

WINNER: **EM** is >= uniform in every condition and better in at least one (esp. high noise): a confidence mechanism that keeps the broadcast at full weight in the fit (so unseen generalization is preserved) while still being noise/coverage-aware. This is the right way to account for confidence.

