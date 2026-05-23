# Precision weighting under heterogeneous teammate noise (sanity check)

Matched MEAN broadcast noise; only the HETEROGENEITY varies. homog = every teammate sigma_obs=1.0; hetero = half sigma_obs=0.1 (clean), half=1.9 (noisy). rho=1.0 (full broadcast). Does precision (1/sigma^2) fit weighting beat uniform when sources DIFFER in noise? 8 seeds, bootstrap 95%% CI.

## Unseen-pair skill

| method | homog | hetero |
|---|---|---|
| uniform | 0.235 [0.213, 0.258] | 0.390 [0.367, 0.410] |
| full | 0.187 [0.165, 0.213] | 0.292 [0.266, 0.316] |
| relcap | 0.172 [0.150, 0.197] | 0.483 [0.454, 0.511] |
| (full - uniform) | -0.048 | -0.098 |
| (relcap - uniform) | -0.063 | +0.093 |

## Anytime earned skill

| method | homog | hetero |
|---|---|---|
| uniform | 0.219 [0.200, 0.236] | 0.267 [0.247, 0.285] |
| full | 0.231 [0.219, 0.244] | 0.266 [0.252, 0.281] |
| relcap | 0.223 [0.209, 0.236] | 0.356 [0.337, 0.375] |
| (full - uniform) | +0.012 | -0.001 |
| (relcap - uniform) | +0.004 | +0.089 |

Sanity: precision's edge over uniform should be ~0 or negative in HOMOG (the known result: coverage, not noise, binds; 'full' over-trusts own clean data) and clearly POSITIVE in HETERO (it correctly down-weights the noisy half). A positive HETERO delta confirms the precision machinery works and SCOPES the negative: precision is the right call exactly when observation sources differ in reliability.

