# CI report (bootstrap 95%, from saved data; no re-run)

Per-seed bootstrap (10k resamples). Paired difference = ours - baseline per seed;
significant if the 95% CI excludes 0. Data: crossover c15 (unseen, 8 seeds),
anytime c16 (8 seeds). For tighter CIs, E1 re-runs at 20 seeds.

## UNSEEN-pair skill, rho=1.0 (c15)
| method | mean | 95% CI |
|---|---|---|
| HybridCF | 0.415 | [0.381, 0.444] |
| RewardCF | 0.388 | [0.361, 0.414] |
| BothCF | 0.360 | [0.336, 0.381] |
| PTF | 0.505 | [0.477, 0.535] |
| ESTR | 0.226 | [0.193, 0.262] |
| BPMF | 0.229 | [0.213, 0.250] |
| UCBIndep | 0.002 | [-0.002, 0.006] |

_Paired (ours - baseline), unseen, rho=1.0:_
| comparison | mean diff | 95% CI | sig |
|---|---|---|---|
| HybridCF - PTF | -0.091 | [-0.124, -0.050] | yes |
| HybridCF - UCBIndep | +0.412 | [+0.380, +0.441] | yes |
| RewardCF - PTF | -0.117 | [-0.161, -0.070] | yes |
| RewardCF - UCBIndep | +0.386 | [+0.358, +0.413] | yes |

## UNSEEN-pair skill, rho=0.25 (c15)
| method | mean | 95% CI |
|---|---|---|
| HybridCF | 0.375 | [0.353, 0.397] |
| RewardCF | 0.337 | [0.309, 0.377] |
| BothCF | 0.325 | [0.291, 0.367] |
| PTF | 0.293 | [0.259, 0.330] |
| ESTR | 0.054 | [0.043, 0.065] |
| BPMF | 0.128 | [0.113, 0.144] |
| UCBIndep | 0.005 | [-0.002, 0.011] |

_Paired (ours - baseline), unseen, rho=0.25:_
| comparison | mean diff | 95% CI | sig |
|---|---|---|---|
| HybridCF - PTF | +0.082 | [+0.060, +0.104] | yes |
| HybridCF - UCBIndep | +0.370 | [+0.346, +0.394] | yes |
| RewardCF - PTF | +0.044 | [+0.019, +0.067] | yes |
| RewardCF - UCBIndep | +0.332 | [+0.303, +0.373] | yes |

## ANYTIME (final-round cumulative) skill, rho=1.0 (c16)
| method | mean | 95% CI |
|---|---|---|
| RewardCF | 0.404 | [0.389, 0.420] |
| BothCF | 0.400 | [0.384, 0.414] |
| HybridCF | 0.357 | [0.343, 0.369] |
| PTF | 0.274 | [0.252, 0.298] |
| ESTR | 0.216 | [0.195, 0.239] |
| Tabular | 0.246 | [0.237, 0.257] |
| UCBIndep | 0.001 | [-0.010, 0.012] |

_Paired (ours - baseline), anytime, rho=1.0:_
| comparison | mean diff | 95% CI | sig |
|---|---|---|---|
| RewardCF - PTF | +0.130 | [+0.096, +0.164] | yes |
| RewardCF - ESTR | +0.188 | [+0.157, +0.217] | yes |
| RewardCF - Tabular | +0.158 | [+0.147, +0.169] | yes |
| RewardCF - UCBIndep | +0.403 | [+0.382, +0.424] | yes |

## ANYTIME (final-round cumulative) skill, rho=0.25 (c16)
| method | mean | 95% CI |
|---|---|---|
| RewardCF | 0.341 | [0.326, 0.356] |
| BothCF | 0.342 | [0.318, 0.365] |
| HybridCF | 0.336 | [0.326, 0.346] |
| PTF | 0.230 | [0.211, 0.248] |
| ESTR | 0.181 | [0.165, 0.199] |
| Tabular | 0.252 | [0.228, 0.274] |
| UCBIndep | -0.006 | [-0.017, 0.005] |

_Paired (ours - baseline), anytime, rho=0.25:_
| comparison | mean diff | 95% CI | sig |
|---|---|---|---|
| RewardCF - PTF | +0.111 | [+0.094, +0.127] | yes |
| RewardCF - ESTR | +0.160 | [+0.146, +0.176] | yes |
| RewardCF - Tabular | +0.089 | [+0.067, +0.113] | yes |
| RewardCF - UCBIndep | +0.347 | [+0.332, +0.363] | yes |

