# Headline results (20 seeds, bootstrap 95% CI)

skill = (method - random)/(oracle - random); 0 = random, 1 = oracle. rho = fraction of broadcast observed. Ours in bold.

## UNSEEN-pair skill
| method | rho=1.0 | rho=0.25 |
|---|---|---|
| UCBIndep | 0.000 [-0.004, 0.004] | -0.001 [-0.005, 0.004] |
| PTF | 0.490 [0.458, 0.517] | 0.272 [0.251, 0.293] |
| **RewardCF** | 0.377 [0.355, 0.398] | 0.326 [0.303, 0.349] |
| **HybridCFconv** | 0.488 [0.468, 0.509] | 0.379 [0.354, 0.401] |
| **ActiveCFconv** | 0.476 [0.461, 0.493] | 0.339 [0.317, 0.358] |

## ANYTIME skill
| method | rho=1.0 | rho=0.25 |
|---|---|---|
| UCBIndep | 0.002 [-0.006, 0.010] | -0.004 [-0.010, 0.002] |
| PTF | 0.273 [0.257, 0.290] | 0.226 [0.215, 0.236] |
| **RewardCF** | 0.383 [0.366, 0.399] | 0.342 [0.332, 0.352] |
| **HybridCFconv** | 0.340 [0.327, 0.354] | 0.300 [0.288, 0.311] |
| **ActiveCFconv** | 0.436 [0.419, 0.451] | 0.341 [0.323, 0.357] |

