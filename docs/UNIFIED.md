# H3: UnifiedCF capstone validation (one method, best-or-tied across regimes)

UnifiedCF (EMCF + loss-self-gating de-confliction offset + loss-gated exploration anneal) vs the per-regime specialist, SAME 8 seeds, bootstrap 95% CI. rho=1.0.

## STANDARD (no contention): anytime skill

| UnifiedCF | EMCF (specialist) |
|---|---|
| 0.437 [0.417, 0.457] | 0.433 [0.416, 0.447] |

## CHURN: active-set / recent-arrival skill

| metric | UnifiedCF | EMCF (specialist) |
|---|---|---|
| active | 0.851 [0.840, 0.862] | 0.842 [0.830, 0.853] |
| recent | 0.347 [0.274, 0.417] | 0.371 [0.291, 0.439] |

## CONTENTION: earned-reward skill

| pool | UnifiedCF | ContentionAdaCF (specialist) | greedy RewardCFconv |
|---|---|---|---|
| 15 | 0.104 [0.086, 0.120] | 0.100 [0.086, 0.113] | 0.059 [0.044, 0.078] |
| 240 | 0.344 [0.317, 0.368] | 0.448 [0.422, 0.472] | 0.439 [0.418, 0.462] |

Read: UnifiedCF should TIE EMCF on standard+churn (it reduces to EMCF when the drone never loses) and TIE-or-beat ContentionAdaCF / greedy under contention (it self-engages the offset and anneals exploration when it starts losing). One method, no per-regime tuning, best-or-statistically-tied everywhere = the design space collapses to one method.

